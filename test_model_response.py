import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch

from deepseek_demo import AgentResponseError, send_messages


def make_payload(finish_reason: str | None = "stop", message=None):
    """构造最小可用的 chat/completions 响应体。"""
    if message is None:
        message = {"role": "assistant", "content": "已完成查询。"}

    choice = {"message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason

    return {"choices": [choice]}


def make_response(payload):
    return BytesIO(json.dumps(payload).encode("utf-8"))


class ModelResponseTests(unittest.TestCase):
    """固定 JSON 代替真实网络：覆盖 send_messages 对上游响应的解析。"""

    def send(self, payload):
        """返回（响应消息，urlopen mock），每个用例只解析一次响应。"""
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}),
            patch(
                "deepseek_demo.urlopen",
                return_value=make_response(payload),
            ) as urlopen,
        ):
            message = send_messages([{"role": "user", "content": "测试"}])

        return message, urlopen

    def test_length_limit_is_rejected(self):
        payload = make_payload(
            "length",
            {"role": "assistant", "content": "尚未输出完整的回答"},
        )

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}),
            patch("deepseek_demo.urlopen", return_value=make_response(payload)),
        ):
            with self.assertRaises(AgentResponseError) as caught:
                send_messages([{"role": "user", "content": "测试"}])

        self.assertEqual(caught.exception.code, "MODEL_OUTPUT_TRUNCATED")

    def test_stop_reason_returns_message(self):
        message = {"role": "assistant", "content": "已完成查询。"}

        result, _ = self.send(make_payload("stop", message))

        self.assertEqual(result, message)

    def test_tool_call_reason_returns_message(self):
        # 工具调用轮次的 finish_reason 是 tool_calls，不能当成截断
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_ticket",
                    "arguments": '{"ticket_id": "T103"}',
                },
            }],
        }

        result, _ = self.send(make_payload("tool_calls", message))

        self.assertEqual(result, message)

    def test_missing_finish_reason_returns_message(self):
        # 有些网关会省略 finish_reason，不能因此报错
        message = {"role": "assistant", "content": "没有 finish_reason。"}

        result, _ = self.send(make_payload(None, message))

        self.assertEqual(result, message)

    def test_unexpected_finish_reason_is_passed_through(self):
        # 目前只拦 length；content_filter 等其它值交给下游协议校验
        message = {"role": "assistant", "content": ""}

        result, _ = self.send(make_payload("content_filter", message))

        self.assertEqual(result, message)

    def test_request_sets_timeout_and_tool_choice(self):
        _, urlopen = self.send(make_payload())

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 60)

        payload = json.loads(request.data)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertTrue(payload["tools"])


if __name__ == "__main__":
    unittest.main()
