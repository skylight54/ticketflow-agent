import unittest
from unittest.mock import patch

from deepseek_demo import run_query
import json
from copy import deepcopy

class QueryAgentTests(unittest.TestCase):
    def test_missing_id_returns_direct_answer(self):
        model_message = {
            "role": "assistant",
            "content": "请提供工单编号。",
        }

        with patch(
            "deepseek_demo.send_messages",
            return_value=model_message,
        ) as send, patch(
            "deepseek_demo.execute_tool",
        ) as execute:
            result = run_query("帮我查一下工单")

        self.assertEqual(result["answer"], "请提供工单编号。")
        self.assertEqual(result["tool_results"], [])
        self.assertEqual(send.call_count, 1)
        execute.assert_not_called()

    def test_normal_query_with_tool_call(self):
        model_message = {
            "role": "assistant",
            "content": "正在查询工单 T001 的状态。",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_ticket",
                        "arguments": '{"ticket_id": "T001"}',
                    },
                }
            ],
        }
        tool_result = {
            "ok": True,
            "data": {
                "ticket_id": "T001",
                "status": "处理中",
                "owner": "技术支持",
                "summary": "无法登录",
            },
            "error": None,
        }
        sent_messages = []
        responses = [
            model_message,
            {
                "role": "assistant",
                "content": "T001 当前处理中，负责人是技术支持。",
            },
        ]
        def fake_send(messages, allow_tools=True, allow_proposals=False):
            self.assertTrue(allow_tools)
            self.assertFalse(allow_proposals)
            sent_messages.append(deepcopy(messages))
            return responses.pop(0)
        with patch(
            "deepseek_demo.send_messages",
            side_effect=fake_send,
        ) as send, patch(
            "deepseek_demo.execute_tool",
            return_value=tool_result,
        ) as execute:
            result = run_query("帮我查一下工单 T001")

        self.assertEqual(
            result["answer"],
            "T001 当前处理中，负责人是技术支持。",
        )
        self.assertEqual(send.call_count, 2)
        execute.assert_called_once()

        messages = sent_messages[1]

        self.assertEqual(
            messages[-2]["tool_calls"][0]["id"],
            "call_1",
        )
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "call_1")

        self.assertEqual(
            send.call_args_list[1].kwargs["allow_tools"],
            True,
        )
        self.assertEqual(
            json.loads(messages[-1]["content"]),
            tool_result,
        )
    def test_tool_call_with_invalid_json_arguments(self):
        model_message = {
            "role": "assistant",
            "content": "正在查询工单 T002 的状态。",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_ticket",
                        "arguments": '{"ticket_id": T002}',  # Invalid JSON
                    },
                }
            ],
        }
        sent_messages = []
        responses = [
            model_message,
            {
                "role": "assistant",
                "content": "查询参数格式有误，暂时无法查询。",
            },
        ]
        def fake_send(messages, allow_tools=True, allow_proposals=False):
            self.assertTrue(allow_tools)
            self.assertFalse(allow_proposals)
            sent_messages.append(deepcopy(messages))
            return responses.pop(0)
        with patch(
            "deepseek_demo.send_messages",
            side_effect=fake_send,
        ) as send, patch(
            "deepseek_demo.execute_tool",
        ) as execute:
            result = run_query("帮我查一下工单 T002")

        self.assertEqual(result["answer"], "查询参数格式有误，暂时无法查询。")
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(
            result["tool_results"][0]["result"]["error"]["code"],
            "INVALID_TOOL_ARGUMENTS",
        )
        execute.assert_not_called()
        self.assertEqual(send.call_count, 2)

        messages = sent_messages[1]
        self.assertEqual(messages[-1]["role"], "tool")

        sent_result = json.loads(messages[-1]["content"])

        # 非法 JSON 测试
        self.assertEqual(
            sent_result["error"]["code"],
            "INVALID_TOOL_ARGUMENTS",
        )
        self.assertIsNone(result["tool_results"][0]["arguments"])
        
    def test_tool_call_with_invalid_tool_name(self):
        model_message = {
            "role": "assistant",
            "content": "正在查询工单 T003 的状态。",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "unknown_tool",  # Invalid tool name
                        "arguments": '{"ticket_id": "T003"}',
                    },
                }
            ],
        }

        with patch(
            "deepseek_demo.send_messages",
            side_effect=[
                model_message,
                {
                    "role": "assistant",
                    "content": "抱歉，没有可用的查询工具。",
                },
            ],
        ) as send:
            result = run_query("帮我查一下工单 T003")

        self.assertEqual(result["answer"], "抱歉，没有可用的查询工具。")
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(
            result["tool_results"][0]["result"]["error"]["code"],
            "UNKNOWN_TOOL",
        )
        self.assertEqual(send.call_count, 2)
    def test_ticket_not_found(self):
        model_message = {
            "role": "assistant",
            "content": "正在查询工单 T004 的状态。",
            "tool_calls": [
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {
                        "name": "get_ticket",
                        "arguments": '{"ticket_id": "T004"}',
                    },
                }
            ],
        }
        tool_result = {
            "ok": False,
            "data": None,
            "error": {"code": "TICKET_NOT_FOUND", "message": "工单未找到"},
        }

        sent_messages = []
        responses = [
            model_message,
            {
                "role": "assistant",
                "content": "没有找到工单 T004，请核对编号。",
            },
        ]
        def fake_send(messages, allow_tools=True, allow_proposals=False):
            self.assertTrue(allow_tools)
            self.assertFalse(allow_proposals)
            sent_messages.append(deepcopy(messages))
            return responses.pop(0)
        with patch(
            "deepseek_demo.send_messages",
            side_effect=fake_send,
        ) as send, patch(
            "deepseek_demo.execute_tool",
            return_value=tool_result,
        ) as execute:
            result = run_query("帮我查一下工单 T004")

        self.assertEqual(result["answer"], "没有找到工单 T004，请核对编号。")
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(result["tool_results"][0]["result"], tool_result)
        execute.assert_called_once()
        self.assertEqual(send.call_count, 2)

        messages = sent_messages[1]
        self.assertEqual(messages[-1]["role"], "tool")

        sent_result = json.loads(messages[-1]["content"])

        # 工单不存在测试
        self.assertEqual(sent_result, tool_result)
        self.assertEqual(messages[-1]["tool_call_id"], "call_4")
