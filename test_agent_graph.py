import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import db
from agent_graph import (
    route_after_model,
    model_node,
    tools_node,
    run_graph_query,
    AgentState,
    AgentResponseError,
    MAX_MODEL_ROUNDS
)


def tool_message(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }
class AgentGraphStreamTest(unittest.TestCase):
    def test_normal_tools_stream(self):
        model_message = {
            "role": "assistant",
            "content": "调用工具流程测试",
            "tool_calls": [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "get_ticket",
                        "arguments": '{"ticket_id": "T103"}',
                    },
                }
            ]
        }
        agentstate:AgentState = {
            "messages": [],
            "session_id": "111",
            "model_rounds": 1,
            "tool_results": [],
            "pending_actions": [],
            "last_message": {}
        }
        before = deepcopy(agentstate)
        with patch(
            "agent_graph.send_messages",
            return_value=model_message,
        ) as send:
            result = model_node(agentstate)

        self.assertEqual(agentstate, before)
        self.assertEqual(result["model_rounds"], 2)
        
        send.assert_called_once_with(
            before["messages"],
            allow_tools=True,
            allow_proposals=True,
        )

        # model_node 不修改入参，路由要读取它返回的新状态
        agentstate["last_message"] = result["last_message"]
        agentstate["model_rounds"] = result["model_rounds"]

        route = route_after_model(agentstate)
        self.assertEqual(route,'tools')
        before = deepcopy(agentstate)
        with patch(
            "agent_graph.execute_tool",
            return_value = {
                "ok": True,
                "data": {
                    "ticket_id": "T114514",
                    "status": "处理中",
                    "owner": "row[2]",
                    "summary": "row[3]"
                },
                "error": None
            }
        ) as execute:
            result = tools_node(agentstate)

        # tools_node 同样不修改入参，新增消息在返回的 state 里
        self.assertEqual(len(result["messages"]), len(before["messages"]) + 2)
        self.assertEqual(before["messages"], [])
        self.assertEqual(agentstate, before)
        execute.assert_called_once()
        self.assertEqual(
            json.loads(execute.call_args.args[0]),
            {"tool": "get_ticket", "args": {"ticket_id": "T103"}},
        )
        self.assertEqual(execute.call_args.kwargs["session_id"], "111")
        self.assertEqual(execute.call_args.kwargs["pending_actions"], [])
        self.assertEqual(result["tool_results"][0]["name"], "get_ticket")
        self.assertEqual(result["pending_actions"], [])

    def test_normal_no_tools_stream(self):
        model_message = {
            "role": "assistant",
            "content": "未调用工具流程测试"
        }
        agentstate:AgentState = {
            "messages": [],
            "session_id": "111",
            "model_rounds": 1,
            "tool_results": [],
            "pending_actions": [],
            "last_message": {}
        }
        before = deepcopy(agentstate)
        with patch(
            "agent_graph.send_messages",
            return_value=model_message,
        ) as send:
            result = model_node(agentstate)

        send.assert_called_once()
        self.assertEqual(agentstate, before)
        self.assertEqual(result["model_rounds"], 2)
        agentstate["last_message"] = result["last_message"]
        agentstate["model_rounds"] = result["model_rounds"]
        route = route_after_model(agentstate)
        self.assertEqual(route,'finish')

    def test_up_MAXROUND(self):
        agentstate:AgentState = {
            "messages": [],
            "session_id": "111",
            "model_rounds": 1,
            "tool_results": [],
            "pending_actions": [],
            "last_message": {}
        }
        agentstate["model_rounds"] = MAX_MODEL_ROUNDS

        with patch("agent_graph.send_messages") as send:
            with self.assertRaises(AgentResponseError) as caught:
                model_node(agentstate)

        self.assertEqual(caught.exception.code, "MODEL_STEP_LIMIT")
        send.assert_not_called()
    def test_invalid_model_message(self):
        state: AgentState = {
            "messages": [],
            "session_id": None,
            "model_rounds": 0,
            "tool_results": [],
            "pending_actions": [],
        }

        with patch(
            "agent_graph.send_messages",
            return_value="错误格式",
        ):
            with self.assertRaises(AgentResponseError) as caught:
                model_node(state)

        self.assertEqual(caught.exception.code, "MODEL_PROTOCOL_ERROR")


    def test_invalid_model_content(self):
        state: AgentState = {
            "messages": [],
            "session_id": None,
            "model_rounds": 0,
            "tool_results": [],
            "pending_actions": [],
        }
        with patch(
            "agent_graph.send_messages",
            return_value={"role": "assistant", "content": ["不是字符串"]},
        ):
            with self.assertRaises(AgentResponseError) as caught:
                model_node(state)
        self.assertEqual(caught.exception.code, "MODEL_PROTOCOL_ERROR")

    def test_invalid_json_arguments_skips_execute_tool(self):
        state: AgentState = {
            "messages": [],
            "session_id": "111",
            "model_rounds": 1,          # 必须小于 MAX_MODEL_ROUNDS
            "tool_results": [],
            "pending_actions": [],
            "last_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "get_ticket",
                        "arguments": '{"ticket_id": T103}',   # 非法 JSON
                    },
                }],
            },
        }
        before = deepcopy(state)

        with patch("agent_graph.execute_tool") as execute:
            result = tools_node(state)

        # 1. 非法参数不执行工具
        execute.assert_not_called()

        # 2. 入参未被修改
        self.assertEqual(state, before)

        # 3. 追加 assistant + tool 两条消息
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "assistant")
        self.assertEqual(result["messages"][1]["role"], "tool")
        self.assertEqual(result["messages"][1]["tool_call_id"], "call_bad")

        # 4. 工具记录保留原始参数和错误
        trace = result["tool_results"][0]
        self.assertEqual(trace["name"], "get_ticket")
        self.assertIsNone(trace["arguments"])
        self.assertEqual(trace["raw_arguments"], '{"ticket_id": T103}')
        self.assertEqual(
            trace["result"]["error"]["code"],
            "INVALID_TOOL_ARGUMENTS",
        )

        # 5. 错误确实写进了要回传给模型的 tool 消息
        tool_payload = json.loads(result["messages"][1]["content"])
        self.assertEqual(
            tool_payload["error"]["code"],
            "INVALID_TOOL_ARGUMENTS",
        )
        self.assertEqual(result["pending_actions"], [])


class AgentGraphIntegrationTests(unittest.TestCase):
    """整图测试：真实 graph + 真实工具，只替换模型调用。"""

    def setUp(self):
        self.original_path = db.DB_PATH
        self.directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        db.DB_PATH = Path(self.directory.name) / "test.db"
        db.init_db()
        db.InsertTicket("T109", "待处理", "技术支持", "无法登录")
        self.sid = db.create_session()

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.directory.cleanup()

    def test_graph_tool_call_then_answer(self):
        replies = [
            tool_message("get_ticket", {"ticket_id": "T109"}, "query_1"),
            {
                "role": "assistant",
                "content": "T109 当前待处理，负责人是技术支持。",
            },
        ]
        snapshots = []

        def fake_send(messages, allow_tools=True, allow_proposals=False):
            self.assertTrue(allow_tools)
            self.assertTrue(allow_proposals)
            snapshots.append(deepcopy(messages))
            return replies.pop(0)

        with patch(
            "agent_graph.send_messages",
            side_effect=fake_send,
        ) as send:
            result = run_graph_query("查询 T109", session_id=self.sid)

        self.assertEqual(send.call_count, 2)
        self.assertEqual(
            result["answer"],
            "T109 当前待处理，负责人是技术支持。",
        )
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(result["tool_results"][0]["name"], "get_ticket")
        self.assertTrue(result["tool_results"][0]["result"]["ok"])
        self.assertEqual(result["pending_actions"], [])

        # 第二轮发给模型的消息里带上了工具结果
        self.assertEqual(snapshots[1][-1]["role"], "tool")
        self.assertEqual(snapshots[1][-1]["tool_call_id"], "query_1")
        self.assertEqual(
            json.loads(snapshots[1][-1]["content"])["data"]["ticket_id"],
            "T109",
        )

        # 最终回答只追加一次
        self.assertEqual(result["messages"][-1]["role"], "assistant")
        self.assertEqual(
            result["messages"][-1]["content"],
            "T109 当前待处理，负责人是技术支持。",
        )

    def test_graph_query_then_propose_keeps_action_pending(self):
        replies = [
            tool_message("get_ticket", {"ticket_id": "T109"}, "query_1"),
            tool_message(
                "propose_status_change",
                {"ticket_id": "T109", "new_status": "处理中"},
                "propose_1",
            ),
            {
                "role": "assistant",
                "content": "建议将 T109 改为处理中，请确认。",
            },
        ]
        snapshots = []

        def fake_send(messages, allow_tools=True, allow_proposals=False):
            self.assertTrue(allow_tools)
            self.assertTrue(allow_proposals)
            snapshots.append(deepcopy(messages))
            return replies.pop(0)

        with patch(
            "agent_graph.send_messages",
            side_effect=fake_send,
        ) as send:
            result = run_graph_query(
                "把 T109 改为处理中",
                session_id=self.sid,
            )

        self.assertEqual(send.call_count, 3)
        self.assertEqual(
            [item["name"] for item in result["tool_results"]],
            ["get_ticket", "propose_status_change"],
        )

        self.assertEqual(len(result["pending_actions"]), 1)
        action = result["pending_actions"][0]
        self.assertEqual(action["ticket_id"], "T109")
        self.assertEqual(action["old_status"], "待处理")
        self.assertEqual(action["new_status"], "处理中")
        self.assertEqual(action["status"], "pending")

        # 建议只存在于图状态里，尚未落库；工单本身也没有被修改
        self.assertIsNone(db.load_proposed_action(action["action_id"]))
        ticket = db.SelectTicket("T109")
        self.assertIsNotNone(ticket)
        assert ticket is not None
        self.assertEqual(ticket["status"], "待处理")

        # 第三轮模型看到了建议结果，且 action_id 与返回的建议一致
        propose_result = json.loads(snapshots[2][-1]["content"])
        self.assertTrue(propose_result["ok"])
        self.assertEqual(
            propose_result["data"]["action_id"],
            action["action_id"],
        )



    



        

        