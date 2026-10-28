from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from db import load_proposed_action



def start_approval(graph, session_id: str, action_id: str) -> dict:
    session_id = session_id.strip()
    action_id = action_id.strip()
    if session_id == "" or action_id == "":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"session_id action_id不能为空"
            }
        }
    propose = load_proposed_action(action_id)
    if not propose or propose["session_id"]!=session_id:
         return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_NOT_FOUND",
                "message":"action不存在或不属于当前会话"
            }
        } 

    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"approval:{action_id}"
        }
    }
    snapshot = graph.get_state(config)
    # 新 thread 的 snapshot.values 是 {}，不是 None
    if snapshot.values:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"APPROVAL_ALREADY_STARTED",
                "message":"审批已经发起"
            }
        }
    """把 pending 检查放到 snapshot.values 检查之后。
    否则已完成的审批再次发起，会返回 ACTION_NOT_PENDING，
    而不是 APPROVAL_ALREADY_STARTED"""
    if propose["status"] != "pending":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_NOT_PENDING",
                "message":"action并非pending状态"
            }
        }
    try:
        state = graph.invoke(
            {"session_id": session_id, "action_id": action_id},
            config,
        )
    except ValueError as exc:
        # 校验与 invoke 之间建议被并发处理，节点会抛 ValueError
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"APPROVAL_START_FAILED",
                "message":str(exc)
            }
        }
    # 图应当暂停等待确认，否则说明执行结果不符合预期
    interrupts = state.get("__interrupt__")
    if not interrupts:
        raise RuntimeError("审批图未按预期暂停")
    return {
        "ok": True,
        "data": {
            "session_id": session_id,
            "action_id": action_id,
            "thread_id": config["configurable"]["thread_id"],
            "status": "waiting",
            "interrupts": [item.value for item in interrupts],
        },
        "error": None,
    }     
def resume_approval(
    graph,
    session_id: str,
    action_id: str,
    approved: bool,
) -> dict:
    session_id = session_id.strip()
    action_id = action_id.strip()
    if session_id == "" or action_id == "":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"session_id action_id不能为空"
            }
        }
    if not isinstance(approved,bool):    
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"approved不为bool"
            }
        }
    propose = load_proposed_action(action_id)
    if not propose or propose["session_id"]!=session_id:
         return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_NOT_FOUND",
                "message":"action不存在或不属于当前会话"
            }
        } 
    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"approval:{action_id}"
        }
    }
    snapshot = graph.get_state(config)
    # 新 thread 的 snapshot.values 是 {}，不是 None
    if not snapshot.values:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"APPROVAL_NOT_STARTED",
                "message":"审批未发起"
            }
        }
    values = snapshot.values
    if (
        values.get("session_id") != session_id
        or values.get("action_id") != action_id
    ):
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"APPROVAL_THREAD_MISMATCH",
                "message":"审批记录与当前建议不匹配"
            }
        }
    if not snapshot.interrupts:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"APPROVAL_NOT_WAITING",
                "message":"审批不在等待确认状态"
            }
        }
    state = graph.invoke(Command(resume=approved), config)
    return state["result"]
    

