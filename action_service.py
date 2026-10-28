from uuid import uuid4
from db import(
    load_session,
    insert_proposed_action,
    load_proposed_action,
    reject_action_transaction,
    confirm_action_transaction
)
from ticket import(
    get_ticket,
    transitions
)


def prepare_status_change(
    session_id: str,
    ticket_id: str,
    new_status: str,
) -> dict:
    session_id = session_id.strip()
    ticket_id = ticket_id.strip()
    new_status = new_status.strip()
    if session_id == "" or ticket_id == "" or new_status == "":
        return {
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"session_id、ticket_id、new_status不能为空"
            }
        }
    if new_status not in transitions:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"new_status状态不存在"
            }
        }
    session = load_session(session_id)
    if not session:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"SESSION_NOT_FOUND",
                "message":"会话不存在"
            }
        }
    ticket = get_ticket(ticket_id)
    if not ticket["ok"]:
        return ticket
    old_status = ticket["data"]["status"]
    if new_status == old_status:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"NO_STATUS_CHANGE",
                "message":"新旧状态相同,不创建建议"
            }
        }
    if new_status not in transitions[old_status]:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_STATUS_TRANSITION",
                "message":"新旧状态不允许流转"
            }
        }
    return {
        "ok": True,
        "data": {
            "action_id": uuid4().hex,
            "session_id": session_id,
            "ticket_id": ticket_id,
            "old_status": old_status,
            "new_status": new_status,
            "status": "pending",
        },
        "error": None,
    }
    ...
def proposed_status_change(
        session_id: str,
        ticket_id: str,
        new_status: str,
)->dict:
    prepared = prepare_status_change(session_id,ticket_id,new_status)
    if not prepared["ok"]:
        return prepared

    action = prepared["data"]

    insert_proposed_action(
        action["session_id"],
        action["ticket_id"],
        action["old_status"],
        action["new_status"],
        action_id=action["action_id"],
    )

    return {
        "ok": True,
        "data": load_proposed_action(action["action_id"]),
        "error": None,
    }

def confirm_proposed_action(
    session_id: str,
    action_id: str,
) -> dict:
    session_id = session_id.strip()
    action_id = action_id.strip()
    if session_id == "" or action_id == "":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"session_id action_id 不能为空"
            }
        }
    result = confirm_action_transaction(session_id,action_id)
    if not result:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_NOT_FOUND",
                "message":f"{session_id}未找到{action_id}建议"
            }
        }
    if result["status"] == "executed":
        action = result
    elif result["status"] == "stale":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_STALE",
                "message":"建议已被别的修改"
            }
        }
    elif result["status"] == "rejected":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_REJECTED",
                "message":"修改被拒绝"
            }
        }
    else:
         raise RuntimeError(
            f"确认后出现非预期建议状态：{result['status']}"
        )
    return{
        "ok":True,
        "data":action,
        "error":None
    }

def reject_proposed_action(
    session_id: str, 
    action_id: str
) -> dict:
    session_id = session_id.strip()
    action_id = action_id.strip()
    if session_id == "" or action_id == "":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"INVALID_ARGUMENT",
                "message":"session_id action_id 不能为空"
            }
        }
    result = reject_action_transaction(session_id,action_id)
    if not result:
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_NOT_FOUND",
                "message":f"{session_id}未找到{action_id}建议"
            }
        }
    if result["status"] == "rejected":
        action = result
    elif result["status"] == "stale":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_STALE",
                "message":"建议已被别的修改"
            }
        }
    elif result["status"] == "executed":
        return{
            "ok":False,
            "data":None,
            "error":{
                "code":"ACTION_ALREADY_EXECUTED",
                "message":"修改已经执行"
            }
        }
    else:
         raise RuntimeError(
            f"确认后出现非预期建议状态：{result['status']}"
        )
 
    return{
        "ok":True,
        "data":action,
        "error":None
    }