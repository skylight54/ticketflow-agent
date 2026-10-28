from db import load_session, save_session_turn
from agent_graph import run_graph_query as run_query

def chat_session(session_id: str, question: str) -> dict:
    # 1. 校验 session_id、question 非空，去除首尾空白
    # 2. 读取会话；不存在返回 SESSION_NOT_FOUND
    # 3. 将已保存的 messages 传给 run_query
    # 4. 用读取时的 version 尝试保存新历史
    # 5. 保存失败返回 SESSION_CONFLICT
    # 6. 保存成功返回回答、本轮工具记录及新版本
    session_id = session_id.strip()
    question = question.strip()
    if not session_id or not question:
        return {
            "ok":False,
            "data":None,
            "error": {
                "code": "INVALID_ARGUMENT", 
                "message": "session_id 和 question 不能为空"
                }
        }
    session = load_session(session_id)
    if not session:
        return {
            "ok":False,
            "data":None,
            "error": {
                "code": "SESSION_NOT_FOUND", 
                "message": "会话不存在"
                }
        }
    messages = session["messages"]
    version = session["version"]
    result = run_query(question, messages,session_id=session_id)
    new_messages = result["messages"]
    pending_actions = result["pending_actions"]
    saved = save_session_turn(session_id, new_messages, version,pending_actions)
    if not saved:
        return {
            "ok":False,
            "data":None,
            "error": {
                "code": "SESSION_CONFLICT", 
                "message": "会话已被修改，请重试"
                }
        }
    return {
        "ok":True,
        "data": {
            "session_id": session_id,
            "version": session["version"] + 1,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "tool_results": result["tool_results"],
            "actions": result["pending_actions"],
        },
        "error": None,
    }