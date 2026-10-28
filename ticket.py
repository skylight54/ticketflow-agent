import sqlite3
from db import InsertTicket, SelectTicket , UpdateTicketStatus

def get_ticket(ticket_id: str) -> dict:
    if ticket_id is None or ticket_id.strip() == "":
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "工单编号不能为空"
            }
        }
    ticket = SelectTicket(ticket_id.strip())
    if ticket is None:
        return {
                "ok": False,
                "data": None,
                "error": {
                    "code": "TICKET_NOT_FOUND",
                    "message": "工单编号不存在"
                }
        }
    return {
        "ok": True,
        "data": ticket,
        "error": None
    }

def create_ticket(ticket_id: str, owner: str, summary: str) -> dict:
    # 1. 去除三个字段首尾空白
    # 2. 任一字段为空：返回 INVALID_ARGUMENT
    # 3. 编号已存在：返回 TICKET_ALREADY_EXISTS
    # 4. 保存新工单，状态为“待处理”
    # 5. 返回成功结果
    if ticket_id is None or ticket_id.strip() == "" or owner is None or owner.strip() == "" or summary is None or summary.strip() == "":
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "工单编号、负责人和摘要不能为空"
            }
        }
    try:
        InsertTicket(ticket_id.strip(), "待处理", owner.strip(), summary.strip())
    except sqlite3.IntegrityError:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "TICKET_ALREADY_EXISTS",
                "message": "工单编号已存在"
            }
        }
    return {
        "ok": True,
        "data": {
            "ticket_id": ticket_id.strip(),
            "status": "待处理",
            "owner": owner.strip(),
            "summary": summary.strip()
        },
        "error": None
    }

transitions = {
    "待处理": ["处理中"],
    "处理中": ["已完成", "已取消"],
    "已完成": [],
    "已取消": []
}
def update_ticket_status(ticket_id: str, new_status: str) -> dict:
    # 1. 校验编号和新状态
    # 2. 查询工单，不存在则返回 TICKET_NOT_FOUND
    # 3. 新旧状态相同，直接返回成功
    # 4. 检查是否允许流转
    # 5. 调用数据库更新函数
    # 6. 更新成功，返回最新工单
    if ticket_id is None or ticket_id.strip() == "" or new_status is None or new_status.strip() == "":
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "工单编号和新状态不能为空"
            }
        }
    ticket_id = ticket_id.strip()
    new_status = new_status.strip()
    if new_status not in transitions:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "未知的工单状态",
            },
        }
    ticket = SelectTicket(ticket_id)
    if ticket is None:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "TICKET_NOT_FOUND",
                "message": "工单编号不存在"
            }
        }
    if ticket["status"] == new_status:
        return {
            "ok": True,
            "data": ticket,
            "error": None
        }
    if new_status not in transitions[ticket["status"]]:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_STATUS_TRANSITION",
                "message": f"工单状态不允许从 {ticket['status']} 流转到 {new_status}"
            }
        }
    if not UpdateTicketStatus(ticket_id, ticket["status"], new_status):
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "TICKET_CONFLICT",
                "message": "工单状态更新失败，可能是因为状态已被其他操作修改"
            }
        }
    ticket_updated = SelectTicket(ticket_id)
    return {
        "ok": True,
        "data": ticket_updated,
        "error": None
    }

if __name__ == "__main__":  
    for value in ["T102", "T999", "", "   ", None, "F001"]:
        print(repr(value), get_ticket(value))