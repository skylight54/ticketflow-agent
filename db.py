import sqlite3
from uuid import uuid4
import json
from config import DATA_DIR

DB_PATH = DATA_DIR / "tickets.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL,
                owner TEXT NOT NULL,
                summary TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY NOT NULL,
                messages TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS proposed_actions (
                action_id TEXT PRIMARY KEY NOT NULL,
                session_id TEXT NOT NULL,
                ticket_id TEXT NOT NULL,
                old_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'executed', 'rejected', 'stale'
                    ))
            )
        """)
        conn.commit()
    finally:
        conn.close()

def InsertTicket(ticket_id, status, owner, summary):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO tickets (ticket_id, status, owner, summary)
            VALUES (?, ?, ?, ?)
        """, (ticket_id, status, owner, summary))
        conn.commit()
    finally:
        conn.close()

def SelectTicket(ticket_id):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("""
            SELECT ticket_id, status, owner, summary
            FROM tickets
            WHERE ticket_id = ?
        """, (ticket_id,))
        row = cursor.fetchone()
        if row:
            return {
                "ticket_id": row[0],
                "status": row[1],
                "owner": row[2],
                "summary": row[3]
            }
        else:
            return None
    finally:
        conn.close()

def UpdateTicketStatus(
        ticket_id,
        old_status,
        new_status):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("""
            UPDATE tickets
            SET status = ?
            WHERE ticket_id = ? AND status = ?
        """, (new_status, ticket_id, old_status))
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()

def create_session() -> str:
    """创建空会话，返回 session_id。"""
    session_id = str(uuid4().hex)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO sessions (session_id, messages)
            VALUES (?, ?)
        """, (session_id, "[]"))
        conn.commit()
    finally:
        conn.close()
    return session_id

def load_session(session_id: str) -> dict | None:
    """返回会话，不存在则返回 None。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("""
            SELECT session_id, messages, version
            FROM sessions
            WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            return {
                "session_id": row[0],
                "messages": json.loads(row[1]),
                "version": row[2]
            }
        else:
            return None
    finally:
        conn.close()    

def save_session(
    session_id: str,
    messages: list,
    expected_version: int,
) -> bool:
    """版本一致才保存，成功返回 True。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("""
            UPDATE sessions
            SET messages = ?, version = version + 1
            WHERE session_id = ? AND version = ?
        """, (json.dumps(messages), session_id, expected_version))
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()

def insert_proposed_action(
    session_id: str,
    ticket_id: str,
    old_status: str,
    new_status: str,
    action_id: str | None = None,
) -> str:
    """保存待确认建议，返回生成的 action_id。"""
    if action_id is None:
        action_id = uuid4().hex
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO proposed_actions (
                action_id, session_id, ticket_id, old_status, new_status
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (action_id, session_id, ticket_id, old_status, new_status),
        )
        conn.commit()
    finally:
        conn.close()
    return action_id

def load_proposed_action(action_id: str) -> dict | None:
    """读取建议，不存在返回 None。"""
    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.execute("""
            SELECT
                action_id, session_id, ticket_id,
                old_status, new_status, status
            FROM proposed_actions
            WHERE action_id = ?
        """, (action_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "action_id": row[0],
            "session_id": row[1],
            "ticket_id": row[2],
            "old_status": row[3],
            "new_status": row[4],
            "status": row[5],
        }
    finally:
        conn.close()

def confirm_action_transaction(
    session_id: str,
    action_id: str,
) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN IMMEDIATE")

        cursor = conn.execute("""
            SELECT
                action_id, session_id, ticket_id,
                old_status, new_status, status
            FROM proposed_actions
            WHERE action_id = ? AND session_id = ?
        """,(action_id,session_id)
        )
        row = cursor.fetchone()
        if not row:
            return None
        propose = {
                "action_id": row[0],
                "session_id": row[1],
                "ticket_id": row[2],
                "old_status": row[3],
                "new_status": row[4],
                "status": row[5],
        }
        if propose["status"] != "pending":
            result = propose
        else:
            cursor = conn.execute("""
                UPDATE tickets
                SET status = ?
                WHERE ticket_id = ? AND status = ?
            """, (
                propose["new_status"], propose["ticket_id"], propose["old_status"])
            )
            if cursor.rowcount == 1:
                new_status = "executed"
            elif cursor.rowcount == 0:
                new_status = "stale"
            cursor = conn.execute("""
                UPDATE proposed_actions
                SET status = ?
                WHERE action_id = ? AND session_id = ? AND status = 'pending'
            """,(new_status,propose["action_id"],propose["session_id"]))
            if cursor.rowcount != 1:
                raise RuntimeError("建议状态更新异常")
            propose["status"]=new_status
            result = propose

        # 使用同一个 conn 查询、判断、更新
        # 得到 result，暂不提前 return

        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def reject_action_transaction(
    session_id: str,
    action_id: str,
) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        conn.execute("BEGIN IMMEDIATE")

        cursor = conn.execute("""
            SELECT *
            FROM proposed_actions
            WHERE action_id = ? AND session_id = ?
        """,(action_id,session_id)
        )
        row = cursor.fetchone()
        if not row:
            return None
        propose = {
                "action_id": row[0],
                "session_id": row[1],
                "ticket_id": row[2],
                "old_status": row[3],
                "new_status": row[4],
                "status": row[5],
        }
        if propose["status"] == "pending":
            cursor = conn.execute("""
                UPDATE proposed_actions
                SET status = 'rejected'
                WHERE action_id = ? AND session_id = ? AND status = 'pending'
            """,(action_id,session_id))
            if cursor.rowcount != 1:
                raise RuntimeError("建议状态更新异常")
            propose["status"] = "rejected"
            result = propose
        else:
            result =propose

        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def save_session_turn(
    session_id: str,
    messages: list,
    expected_version: int,
    actions: list,
) -> bool:
    """
    开始事务
        → 按旧版本更新会话
        → 版本不匹配：回滚，返回 False
        → 插入本轮全部建议
        → 全部成功：提交，返回 True
        → 任一步异常：回滚，重新抛出
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute("""
            UPDATE sessions
            SET messages = ?, version = version + 1
            WHERE session_id = ? AND version = ?
        """,(json.dumps(messages, ensure_ascii=False),
             session_id,expected_version))
        if cursor.rowcount != 1:
            conn.rollback()
            return False
        for action in actions:
            if action["session_id"] != session_id:
                raise ValueError("建议不属于当前会话")
            if action["status"] != "pending":
                raise ValueError("只能保存待确认建议")
            cursor = conn.execute("""
                INSERT INTO proposed_actions(
                    action_id, session_id, 
                    ticket_id, old_status,
                    new_status
                )
                VALUES( ? , ? , ? , ? , ?)
            """,(action["action_id"],action["session_id"],
                 action["ticket_id"],action["old_status"],
                 action["new_status"]))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print(f"数据库初始化完成：{DB_PATH}")

