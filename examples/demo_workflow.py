"""演示完整审批流程：创建工单 → 建议 → 审批暂停 → 确认 → 执行。

不调用模型，不需要 API key，但会在目标服务中创建一条演示工单。
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 10


def read_body(raw: bytes):
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def format_body(body) -> str:
    if isinstance(body, (dict, list)):
        return json.dumps(body, ensure_ascii=False)
    return repr(body)


def call(base_url: str, method: str, path: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(base_url + path, data=data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.status, read_body(response.read())
    except HTTPError as exc:
        with exc:
            return exc.code, read_body(exc.read())
    except URLError as exc:
        print(f"[FAIL] 无法连接 {base_url}{path}：{exc.reason}")
        print("  请先启动服务，或用 --base-url 指向正在运行的服务")
        sys.exit(1)


def require(condition: bool, step: str, status: int, body) -> None:
    if condition:
        return
    print(f"[FAIL] {step}")
    print(f"  状态码：{status}")
    print(f"  响应：{format_body(body)}")
    sys.exit(1)


def get_data(body):
    return body.get("data") if isinstance(body, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="演示审批流程：创建演示工单，经人工确认后执行状态变更。",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"服务地址，默认 {DEFAULT_BASE_URL}",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    # 每次运行使用新编号，避免与上次演示数据冲突
    ticket_id = f"DEMO-{uuid4().hex[:8]}"
    print(f"演示工单：{ticket_id}")
    print(f"服务地址：{base_url}")

    # 1. 创建工单
    status, body = call(base_url, "POST", "/tickets", {
        "ticket_id": ticket_id,
        "owner": "技术支持",
        "summary": "演示审批流程（可删除）",
    })
    data = get_data(body)
    require(
        status == 201
        and isinstance(data, dict)
        and data.get("ticket_id") == ticket_id
        and data.get("status") == "待处理",
        "创建工单",
        status,
        body,
    )
    print(f"[PASS] 创建工单 {ticket_id}")

    # 2. 创建会话
    status, body = call(base_url, "POST", "/sessions")
    data = get_data(body)
    session_id = data.get("session_id") if isinstance(data, dict) else None
    require(
        status == 201
        and isinstance(session_id, str)
        and session_id != "",
        "创建会话",
        status,
        body,
    )
    print(f"[PASS] 创建会话 {session_id}")

    # 3. 手动创建“待处理 → 处理中”建议，不调用模型
    status, body = call(
        base_url,
        "POST",
        f"/sessions/{session_id}/actions",
        {"ticket_id": ticket_id, "new_status": "处理中"},
    )
    data = get_data(body)
    action_id = data.get("action_id") if isinstance(data, dict) else None
    require(
        status == 201
        and isinstance(data, dict)
        and isinstance(action_id, str)
        and action_id != ""
        and data.get("old_status") == "待处理"
        and data.get("new_status") == "处理中"
        and data.get("status") == "pending",
        "创建建议",
        status,
        body,
    )
    assert isinstance(action_id, str)
    print(f"[PASS] 创建建议 {action_id[:8]}")

    approval_path = f"/sessions/{session_id}/actions/{action_id}/approval"

    # 4. 发起审批，图应暂停在确认节点
    status, body = call(base_url, "POST", f"{approval_path}/start")
    data = get_data(body)
    require(
        status == 200
        and isinstance(data, dict)
        and data.get("status") == "waiting"
        and data.get("action_id") == action_id,
        "发起审批",
        status,
        body,
    )
    print("[PASS] 发起审批，等待确认")

    # 5. 确认前工单不应被修改
    status, body = call(base_url, "GET", f"/tickets/{ticket_id}")
    data = get_data(body)
    require(
        status == 200
        and isinstance(data, dict)
        and data.get("status") == "待处理",
        "查询确认前工单",
        status,
        body,
    )
    print("[PASS] 确认前工单未修改")

    # 6. 恢复审批并同意
    status, body = call(
        base_url,
        "POST",
        f"{approval_path}/resume",
        {"approved": True},
    )
    data = get_data(body)
    require(
        status == 200
        and isinstance(data, dict)
        and data.get("status") == "executed",
        "恢复审批",
        status,
        body,
    )
    print("[PASS] 恢复审批，建议已执行")

    # 7. 确认后工单已更新
    status, body = call(base_url, "GET", f"/tickets/{ticket_id}")
    data = get_data(body)
    require(
        status == 200
        and isinstance(data, dict)
        and data.get("status") == "处理中",
        "查询确认后工单",
        status,
        body,
    )
    print("[PASS] 确认后工单已更新")

    # 8. 同一审批不能恢复两次
    status, body = call(
        base_url,
        "POST",
        f"{approval_path}/resume",
        {"approved": True},
    )
    error = body.get("error") if isinstance(body, dict) else None
    require(
        status == 409
        and isinstance(error, dict)
        and error.get("code") == "APPROVAL_NOT_WAITING",
        "重复恢复",
        status,
        body,
    )
    print("[PASS] 重复恢复被拒绝（409 APPROVAL_NOT_WAITING）")

    print(f"演示完成：工单 {ticket_id} 状态为处理中，演示数据保留在服务中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
