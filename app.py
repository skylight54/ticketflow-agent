from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
from ticket import get_ticket, create_ticket , update_ticket_status
import os
from urllib.error import HTTPError, URLError
from langgraph.checkpoint.sqlite import SqliteSaver
import logging
from time import perf_counter
from uuid import uuid4

from approval_graph import build_approval_graph
from approval_service import start_approval,resume_approval
from deepseek_demo import AgentResponseError, run_query
from db import create_session, init_db
from session_service import chat_session
from action_service import (
    proposed_status_change,
    confirm_proposed_action,
    reject_proposed_action
)
from config import DATA_DIR

CHECKPOINT_DB_PATH = DATA_DIR / "approval_checkpoints.sqlite"

@asynccontextmanager
async def lifespan(app: FastAPI):
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        app.state.approval_graph = build_approval_graph(saver)
        app.state.approval_lock = Lock()
        yield

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("tickets.api")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "ok": True,
        "data": {
            "service": "ticketflow-agent",
            "status": "alive",
        },
        "error": None,
    }

@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s "
            "status=500 elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            (perf_counter() - started) * 1000,
        )
        response = JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "data": None,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "服务内部错误，请提供请求编号以便排查",
                    "request_id": request_id,
                },
            },
        )
        # 异常路径也要带响应头，调用方才能把正文和日志关联起来
        response.headers["X-Request-ID"] = request_id
        return response

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_completed request_id=%s method=%s path=%s "
        "status=%s elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - started) * 1000,
    )
    return response

class AssistantQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class TicketCreateRequest(BaseModel):
    ticket_id: str = Field(
        ..., min_length=1, description="工单编号", examples=["T102"]
        )
    owner: str = Field(
        ..., min_length=1, description="工单负责人", examples=["技术支持"]
        )
    summary: str = Field(
        ..., min_length=1, description="工单摘要", examples=["用户无法登录"]
        )

class TicketStatusRequest(BaseModel):
    status: str = Field(..., min_length=1)

class ProposedActionRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1)
    new_status: str = Field(..., min_length=1)

class ApprovalDecisionRequest(BaseModel):
    approved: bool = Field(..., strict=True)

def api_error(status_code: int, code: str, message: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "data": None,
            "error": {"code": code, "message": message},
        },
    )

@app.post("/assistant/query")
def query_assistant(request: AssistantQueryRequest):
    question = request.question.strip()

    # 1. 纯空白问题 → 400 INVALID_ARGUMENT
    # 2. 未配置 DEEPSEEK_API_KEY → 503 MODEL_NOT_CONFIGURED
    # 3. 调用 run_query(question)
    # 4. 成功 → 返回统一响应
    # 5. 将网络异常转换为 HTTP 响应
    if not question or question == "":
        return api_error(400, "INVALID_ARGUMENT", "问题不能为空")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return api_error(503, "MODEL_NOT_CONFIGURED", "模型未配置")
    try:
        run_result = run_query(question)
    except AgentResponseError as exc:
        return api_error(502, exc.code, str(exc))
    except HTTPError:
        return api_error(
            502, "MODEL_UPSTREAM_ERROR", "模型服务请求失败"
        )
    except TimeoutError:
        return api_error(
            504, "MODEL_TIMEOUT", "模型请求超时"
        )
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return api_error(
                504, "MODEL_TIMEOUT", "模型请求超时"
            )

        return api_error(
            502, "MODEL_CONNECTION_ERROR", "无法连接模型服务"
        )

    return {
        "ok": True,
        "data": run_result,
        "error": None,
    }


@app.get("/tickets/{ticket_id}")
def read_ticket(ticket_id: str):
    result = get_ticket(ticket_id)

    if result["ok"]:
        return JSONResponse(
            status_code=200,
            content=result
        )

    error_code = result["error"]["code"]
    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(
            status_code=400,
            content=result
        )
    elif error_code == "TICKET_NOT_FOUND":
        return JSONResponse(
            status_code=404,
            content=result
        )

@app.post("/tickets",status_code=201)
def add_ticket(request: TicketCreateRequest):
    result = create_ticket(request.ticket_id, request.owner, request.summary)

    if result["ok"]:
        return JSONResponse(
            status_code=201,
            content=result
        )

    error_code = result["error"]["code"]
    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(
            status_code=400,
            content=result
        )
    elif error_code == "TICKET_ALREADY_EXISTS":
        return JSONResponse(
            status_code=409,
            content=result
        )

@app.patch("/tickets/{ticket_id}/status")
def change_ticket_status(
    ticket_id: str,
    request: TicketStatusRequest
):
    result = update_ticket_status(ticket_id, request.status)

    if result["ok"]:
        return JSONResponse(
            status_code=200,
            content=result
        )

    error_code = result["error"]["code"]
    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(
            status_code=400,
            content=result
        )
    elif error_code == "TICKET_NOT_FOUND":
        return JSONResponse(
            status_code=404,
            content=result
        )
    elif error_code == "INVALID_STATUS_TRANSITION":
        return JSONResponse(
            status_code=409,
            content=result
        )
    elif error_code == "TICKET_CONFLICT":
        return JSONResponse(
            status_code=409,
            content=result
        )


@app.post("/sessions", status_code=201)
def new_session():
    session_id = create_session()

    return {
        "ok": True,
        "data": {
            "session_id": session_id,
            "version": 0,
        },
        "error": None,
    }

@app.post("/sessions/{session_id}/messages")
def send_session_message(
    session_id: str,
    request: AssistantQueryRequest,
    http_request: Request,
):
    # 1. 检查模型密钥，未配置返回 503
    # 2. try 中调用 chat_session(session_id, request.question)
    # 3. 将模型网络异常转换成 502/504
    # 4. 根据业务结果返回 HTTP 响应
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return api_error(503, "MODEL_NOT_CONFIGURED", "模型未配置")

    request_id = http_request.state.request_id

    logger.info(
        "session_turn_started request_id=%s session_id=%s",
        request_id,
        session_id,
    )
    try:
        result = chat_session(session_id, request.question)
    except AgentResponseError as exc:
        logger.warning(
            "session_turn_failed request_id=%s session_id=%s "
            f"error_code={exc.code}",
            request_id,
            session_id,
        )        
        return api_error(502, exc.code, str(exc))
    except HTTPError:
        logger.warning(
            "session_turn_failed request_id=%s session_id=%s "
            "error_code=MODEL_UPSTREAM_ERROR",
            request_id,
            session_id,
        )
        return api_error(502, "MODEL_UPSTREAM_ERROR", "模型服务请求失败")
    except TimeoutError:
        logger.warning(
            "session_turn_failed request_id=%s session_id=%s "
            "error_code=MODEL_TIMEOUT",
            request_id,
            session_id,
        )
        return api_error(504, "MODEL_TIMEOUT", "模型请求超时")
    except URLError as exc:
        is_timeout = isinstance(exc.reason, TimeoutError)
        code = "MODEL_TIMEOUT" if is_timeout else "MODEL_CONNECTION_ERROR"

        logger.warning(
            "session_turn_failed request_id=%s session_id=%s error_code=%s",
            request_id,
            session_id,
            code,
        )

        return api_error(
            504 if is_timeout else 502,
            code,
            "模型请求超时" if is_timeout else "无法连接模型服务",
        )

    data = result.get("data") or {}
    error = result.get("error") or {}

    logger.info(
        "session_turn_completed request_id=%s session_id=%s "
        "ok=%s error_code=%s tool_count=%s action_count=%s",
        request_id,
        session_id,
        result["ok"],
        error.get("code", "-"),
        len(data.get("tool_results", [])),
        len(data.get("actions", [])),
    )
    if result["ok"]:
        return result

    error_code = result["error"]["code"]
    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(status_code=400, content=result)
    if error_code == "SESSION_NOT_FOUND":
        return JSONResponse(status_code=404, content=result)
    if error_code == "SESSION_CONFLICT":
        return JSONResponse(status_code=409, content=result)
    return JSONResponse(status_code=500, content=result)

@app.post("/sessions/{session_id}/actions", status_code=201)
def propose_action(
    session_id: str,
    request: ProposedActionRequest,
):
    result = proposed_status_change(
        session_id,
        request.ticket_id,
        request.new_status,
    )
    # 成功返回 result；失败用 JSONResponse 映射状态码
    if result["ok"]:
        return result

    error_code = result["error"]["code"]

    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(status_code=400,content=result)
    if error_code == "SESSION_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)
    if error_code == "TICKET_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)
    if error_code == "ACTION_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)   
    if error_code == "NO_STATUS_CHANGE":
        return JSONResponse(status_code=409,content=result)
    if error_code == "INVALID_STATUS_TRANSITION":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_STALE":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_REJECTED":
        return JSONResponse(status_code=409,content=result)
    return api_error(500, "INTERNAL_ERROR", "操作处理异常")

@app.post("/sessions/{session_id}/actions/{action_id}/confirm")
def confirm_action(session_id: str, action_id: str):
    result = confirm_proposed_action(session_id, action_id)
    # 成功返回 result；失败用 JSONResponse 映射状态码
    if result["ok"]:
        return result
    
    error_code = result["error"]["code"]

    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(status_code=400,content=result)
    if error_code == "SESSION_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)
    if error_code == "TICKET_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)
    if error_code == "ACTION_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)   
    if error_code == "NO_STATUS_CHANGE":
        return JSONResponse(status_code=409,content=result)
    if error_code == "INVALID_STATUS_TRANSITION":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_STALE":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_REJECTED":
        return JSONResponse(status_code=409,content=result)
    return api_error(500, "INTERNAL_ERROR", "操作处理异常")

@app.post("/sessions/{session_id}/actions/{action_id}/reject",status_code=200)
def reject_action(session_id: str, action_id: str):
    result = reject_proposed_action(session_id,action_id)

    if result["ok"]:
        return result
    error_code = result["error"]["code"]

    if error_code == "INVALID_ARGUMENT":
        return JSONResponse(status_code=400,content=result)
    if error_code == "SESSION_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)
    if error_code == "TICKET_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)
    if error_code == "ACTION_NOT_FOUND":
        return JSONResponse(status_code=404,content=result)   
    if error_code == "NO_STATUS_CHANGE":
        return JSONResponse(status_code=409,content=result)
    if error_code == "INVALID_STATUS_TRANSITION":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_STALE":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_REJECTED":
        return JSONResponse(status_code=409,content=result)
    if error_code == "ACTION_ALREADY_EXECUTED":
        return JSONResponse(status_code=409, content=result)
    return api_error(500, "INTERNAL_ERROR", "操作处理异常")


@app.post("/sessions/{session_id}/actions/{action_id}/approval/start")
def start_action_approval(
    session_id: str,
    action_id: str,
    request: Request,
):
    with request.app.state.approval_lock:
        result = start_approval(
            request.app.state.approval_graph,
            session_id,
            action_id,
        )

    if result["ok"]:
        return result

    code = result["error"]["code"]
    status_code = {
        "INVALID_ARGUMENT": 400,
        "ACTION_NOT_FOUND": 404,
        "APPROVAL_ALREADY_STARTED": 409,
        "ACTION_NOT_PENDING": 409,
        "APPROVAL_START_FAILED": 409,
    }.get(code, 500)

    return JSONResponse(status_code=status_code, content=result)

@app.post("/sessions/{session_id}/actions/{action_id}/approval/resume")
def resume_action_approval(
    session_id: str,
    action_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
):
    with request.app.state.approval_lock:
        result = resume_approval(
            request.app.state.approval_graph,
            session_id,
            action_id,
            body.approved,
        )

    if result["ok"]:
        return result

    code = result["error"]["code"]
    status_code = {
        "INVALID_ARGUMENT": 400,
        "ACTION_NOT_FOUND": 404,
        "APPROVAL_NOT_STARTED": 409,
        "APPROVAL_NOT_WAITING": 409,
        "APPROVAL_THREAD_MISMATCH": 409,
        "ACTION_STALE": 409,
        "ACTION_REJECTED": 409,
        "ACTION_ALREADY_EXECUTED": 409,
    }.get(code, 500)

    return JSONResponse(status_code=status_code, content=result)