# TicketFlow Agent · 智能工单助手

[![Tests](https://github.com/skylight54/ticketflow-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/skylight54/ticketflow-agent/actions/workflows/tests.yml)

面向学习与作品展示的后端项目，支持工具调用、人工审批、关键词 RAG、引用校验与容器部署。尚未实现用户认证，请在本地演示环境使用。

基于 FastAPI + SQLite + LangGraph 的工单助手。用户可以用自然语言查询工单，由模型提出状态变更建议，再经人工确认后执行。审批流程可以暂停，并在服务重启后继续。回答处理规范时，助手先检索项目内的知识库，并在回答中标注文档编号；来源列表由程序根据真实检索结果生成。

## 一、功能概览

核心流程：

```text
用户提问
   ↓
Agent 图（LangGraph）
   model ──工具调用──→ tools ──工具结果──→ model
     └──无工具调用──→ finish（校验引用 + 生成 sources）
   ↓
建议写入 proposed_actions（pending）
   ↓
审批图（LangGraph）
   load → wait（interrupt 暂停） → apply
                    ↑
               人工确认 / 拒绝
   ↓
业务事务：条件更新工单 + 更新建议状态
```

设计要点：

- **模型只能提出建议。** 模型可调用的修改工具是 `propose_status_change`，它只校验并生成建议，不直接修改工单。真正执行必须经过独立的人工确认入口。
- **人工确认走审批图。** `interrupt()` 把执行状态写入 LangGraph 检查点，HTTP 请求先返回 `waiting`；之后的 `resume` 请求再根据决定继续执行，因此审批可以跨请求、跨进程重启。
- **执行前重新检查工单状态。** 建议基于创建时的工单状态；审批期间工单可能已被其他操作修改。执行时用 `UPDATE ... WHERE status = old_status` 做条件更新，旧状态不存在则把建议标记为 `stale`，避免执行过期建议。
- **规范回答必须带来源。** `search_knowledge` 检索 `knowledge/` 下的文档；`finish` 节点用 `citation_guard` 校验回答只能引用本轮检索到的文档，并从真实工具结果生成 `sources`。模型无法填写文档标题和文件名。
- **会话历史与图检查点分开存储。** 会话历史保存对话消息，供后续模型调用；图检查点保存审批图的执行状态和中断信息。两者生命周期不同，分别存放在两个 SQLite 文件中。
- **模型输出按失败关闭处理。** 工具调用格式非法、轮次或工具次数超限、输出被 `max_tokens` 截断、最终回答包含 DSML 工具协议标记等情况会抛出 `AgentResponseError`，本轮会话不落库。

## 二、运行机制

### 1. Agent 工具调用图

```text
START → model ──有 tool_calls──→ tools → model
           └──无 tool_calls──→ finish → END
```

- `model`：调用 DeepSeek，校验返回消息的结构和内容类型；最多执行 `MAX_MODEL_ROUNDS = 4` 轮。上游 `finish_reason` 为 `length` 时判定输出被截断，直接抛出 `MODEL_OUTPUT_TRUNCATED`。
- `tools`：校验工具调用字段、调用 ID 唯一性和参数 JSON，执行工具；单轮会话最多累计 `MAX_TOOL_CALLS = 8` 次调用。
- `finish`：要求最终回答为非空字符串，拒绝包含 `<｜｜DSML｜｜` / `<|DSML|` 等工具协议标记的文本；随后调用 `build_sources()` 校验引用并生成来源列表。
- 可用工具：
  - `get_ticket`：查询真实工单。
  - `search_knowledge`：检索项目模拟知识库，返回文档编号、标题、来源文件和正文。
  - `propose_status_change`：创建状态变更建议，仅在带会话上下文时可用。
- 本轮消息（包含工具结果）与待确认建议原子保存；`sources` 从本轮检索结果生成并随会话响应返回，不单独落表。出现协议或引用校验错误时整轮不保存。

### 2. 知识库与引用校验

**文档组织。** `knowledge/` 下每份 Markdown 是一篇规范，头部固定元信息：

```text
# 无法登录处理规范

文档编号：KB-LOGIN-001
适用场景：用户反馈无法登录、登录报错、账号被锁定、收不到验证码
关键词：登录、密码、验证码、账号锁定、认证、SSO
相关文档：KB-STATUS-001、KB-PAYMENT-001
```

当前三份文档：

| 文档编号 | 文件 | 内容 |
|---|---|---|
| KB-LOGIN-001 | `knowledge/login.md` | 无法登录的排查步骤、信息收集与转人工条件 |
| KB-PAYMENT-001 | `knowledge/payment.md` | 支付成功但订单未更新、重复扣款与退款时效 |
| KB-STATUS-001 | `knowledge/ticket_status.md` | 工单状态定义、允许的流转、错误码对照 |

**检索。** `knowledge_search.py` 每次读取目录下全部 Markdown，按 `关键词：` 拆词；问题 `casefold()` 后做子串匹配，关键词命中 3 分、标题整体命中 1 分；零分文档直接丢弃，同分按文档编号排序，取前 `top_k` 篇（工具调用固定 `top_k=2`）。

**引用校验。** `citation_guard.validate_citations()` 只认可**本轮**成功执行的 `search_knowledge` 结果：

- 引用了本轮未检索到的编号 → `INVALID_CITATION`；
- 检索到了文档但回答没有标注来源 → `MISSING_CITATION`；
- 检索失败的结果不算依据；编号大小写不敏感，但必须出现在白名单中。

因此后续追问如果仍需引用资料，必须重新检索，不能沿用上一轮的结果。

**来源生成。** `build_sources()` 在 `finish` 节点调用：先执行校验，再按 `[KB-...]` 在回答中出现的顺序去重生成 `sources`。每项只含 `doc_id`、`title`、`source`，全部取自工具结果；回答没有引用时返回空列表。`sources` 表示**引用来源**，不是本轮检索到的全部候选文档。

### 3. 审批图与检查点

```text
START → load → wait（interrupt） → apply → END
```

- `load`：读取建议，校验建议存在、属于当前会话且状态为 `pending`。
- `wait`：调用 `interrupt()` 暂停，返回确认问题；恢复结果必须是布尔值。
- `apply`：同意时调用确认事务，拒绝时调用拒绝事务。
- 检查点保存在 `approval_checkpoints.sqlite`，线程编号为 `approval:{action_id}`。
- 应用启动时通过 `SqliteSaver` 打开检查点文件；同一进程内用锁串行化发起和恢复操作。

### 4. 并发与一致性

- **会话**：`sessions.version` 做乐观锁。保存本轮消息和建议时按旧版本更新，版本不一致返回 `409 SESSION_CONFLICT`。
- **确认建议**：`BEGIN IMMEDIATE` 事务 + 条件更新工单。工单状态已被修改时返回 `409 ACTION_STALE`。
- **拒绝建议**：只把建议从 `pending` 改为 `rejected`，不修改工单；工单是否被改过不影响拒绝。
- **建议状态机**：`pending → executed / rejected / stale`，之后不可再变。

## 三、接口说明

### 1. 接口列表

| 方法 | 路径 | 用途 | 成功状态码 |
|---|---|---|---|
| GET | `/health` | HTTP 存活检查，不检查数据库或模型依赖 | 200 |
| POST | `/assistant/query` | 无状态查询（不保存会话，不提建议，不返回来源） | 200 |
| GET | `/tickets/{ticket_id}` | 查询工单 | 200 |
| POST | `/tickets` | 创建工单 | 201 |
| PATCH | `/tickets/{ticket_id}/status` | 直接更新工单状态 | 200 |
| POST | `/sessions` | 创建会话 | 201 |
| POST | `/sessions/{session_id}/messages` | 发送消息，由 Agent 回答并可能提出建议 | 200 |
| POST | `/sessions/{session_id}/actions` | 直接创建状态变更建议（不调用模型） | 201 |
| POST | `/sessions/{session_id}/actions/{action_id}/confirm` | 直接确认建议（不使用审批图） | 200 |
| POST | `/sessions/{session_id}/actions/{action_id}/reject` | 直接拒绝建议 | 200 |
| POST | `/sessions/{session_id}/actions/{action_id}/approval/start` | 发起审批并暂停在确认节点 | 200 |
| POST | `/sessions/{session_id}/actions/{action_id}/approval/resume` | 提交审批决定并继续执行 | 200 |

两种确认方式（`confirm/reject` 与 `approval/start + resume`）最终都调用同一套业务事务，状态校验和并发语义一致。区别是后者通过 LangGraph 检查点持久化暂停状态，适合审批需要跨请求或重启的场景。

### 2. 统一响应格式

业务接口统一返回以下结构：

```json
{
  "ok": true,
  "data": {
    "ticket_id": "T102",
    "status": "待处理",
    "owner": "技术支持",
    "summary": "用户无法登录"
  },
  "error": null
}
```

失败时 `ok` 为 `false`，`data` 为 `null`，`error` 中包含 `code` 与 `message`：

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "TICKET_NOT_FOUND",
    "message": "工单编号不存在"
  }
}
```

会话消息接口的 `data` 额外包含 `sources`，只列出回答实际引用的文档：

```json
{
  "ok": true,
  "data": {
    "session_id": "3f2c...",
    "version": 1,
    "answer": "先核对账号是否被锁定……[KB-LOGIN-001]",
    "sources": [
      {
        "doc_id": "KB-LOGIN-001",
        "title": "无法登录处理规范",
        "source": "login.md"
      }
    ],
    "tool_results": [],
    "actions": []
  },
  "error": null
}
```

`/assistant/query` 走无状态的 `run_query()`，不经过 Agent 图，因此不返回 `sources`，也不做引用校验。

> 请求体缺少必填字段、类型不合法时，由 FastAPI 返回 `422`，响应体为框架默认的 `{"detail": [...]}` 格式，不使用上述统一结构。例如 `approved` 字段使用严格布尔类型，传入 `1` 也会返回 `422`。

### 3. 错误码

| HTTP 状态码 | 错误码 | 说明 |
|---|---|---|
| 400 | `INVALID_ARGUMENT` | 参数为空、状态值未知或参数不合法 |
| 404 | `TICKET_NOT_FOUND` | 工单编号不存在 |
| 404 | `SESSION_NOT_FOUND` | 会话不存在 |
| 404 | `ACTION_NOT_FOUND` | 建议不存在，或不属于当前会话 |
| 409 | `TICKET_ALREADY_EXISTS` | 创建时工单编号已存在 |
| 409 | `INVALID_STATUS_TRANSITION` | 不允许的工单状态流转 |
| 409 | `TICKET_CONFLICT` | 更新工单时状态已被其他操作修改 |
| 409 | `NO_STATUS_CHANGE` | 目标状态与当前状态相同，不创建建议 |
| 409 | `SESSION_CONFLICT` | 会话版本冲突，需要重试 |
| 409 | `ACTION_STALE` | 审批期间工单旧状态已不存在，建议作废 |
| 409 | `ACTION_REJECTED` | 建议已被拒绝，不能再确认 |
| 409 | `ACTION_ALREADY_EXECUTED` | 建议已经执行，不能再拒绝 |
| 409 | `APPROVAL_ALREADY_STARTED` | 该建议的审批已经发起 |
| 409 | `ACTION_NOT_PENDING` | 建议不是 `pending` 状态，无法发起审批 |
| 409 | `APPROVAL_START_FAILED` | 发起审批时校验失败（如建议被并发处理） |
| 409 | `APPROVAL_NOT_STARTED` | 恢复前没有找到对应审批 |
| 409 | `APPROVAL_NOT_WAITING` | 审批不在等待确认状态（已完成或已恢复过） |
| 409 | `APPROVAL_THREAD_MISMATCH` | 检查点中的建议与请求不一致 |
| 502 | `MODEL_PROTOCOL_ERROR` | 模型返回内容不符合工具协议 |
| 502 | `MODEL_STEP_LIMIT` | 模型轮次或工具调用次数超限 |
| 502 | `MODEL_OUTPUT_TRUNCATED` | 模型输出达到 `max_tokens` 上限被截断 |
| 502 | `INVALID_CITATION` | 回答引用了本轮未检索到的文档 |
| 502 | `MISSING_CITATION` | 检索到文档但回答没有标注来源 |
| 502 | `MODEL_UPSTREAM_ERROR` / `MODEL_CONNECTION_ERROR` | 模型服务调用失败 |
| 503 | `MODEL_NOT_CONFIGURED` | 未配置 `DEEPSEEK_API_KEY` |
| 504 | `MODEL_TIMEOUT` | 模型请求超时 |

工具执行中的错误（如 `INVALID_TOOL_ARGUMENTS`、`UNKNOWN_TOOL`）不会直接变成 HTTP 错误，而是作为工具结果返回给模型，由模型决定如何继续。

### 4. 工单状态机

| 当前状态 | 允许流转到 |
|---|---|
| 待处理 | 处理中 |
| 处理中 | 已完成、已取消 |
| 已完成 | 无（终态） |
| 已取消 | 无（终态） |

- `PATCH /tickets/{id}/status` 更新为与当前相同的状态视为幂等操作，直接返回成功。
- 创建建议时目标状态与当前状态相同返回 `NO_STATUS_CHANGE`；不允许的流转返回 `INVALID_STATUS_TRANSITION`。
- 传入状态机之外的状态值返回 `400 INVALID_ARGUMENT`。

### 5. 完整调用示例

以下示例使用 Bash 语法（Windows 可在 Git Bash 执行）；PowerShell 用户也可通过 Swagger UI 操作。用环境变量保存创建时返回的 ID，工单编号需选择尚未存在的新编号。容器服务请把 `BASE` 改为 `http://127.0.0.1:8001`。

```bash
BASE=http://127.0.0.1:8000

# 1. 创建工单
curl -X POST "$BASE/tickets" \
  -H 'Content-Type: application/json' \
  -d '{
    "ticket_id": "T106",
    "owner": "技术支持",
    "summary": "用户无法登录"
  }'

# 2. 创建会话
SESSION_ID=$(curl -s -X POST "$BASE/sessions" | python -c 'import json,sys; print(json.load(sys.stdin)["data"]["session_id"])')
```

提出建议有两种方式，二选一：

```bash
# 3A. 通过模型提出建议（需要 DEEPSEEK_API_KEY）
curl -X POST "$BASE/sessions/$SESSION_ID/messages" \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "请为工单T106提出从待处理改为处理中的建议，等待我确认。"
  }'

# 3B. 不调用模型，直接创建建议
ACTION_ID=$(curl -s -X POST "$BASE/sessions/$SESSION_ID/actions" \
  -H 'Content-Type: application/json' \
  -d '{
    "ticket_id": "T106",
    "new_status": "处理中"
  }' | python -c 'import json,sys; print(json.load(sys.stdin)["data"]["action_id"])')
```

方式 3A 的 `ACTION_ID` 从响应里的 `data.actions[0].action_id` 取得。问规范类问题时，同一响应里的 `data.answer` 会带 `[KB-...]` 标注，`data.sources` 列出对应文档。

```bash
# 4. 发起审批，返回 status=waiting 和确认问题
curl -X POST "$BASE/sessions/$SESSION_ID/actions/$ACTION_ID/approval/start"

# 5. 同意审批（拒绝时传 false）
curl -X POST "$BASE/sessions/$SESSION_ID/actions/$ACTION_ID/approval/resume" \
  -H 'Content-Type: application/json' \
  -d '{"approved": true}'

# 6. 查询更新后的工单
curl "$BASE/tickets/T106"
```

## 四、部署与运行

### 环境要求

- Python 3.12（已在 3.12.4 上验证）
- 依赖版本见 `requirements.txt`

### 1. 创建并激活虚拟环境

建议为项目创建独立的虚拟环境，避免与系统 Python 的依赖互相影响。在项目根目录执行：

```bash
python -m venv .venv
```

按所用终端选择一种方式激活：

```bash
# Git Bash
source .venv/Scripts/activate

# CMD
.venv\Scripts\activate.bat
```

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

激活成功后，终端提示符前会出现 `(.venv)`，此后命令中的 `python` 和 `pip` 均指向虚拟环境内的解释器。

- 不想激活时，可直接使用 `.venv\Scripts\python.exe` 和 `.venv\Scripts\pip.exe` 执行命令。
- 退出虚拟环境：执行 `deactivate`。
- `.venv/` 已在 `.gitignore` 中忽略，无需提交到版本库。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

模型参数和数据目录在 `config.py` 中读取，并在进程启动时求值；模型密钥由请求处理代码读取：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 无 | 模型密钥；未配置时模型相关接口返回 `503` |
| `DEEPSEEK_MODEL` | `deepseek-flash` | 模型名称 |
| `MODEL_TIMEOUT_SECONDS` | `60` | 单次模型请求超时（秒），必须大于 0 |
| `MODEL_MAX_TOKENS` | `1200` | 模型单次输出上限，必须大于 0；超过会返回 `MODEL_OUTPUT_TRUNCATED` |
| `TICKETFLOW_DATA_DIR` | 项目根目录 | 数据库和检查点文件的存放目录，不存在时自动创建 |

```bash
# Git Bash
export DEEPSEEK_API_KEY="sk-..."
export TICKETFLOW_DATA_DIR="./data"
```

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:TICKETFLOW_DATA_DIR = ".\data"
```

未配置密钥时，`/assistant/query` 和 `/sessions/{session_id}/messages` 返回 `503 MODEL_NOT_CONFIGURED`；工单、建议、审批和知识库检索工具不依赖该密钥。

### 4. 初始化数据库

```bash
python db.py
```

在数据目录创建 `tickets.db` 并建表；重复执行不会清空已有数据。应用启动时也会执行同样的初始化，因此这一步可以省略。数据目录不存在时会自动创建。

审批图检查点文件 `approval_checkpoints.sqlite` 会在应用启动时自动创建，位置同样在数据目录下。

相对数据目录按项目根目录解析，不依赖启动命令所在目录。切换数据目录会使用另一个数据库，不会自动迁移原数据。本地 Python 启动不会自动加载 `.env`，请在终端设置环境变量；下文 Compose 通过 `env_file` 加载它。

### 5. 启动服务

```bash
python -m uvicorn app:app --reload
```

`--reload` 仅用于开发环境，生产环境请去掉该参数。当前审批锁是进程内的，建议以单 worker 模式运行。

### 6. 演示完整审批流程

服务启动后，另开一个终端执行：

```powershell
.\.venv\Scripts\python.exe -m examples.demo_workflow --base-url http://127.0.0.1:8001
```

`--base-url` 默认是 `http://127.0.0.1:8000`，请指向实际运行的服务。脚本每次用 `uuid4().hex[:8]` 生成新的工单编号，依次验证：

1. 创建工单，检查 `201` 和初始状态 `待处理`；
2. 创建会话，取得 `session_id`；
3. 手动创建“待处理 → 处理中”建议（不调用模型），检查建议为 `pending`；
4. 发起审批，检查返回 `waiting`；
5. 查询工单，检查仍为 `待处理`；
6. 恢复审批（`approved: true`），检查建议为 `executed`；
7. 查询工单，检查已变为 `处理中`；
8. 再次恢复，检查返回 `409 APPROVAL_NOT_WAITING`。

每一步都检查状态码和关键字段，通过时打印 `[PASS]`，例如：

```text
[PASS] 创建工单 DEMO-b95a654b
[PASS] 发起审批，等待确认
[PASS] 确认前工单未修改
[PASS] 确认后工单已更新
[PASS] 重复恢复被拒绝（409 APPROVAL_NOT_WAITING）
```

任何一步不符合预期，脚本会打印该步的状态码和完整响应，并以非零退出码结束，不会在未校验的情况下输出 `PASS`。脚本不调用模型、不需要 `DEEPSEEK_API_KEY`，但会在目标服务中留下一条 `DEMO-xxxxxxxx` 演示工单；当前没有删除接口，演示数据会保留。该脚本覆盖的是确定性的业务与审批流程，RAG 回答和真实模型表现仍由第五节评测脚本验证。

### 7. 接口文档

- Swagger UI：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>

### 8. 运行测试

```bash
python -m unittest discover -s tests -t . -v
```

测试使用临时数据库和临时检查点文件，不会修改 `tickets.db` 或 `approval_checkpoints.sqlite`；模型调用、上游响应和检查点恢复使用 mock 或子进程隔离，不需要真实 API key，也不产生模型费用。

### 9. 停止服务

在启动服务的终端中按 `Ctrl+C`。

### 10. Docker Compose 启动（推荐演示方式）

前提：Docker Desktop 的 Linux 引擎或 Linux Docker Engine 正常运行，且 `docker compose version` 可用。在仓库根目录执行：

```powershell
Copy-Item .env.example .env
docker volume create ticketflow-data
docker compose config --quiet
docker compose up -d --build --wait --wait-timeout 60
docker compose ps
```

Linux / Git Bash 将第一条改为 `cp .env.example .env`。已有 `.env` 时不要再次复制覆盖。需要模型问答时，在本地 `.env` 填写 `DEEPSEEK_API_KEY`；留空仍可使用健康检查、工单 API 和不调用模型的审批接口。真实密钥不应放进 `.env.example`、Dockerfile 或 Git；`.env` 已由 `.gitignore` 和 `.dockerignore` 排除。

Compose 固定使用单 worker，端口只绑定本机；容器数据目录为 `/data`，挂载已有命名卷 `ticketflow-data`。该卷配置为 `external: true`，首次运行必须创建，后续可重复使用。

- 健康检查：[http://127.0.0.1:8001/health](http://127.0.0.1:8001/health)
- 接口文档：[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
- 日志：`docker compose logs --tail 100 api`
- 更新代码后重新构建并启动：`docker compose up -d --build --wait`
- 停止并删除本项目容器：`docker compose down`（外部数据卷保留）

若之前通过 `docker run` 启动了 `ticketflow-agent-dev`，先停止并删除这个旧容器，释放 8001 端口，再启动 Compose。不要删除数据卷。镜像构建成功不代表服务已启动，以 `docker compose ps` 的状态和 HTTP 检查结果为准。

### 11. 数据持久化与审批恢复演示

1. 在 8001 的 Swagger UI 创建新工单、会话和状态变更建议；保存返回的 `session_id` 和 `action_id`。
2. 调用 `/approval/start`，确认返回 `waiting`，工单仍处于原状态。
3. 执行 `docker compose down`，再执行 `docker compose up -d --wait`，使用同一个数据卷。
4. 用原 ID 调用 `/approval/resume`，请求体为 `{"approved": true}`。
5. 检查建议为 `executed`，并通过查询接口验证工单状态改变；再次恢复应返回 `APPROVAL_NOT_WAITING`。

该演示不需要模型密钥，可以通过 `/sessions/{session_id}/actions` 手动创建建议。其中不涉及容器重启的部分（创建工单、建议、发起审批到确认执行）可以直接用 `python -m examples.demo_workflow --base-url http://127.0.0.1:8001` 自动跑完；重启后的恢复仍需按上述步骤手工验证。用户已完成同卷删除、重建容器后的审批恢复验证；持久化依赖业务库与检查点库都保存在同一个卷中。数据卷不能替代备份，丢失卷仍会丢失运行数据。

### 12. 请求日志与 CI

每个请求由中间件生成 `X-Request-ID`，日志记录方法、路径、状态码和耗时。会话日志使用相同编号关联开始、业务结果或模型异常类别。未捕获异常返回 `500 INTERNAL_ERROR`，正文包含请求编号，详细异常留在服务日志中。`/health` 只检查 HTTP 存活，不验证模型可用性或数据库读写能力。

`.github/workflows/tests.yml` 在 push、PR 或手动触发时执行：

1. `test`：Python 3.12 安装依赖并运行 `python -m unittest discover -s tests -t . -p "test_*.py" -v`。
2. `docker-smoke`：前一任务通过后，从 `.env.example` 创建空密钥配置与数据卷，构建镜像、等待健康检查并请求 `/health`；无论成功失败均尝试输出日志、停止服务。

CI 不运行真实模型评测，也不需要真实模型密钥。Docker 冒烟测试验证镜像构建和 HTTP 启动，不等于完整业务或真实模型测试；以仓库 Actions 中对应提交的运行结果为准。

## 五、评测

### 1. Agent 任务评测

```bash
python -m evaluations.eval_agent
```

需要先设置 `DEEPSEEK_API_KEY`。脚本读取 `evaluations/cases/eval_cases.json`，逐条调用 Agent，结果写入 `evaluations/results/eval_results.json`（已在 `.gitignore` 中忽略），并在终端打印每条用例的耗时、失败原因和汇总指标。

当前共 5 条用例：

| 用例 | 输入 | 期望行为 |
|---|---|---|
| `existing_ticket` | 查询 T103 的状态和负责人 | 调用 `get_ticket`，回答包含工单号、状态、负责人 |
| `missing_ticket` | 查询不存在的 NOT_EXIST_001 | 调用 `get_ticket`，返回 `TICKET_NOT_FOUND` |
| `missing_id` | “帮我查询工单” | 不调用工具，反问工单编号 |
| `indirect_query` | “T103 现在归谁负责？进展到哪一步了？” | 调用 `get_ticket`，回答包含工单号、状态、负责人 |
| `missing_id_with_urgency` | “急，帮我看看那个一直没解决的工单！” | 不调用工具，反问工单编号 |

评测不读取业务库。`evaluation_database()` 会记录当前 `db.DB_PATH`，创建临时目录和 `eval.db` 并执行 `init_db()`，插入固定工单 T103（状态 `待处理`、负责人 `技术支持`、摘要 `用户无法登录`），结束后恢复路径并删除临时目录。因此评测不会修改 `tickets.db`，相同用例每次面对的数据一致。

本次结果（取自 `evaluations/results/eval_results.json`，保留 3 位小数）：

| 指标 | 数值 |
|---|---|
| 通过率 | 5/5（100%） |
| 平均耗时 | 2.072s |
| 最大耗时 | 2.591s |

`score_case()` 做的是自动可验证的检查：

- **工具选择**：`expected_tool` 是否匹配；为 `null` 时不应调用任何工具。
- **调用次数与参数**：预期恰好一次工具调用，工具名和 `ticket_id` 必须匹配。
- **工具结果**：成功时 `ok` 必须为 `true` 且无 `error`；失败时错误码必须等于 `expected_error`。
- **回答字段覆盖**：`expected_answer_fields` 中的字段值必须出现在最终回答里。

局限：样本只有 5 条；字段覆盖是字符串包含判断，不能证明语义正确；当前只覆盖查询路径，没有覆盖建议、审批和多轮对话。

### 2. 知识检索评测

```bash
python -m evaluations.eval_retrieval
```

不需要 API key，也不调用模型。脚本读取 `evaluations/cases/retrieval_cases.json`，逐条调用 `search_knowledge(question, top_k=1)`，结果写入 `evaluations/results/eval_retrieval_results.json`（已忽略）。

判定只看**第一条结果**：`expected_doc_id` 为 `null` 的用例要求返回空；否则第一条的文档编号必须与预期相同。只取 top-1 是为了避免“把三篇都返回也算通过”，让评测真正衡量排序质量。

| 用例 | 问题 | 预期文档 |
|---|---|---|
| 登录排查 | 用户一直登录失败，该怎么排查？ | `KB-LOGIN-001` |
| 支付订单 | 已经扣款，但订单还显示未支付怎么办？ | `KB-PAYMENT-001` |
| 状态流转 | 待处理工单能直接变成已完成吗？ | `KB-STATUS-001` |
| 知识库外问题 | 公司今年年假可以休几天？ | 无（返回空） |
| 大小写 | `sso` | `KB-LOGIN-001` |

本次结果：5/5（100%）。局限：用例关键词比较直给，检索是子串匹配且没有分词，真实用户使用同义词或口语化表达（例如“钱扣了单子还是待支付”）时可能漏检；这批用例还不能说明泛化效果。

### 3. RAG 回答评测

```bash
python -m evaluations.eval_rag
```

需要 `DEEPSEEK_API_KEY`。脚本读取 `evaluations/cases/rag_cases.json`，在独立评测库中逐条调用 `run_graph_query()`（不传历史），记录回答、来源、工具记录和耗时，结果写入 `evaluations/results/eval_rag_results.json`（已忽略），并打印逐条结果与汇总。用例来自知识库已经覆盖的三个场景，例如登录排查、状态规则和知识库之外的“年假天数”。

`score_rag_case()` 检查五项：

1. 回答是非空字符串；
2. 至少调用一次 `search_knowledge` 且执行成功；
3. `sources` 的文档编号集合与 `expected_source_ids` 完全相同；
4. `pending_actions` 为空，问规范不应产生修改建议；
5. 没有调用 `propose_status_change`。

第三条用集合相等，多给一篇未被引用的来源也算失败——`sources` 是引用来源，不是检索候选。

注意：`expected_source_ids` 为空的用例只能证明回答没有标注来源，**不能证明模型没有编造内容**。例如年假问题，模型可能编出天数却不标来源，评分仍会通过。运行后必须人工检查这类用例的 `answer`，确认它明确说明“知识库暂无依据”。

本次结果（取自 `evaluations/results/eval_rag_results.json`，保留 3 位小数）：

| 用例 | 结果 | 来源 | 说明 |
|---|---|---|---|
| `status_rule` | 通过 | `KB-STATUS-001` | 回答引用状态规则文档并说明流转限制 |
| `unknown_policy` | 通过 | 无 | 人工复核：回答明确说明知识库暂无依据，未编造年假天数 |
| `login_guide` | 失败 | 无 | 检索到了 `KB-LOGIN-001`，但回答没有标注来源，被 `MISSING_CITATION` 拒绝，本轮未提交 |

通过率 2/3（66.7%），平均耗时 3.670s，最大 4.857s。

`login_guide` 的失败不是评分误判，而是引用守卫按设计工作：检索到文档却没有标注来源时，即使回答内容可能正确，也不能提交给用户。这一条也说明「要求模型遵守引用格式」在真实调用中并不稳定，需要人工复核失败原因，而不是放宽校验。

### 4. 评测局限

- 三个评测脚本的样本量都小（5 / 5 / 3 条），通过率只能说明固定场景可用。
- 自动评分只覆盖可机械判断的部分：工具名、参数、错误码、字段和来源集合；回答是否编造、推理是否合理仍需人工复核。
- 耗时受模型响应和网络波动影响，重复运行不会完全一致。

## 六、项目文件

```text
ticketflow-agent/
├── app.py                    # HTTP 服务入口
├── *_service.py              # 业务服务
├── agent_graph.py / approval_graph.py
├── db.py / config.py / ticket.py
├── deepseek_demo.py          # 运行时模型调用模块，业务代码仍使用它
├── knowledge_search.py / knowledge_chunks.py / citation_guard.py
├── knowledge/               # 规范原文
├── tests/                   # 自动化测试（test_*.py）
├── evaluations/             # 评测脚本
│   ├── cases/               # 人工标注的 JSON 用例，纳入版本控制
│   └── results/             # 本地评测输出，不提交
├── examples/                # 学习实验与端到端演示
└── .github/workflows/       # CI
```

以下命令均在项目根目录运行。子目录是 Python 包，使用 `-m` 启动，无需手动修改 `sys.path` 或设置 `PYTHONPATH`：

```bash
# 全部自动化测试；测试使用临时数据库和模型桩，不调用真实模型
python -m unittest discover -s tests -t . -v
# 单个测试模块
python -m unittest tests.test_approval_graph -v
# 关键词检索评测（不需要模型）
python -m evaluations.eval_retrieval
# 指定用例和输出；相对路径以当前工作目录为基准
python -m evaluations.eval_retrieval --cases evaluations/cases/retrieval_challenge_cases.json --output evaluations/results/challenge_expanded.json --expand
# 学习示例
python -m examples.reducer_demo
python -m examples.demo_workflow --base-url http://127.0.0.1:8001
```

向量检索仍是独立实验，需要 `.venv-retrieval` 环境中的 `sentence-transformers` 和本地模型缓存；尚未接入 HTTP 服务，也不加入生产镜像：

```powershell
.\.venv-retrieval\Scripts\python.exe -m examples.check_chunk_tokens
.\.venv-retrieval\Scripts\python.exe -m examples.embedding_search_demo
.\.venv-retrieval\Scripts\python.exe -m evaluations.eval_embedding --offline --rerank --output evaluations/results/embedding_reranked.json
```

评测默认从 `evaluations/cases/` 读取用例，向 `evaluations/results/` 写入结果，输出目录会自动创建。`evaluations.eval_agent` 和 `evaluations.eval_rag` 仍需要真实模型密钥。原有本地报告已移入结果目录，业务数据库、检查点和环境配置的位置不受本次整理影响。

| 文件 | 职责 |
|---|---|
| `app.py` | FastAPI 应用：请求模型、路由、HTTP 状态码映射、审批图生命周期、请求日志 |
| `config.py` | 模型名、超时、输出上限和数据目录配置 |
| `Dockerfile` / `.dockerignore` | 构建 Python 3.12 镜像并排除本机运行数据 |
| `compose.yaml` | 单 worker 服务、8001 端口、健康检查和外部数据卷 |
| `.env.example` | 不含真实密钥的容器环境变量模板 |
| `.github/workflows/tests.yml` | 自动化测试与 Docker 启动冒烟测试 |
| `ticket.py` | 工单查询、创建、状态机校验和直接状态更新 |
| `action_service.py` | 变更建议的创建、确认、拒绝及业务错误处理 |
| `session_service.py` | 会话轮次：读取历史、调用 Agent、按版本保存结果（含来源） |
| `approval_service.py` | 发起/恢复审批，校验检查点与请求的一致性 |
| `agent_graph.py` | LangGraph 版 Agent 图（model / tools / finish） |
| `approval_graph.py` | LangGraph 版审批图（load / wait / apply） |
| `deepseek_demo.py` | DeepSeek HTTP 调用、工具协议、无状态 `run_query` |
| `tool_runner.py` | 工具注册、参数校验和执行分派 |
| `knowledge_search.py` | 读取 `knowledge/` 并做关键词检索 |
| `citation_guard.py` | 引用校验与 `sources` 生成 |
| `knowledge/` | 三份模拟规范文档（登录、支付、工单状态） |
| `db.py` | SQLite 建表与存取、会话版本更新、建议事务 |
| `examples/checkpoint_demo.py` | 审批暂停/恢复的最小示例，支持跨进程恢复 |
| `examples/graph_demo.py` | LangGraph 条件边示例 |
| `examples/reducer_demo.py` | LangGraph 状态覆盖与 `Annotated` 累加示例 |
| `examples/demo_workflow.py` | 端到端演示脚本：创建工单和建议，走审批暂停/恢复并逐项校验 |
| `evaluations/eval_agent.py` | Agent 任务评测脚本（需要 API key） |
| `evaluations/cases/eval_cases.json` | Agent 评测用例定义 |
| `evaluations/results/eval_results.json` | Agent 评测输出（运行时生成，已忽略） |
| `evaluations/eval_retrieval.py` | 知识检索评测脚本（不需要 API key） |
| `evaluations/cases/retrieval_cases.json` | 检索评测用例定义 |
| `evaluations/results/eval_retrieval_results.json` | 检索评测输出（运行时生成，已忽略） |
| `evaluations/eval_rag.py` | RAG 回答评测脚本（需要 API key） |
| `evaluations/eval_embedding.py` | 向量检索、可选重排序及文档/章节/候选命中评测 |
| `examples/embedding_demo.py` / `examples/embedding_search_demo.py` / `examples/rerank_demo.py` | 向量编码、知识库检索和重排序学习实验 |
| `examples/check_chunk_tokens.py` | 检查知识块是否超过向量模型的编码长度 |
| `evaluations/cases/rag_cases.json` | RAG 评测用例定义 |
| `evaluations/results/eval_rag_results.json` | RAG 评测输出（运行时生成，已忽略） |
| `tests/test_api.py` | 工单 HTTP 接口测试 |
| `tests/test_tickets.py` | 工单业务行为测试 |
| `tests/test_sessions.py` | 会话轮次保存、版本冲突测试 |
| `tests/test_agent.py` | 无状态 Agent 循环和工具协议测试 |
| `tests/test_agent_graph.py` | Agent 图节点、整图集成和多轮来源测试 |
| `tests/test_query_loop.py` | 会话问答循环和 HTTP 接口测试 |
| `tests/test_assistant_api.py` | `/assistant/query` 接口测试 |
| `tests/test_approval_graph.py` | 审批图节点、暂停、恢复和过期建议测试 |
| `tests/test_approval_service.py` | 审批服务各错误分支测试 |
| `tests/test_approval_api.py` | 审批 HTTP 接口和应用重启后恢复测试 |
| `tests/test_checkpoint.py` | 子进程级暂停/恢复与重复恢复测试 |
| `tests/test_citation_guard.py` | 引用校验和来源生成的固定数据测试 |
| `tests/test_model_response.py` | 上游响应解析（`finish_reason`、超时、请求参数）测试 |
| `tests/test_eval_agent.py` | 评测评分与汇总函数测试 |
| `tests/test_request_logging.py` | 请求编号、异常日志和 500 响应测试 |
| `requirements.txt` | 依赖清单 |

## 七、数据存储

`db.py` 和检查点都位于数据目录 `DATA_DIR`（默认项目根目录，可用 `TICKETFLOW_DATA_DIR` 指定），目录不存在时自动创建。

### 1. `tickets.db`

`tickets`：

| 字段 | 类型 | 约束 |
|---|---|---|
| `ticket_id` | TEXT | 主键，非空 |
| `status` | TEXT | 非空 |
| `owner` | TEXT | 非空 |
| `summary` | TEXT | 非空 |

`sessions`：

| 字段 | 类型 | 约束 |
|---|---|---|
| `session_id` | TEXT | 主键，非空 |
| `messages` | TEXT | 非空，JSON 格式的对话历史 |
| `version` | INTEGER | 非空，默认 0，用于乐观锁 |

`proposed_actions`：

| 字段 | 类型 | 约束 |
|---|---|---|
| `action_id` | TEXT | 主键，非空 |
| `session_id` | TEXT | 非空 |
| `ticket_id` | TEXT | 非空 |
| `old_status` | TEXT | 非空，创建建议时的工单状态 |
| `new_status` | TEXT | 非空 |
| `status` | TEXT | 非空，默认 `pending`，取值 `pending/executed/rejected/stale` |

### 2. `approval_checkpoints.sqlite`

LangGraph `SqliteSaver` 管理的检查点文件，保存：

- 审批图的当前状态（建议摘要、审批结果）；
- 等待执行的节点（`wait`）；
- 中断信息和线程编号 `approval:{action_id}`。

该文件由框架维护，表结构不属于业务模型；删除它会丢失所有等待中的审批。`tickets.db`、`approval_checkpoints.sqlite` 及其 WAL/SHM 文件都是运行时数据，不应提交到版本库。

### 3. 知识库

`knowledge/*.md` 是随代码发布的规范内容，不属于运行时数据。修改文档后无需迁移，下一次检索直接生效；新增文档只要包含 `文档编号`、`关键词` 两行元信息即可被 `load_documents()` 读取。

## 八、已知限制

- **单进程运行。** 审批发起/恢复使用进程内 `threading.Lock` 串行化；多 worker 部署时锁不共享，需要换成数据库锁或外部协调机制。
- **SQLite 并发能力有限。** 当前依赖事务和条件更新保证一致性，未实现重试队列或连接池。
- **没有用户认证。** 会话归属校验只依赖 `session_id`，拿到会话编号即可操作；尚未实现登录和权限体系。
- **审批由独立请求恢复。** 客户端通过 `start` 发起、通过 `resume` 提交决定；未实现审批状态轮询接口或服务端推送通知。
- **检查点不会自动清理。** 已完成的审批记录会保留在 `approval_checkpoints.sqlite` 中，长期运行需要定期归档。
- **模型输出受限制。** 单轮最多 4 次模型调用、8 次工具调用；超出后返回 `MODEL_STEP_LIMIT`，本轮不保存。输出被 `max_tokens` 截断时返回 `MODEL_OUTPUT_TRUNCATED`。
- **检索是关键词子串匹配。** 没有分词和向量召回，同义词、口语化表达可能漏检；命中后按关键词数量和标题命中排序，不保证语义最优。
- **引用只认本轮检索。** 多轮追问如果仍需引用资料必须重新检索；否则回答会被 `INVALID_CITATION` 拒绝。`/assistant/query` 不经过 Agent 图，不返回来源，也不做引用校验。
- **未标注来源的回答会被拒绝。** 检索到文档但模型没有写 `[KB-...]` 时返回 `MISSING_CITATION`，本轮不落库，不会降级成无来源回答；这类失败需要重试或人工处理。
- **配置在进程启动时读取。** 修改环境变量后需要重启服务才生效。
- **文档知识是模拟规范。** 登录与支付规则仅用于项目演示，不应当作真实公司的制度或生产操作依据。
- **引用校验不等于事实校验。** 只能验证识别到的文档编号属于本轮检索、并生成对应来源，不能保证每句话都受资料支持。
