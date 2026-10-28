# 工单系统后端

基于 FastAPI + SQLite + LangGraph 的工单助手。用户可以用自然语言查询工单，由模型提出状态变更建议，再经人工确认后执行。审批流程可以暂停，并在服务重启后继续。

## 一、功能概览

核心流程：

```text
用户提问
   ↓
Agent 图（LangGraph）
   model ──工具调用──→ tools ──工具结果──→ model
     └──无工具调用──→ finish
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
- **会话历史与图检查点分开存储。** 会话历史保存对话消息，供后续模型调用；图检查点保存审批图的执行状态和中断信息。两者生命周期不同，分别存放在两个 SQLite 文件中。
- **模型输出按失败关闭处理。** 工具调用格式非法、轮次或工具次数超限、最终回答包含 DSML 工具协议标记等情况会抛出 `AgentResponseError`，本轮会话不落库。

## 二、运行机制

### 1. Agent 工具调用图

```text
START → model ──有 tool_calls──→ tools → model
           └──无 tool_calls──→ finish → END
```

- `model`：调用 DeepSeek，校验返回消息的结构和内容类型；最多执行 `MAX_MODEL_ROUNDS = 4` 轮。
- `tools`：校验工具调用字段、调用 ID 唯一性和参数 JSON，执行工具；单轮会话最多累计 `MAX_TOOL_CALLS = 8` 次调用。
- `finish`：要求最终回答为非空字符串，并拒绝包含 `<｜｜DSML｜｜` / `<|DSML|` 等工具协议标记的文本。
- 可用工具：
  - `get_ticket`：查询真实工单。
  - `propose_status_change`：创建状态变更建议，仅在带会话上下文时可用。
- 本轮工具记录和待确认建议随会话历史一起原子保存；出现协议错误时整轮不保存。

### 2. 审批图与检查点

```text
START → load → wait（interrupt） → apply → END
```

- `load`：读取建议，校验建议存在、属于当前会话且状态为 `pending`。
- `wait`：调用 `interrupt()` 暂停，返回确认问题；恢复结果必须是布尔值。
- `apply`：同意时调用确认事务，拒绝时调用拒绝事务。
- 检查点保存在 `approval_checkpoints.sqlite`，线程编号为 `approval:{action_id}`。
- 应用启动时通过 `SqliteSaver` 打开检查点文件；同一进程内用锁串行化发起和恢复操作。

### 3. 并发与一致性

- **会话**：`sessions.version` 做乐观锁。保存本轮消息和建议时按旧版本更新，版本不一致返回 `409 SESSION_CONFLICT`。
- **确认建议**：`BEGIN IMMEDIATE` 事务 + 条件更新工单。工单状态已被修改时返回 `409 ACTION_STALE`。
- **拒绝建议**：只把建议从 `pending` 改为 `rejected`，不修改工单；工单是否被改过不影响拒绝。
- **建议状态机**：`pending → executed / rejected / stale`，之后不可再变。

## 三、接口说明

### 1. 接口列表

| 方法 | 路径 | 用途 | 成功状态码 |
|---|---|---|---|
| POST | `/assistant/query` | 无状态查询（不保存会话，不提建议） | 200 |
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

以下用环境变量保存创建时返回的 ID，实际使用时替换即可。

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

方式 3A 的 `ACTION_ID` 从响应里的 `data.actions[0].action_id` 取得。

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

### 3. 配置模型密钥

只有调用模型的接口需要密钥：

```bash
# Git Bash
export DEEPSEEK_API_KEY="sk-..."
```

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "sk-..."
```

未配置时，`/assistant/query` 和 `/sessions/{session_id}/messages` 返回 `503 MODEL_NOT_CONFIGURED`；工单、建议和审批接口不依赖该密钥。

### 4. 初始化数据库

```bash
python db.py
```

在脚本所在目录创建 `tickets.db` 并建表；重复执行不会清空已有数据。应用启动时也会执行同样的初始化，因此这一步可以省略。

审批图检查点文件 `approval_checkpoints.sqlite` 会在应用启动时自动创建。

### 5. 启动服务

```bash
python -m uvicorn app:app --reload
```

`--reload` 仅用于开发环境，生产环境请去掉该参数。当前审批锁是进程内的，建议以单 worker 模式运行。

### 6. 接口文档

- Swagger UI：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>

### 7. 运行测试

```bash
python -m unittest -v
```

测试使用临时数据库和临时检查点文件，不会修改 `tickets.db` 或 `approval_checkpoints.sqlite`；模型调用和检查点恢复使用 mock 或子进程隔离，不需要真实 API key。

### 8. 停止服务

在启动服务的终端中按 `Ctrl+C`。

## 五、评测

### 1. 运行方式

```bash
python eval_agent.py
```

需要先设置 `DEEPSEEK_API_KEY`。脚本读取 `eval_cases.json`，逐条调用 Agent，结果写入 `eval_results.json`（已在 `.gitignore` 中忽略），并在终端打印每条用例的耗时、失败原因和汇总指标。

### 2. 用例说明

当前共 5 条用例：

| 用例 | 输入 | 期望行为 |
|---|---|---|
| `existing_ticket` | 查询 T103 的状态和负责人 | 调用 `get_ticket`，回答包含工单号、状态、负责人 |
| `missing_ticket` | 查询不存在的 NOT_EXIST_001 | 调用 `get_ticket`，返回 `TICKET_NOT_FOUND` |
| `missing_id` | “帮我查询工单” | 不调用工具，反问工单编号 |
| `indirect_query` | “T103 现在归谁负责？进展到哪一步了？” | 调用 `get_ticket`，回答包含工单号、状态、负责人 |
| `missing_id_with_urgency` | “急，帮我看看那个一直没解决的工单！” | 不调用工具，反问工单编号 |

### 3. 固定数据

评测不读取业务库。`evaluation_database()` 会：

1. 记录当前 `db.DB_PATH`；
2. 创建临时目录和 `eval.db`，执行 `init_db()`；
3. 插入固定工单 T103（状态 `待处理`、负责人 `技术支持`、摘要 `用户无法登录`）；
4. 评测结束后恢复 `db.DB_PATH` 并删除临时目录。

因此评测不会修改 `tickets.db`，相同用例每次面对的数据一致。

### 4. 本次结果

| 指标 | 数值 |
|---|---|
| 通过率 | 5/5（100%） |
| 平均耗时 | 2.072s |
| 最大耗时 | 2.591s |

耗时取自 `eval_results.json` 中每条用例的记录（保留 3 位小数）。结果会受模型响应和网络波动影响，再次运行不一定完全一致。

### 5. 评分范围

`score_case()` 做的是自动可验证的检查：

- **工具选择**：`expected_tool` 是否匹配；为 `null` 时不应调用任何工具。
- **调用次数与参数**：预期恰好一次工具调用，工具名和 `ticket_id` 必须匹配。
- **工具结果**：成功时 `ok` 必须为 `true` 且无 `error`；失败时错误码必须等于 `expected_error`。
- **回答字段覆盖**：`expected_answer_fields` 中的字段值必须出现在最终回答里。

### 6. 局限

- **样本量小。** 目前只有 5 条用例，通过率只能说明这些固定场景可用，不能代表整体效果。
- **字段覆盖不等于语义正确。** 检查是字符串包含判断，只要回答里出现工单号、状态和负责人就算通过；回答的推理是否合理、结论是否自洽、是否编造额外信息，都无法自动判断。
- **场景覆盖有限。** 当前只覆盖查询路径，没有覆盖建议创建、审批、暂停恢复、多轮对话和并发冲突；后续可以补充 `propose_status_change` 和审批相关用例。

## 六、项目文件

| 文件 | 职责 |
|---|---|
| `app.py` | FastAPI 应用：请求模型、路由、HTTP 状态码映射、审批图生命周期 |
| `ticket.py` | 工单查询、创建、状态机校验和直接状态更新 |
| `action_service.py` | 变更建议的创建、确认、拒绝及业务错误处理 |
| `session_service.py` | 会话轮次：读取历史、调用 Agent、按版本保存结果 |
| `approval_service.py` | 发起/恢复审批，校验检查点与请求的一致性 |
| `agent_graph.py` | LangGraph 版 Agent 图（model / tools / finish） |
| `approval_graph.py` | LangGraph 版审批图（load / wait / apply） |
| `deepseek_demo.py` | DeepSeek HTTP 调用、工具协议、无状态 `run_query` |
| `tool_runner.py` | 工具注册、参数校验和执行分派 |
| `db.py` | SQLite 建表与存取、会话版本更新、建议事务 |
| `checkpoint_demo.py` | 审批暂停/恢复的最小示例，支持跨进程恢复 |
| `graph_demo.py` | LangGraph 条件边示例 |
| `reducer_demo.py` | LangGraph 状态覆盖与 `Annotated` 累加示例 |
| `eval_agent.py` | 离线评估脚本（需要 API key） |
| `eval_cases.json` | 评测用例定义 |
| `eval_results.json` | 上次评测输出（运行时生成，已忽略） |
| `test_api.py` | 工单 HTTP 接口测试 |
| `test_tickets.py` | 工单业务行为测试 |
| `test_sessions.py` | 会话轮次保存、版本冲突测试 |
| `test_agent.py` | 无状态 Agent 循环和工具协议测试 |
| `test_agent_graph.py` | Agent 图节点与整图集成测试 |
| `test_query_loop.py` | 会话问答循环和 HTTP 接口测试 |
| `test_assistant_api.py` | `/assistant/query` 接口测试 |
| `test_approval_graph.py` | 审批图节点、暂停、恢复和过期建议测试 |
| `test_approval_service.py` | 审批服务各错误分支测试 |
| `test_approval_api.py` | 审批 HTTP 接口和应用重启后恢复测试 |
| `test_checkpoint.py` | 子进程级暂停/恢复与重复恢复测试 |
| `requirements.txt` | 依赖清单 |

## 七、数据存储

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

## 八、已知限制

- **单进程运行。** 审批发起/恢复使用进程内 `threading.Lock` 串行化；多 worker 部署时锁不共享，需要换成数据库锁或外部协调机制。
- **SQLite 并发能力有限。** 当前依赖事务和条件更新保证一致性，未实现重试队列或连接池。
- **没有用户认证。** 会话归属校验只依赖 `session_id`，拿到会话编号即可操作；尚未实现登录和权限体系。
- **审批是轮询模式。** 发起后客户端需要自行决定何时调用 `resume`，服务端不推送通知。
- **检查点不会自动清理。** 已完成的审批记录会保留在 `approval_checkpoints.sqlite` 中，长期运行需要定期归档。
- **模型输出受限制。** 单轮最多 4 次模型调用、8 次工具调用；超出后返回 `MODEL_STEP_LIMIT`，本轮不保存。
