# DocuMind-Agent 改造路线图（交付执行版）

> **交付对象**：代码修改执行方（Gemini 等）。
> **项目根目录**：`C:\Users\ZhuanZ\Desktop\DocuMind-Agent`
> **目标**：在**保持现有架构与「可审计性」卖点**的前提下，补齐「**评测 / 可观测性 / 记忆**」三块能力，并修掉若干正确性硬伤。
> **硬约束**：
> 1. 每项改动必须可独立验证；
> 2. 不要重写 README 的叙事与实测数据表（已在别的会话定稿）；
> 3. 沿用现有代码风格：中文注释、类型注解、Pydantic v2；
> 4. 不引入重型依赖，除非本文明确要求。

---

## 0. 给执行方的须知

- **先读后改**：动手前先通读 `src/graph_agent.py`、`src/rag/hybrid_retriever.py`、`src/chat_engine.py`、`src/api.py`、`src/tools/compliance_tools.py`、`app.py`。
- **先备份**：项目当前没有 `.git` 目录，建议先 `git init && git add -A && git commit -m "baseline"` 再改。
- **逐项验证**：每完成一个 P 级任务，跑本文对应「验证」命令，确认通过再进下一项。
- **行号会漂移**：本文给的行号仅作参考，**以函数名 / 代码片段为准**（`Ctrl+F` 搜函数名更快）。

---

## 1. 项目现状速览

```
DocuMind-Agent/
├── app.py                      # Streamlit 双工作台（审计报告 + Copilot 问答）
├── src/
│   ├── api.py                  # FastAPI 端点（upload / sync / stream SSE）
│   ├── graph_agent.py          # LangGraph 6 节点有环状态机
│   ├── agent.py                # 独立 ReAct 智能体（多工具自主调度）
│   ├── chat_engine.py          # 流式问答引擎（复用检索底座）
│   ├── parser.py               # PDF/MD 解析、表格还原、OCR 降级
│   ├── schemas.py              # Pydantic v2 契约
│   ├── config.py               # OpenAI 兼容 / Gemini 双通道配置
│   ├── rag/
│   │   ├── hybrid_retriever.py # BM25 + 向量 + RRF + Reranker
│   │   ├── chunking.py         # Small-to-Big 父子块
│   │   └── reranker.py         # BGE-Reranker
│   └── tools/
│       └── compliance_tools.py # 违约金确定性算子 + 检索工具工厂
├── tests/                      # 证据链 / PDF / API 测试 + 检索消融
└── training/                   # 免密钥算法自测
```

**当前缺口**（本次要补的）：
| 能力 | 现状 |
| --- | --- |
| 评测 | 只有 `tests/eval_benchmark.py` 的**检索层**指标（Hit@k/NDCG/MRR，8 条已饱和）；**没有答案层评测**（faithfulness / answer relevance） |
| 可观测性 | `src/` 里 92 处 `print()`；`trace_logs` 只是字符串列表，**不落库、无 trace_id、无 token/成本统计** |
| 记忆 | `MemorySaver()` 内存版，进程重启即丢；`chat_engine` 只带最近 4 条历史 |

---

## 2. 改动总览（按优先级）

| 编号 | 任务 | 优先级 | 预估 | 产出 |
| --- | --- | --- | --- | --- |
| P0 | 修 6 个正确性 bug | 🔴 必做 | 1–2h | 行为正确、可信 |
| P1 | 评测体系（`evals/`） | 🟠 核心交付物 | 1–2 天 | 评测报告（可写进简历） |
| P2 | 可观测性（trace_id + JSONL） | 🟠 | 半天 | 结构化 trace |
| P3 | 持久化记忆（Sqlite checkpointer） | 🟡 | 半天 | 跨会话恢复 |
| P4 | 生产就绪（单例/依赖/重试） | 🟢 | 半天 | 稳定性 |

> **执行顺序建议**：P0 → P1 → P2 → P3 → P4。P1 是简历最缺的一块，务必优先完成。

---

## 3. P0 · 正确性修复（必做）

### P0-1 违约金「逾期天数」不再偷偷默认 30 天

**问题**：`src/graph_agent.py` 的 `penalty_calculator_node` 里 `days = ... else 30`；`calc_penalty_risk` 签名默认 `overdue_days=30`，内部还 `else 30`。这违背 README「拒绝编造默认值」的承诺——合同没写天数时会被静默按 30 天算。

**改法 A：`src/tools/compliance_tools.py` 整个替换 `calc_penalty_risk` 函数体**

```python
def calc_penalty_risk(
    contract_total_amount: Optional[float] = None,
    daily_rate: Optional[float] = None,
    overdue_days: Optional[int] = None,
) -> Dict[str, Any]:
    # 合同总额或日费率缺失 -> 无法测算
    if contract_total_amount is None or daily_rate is None \
            or contract_total_amount <= 0 or daily_rate <= 0:
        return {
            "status": "incomplete_data",
            "contract_total_amount": contract_total_amount,
            "daily_rate": daily_rate,
            "overdue_days": overdue_days,
            "calculated_total_penalty": None,
            "penalty_ratio_of_contract": None,
            "is_commercial_excessive": False,
            "legal_risk_warning": "【要素缺失需人工核查】合同条款未载明明确的合同总金额或日违约金费率，"
                                   "无法测算确定性金额。需法务线下核对违约金起算基数。",
        }

    # 只给了日费率、没约定天数：只给日违约金，不编造累计
    if overdue_days is None or overdue_days <= 0:
        daily_penalty = round(contract_total_amount * daily_rate, 2)
        return {
            "status": "partial_data",
            "contract_total_amount": contract_total_amount,
            "daily_rate_percent": f"{daily_rate * 100:.2f}%",
            "overdue_days": None,
            "calculated_total_penalty": None,
            "daily_penalty": daily_penalty,
            "penalty_ratio_of_contract": None,
            "is_commercial_excessive": False,
            "legal_reference": "《民法典》第585条及最高人民法院《合同编通则解释》第65条",
            "legal_risk_warning": f"合同未约定逾期天数，仅能测算【每日违约金 ￥{daily_penalty}"
                                  f"（日费率 {daily_rate * 100:.2f}%）】。累计金额需法务确认逾期天数后"
                                  f"再测算，系统拒绝编造天数。",
        }

    days = int(overdue_days)
    total_penalty = round(contract_total_amount * daily_rate * days, 2)
    penalty_ratio = total_penalty / contract_total_amount
    is_commercial_excessive = penalty_ratio > 0.30

    return {
        "status": "success",
        "contract_total_amount": contract_total_amount,
        "daily_rate_percent": f"{daily_rate * 100:.2f}%",
        "overdue_days": days,
        "calculated_total_penalty": total_penalty,
        "penalty_ratio_of_contract": f"{penalty_ratio * 100:.2f}%",
        "is_commercial_excessive": is_commercial_excessive,
        "legal_reference": "《民法典》第585条及最高人民法院《合同编通则解释》第65条",
        "legal_risk_warning": (
            "【商业风险预警】累计违约金占合同总金额比例达 "
            f"{penalty_ratio * 100:.2f}%（已超30%商业警戒线）。"
            "法务要点提示：根据《民法典》第585条，司法实践中违约金调减的法定参照基准为守约方的"
            "【实际损失】（超过损失30%可申请法院调减）。当前约定比例显著过高，履约中极易被对方"
            "主张显失公平并申请司法酌减，建议设定累计违约金封顶上限（如合同总额的10%~20%）。"
            if is_commercial_excessive else
            f"违约金占合同总额 {penalty_ratio * 100:.2f}%，处于常见商业可控区间。"
            "实际裁判中仍将结合履约实际损失进行个案权衡。"
        ),
    }
```

**改法 B：`src/graph_agent.py` 的 `penalty_calculator_node` 里参数解析段整体替换为**

```python
            args = json.loads(resp.choices[0].message.content)

            def _safe_float(value):
                """稳健数值解析：支持 int/float、千分位逗号、百分号、负号；失败返回 None。"""
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                s = str(value).strip().replace(",", "").replace("，", "").replace("％", "%")
                if s.endswith("%"):
                    s = s[:-1]
                    if s in ("", "-", "."):
                        return None
                    try:
                        return float(s) / 100.0
                    except ValueError:
                        return None
                try:
                    return float(s)
                except ValueError:
                    return None

            amt = _safe_float(args.get("contract_total_amount"))
            rate = _safe_float(args.get("daily_rate"))
            days = args.get("overdue_days")
            days = int(days) if isinstance(days, (int, float)) and not isinstance(days, bool) else \
                   (int(str(days)) if isinstance(days, str) and str(days).strip().isdigit() else None)

            # 防御：若 LLM 把 "1.5%" 误写成数字 1.5，按百分数修正（日费率不可能 > 100%）
            if rate is not None and rate > 1:
                rate = rate / 100.0

            cert = calc_penalty_risk(
                contract_total_amount=amt,
                daily_rate=rate,
                overdue_days=days,
            )
```

> 注意：下游打印逻辑里 `if cert["status"] == "success"` 无需改；`partial_data` 会走 else 分支打印 `legal_risk_warning`，符合预期。

**改法 C：`src/tools/compliance_tools.py` 里 `CALC_PENALTY_SCHEMA` 的 `overdue_days` 描述改为**：

```python
"overdue_days": {"type": ["integer", "null"], "description": "预估或约定的违约天数；合同未约定则传 null，禁止编造"},
```

**验证**：
- 合同明确写了「延期 30 天、日 1.5%」→ `status == "success"`；
- 合同只写「日 1.5%」不写天数 → `status == "partial_data"`，`calculated_total_penalty is None`；
- 合同啥都没写 → `status == "incomplete_data"`。

---

### P0-2 `infra_degraded` 不再被计入「需人工复核」

**问题**：`src/graph_agent.py` 的 `report_aggregator_node` 里，未闭环统计条件是 `st_val != VERIFIED`，会把 `INFRA_DEGRADED`（精排降级、但证据已 grounding）也计成 `unverified_count` / `needs_human_review`，与 README「基础设施降级与内容造假解耦」矛盾。

**改法**：把该统计段替换为：

```python
        unverified_cnt = 0
        infra_degraded_cnt = 0
        for i in issues_raw:
            raw = i.get("evidence", {}).get("verification_status")
            st = raw.value if isinstance(raw, VerificationStatus) else str(raw)
            if st == VerificationStatus.VERIFIED.value:
                continue
            elif st == VerificationStatus.INFRA_DEGRADED.value:
                infra_degraded_cnt += 1
            else:
                unverified_cnt += 1
```

保持 `needs_human_review=(unverified_cnt > 0)` 与 `overall_risk` 的判定不变（`infra_degraded` 不强制抬升风险等级，因为它仍基于已 grounding 的 RRF 证据）。

**验证**：构造一个只有 `infra_degraded` 的议题 → `report.unverified_count == 0`、`needs_human_review == False`；构造一个 `unverified` 议题 → 两者都相应置位。

---

### P0-3 Critic 通过分支不再「就地改 state」

**问题**：`src/graph_agent.py` 的 `critic_node`，通过分支里直接 `latest_issue["evidence"]["verification_status"] = ...` 改嵌套 dict，但**返回的 dict 不含 `accumulated_issues`**，依赖 LangGraph 不深拷贝的实现细节，脆弱。

**改法**：把通过分支（`if confidence_ok:` 内）改为：

```python
            updated_issues = list(state.get("accumulated_issues", []))
            if updated_issues and "evidence" in updated_issues[-1]:
                updated_issues[-1] = {
                    **updated_issues[-1],
                    "evidence": {
                        **updated_issues[-1]["evidence"],
                        "verification_status": status_to_set,
                    },
                }

            next_idx = task_idx + 1
            is_all_done = next_idx >= len(state["plan_tasks"])
            next_query = state["plan_tasks"][next_idx]["query"] if not is_all_done else ""

            return {
                "critic_approved": True,
                "current_task_idx": next_idx,
                "current_query": next_query,
                "retry_count": 0,
                "accumulated_issues": updated_issues,
                "trace_logs": state.get("trace_logs", [])
                + [f"{log_tag}，总进度 {next_idx}/{len(state['plan_tasks'])}"],
            }
```

**验证**：跑一遍完整 `agent.audit(...)`，最终 `report.issues[*].evidence.verification_status` 与原文定位一致。

---

### P0-4 聊天引擎不再把当前问题注入两次

**问题**：`app.py` 先把当前用户问题 `append` 进 `chat_messages`，再整体当 `history` 传给 `chat_engine.query()`；而 `query()` 内部又显式追加一次当前问题 → 提示词里问题出现两遍。

**改法**：`app.py` 里调用处改为传入**排除当前条**的历史：

```python
                token_stream, citations = chat_engine.query(
                    active_prompt,
                    history=st.session_state["chat_messages"][:-1],
                )
```

**验证**：在 Streamlit 里连问两轮，检查发给模型的 `prompt_messages`（打印或断点）里当前问题只出现一次。

---

### P0-5 修正 CORS 配置

**问题**：`src/api.py` 里 `allow_origins=["*"]` + `allow_credentials=True` 是非法组合，浏览器会拒绝，也是安全隐患。

**改法**：`add_middleware(CORSMiddleware, ...)` 里把 `allow_credentials=True` 改为 `False`（若确需带凭据，则把 `allow_origins` 改为从环境变量读取的显式域名列表，如 `os.getenv("CORS_ORIGINS", "").split(",")`）。

**验证**：`curl` 或浏览器跨域请求不再被拒；`/docs` 正常。

---

### P0-6 上传文件名清洗 + 大小限制

**问题**：`src/api.py` 的 `upload_contract` 把 `file.filename` 直接拼进路径，且无体积上限（DoS 风险）。

**改法**：

```python
        safe_name = Path(file.filename or "upload").name  # 去掉任何路径分隔符
        temp_file_path = temp_dir / f"{uuid.uuid4().hex}_{safe_name}"
```

并在 `await file.read()` 之后加体积校验：

```python
        MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 20MB 上限，请拆分后重试")
```

**验证**：上传 `..\..\evil.md` 不会越界写；上传 >20MB 文件返回 413。

---

## 4. P1 · 评测体系（核心交付物）

> 目标：把「检索层 + 答案层」评测补齐，产出一份可写进简历的评测报告。
> 设计原则：**不依赖真实案例**——用项目自带的三份合同当语料，人工/LLM 生成 30–50 个带「金标准证据片段」的问题即可。

### 4.1 目录结构

```
evals/
├── testset.jsonl          # 30–50 条测试样本
├── eval_rag.py            # 评测主入口
├── judge.py               # LLM-as-judge（faithfulness / answer relevance）
└── reports/
    └── eval_report.md     # 运行后生成
```

### 4.2 测试集 schema（`evals/testset.jsonl`，每行一个 JSON）

```json
{"id": "q001", "source": "tests/sample_contracts/large_scale_procurement_contract.md", "question": "乙方逾期交付每日违约金比例是多少？", "gold_evidence": "每逾期一日，按合同总额的 1.5% 向甲方支付违约金"}
```

- `gold_evidence` 必须是**合同里逐字存在的片段**（用于检索召回判定）。
- 建议从三份样本合同里各抽 10–15 条，覆盖：违约金、验收、知识产权、保密、责任限制等。

### 4.3 指标定义

**检索层（确定性，无需 LLM）**
- `Context Recall` = `gold_evidence` 是否出现在检索回填的上下文里（命中率）。
- `Context Precision@k` = 命中 `gold_evidence` 的块在 Top-k 中的排名（rank 1 记 1.0，rank 2 记 0.5 …… 未命中记 0）。
- `MRR` = 所有样本 `1/rank` 的均值。

**答案层（LLM-as-judge）**
- `Faithfulness` = 答案中每条事实陈述能否在检索上下文中找到依据（0–1）。
- `Answer Relevance` = 答案是否直接、完整回答问题（0–1）。

### 4.4 `evals/judge.py` 参考实现

```python
"""LLM-as-judge：Faithfulness 与 Answer Relevance。复用项目现有 OpenAI 兼容客户端。"""

def judge_faithfulness(client, model, context: str, answer: str) -> dict:
    prompt = f"""你是严格的 RAG 评测员。给定【检索上下文】和【模型答案】，判断答案中的每一条事实性陈述是否都能在上下文中找到依据。
只输出 JSON：{{"score": 0.0-1.0, "reason": "简短说明", "unsupported_claims": ["无依据的陈述"]}}
score = 有依据的陈述数 / 总陈述数。

【检索上下文】：{context}
【模型答案】：{answer}"""
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.0,
    )
    import json
    return json.loads(resp.choices[0].message.content)


def judge_answer_relevance(client, model, question: str, answer: str) -> dict:
    prompt = f"""你是严格的 RAG 评测员。给定【用户问题】和【模型答案】，判断答案是否直接、完整地回答了问题（不跑题、不含无关信息）。
只输出 JSON：{{"score": 0.0-1.0, "reason": "简短说明"}}

【用户问题】：{question}
【模型答案】：{answer}"""
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.0,
    )
    import json
    return json.loads(resp.choices[0].message.content)
```

### 4.5 `evals/eval_rag.py` 参考骨架

```python
"""DocuMind 评测入口：检索层 + 答案层。"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parser import DocumentParser
from src.rag.hybrid_retriever import HybridRetriever
from src.chat_engine import ContractChatEngine
from src.config import get_llm_client
from evals.judge import judge_faithfulness, judge_answer_relevance


def load_testset(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_retrieval(retriever, items, top_k=3):
    hits, ranks = 0, []
    for it in items:
        results = retriever.retrieve(it["question"], top_k=top_k, mode="hybrid_rerank", return_parent=True)
        rank = 0
        for i, (text, _score) in enumerate(results, start=1):
            if it["gold_evidence"][:20] in text:
                rank = i
                break
        if rank:
            hits += 1
            ranks.append(1.0 / rank)
    n = len(items)
    return {"recall": hits / n if n else 0.0, "mrr": sum(ranks) / n if n else 0.0}


def eval_generation(chat_engine, client, model, items):
    faith_scores, rel_scores, failures = [], [], []
    for it in items:
        stream, citations = chat_engine.query(it["question"], history=[])
        answer = "".join(stream)
        context = "\n\n".join(c["text"] for c in citations)
        f = judge_faithfulness(client, model, context, answer)
        r = judge_answer_relevance(client, model, it["question"], answer)
        faith_scores.append(float(f.get("score", 0)))
        rel_scores.append(float(r.get("score", 0)))
        if f.get("score", 1) < 0.8:
            failures.append({"id": it["id"], "type": "faithfulness", "detail": f.get("reason")})
        if r.get("score", 1) < 0.8:
            failures.append({"id": it["id"], "type": "relevance", "detail": r.get("reason")})
    return {
        "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
        "answer_relevance": sum(rel_scores) / len(rel_scores) if rel_scores else 0.0,
        "failures": failures,
    }


def main():
    # 按 source 分组：同一合同只构建一次 retriever
    items = load_testset(ROOT / "evals" / "testset.jsonl")
    by_source = {}
    for it in items:
        by_source.setdefault(it["source"], []).append(it)

    _, client, model = get_llm_client()  # 需在 src/config.py 补一个返回名字的版本，见 4.6
    all_ret, all_gen = [], []
    for source, group in by_source.items():
        file_path = ROOT / source
        sections = DocumentParser.parse_file(file_path)
        text = DocumentParser.extract_full_text(file_path)
        retriever = HybridRetriever(text, chunk_size=180, overlap=30,
                                    use_parent_child=True, sections=sections)
        chat = ContractChatEngine(retriever)
        all_ret.append((source, eval_retrieval(retriever, group)))
        all_gen.append((source, eval_generation(chat, client, model, group)))

    # 汇总写报告（表格 + 失败 case 分类）
    write_report(all_ret, all_gen)
```

> 注：`eval_generation` 里 `chat_engine.query` 的 `history=[]` 即可（评测不需要多轮）。

### 4.6 需要的小前置：`src/config.py` 暴露 `get_llm_client`

`config.py` 里已有 `get_llm_client()`，返回 `(provider, client, model)`。评测直接复用它即可，无需改动——但请注意它当前**只被 config 自己使用**，其它模块是硬编码 `OpenAI(...)`。本次不改这一点（见 P4），评测直接调 `get_llm_client()` 即可。

### 4.7 报告格式（`evals/reports/eval_report.md`）

```markdown
# DocuMind-Agent 评测报告

| 合同 | Context Recall | MRR | Faithfulness | Answer Relevance |
| --- | --- | --- | --- | --- |
| xxx.md | 0.86 | 0.79 | 0.91 | 0.88 |

## 失败 case 分类
### 检索没召回（Context Recall 未命中）
- q012：……
### 生成幻觉（Faithfulness < 0.8）
- q023：……
```

### 4.8 验证

```powershell
# 先准备 30–50 条 testset.jsonl（可先用 10 条跑通链路）
.\.venv\Scripts\python.exe evals\eval_rag.py
# 预期：生成 evals/reports/eval_report.md，含指标表与失败 case
```

> 若暂时没有 API Key，可先只跑 `eval_retrieval`（不调 LLM），验证链路。

---

## 5. P2 · 可观测性（trace_id + 结构化落库）

**目标**：给每次审查/问答一个 `trace_id`，把节点、耗时、token、状态落成 JSONL，可回查。不引入 Langfuse 硬依赖（可选接入）。

### 5.1 新文件 `src/observability.py`

```python
"""轻量可观测性：trace_id + 结构化 JSONL 落盘（可选 Langfuse 上报）。"""
import json, os, time, uuid
from pathlib import Path

TRACE_DIR = Path("tmp_contracts") / "traces"


def new_trace_id() -> str:
    return uuid.uuid4().hex


class Tracer:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or new_trace_id()
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = TRACE_DIR / f"{self.trace_id}.jsonl"

    def span(self, node: str, status: str = "ok", duration_ms: float = 0.0, **extra):
        record = {"trace_id": self.trace_id, "ts": time.time(), "node": node,
                  "status": status, "duration_ms": round(duration_ms, 2), **extra}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
```

### 5.2 接入 `src/graph_agent.py`

- `AuditState` 增加 `trace_id: str`（或让 `LangGraphComplianceAgent` 持有 `Tracer`）。
- `audit()` / `stream_audit()` 开头 `tracer = Tracer()`，把 `trace_id` 写进初始 state 或返回给调用方。
- 各节点用 `t0 = time.time()` … `tracer.span(node_name, duration_ms=(time.time()-t0)*1000, **摘要)` 记录。
- `print(...)` 逐步替换为 `logging.getLogger("documind")`，并配置 `logging.basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")`。

### 5.3 接入 `src/api.py`

- 每个请求生成/接收 `trace_id`，在响应体或 SSE 事件里带 `trace_id`（已有一部分），并确保 `LangGraphComplianceAgent` 使用它。

### 5.4 验证

```powershell
.\.venv\Scripts\python.exe src\api.py
# 或直接跑一次 audit，检查 tmp_contracts/traces/<trace_id>.jsonl 是否含各节点记录
```

---

## 6. P3 · 持久化记忆

**目标**：`MemorySaver` → 磁盘持久化，`thread_id` 可跨进程恢复。

### 6.1 依赖

`requirements.txt` 增加：
```
langgraph-checkpoint-sqlite>=2.0.0
```

### 6.2 改造 `src/graph_agent.py` 的 `build_compliance_graph`

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

_CHECKPOINT_CONN = None

def build_compliance_graph(retriever):
    global _CHECKPOINT_CONN
    nodes = AuditNodes(retriever)
    ...
    if _CHECKPOINT_CONN is None:
        Path("tmp_contracts").mkdir(parents=True, exist_ok=True)
        _CHECKPOINT_CONN = sqlite3.connect("tmp_contracts/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(_CHECKPOINT_CONN)
    return workflow.compile(checkpointer=checkpointer)
```

### 6.3 `ContractChatEngine` 的会话记忆

- 增加 `thread_id` 参数与一个轻量记忆存储（SQLite 表 `chat_memory(thread_id, role, content, ts)`），检索前按 `thread_id` 读出最近 N 条；
- 可选：用 LLM 把长历史压成摘要（摘要预算 200 字），避免上下文无限增长。

### 6.4 验证

- 同一 `thread_id` 连续两次 `agent.audit(...)`，第二次能读到上一轮 checkpoint（不重复建检索器成本可另议）。
- 重启进程后，`thread_id` 的历史仍在 `tmp_contracts/checkpoints.sqlite`。

---

## 7. P4 · 生产就绪

1. **检索器单例**：`src/api.py` 每次请求都 `LangGraphComplianceAgent(...)` → new 一个 `HybridRetriever`。改为按 `(doc_hash, use_parent_child)` 做进程内缓存（`functools.lru_cache` 或模块级 dict），避免重复建索引/重复 Embedding。
2. **依赖可复现**：新增 `pyproject.toml`（含 `[project]`、依赖 pin、`[tool.ruff]`/`[tool.pytest.ini_options]`）；或至少把 `requirements.txt` 的 `>=` 改为 `==` 并锁定版本。
3. **LLM 重试**：给所有 `chat.completions.create` 加 `timeout=60` 与 `tenacity` 指数退避（venv 已装 `tenacity`）。
4. **统一关代理**：`os.environ["http_proxy"]=""` 目前散落 4 个文件，收敛到 `src/config.py` 顶部执行一次。
5. **删除死代码/重复代码**：`config.py` 的 `Settings`/`get_llm_client` 与各模块硬编码客户端二选一（建议统一走 `get_llm_client`，并让 Gemini 路径真正可用，或删掉 Gemini 承诺）。
6. **`training/practice_exam.py`** 里每个函数被定义了两次（先 TODO 空壳、再真实实现），删除空壳定义即可（不影响断言）。

---

## 8. 明确不要改的部分

- **README.md 的叙事、实测数据表（Hit@1/NDCG 等）**：已定稿，不要重写。
- **面试问答手册**：已移到 `C:\Users\ZhuanZ\Desktop\答辩\`，项目里不要再生成。
- **`tests/` 下现有断言**：保持通过；新增测试放 `evals/`，不要动 `tests/` 的既有语义。
- **`docs/DEBUG_GUIDE.md`**：行号会漂移，但结构不用改。

---

## 9. 最终验收清单

| # | 命令 | 预期 |
| --- | --- | --- |
| 1 | `.\.venv\Scripts\python.exe training\practice_exam.py` | 3 题断言全 PASS（免密钥） |
| 2 | `.\.venv\Scripts\python.exe tests\test_evidence_anchor.py` | Δ=0 + 虚假证据拦截通过（需 Key） |
| 3 | `.\.venv\Scripts\python.exe tests\test_pdf_parser.py` | 通过 |
| 4 | `.\.venv\Scripts\python.exe tests\test_api.py` | 通过 |
| 5 | `.\.venv\Scripts\python.exe evals\eval_rag.py` | 生成 `evals/reports/eval_report.md` |
| 6 | 手动：合同只写日费率、不写天数 | `calc_penalty_risk` 返回 `partial_data` |
| 7 | 手动：`tmp_contracts/traces/<trace_id>.jsonl` 存在且含节点记录 | 可观测性生效 |
| 8 | 手动：重启进程后同 `thread_id` 的 checkpoint 仍在 | 记忆持久化生效 |

---

## 10. 交付物清单（给简历/汇报用）

1. `evals/`（测试集 + 评测脚本 + 评测报告）——**「评测」能力的证明**
2. `src/observability.py` + `tmp_contracts/traces/`——**「可观测性」能力的证明**
3. `SqliteSaver` 持久化记忆——**「记忆」能力的证明**
4. P0 的 6 个 bug 修复——**「代码质量 / 可审计性」的证明**

> 一句话总结（可直接写进简历）：
> 「为 DocuMind-Agent 补齐 RAG 评测（Context Recall / MRR / Faithfulness / Answer Relevance 四指标）、全链路 trace 与持久化会话记忆，并修复 6 处会导致证据链失真的确定性缺陷。」
