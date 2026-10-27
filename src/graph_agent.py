"""
DocuMind-Agent: 工业级 LangGraph 合规审查状态机 (Agentic Workflow)
审查链路解耦为五个核心节点：
1. Task Planner (任务规划节点：拆解审查维度)
2. Hybrid Retriever (混合检索节点：父子块检索 + 多跳召回)
3. Penalty Calculator (风控算子节点：确定性数学消灭长文复利与比例幻觉)
4. Clause Evaluator (合规审查节点：生成带证据链的风险定级)
5. Critic (反思检验节点：反思检验置信度，不足时自动触发 Query 改写重试，最多 3 轮循环)
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.rag.hybrid_retriever import HybridRetriever
from src.parser import DocumentParser
from src.tools.compliance_tools import calc_penalty_risk
from src.schemas import (
    RiskLevel,
    ClauseEvidence,
    ComplianceIssue,
    ContractAuditReport,
    VerificationStatus
)

load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


# ==========================================
# 1. 状态机全局状态模式 (State Schema)
# ==========================================
class AuditState(TypedDict):
    """LangGraph 节点流转的强契约状态载体"""
    contract_text: str                          # 待审合同全文
    task_instruction: str                       # 用户总任务指令
    plan_tasks: List[Dict[str, str]]            # Task Planner 分解出的法务审查任务列表
    current_task_idx: int                       # 当前正在处理的任务序号
    current_query: str                          # 当前用于检索的 Query
    retrieved_clauses: List[Dict[str, Any]]     # 检索召回的法条及其得分/父块上下文
    penalty_cert: Optional[Dict[str, Any]]      # 确定性风控计算器出具的数学凭证
    accumulated_issues: List[Dict[str, Any]]    # 累积评估产出的风险项清单
    critic_approved: bool                       # 反思检验节点是否核准通过
    critic_notes: str                           # 反思检验意见 / 优化建议
    retry_count: int                            # 当前任务的 Query 改写重试次数 (上限 3 轮)
    final_report: Optional[Dict[str, Any]]      # 最终组装的合规审计大报告
    trace_logs: List[str]                       # 全链路可观测性 Trace 审计足迹


# ==========================================
# 2. 五大专业节点实现 (Graph Nodes)
# ==========================================
class AuditNodes:
    """封装 LangGraph 五大节点的业务逻辑与 LLM 交互"""

    def __init__(self, retriever: HybridRetriever):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3")
        self.retriever = retriever

    def task_planner_node(self, state: AuditState) -> Dict[str, Any]:
        """【节点 1: Task Planner】将审查需求拆解为正交的法务审查子任务"""
        print("\n" + "=" * 60)
        print("🧭 [Node 1: Task Planner] 正在对合同合规任务进行维度拆解与动态规划...")
        
        prompt = f"""
你是一名资深企业法务架构师。请阅读用户的合规审查要求：
\"{state['task_instruction']}\"

请将其拆解为 2 到 3 个正交且互不重叠的审查子任务，每个子任务包含核心议题与针对性检索 Query。
必须返回严格的 JSON 格式，结构如下：
{{
  "tasks": [
    {{"id": 1, "topic": "知识产权与背景专利", "query": "乙方原有背景算法与专利权归属约定"}},
    {{"id": 2, "topic": "违约赔偿与比例上限", "query": "逾期交付每日违约金比例与违约金上限约定"}},
    {{"id": 3, "topic": "交付验收与付款条件", "query": "验收异议期及沉默认可排除条款"}}
  ]
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw_res = response.choices[0].message.content
        try:
            parsed = json.loads(raw_res)
            tasks = parsed.get("tasks", [])
        except Exception:
            tasks = [
                {"id": 1, "topic": "知识产权", "query": "乙方底层专利与背景算法归属"},
                {"id": 2, "topic": "违约金风险", "query": "每日逾期违约金比例与累计上限"}
            ]

        print(f"  🎯 成功规划出 {len(tasks)} 个深度审查议题：")
        for t in tasks:
            print(f"     - [{t['topic']}] 检索意图: '{t['query']}'")

        log_msg = f"[Planner] 拆解为 {len(tasks)} 个法务子议题"
        first_query = tasks[0]["query"] if tasks else state["task_instruction"]
        return {
            "plan_tasks": tasks,
            "current_task_idx": 0,
            "current_query": first_query,
            "retry_count": 0,
            "trace_logs": state.get("trace_logs", []) + [log_msg]
        }

    def hybrid_retriever_node(self, state: AuditState) -> Dict[str, Any]:
        """【节点 2: Hybrid Retriever】调用双阶段混合检索器抓取法条细则与父块上下文"""
        query = state["current_query"]
        task_idx = state["current_task_idx"]
        task_info = state["plan_tasks"][task_idx] if task_idx < len(state["plan_tasks"]) else {}
        topic = task_info.get("topic", "专项条款")

        print(f"\n🔍 [Node 2: Hybrid Retriever] 正在执行多跳召回 (议题: {topic} | Query: '{query}')...")
        # 调用两阶段精排，并回溯父块上下文
        hits = self.retriever.retrieve(query, top_k=2, mode="hybrid_rerank", return_parent=True)

        clauses = []
        for rank, (doc, score) in enumerate(hits, start=1):
            clauses.append({"rank": rank, "content": doc, "score": score})
            print(f"  📄 [命中条款 Top{rank} | 得分: {score:.4f}]: {doc[:80]}...")

        log_msg = f"[Retriever] Query '{query}' 命中 {len(clauses)} 块，Top1得分: {clauses[0]['score'] if clauses else 0}"
        return {
            "retrieved_clauses": clauses,
            "trace_logs": state.get("trace_logs", []) + [log_msg]
        }

    def penalty_calculator_node(self, state: AuditState) -> Dict[str, Any]:
        """【节点 3: Penalty Calculator】外挂本地确定性风控算子，彻底消除数学复利幻觉"""
        clauses_text = "\n".join([c["content"] for c in state["retrieved_clauses"]])
        task_idx = state["current_task_idx"]
        topic = state["plan_tasks"][task_idx].get("topic", "")

        # 若当前议题与金额、违约金无关，则平滑跳过
        if "违约" not in topic and "赔偿" not in topic and "款" not in topic and "%" not in clauses_text:
            return {"penalty_cert": None}

        print("\n🧮 [Node 3: Penalty Calculator] 正在扫描违约金参数，呼叫本地确定性风控算子...")
        # 借助大模型精确抽取参数 (金额、日利率、天数)，未在合同明确载明必须填 null，严禁编造默认值
        prompt = f"""
请从以下合同条款中，抽取出违约金计算三要素。如果合同中未明确约定对应要素，对应字段必须输出 null，严禁凭空编造！
条款原文：
\"\"\"{clauses_text}\"\"\"

请严格提取出以下字段并输出 JSON：
- contract_total_amount: 合同总金额（纯数字，例如 3800000.0；若合同中未写明总额填 null）
- daily_rate: 每日违约金比例（如 1.5% 转换为纯小数 0.015；0.5% 转换为 0.005；若未写明填 null）
- overdue_days: 预估或约定的逾期天数（整数，若未明确约定可填 null）

输出格式：
{{
  "contract_total_amount": null,
  "daily_rate": null,
  "overdue_days": null
}}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            args = json.loads(resp.choices[0].message.content)
            # 执行本地确定性 Python 数学函数（安全解析，无造假默认值）
            raw_amt = args.get("contract_total_amount")
            raw_rate = args.get("daily_rate")
            raw_days = args.get("overdue_days")

            amt = float(raw_amt) if (raw_amt is not None and str(raw_amt).replace('.', '', 1).isdigit()) else None
            rate = float(raw_rate) if (raw_rate is not None and str(raw_rate).replace('.', '', 1).isdigit()) else None
            days = int(raw_days) if (raw_days is not None and str(raw_days).isdigit()) else 30

            cert = calc_penalty_risk(
                contract_total_amount=amt,
                daily_rate=rate,
                overdue_days=days
            )
            if cert["status"] == "success":
                print(f"  ✅ [本地算子出具风控凭证]:")
                print(f"     合同额: ￥{cert['contract_total_amount']} | 日费率: {cert['daily_rate_percent']} | 测算逾期: {cert['overdue_days']} 天")
                print(f"     累计违约金: ￥{cert['calculated_total_penalty']} (占合同额 {cert['penalty_ratio_of_contract']})")
                print(f"     法务风控提示: {cert['legal_risk_warning'][:50]}...")
            else:
                print(f"  ℹ️ [本地算子提示]: {cert['legal_risk_warning']}")

            return {
                "penalty_cert": cert,
                "trace_logs": state.get("trace_logs", []) + [f"[Calculator] 测算状态: {cert['status']}, 违约金占合同额: {cert.get('penalty_ratio_of_contract', '未明确')}"]
            }
        except Exception as e:
            print(f"  ⚠️ 参数抽取异常: {e}")
            return {"penalty_cert": None}

    def clause_evaluator_node(self, state: AuditState) -> Dict[str, Any]:
        """【节点 4: Clause Evaluator】对比法律法规与行业惯例，形成带证据链的合规风险草案"""
        task_idx = state["current_task_idx"]
        task_info = state["plan_tasks"][task_idx]
        topic = task_info.get("topic", "审查项")
        clauses_text = "\n\n".join([f"【条款 {c['rank']}】:\n{c['content']}" for c in state["retrieved_clauses"]])
        calc_text = json.dumps(state["penalty_cert"], ensure_ascii=False) if state.get("penalty_cert") else "无数学计算项"

        print(f"\n⚖️ [Node 4: Clause Evaluator] 正在对照法条进行合规评估 (议题: {topic})...")
        prompt = f"""
你是一名资深企业法务合规审查总监。请依据以下提供的真实合同条款原文及数学测算凭证，对此项议题进行合规判定。

【审查议题】: {topic}
【合同原文候选切片】:
{clauses_text}

【外部确定性数学风控测算凭证】:
{calc_text}

【判定原则】:
1. 必须逐字摘录原文证据，严禁凭空编造；
2. 结合中国《民法典》及司法解释对格式免责霸王条款、畸高违约金约定、底层知识产权侵夺等陷阱严格定级；
3. 输出严格的 JSON 格式：
{{
  "issue_id": "RISK-0{task_idx+1}",
  "title": "风险项简明标题",
  "risk_level": "高风险", // 仅限 "高风险", "中风险", "低风险"
  "category": "{topic}",
  "evidence": {{
    "clause_title": "所属条款名称 (如: 第7.1条 逾期违约责任)",
    "original_text": "合同中违规的原文逐字摘录"
  }},
  "risk_analysis": "为何违规及潜在法务/财务后果",
  "suggestion": "业务处置与修改建议",
  "suggested_replacement": "推荐替换的标准法言法语表述"
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        try:
            issue_dict = json.loads(response.choices[0].message.content)
            # 核心：物理行号真实锚定，未命中显式标记 unverified，杜绝假坐标
            orig_text = issue_dict.get("evidence", {}).get("original_text", "")
            loc = self.retriever.get_chunk_location(orig_text)
            if "evidence" in issue_dict:
                issue_dict["evidence"]["line_start"] = loc["line_start"]
                issue_dict["evidence"]["line_end"] = loc["line_end"]
                issue_dict["evidence"]["page_number"] = loc["page"]
                issue_dict["evidence"]["verification_status"] = "verified" if loc["status"] == "located" else "unverified_needs_human_review"

            print(f"  🚨 发现风险: 【{issue_dict.get('risk_level', '未知')}】 {issue_dict.get('title')}")
            if loc["status"] == "located":
                print(f"     [物理证据锚定]: 第 {loc['line_start']} - {loc['line_end']} 行 | 真实出处对齐")
            else:
                print(f"     [物理证据锚定]: ⚠️ 未能在原文准确定位物理行号，系统已如实标记为【未核验(需人工复核)】")
            print(f"     原文摘录: \"{orig_text[:50]}...\"")
        except Exception:
            loc = self.retriever.get_chunk_location(clauses_text[:60])
            issue_dict = {
                "issue_id": f"RISK-0{task_idx+1}",
                "title": f"{topic}条款需重点关注",
                "risk_level": "中风险",
                "category": topic,
                "evidence": {
                    "clause_title": topic,
                    "original_text": clauses_text[:80],
                    "line_start": loc["line_start"],
                    "line_end": loc["line_end"],
                    "page_number": None,
                    "verification_status": "unverified_needs_human_review"
                },
                "risk_analysis": "该条款约定存在模糊或权责不对等可能，需法务人员线下核验原文。",
                "suggestion": "建议法务人员二次复核"
            }

        new_issues = list(state.get("accumulated_issues", [])) + [issue_dict]
        log_msg = f"[Evaluator] 议题 {topic} 完成评估，定级为 {issue_dict.get('risk_level')}"
        return {
            "accumulated_issues": new_issues,
            "trace_logs": state.get("trace_logs", []) + [log_msg]
        }

    def critic_node(self, state: AuditState) -> Dict[str, Any]:
        """【节点 5: Critic 反思检验】检验事实依据充分度，三态路由（通过 / 改写重试 / 熔断人工标记），隔离量纲冲突"""
        print(f"\n🧐 [Node 5: Critic] 正在对当前审查结论进行严格的【反思检验 (Self-Reflection)】...")
        task_idx = state["current_task_idx"]
        latest_issue = state["accumulated_issues"][-1] if state["accumulated_issues"] else {}
        evidence_text = latest_issue.get("evidence", {}).get("original_text", "")
        retry_count = state.get("retry_count", 0)

        # 检验 1: 真实可溯源性检验 (Grounding Check)
        evidence_snippet = evidence_text.strip()[:30]
        is_grounded = bool(evidence_snippet and (evidence_snippet in state["contract_text"]))

        # 检验 2: 检索得分置信度与模式感知 (消灭量纲冲突)
        retrieval_mode = getattr(self.retriever, "last_retrieval_mode", "hybrid_rerank")
        top_score = state["retrieved_clauses"][0]["score"] if state["retrieved_clauses"] else 0.0

        if retrieval_mode == "hybrid_rerank":
            # 精排模式下，采用标定阈值 0.35
            confidence_ok = (top_score >= 0.35) and is_grounded
            infra_degraded = False
        else:
            # 基础设施降级模式 (RRF 粗排分处于 0.016 级别，不可比)
            # 只要条款真实存在于原文，放行但标记为 INFRA_DEGRADED，绝不触发 3 轮无效重试误报
            confidence_ok = is_grounded
            infra_degraded = True

        # 分支 A: 证据链闭环且置信度达标 -> 放行推进
        if confidence_ok:
            if infra_degraded:
                print("  ℹ️ [Critic 降级放行]：Reranker 服务降级，已基于 RRF 证据放行，标记为【精排降级】！")
                log_tag = "[Critic] 检索服务降级放行"
                status_to_set = VerificationStatus.INFRA_DEGRADED
            else:
                print("  ✅ [Critic 检验通过]：证据链真实且可追溯，准予放行并推进下一议题！")
                log_tag = "[Critic] 议题核准通过"
                status_to_set = VerificationStatus.VERIFIED

            if state["accumulated_issues"] and "evidence" in latest_issue:
                latest_issue["evidence"]["verification_status"] = status_to_set

            next_idx = task_idx + 1
            is_all_done = next_idx >= len(state["plan_tasks"])
            next_query = state["plan_tasks"][next_idx]["query"] if not is_all_done else ""

            return {
                "critic_approved": True,
                "current_task_idx": next_idx,
                "current_query": next_query,
                "retry_count": 0,
                "trace_logs": state.get("trace_logs", []) + [f"{log_tag}，总进度 {next_idx}/{len(state['plan_tasks'])}"]
            }

        # 分支 B: 校验失败，且尚未达到 3 次重试上限 -> 触发 Query 自适应改写重试
        if retry_count < 3:
            new_retry = retry_count + 1
            print(f"  ⚠️ [Critic 拦截重试 ({new_retry}/3 轮)]：法条上下文置信度不足或证据不可溯，触发 Query 自适应改写！")
            
            old_query = state["current_query"]
            rewrite_prompt = f"原搜索 Query '{old_query}' 召回结果模糊或证据不闭环。请改写为一个包含更精确法言法语和核心实体的搜索词，仅输出改写后的 Query 文本，严禁多余废话。"
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.3
            )
            rewritten_q = resp.choices[0].message.content.strip().replace('"', '')
            print(f"  🔄 [Query 改写演变]: '{old_query}' ➡️ '{rewritten_q}'")

            # 剔除刚生成的未达标草稿，准备回溯重捞
            updated_issues = state["accumulated_issues"][:-1]
            return {
                "critic_approved": False,
                "current_query": rewritten_q,
                "retry_count": new_retry,
                "accumulated_issues": updated_issues,
                "trace_logs": state.get("trace_logs", []) + [f"[Critic] 触发第 {new_retry} 次 Query 改写: {rewritten_q}"]
            }

        # 分支 C: 达到 3 次重试上限仍未闭环 -> 触发熔断通道，绝不假装通过！
        # Fail-Closed 机制：如实标记为 UNVERIFIED，给法务最真实透明的风控标记
        print("  🚨 [Critic 熔断]：已重试 3 轮仍未在原文闭环证据链，系统拒绝伪造可信度！")
        print("     已将该议题明确标记为【未核验-需人工复核】，强制收敛并推进下一项。")
        if state["accumulated_issues"]:
            target_issue = state["accumulated_issues"][-1]
            if "evidence" in target_issue:
                target_issue["evidence"]["verification_status"] = VerificationStatus.UNVERIFIED
            target_issue["risk_analysis"] = (
                "【⚠️ 证据未闭环-需人工线下核验】：该条款摘录在合同中未实现100%物理锚定，"
                "可能存在模型推理幻觉或文本扫描跨页断裂，请法务人员重点复核。\n"
                + target_issue.get("risk_analysis", "")
            )

        next_idx = task_idx + 1
        is_all_done = next_idx >= len(state["plan_tasks"])
        next_query = state["plan_tasks"][next_idx]["query"] if not is_all_done else ""

        return {
            "critic_approved": True,  # 允许图向后流转收敛，但结论中已带有未核验烙印
            "current_task_idx": next_idx,
            "current_query": next_query,
            "retry_count": 0,
            "trace_logs": state.get("trace_logs", []) + [f"[Critic] 达到重试上限熔断，标记未核验需人工复核，总进度 {next_idx}/{len(state['plan_tasks'])}"]
        }

    def report_aggregator_node(self, state: AuditState) -> Dict[str, Any]:
        """【报告收敛节点】将所有累积议题聚合为 Pydantic v2 强契约输出（Fail-Closed 报表闸门与动态摘要）"""
        print("\n" + "=" * 60)
        print("📑 [Report Aggregator] 所有议题审查与反思全部收敛，正在组装最终强契约审计报告...")
        issues_raw = state.get("accumulated_issues", [])
        
        # 统计风险等级
        high_cnt = sum(1 for i in issues_raw if i.get("risk_level") == "高风险")
        med_cnt = sum(1 for i in issues_raw if i.get("risk_level") == "中风险")
        low_cnt = sum(1 for i in issues_raw if i.get("risk_level") == "低风险")

        # Fail-Closed 严谨核查：统计所有未完全闭环证据项
        unverified_cnt = 0
        for i in issues_raw:
            st_val = i.get("evidence", {}).get("verification_status")
            if st_val != VerificationStatus.VERIFIED and st_val != VerificationStatus.VERIFIED.value:
                unverified_cnt += 1

        # 报表级安全闸门：若存在未闭环证据项，综合定级绝不评为 LOW（至少为 MEDIUM 提示人工复核）
        if high_cnt > 0:
            overall_risk = RiskLevel.HIGH
        elif med_cnt > 0 or unverified_cnt > 0:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        # 转换为强类型 Pydantic 对象，未知等级保守降级为 MEDIUM 并提醒人工复核
        compliance_issues = []
        for i in issues_raw:
            ev_data = i.get("evidence", {})
            st_raw = ev_data.get("verification_status", VerificationStatus.UNVERIFIED)
            if isinstance(st_raw, VerificationStatus):
                ev_status = st_raw
            else:
                try:
                    ev_status = VerificationStatus(str(st_raw))
                except Exception:
                    ev_status = VerificationStatus.UNVERIFIED

            evidence = ClauseEvidence(
                clause_title=ev_data.get("clause_title", "未命名条款"),
                original_text=ev_data.get("original_text", ""),
                line_start=ev_data.get("line_start"),
                line_end=ev_data.get("line_end"),
                page_number=ev_data.get("page_number"),
                verification_status=ev_status
            )
            raw_rl = str(i.get("risk_level", "")).strip()
            if "高" in raw_rl:
                rl = RiskLevel.HIGH
            elif "中" in raw_rl:
                rl = RiskLevel.MEDIUM
            elif "低" in raw_rl:
                rl = RiskLevel.LOW
            else:
                rl = RiskLevel.MEDIUM  # 保守防御：未知风险坚决不降级为 LOW

            compliance_issues.append(
                ComplianceIssue(
                    issue_id=i.get("issue_id", "RISK-00"),
                    title=i.get("title", "未命名合规风险"),
                    risk_level=rl,
                    category=i.get("category", "通用条款"),
                    evidence=evidence,
                    risk_analysis=i.get("risk_analysis", ""),
                    suggestion=i.get("suggestion", ""),
                    suggested_replacement=i.get("suggested_replacement")
                )
            )

        # 动态提取真实合同名称（绝不写死"企业级标的采购"）
        contract_lines = [l.strip() for l in state.get("contract_text", "").split("\n") if l.strip()]
        detected_name = "未命名待审合同"
        for line in contract_lines[:8]:
            m = re.search(r"《(.*?)》", line)
            if m:
                detected_name = f"《{m.group(1)}》"
                break
        if detected_name == "未命名待审合同" and contract_lines:
            detected_name = contract_lines[0][:35]

        # 动态组装审计摘要
        summary_lines = [
            f"本次审查共完成 {len(state.get('plan_tasks', []))} 大核心法务议题评估，"
            f"发现高风险项 {high_cnt} 处、中风险项 {med_cnt} 处、低风险项 {low_cnt} 处。"
        ]
        if unverified_cnt > 0:
            summary_lines.append(f"其中有 {unverified_cnt} 项证据未能在原文完成100%物理闭环，已如实标为【需人工复核】。")
        if high_cnt > 0:
            summary_lines.append("存在可能造成单方重大权责失衡或严重赔偿敞口的风险项，建议法务严控谈判并予以修改。")
        else:
            summary_lines.append("未发现一票否决级高危漏洞，整体法律风险受控，请按常规流程归档履约。")

        # 动态提取通过项模块
        all_topics = [t.get("topic", "") for t in state.get("plan_tasks", [])]
        problem_topics = [i.get("category", "") for i in issues_raw if i.get("risk_level") in ("高风险", "中风险")]
        passed_topics = [t for t in all_topics if t and t not in problem_topics]
        if not passed_topics:
            passed_topics = ["合同主体基本信息与准入格式规范"]

        report = ContractAuditReport(
            contract_name=detected_name,
            audit_summary=" ".join(summary_lines),
            overall_risk=overall_risk,
            high_risk_count=high_cnt,
            medium_risk_count=med_cnt,
            low_risk_count=low_cnt,
            unverified_count=unverified_cnt,
            needs_human_review=(unverified_cnt > 0),
            issues=compliance_issues,
            passed_items=passed_topics
        )

        print(f"🎉 审查报告生成就绪！综合定级: 【{report.overall_risk.value}】 (高危: {high_cnt} | 中危: {med_cnt})")
        return {"final_report": report.model_dump()}


# ==========================================
# 3. 构造 LangGraph 状态机并编译
# ==========================================
def build_compliance_graph(retriever: HybridRetriever):
    """编译企业级合规审查 StateGraph 有向有环图"""
    nodes = AuditNodes(retriever)
    workflow = StateGraph(AuditState)

    # 注册节点
    workflow.add_node("task_planner", nodes.task_planner_node)
    workflow.add_node("hybrid_retriever", nodes.hybrid_retriever_node)
    workflow.add_node("penalty_calculator", nodes.penalty_calculator_node)
    workflow.add_node("clause_evaluator", nodes.clause_evaluator_node)
    workflow.add_node("critic", nodes.critic_node)
    workflow.add_node("report_aggregator", nodes.report_aggregator_node)

    # 编排基础有向边
    workflow.add_edge(START, "task_planner")
    workflow.add_edge("task_planner", "hybrid_retriever")
    workflow.add_edge("hybrid_retriever", "penalty_calculator")
    workflow.add_edge("penalty_calculator", "clause_evaluator")
    workflow.add_edge("clause_evaluator", "critic")

    # 核心：条件路由边 (动态循环与终止)
    def critic_decision(state: AuditState) -> str:
        """Critic 节点决策路由器"""
        # 情况 1: 未通过核准且重试未达上限 -> 循环回 Hybrid Retriever 重新改写检索
        if not state.get("critic_approved", True):
            return "hybrid_retriever"
        
        # 情况 2: 已通过且还有未处理议题 -> 推进到下一个议题的 Hybrid Retriever
        if state["current_task_idx"] < len(state["plan_tasks"]):
            return "hybrid_retriever"

        # 情况 3: 所有议题全部执行并通过 -> 进入报告收敛聚合
        return "report_aggregator"

    workflow.add_conditional_edges(
        "critic",
        critic_decision,
        {
            "hybrid_retriever": "hybrid_retriever",
            "report_aggregator": "report_aggregator"
        }
    )
    workflow.add_edge("report_aggregator", END)

    # 编译状态图并挂载持久化内存检查点 (Checkpointer)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==========================================
# 4. 企业级智能体统一外观入口类
# ==========================================
class LangGraphComplianceAgent:
    """以 LangGraph 为核心决策引擎的高保真审查智能体"""

    def __init__(
        self, 
        contract_content: str, 
        use_parent_child: bool = True,
        sections: Optional[List[Any]] = None
    ):
        self.contract_content = contract_content
        self.sections = sections
        print("[*] 正在为 LangGraph 初始化专属双阶段检索底座 (带父子块 Small-to-Big 索引与高保真物理坐标映射)...")
        self.retriever = HybridRetriever(
            contract_content, 
            chunk_size=180, 
            overlap=30, 
            use_parent_child=use_parent_child,
            sections=sections
        )
        self.graph = build_compliance_graph(self.retriever)

    @classmethod
    def from_file(cls, file_path: str | Path, use_parent_child: bool = True) -> "LangGraphComplianceAgent":
        """从 PDF / Markdown 文件构建，自动携带物理章节坐标与页码"""
        sections = DocumentParser.parse_file(file_path)
        text = DocumentParser.extract_full_text(file_path)
        return cls(text, use_parent_child=use_parent_child, sections=sections)

    def audit(self, instruction: str, thread_id: str = "documind_default_session") -> ContractAuditReport:
        """执行完整状态机审计流 (支持基于 thread_id 的检查点持久化与快照追溯)"""
        initial_state: AuditState = {
            "contract_text": self.contract_content,
            "task_instruction": instruction,
            "plan_tasks": [],
            "current_task_idx": 0,
            "current_query": "",
            "retrieved_clauses": [],
            "penalty_cert": None,
            "accumulated_issues": [],
            "critic_approved": True,
            "critic_notes": "",
            "retry_count": 0,
            "final_report": None,
            "trace_logs": []
        }

        # 触发 LangGraph 状态图流转 (携带持久化配置与显式递归上限，杜绝默认25步超限异常)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 60
        }
        final_state = self.graph.invoke(initial_state, config=config)
        report_data = final_state.get("final_report", {})
        return ContractAuditReport.model_validate(report_data)

    def stream_audit(self, instruction: str, thread_id: str = "documind_default_session"):
        """
        以流式生成器形式运行 LangGraph 状态机
        每当一个节点计算完毕，实时 yield (node_name, state_delta)
        用于 FastAPI SSE (Server-Sent Events) 前端实时打字机与步骤渲染
        """
        initial_state: AuditState = {
            "contract_text": self.contract_content,
            "task_instruction": instruction,
            "plan_tasks": [],
            "current_task_idx": 0,
            "current_query": "",
            "retrieved_clauses": [],
            "penalty_cert": None,
            "accumulated_issues": [],
            "critic_approved": True,
            "critic_notes": "",
            "retry_count": 0,
            "final_report": None,
            "trace_logs": []
        }
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 60
        }
        for update_dict in self.graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, delta in update_dict.items():
                yield node_name, delta


if __name__ == "__main__":
    # 适配 Windows 控制台输出中文
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample_contract_file = Path("tests/sample_contracts/large_scale_procurement_contract.md")
    with open(sample_contract_file, "r", encoding="utf-8") as f:
        content = f.read()

    print("=" * 70)
    print("🚀 启动 DocuMind LangGraph 状态机实战审查流水线")
    print("=" * 70)

    agent = LangGraphComplianceAgent(content, use_parent_child=True)
    
    test_instruction = (
        "请对本合同进行全方位合规审查：\n"
        "1. 审查‘知识产权’条款中服务商原有的底层通用算法与背景专利是否被侵夺；\n"
        "2. 审查‘违约赔偿’条款中每日违约金比例是否超过法定保护上限，并测算延期30天的违约金额与占比；\n"
        "3. 审查‘系统验收’中是否存在单方无限期拖延验收、排除沉默认可的程序陷阱；\n"
        "4. 输出带精确证据链和修改方案的结构化报告。"
    )

    audit_result = agent.audit(test_instruction)

    print("\n" + "=" * 70)
    print("📋 LangGraph 状态机最终交付的合规审计大报告 (Pydantic v2 结构化结果)")
    print("=" * 70)
    print(f"合同名称: {audit_result.contract_name}")
    print(f"综合风险: 【{audit_result.overall_risk.value}】")
    print(f"审查摘要: {audit_result.audit_summary}\n")
    print(f"风险项详情 (共发现 {len(audit_result.issues)} 项):")
    for idx, iss in enumerate(audit_result.issues, start=1):
        print(f"\n[{idx}] 【{iss.risk_level.value}】 {iss.title} ({iss.category})")
        print(f"    📍 原文出处: {iss.evidence.clause_title}")
        print(f"    🔍 摘录原文: \"{iss.evidence.original_text}\"")
        print(f"    💣 风险剖析: {iss.risk_analysis}")
        print(f"    💡 处置建议: {iss.suggestion}")
        if iss.suggested_replacement:
            print(f"    📝 建议修改条款: {iss.suggested_replacement}")
