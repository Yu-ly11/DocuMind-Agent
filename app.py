"""
DocuMind-Agent: 企业级长文档智能合规审查工作台 (Final Form)
双引擎驱动：
- 引擎 A: LangGraph 5节点状态机全自动合规大体检 (Agentic Audit)
- 引擎 B: 毫秒级双阶段检索 + 流式打字机即问即答 Copilot (Interactive Streaming RAG)
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from src.parser import DocumentParser, ScannedPDFError
from src.rag.hybrid_retriever import HybridRetriever
from src.chat_engine import ContractChatEngine
from src.graph_agent import LangGraphComplianceAgent
from src.schemas import RiskLevel, ContractAuditReport

# 页面基础元数据配置
st.set_page_config(
    page_title="DocuMind —— 企业级长文档智能审查工作台",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 美化徽章与卡片，并注入 notranslate 防御浏览器翻译篡改 React DOM
st.markdown("""
<style>
    /* 防御浏览器自动翻译插件篡改 React 虚拟 DOM 导致 removeChild 崩溃 */
    body, [data-testid="stAppViewContainer"], [data-testid="stChatMessage"] {
        translate: no !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .stCodeBlock {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 顶部导航标题与技术标签
st.title("⚖️ DocuMind-Agent: 企业级合同智能审查工作台")
st.caption("🚀 架构支撑：**双引擎驱动** ｜ **LangGraph 5节点自治状态机** ｜ **Small-to-Big 双阶段检索** ｜ **秒级流式问答 Copilot** ｜ **Pydantic v2 强契约**")

# 初始化 Session State
if "current_file_path" not in st.session_state:
    st.session_state["current_file_path"] = None
if "file_name" not in st.session_state:
    st.session_state["file_name"] = None
if "audit_report" not in st.session_state:
    st.session_state["audit_report"] = None
if "parsed_text" not in st.session_state:
    st.session_state["parsed_text"] = None
if "parsed_sections" not in st.session_state:
    st.session_state["parsed_sections"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

temp_dir = Path("tmp_contracts")
temp_dir.mkdir(exist_ok=True)

# ==========================================
# 侧边栏：真实案例与配置
# ==========================================
with st.sidebar:
    st.header("📂 合同文件载入")
    
    uploaded_file = st.file_uploader(
        "上传待审合同（支持 .pdf / .md / .txt）",
        type=["pdf", "md", "txt"],
        help="支持上传任意文字版企业合同、标书或外包技术协议"
    )

    if uploaded_file is not None:
        saved_path = temp_dir / uploaded_file.name
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["current_file_path"] = saved_path
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["audit_report"] = None
        st.session_state["parsed_text"] = None
        st.session_state["parsed_sections"] = None
        st.session_state["retriever"] = None
        st.session_state["chat_messages"] = []

    st.divider()
    st.markdown("### 🏆 工业级真实长合同样本")
    
    # 真实纠纷大合同 (PDF 与 Markdown 双版本)
    if st.button("📑【真实长PDF】680万涉诉协议.pdf (7页真实排版)", use_container_width=True, help="真正的多页 PDF 格式：包含每页绝密页眉横线、页码存证号、4大复杂表格与跨页条款，检验 pdfplumber 几何裁剪与表格还原"):
        sample_pdf = Path("tests/sample_contracts/real_judicial_software_dispute_contract.pdf")
        if sample_pdf.exists():
            st.session_state["current_file_path"] = sample_pdf
            st.session_state["file_name"] = sample_pdf.name
            st.session_state["audit_report"] = None
            st.session_state["parsed_text"] = None
            st.session_state["parsed_sections"] = None
            st.session_state["retriever"] = None
            st.session_state["chat_messages"] = []
            st.success("已载入 7 页真实长 PDF 合同！")

    if st.button("🔥【真实涉诉】680万软件定制大协议.md", use_container_width=True, help="取材自最高法真实技术集成纠纷判决书：7000字、16个章节，包含60日沉默认可排除、Background IP无偿独占剥夺、每日1%超额违约金与无上限连带穿透"):
        sample_p0 = Path("tests/sample_contracts/real_judicial_software_dispute_contract.md")
        if sample_p0.exists():
            st.session_state["current_file_path"] = sample_p0
            st.session_state["file_name"] = sample_p0.name
            st.session_state["audit_report"] = None
            st.session_state["parsed_text"] = None
            st.session_state["parsed_sections"] = None
            st.session_state["retriever"] = None
            st.session_state["chat_messages"] = []
            st.success("已载入 680 万真实涉诉技术开发大协议！")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("万字采购大合同", use_container_width=True, help="5067字大标的采购协议"):
            sample_p = Path("tests/sample_contracts/large_scale_procurement_contract.md")
            if sample_p.exists():
                st.session_state["current_file_path"] = sample_p
                st.session_state["file_name"] = sample_p.name
                st.session_state["audit_report"] = None
                st.session_state["parsed_text"] = None
                st.session_state["retriever"] = None
                st.session_state["chat_messages"] = []
                st.success("已载入万字采购协议！")
    with col_s2:
        if st.button("技术外包协议", use_container_width=True, help="IT外包技术服务合同"):
            sample_p2 = Path("tests/sample_contracts/sample_procurement_contract.md")
            if sample_p2.exists():
                st.session_state["current_file_path"] = sample_p2
                st.session_state["file_name"] = sample_p2.name
                st.session_state["audit_report"] = None
                st.session_state["parsed_text"] = None
                st.session_state["retriever"] = None
                st.session_state["chat_messages"] = []
                st.success("已载入外包协议！")

    st.divider()
    st.markdown("### ⚙️ 智能体运行策略")
    use_pc = st.checkbox("启用父子块两阶段检索 (Small-to-Big)", value=True, help="子块微观精准匹配，父块宏观上下文还原")
    st.caption("🛡️ 内置确定性风控数学算子（依据民法典585条及合同编通则司法解释进行边界校验）")

    if st.session_state["current_file_path"] is not None:
        if st.button("🔄 清空工作台", use_container_width=True):
            st.session_state["current_file_path"] = None
            st.session_state["file_name"] = None
            st.session_state["audit_report"] = None
            st.session_state["parsed_text"] = None
            st.session_state["parsed_sections"] = None
            st.session_state["retriever"] = None
            st.session_state["chat_messages"] = []
            st.rerun()

# ==========================================
# 主界面渲染
# ==========================================
target_path = st.session_state["current_file_path"]

if target_path and Path(target_path).exists():
    # 自动解析合同文本与结构化章节 (一次解析，全量缓存)
    if st.session_state["parsed_text"] is None or st.session_state.get("parsed_sections") is None:
        try:
            with st.spinner("📑 正在调用 DocumentParser 执行高保真清洗与版面解析（检测到扫描件将自动启动 RapidOCR 引擎逐页转录，请稍候）..."):
                parsed_sections = DocumentParser.parse_file(target_path)
                st.session_state["parsed_sections"] = parsed_sections
                st.session_state["parsed_text"] = "\n\n".join(sec.content for sec in parsed_sections)
        except ScannedPDFError as err:
            st.error(f"🚨 PDF 拦截防御触发: {err}")
            st.stop()
        except Exception as e:
            st.error(f"解析失败: {e}")
            st.stop()

    contract_text = st.session_state["parsed_text"]
    all_sections = st.session_state.get("parsed_sections") or []

    # 初始化全局单例检索器（保证问答与审查共享索引，秒级复用）
    if st.session_state["retriever"] is None:
        with st.spinner("⚡ 正在构建/加载两阶段混合检索索引 (Small-to-Big)..."):
            st.session_state["retriever"] = HybridRetriever(
                contract_text,
                chunk_size=180,
                overlap=30,
                use_parent_child=use_pc,
                sections=all_sections
            )

    col_left, col_right = st.columns([1, 1], gap="large")

    # --- 左侧：合同原文高保真预览 ---
    with col_left:
        st.subheader(f"📄 合同原文预览: `{st.session_state['file_name']}`")
        st.caption(f"全文总字数: **{len(contract_text)}** 字 ｜ 行数: **{contract_text.count(chr(10)) + 1}** 行 ｜ 格式: `{Path(target_path).suffix.upper()}`")
        
        tab_full, tab_sections = st.tabs(["📝 合同清洗正文", "📑 章节结构化切片"])
        with tab_full:
            st.text_area("合同全文", contract_text, height=680)
        with tab_sections:
            if all_sections:
                st.info(f"共解析出 **{len(all_sections)}** 个独立条款章节：")
                for s in all_sections[:12]:
                    with st.expander(f"📌 {s.section_title} (P{s.page_start}-P{s.page_end} | 第{s.line_start}-{s.line_end}行)"):
                        st.text(s.content)
            else:
                st.warning("章节结构化视图暂不可用")

    # --- 右侧：双工作流标签页（大报告 vs 秒级问答） ---
    with col_right:
        work_tab_audit, work_tab_chat = st.tabs([
            "📋 全流程智能审计大报告 (Agentic Audit)", 
            "💬 合同即问即答 Copilot (秒级交互 RAG)"
        ])

        # ==========================================
        # 工作流 A: LangGraph 状态机全面审计大报告
        # ==========================================
        with work_tab_audit:
            st.caption("适合场景：法务一键交卷，自主驱动 5 节点状态机出具带证据链的合规诊断书。")
            default_instruction = (
                "请对本合同进行全方位的深度合规风控审查：\n"
                "1. 审查‘知识产权’条款中服务商原有的底层通用算法与既有专利（Background IP）是否被无偿或强制买断；\n"
                "2. 审查‘违约赔偿’条款中每日违约金比例是否超过法定保护上限，并测算延期30天的违约金额与占比；\n"
                "3. 审查‘系统验收’中是否存在单方无限期拖延验收、排除沉默认可的程序陷阱；\n"
                "4. 审查‘责任限制’中是否存在隐蔽连环穿透导致无上限连带责任的风险。"
            )
            task_prompt = st.text_area("审查指令要求 (Prompt)", default_instruction, height=120)

            if st.button("🚀 启动 LangGraph 5节点状态机全流程审查", type="primary", use_container_width=True):
                with st.status("🤖 DocuMind LangGraph 状态机正在流转中...", expanded=True) as status:
                    try:
                        agent = LangGraphComplianceAgent(contract_text, use_parent_child=use_pc, sections=all_sections)
                        node_labels = {
                            "task_planner": "🧭 **[Node 1: Task Planner]** 审查需求正交分解与规划完成",
                            "hybrid_retriever": "🔍 **[Node 2: Hybrid Retriever]** 双路检索与父子块回溯完成",
                            "penalty_calculator": "🧮 **[Node 3: Penalty Calculator]** 违约金风控数学算子执行完成",
                            "clause_evaluator": "⚖️ **[Node 4: Clause Evaluator]** 对照民法典条款评估完成",
                            "critic": "🧐 **[Node 5: Critic]** 事实依据反思核验与路由决策完成",
                            "report_aggregator": "📑 **[Node 6: Report Aggregator]** 聚合生成强契约大报告"
                        }
                        
                        final_report_data = None
                        # 真正基于 LangGraph 状态事件流驱动进度渲染（单次执行，保证幂等与成本可控）
                        for node_name, delta in agent.stream_audit(task_prompt):
                            msg = node_labels.get(node_name, f"⚙️ 节点 [{node_name}] 流转完成")
                            st.write(msg)
                            if "final_report" in delta and delta["final_report"]:
                                final_report_data = delta["final_report"]

                        if final_report_data:
                            report = ContractAuditReport.model_validate(final_report_data)
                            st.session_state["audit_report"] = report
                            status.update(label="🎉 LangGraph 审查流水线顺利收敛并成功完成！", state="complete", expanded=False)
                        else:
                            status.update(label="❌ 审查状态机未产出最终聚合报告", state="error")
                            st.error("未能获取最终报告，请重试")

                    except Exception as e:
                        status.update(label=f"❌ 审查状态机执行异常: {e}", state="error")
                        st.error(f"详细错误日志: {e}")

            # 呈现审查报告结果
            if st.session_state.get("audit_report") is not None:
                report: ContractAuditReport = st.session_state["audit_report"]

                st.divider()
                st.markdown("### 📊 合规审计综合结论")

                # 指标卡片 (5维指标，包含证据链闭环率)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("综合定级", report.overall_risk.value)
                m2.metric("高风险项", f"{report.high_risk_count} 项", delta="重点拦截" if report.high_risk_count > 0 else None, delta_color="inverse")
                m3.metric("中风险项", f"{report.medium_risk_count} 项")
                m4.metric("低风险项", f"{report.low_risk_count} 项")
                m5.metric(
                    "待人工复核", 
                    f"{report.unverified_count} 项", 
                    delta="需线下核实" if report.unverified_count > 0 else "全部物理闭环",
                    delta_color="inverse" if report.unverified_count > 0 else "normal"
                )

                st.info(f"**高管摘要 (Executive Summary)**: {report.audit_summary}")

                # 逐项展开风险详情
                st.markdown("### ⚠️ 风险条款逐项剖析与处置方案")
                for idx, issue in enumerate(report.issues, start=1):
                    badge = "🔴" if issue.risk_level == RiskLevel.HIGH else ("🟡" if issue.risk_level == RiskLevel.MEDIUM else "🟢")
                    with st.expander(f"{badge} 【{issue.risk_level.value}】 {issue.title} ({issue.category})", expanded=True):
                        loc_desc = f"物理行号: **第 {issue.evidence.line_start} - {issue.evidence.line_end} 行**" if issue.evidence.line_start is not None else "物理行号: **⚠️ 全文未精确定位**"
                        st.markdown(f"**📍 条款出处**: `{issue.evidence.clause_title}` ({loc_desc})")
                        
                        if getattr(issue.evidence, "verification_status", "verified") == "unverified_needs_human_review":
                            st.error("⚠️ **【证据未闭环警示】**：本项引用的原文在合同全文中未实现100%物理对齐，存在模型推理幻觉可能，已如实标记为【需人工核验】。")

                        st.warning(f"**逐字原文证据**: \"{issue.evidence.original_text}\"")
                        st.markdown(f"**💣 风险深度剖析**: {issue.risk_analysis}")
                        st.markdown(f"**💡 商务与谈判建议**: {issue.suggestion}")
                        if issue.suggested_replacement:
                            st.markdown("**📝 建议修改法律表述 (可一键复制用于谈判修改)**:")
                            st.code(issue.suggested_replacement, language="text")

                # 导出报告
                st.divider()
                report_markdown = f"""# {report.contract_name} - 合规审查报告
**综合定级**: {report.overall_risk.value}  
**审查时间**: {report.audit_time}  
**高管摘要**: {report.audit_summary}  

---
## 风险问题清单
"""
                for idx, issue in enumerate(report.issues, start=1):
                    loc_md = f"第 {issue.evidence.line_start}-{issue.evidence.line_end} 行" if issue.evidence.line_start is not None else (f"第 {issue.evidence.page_number} 页" if issue.evidence.page_number is not None else "全文检索")
                    report_markdown += f"""
### {idx}. [{issue.risk_level.value}] {issue.title}
- **条款出处**: {issue.evidence.clause_title} ({loc_md})
- **原文证据**: {issue.evidence.original_text}
- **风险剖析**: {issue.risk_analysis}
- **修改建议**: {issue.suggestion}
- **推荐条款**: {issue.suggested_replacement or '无'}

"""
                st.download_button(
                    label="📥 导出完整合规审查报告 (Markdown 格式)",
                    data=report_markdown,
                    file_name=f"DocuMind_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # ==========================================
        # 工作流 B: 秒级交互 RAG 即问即答 Copilot
        # ==========================================
        with work_tab_chat:
            st.caption("⚡ 适合场景：秒级提问互动。检索底座毫秒响应，打字机即时输出，精确锁定行号。")

            chat_engine = ContractChatEngine(st.session_state["retriever"])

            # 快捷提问药丸
            st.markdown("**💡 快速提问推荐：**")
            qp_col1, qp_col2 = st.columns(2)
            prompt_to_submit = None
            with qp_col1:
                if st.button("💰 违约金每日扣多少？有没有上限？", use_container_width=True):
                    prompt_to_submit = "违约金每日扣多少？有没有上限？如果延期了怎么罚？"
                if st.button("⚖️ 验收标准怎么规定的？有拖延陷阱吗？", use_container_width=True):
                    prompt_to_submit = "合同中系统验收流程和期限怎么规定的？甲方未答复是否视为合格？"
            with qp_col2:
                if st.button("🧠 既有算法专利(Background IP)归谁？", use_container_width=True):
                    prompt_to_submit = "乙方原有的既有知识产权(Background IP)和底层算法是如何约定的？是否被强制独占或买断？"
                if st.button("🚫 责任限制条款是否存在穿透陷阱？", use_container_width=True):
                    prompt_to_submit = "合同中约定的赔偿责任上限是多少？存在哪些穿透或无上限的例外情形？"

            st.divider()

            # 显示历史消息
            for msg in st.session_state["chat_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "citations" in msg and msg["citations"]:
                        with st.expander(f"🔍 查看本回答引用的合同条款出处 (共 {len(msg['citations'])} 处证据)", expanded=False):
                            for c in msg["citations"]:
                                loc_label = f"第 **{c['line_start']} - {c['line_end']}** 行" if c.get('line_start') is not None else (f"第 **{c.get('page')}** 页" if c.get('page') is not None else "全文精准匹配")
                                st.markdown(f"🏷️ **[依据 {c['rank']}]** {loc_label} (匹配度得分: `{c['score']}`)")
                                st.caption(f"\"{c['text'][:160]}...\"")

            # 处理用户输入
            user_input = st.chat_input("向当前合同提问，例如：若延期交付10天，违约金如何计算？...")
            active_prompt = prompt_to_submit or user_input

            if active_prompt:
                # 记录用户消息
                st.session_state["chat_messages"].append({"role": "user", "content": active_prompt})
                with st.chat_message("user"):
                    st.markdown(active_prompt)

                # 1. 在外部直接调用毫秒级混合检索，获得 token 流和引用 (避免动态 spinner 嵌套导致 DOM 异常)
                start_time = time.time()
                token_stream, citations = chat_engine.query(
                    active_prompt, 
                    history=st.session_state["chat_messages"]
                )
                elapsed = time.time() - start_time

                # 2. 进入 assistant 气泡，直接纯打字机流式渲染
                with st.chat_message("assistant"):
                    response_text = st.write_stream(token_stream)
                    st.caption(f"⏱️ 检索与生成响应耗时: **{elapsed:.2f} 秒**")

                    # 渲染证据出处
                    if citations:
                        with st.expander(f"🔍 查看本回答引用的合同条款出处 (共 {len(citations)} 处证据)", expanded=True):
                            for c in citations:
                                loc_label = f"第 **{c['line_start']} - {c['line_end']}** 行" if c.get('line_start') is not None else (f"第 **{c.get('page')}** 页" if c.get('page') is not None else "全文精准匹配")
                                st.markdown(f"🏷️ **[依据 {c['rank']}]** {loc_label} (相关度得分: `{c['score']}`)")
                                st.caption(f"\"{c['text'][:180]}...\"")

                    # 保存历史记录
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "citations": citations
                    })

else:
    st.info("👈 请在左侧上传待审合同（支持 .pdf / .md / .txt），或直接点击左侧【🔥 真实涉诉大协议】一键载入样例开始体验！")
