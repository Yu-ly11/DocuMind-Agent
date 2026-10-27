"""
DocuMind-Agent: 工业级 ReAct 合规审查自主智能体
具备多工具调度 (两阶段 HybridRetriever + 计算器)、自主决策循环与最大迭代防御。
已完成架构重构：工具定义与执行完全收纳在 src/tools/ 模块中。
"""
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# 确保项目根目录在 sys.path 中
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from openai import OpenAI
from dotenv import load_dotenv
from src.rag.hybrid_retriever import HybridRetriever
from src.parser import DocumentParser
from src.tools.compliance_tools import (
    calc_penalty_risk,
    CALC_PENALTY_SCHEMA,
    build_search_contract_tool
)

load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


class ReActComplianceAgent:
    """具备自主多步推理与工具调用能力的合同审查智能体"""

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ReActComplianceAgent":
        """工厂方法：通过高保真解析器自动识别并抽取 PDF / Markdown / TXT 文档全文"""
        print(f"[*] 正在调用 DocumentParser 高保真解析文档: {file_path}")
        content = DocumentParser.extract_full_text(file_path)
        return cls(content)

    def __init__(self, contract_content: str):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = os.getenv("MODEL_NAME")
        
        # 1. 为当前合同初始化最强双阶段混合检索器 (jieba + BGE-M3 + 磁盘持久化 + Reranker)
        print("[*] 正在为合同构建/加载两阶段混合检索持久化索引...")
        self.retriever = HybridRetriever(contract_content, chunk_size=350, overlap=50)
        print(f"[*] 索引就绪！共管理 {len(self.retriever.chunks)} 个语义块。\n")

        # 2. 优雅组装工具箱 (架构重构：从 tools 模块动态获取)
        search_schema, self.search_executor = build_search_contract_tool(self.retriever)
        self.tools = [search_schema, CALC_PENALTY_SCHEMA]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """根据模型给出的名字，分发执行真实本地函数"""
        if name == "search_contract_clauses":
            return self.search_executor(args.get("query", ""))
            
        elif name == "calc_penalty_risk":
            res = calc_penalty_risk(**args)
            return json.dumps(res, ensure_ascii=False)
            
        return f"错误：未知的工具名 {name}"

    def audit(self, task_instruction: str, max_steps: int = 6) -> str:
        """
        核心：ReAct 智能体自主决策循环 (Agent Loop)
        :param task_instruction: 审查指令
        :param max_steps: 最大循环步数防死循环
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名拥有最高执业水准的合同法务风控审查智能体（DocuMind ReAct Agent）。\n"
                    "你手握两个专业工具：\n"
                    "1. `search_contract_clauses`：用于查阅长合同中的具体条款原文；\n"
                    "2. `calc_penalty_risk`：用于精确计算违约金数学比例与法律风险。\n\n"
                    "【工作守则】：\n"
                    "- 面对复杂的长合同审查，不要凭空猜测，先调用检索工具查明事实！\n"
                    "- 涉及违约金和金额计算时，必须调用计算器工具拿出准确数据！\n"
                    "- 收集齐所有事实与计算凭据后，直接输出条理清晰的最终合规审查结论报告！"
                )
            },
            {"role": "user", "content": task_instruction}
        ]

        step = 0
        while step < max_steps:
            step += 1
            print(f"--- 🔄 [第 {step}/{max_steps} 轮思考] 智能体正在决策下一步动作... ---")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            messages.append(msg)

            # 检查大模型是想交卷，还是想继续调工具？
            if not msg.tool_calls:
                print("✅ 智能体判断所有事实和计算均已齐备，决定退出循环，交付最终审查结论！\n")
                return msg.content

            # 如果大模型呼叫了工具，逐一执行并把结果递回
            for tool_call in msg.tool_calls:
                f_name = tool_call.function.name
                f_args = json.loads(tool_call.function.arguments)
                print(f"  👉 [自主决策] 触发工具: 【{f_name}】")
                print(f"     [下发参数]: {f_args}")

                # 本地执行真实代码
                tool_output = self._execute_tool(f_name, f_args)
                print(f"     [工具执行反馈 (截取60字)]: {tool_output[:60]}...")

                # 将工具产出以 role: tool 角色回传给大模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": f_name,
                    "content": tool_output
                })

        return "⚠️ 审查达到最大安全步数，被系统强制熔断拦截。"


if __name__ == "__main__":
    # 1. 载入万字真实长篇大合同
    sample_path = Path("tests/sample_contracts/large_scale_procurement_contract.md")
    with open(sample_path, "r", encoding="utf-8") as f:
        contract_text = f.read()

    print(f"[*] 🚀 启动 DocuMind-Agent，正在载入长篇标的合同: {sample_path.name} (全文共 {len(contract_text)} 字)...")

    # 2. 实例化智能体
    agent = ReActComplianceAgent(contract_text)

    # 3. 下达全维度长篇深度审查任务
    task = (
        "请对本合同进行全方位的深度合规风控审查：\n"
        "1. 审查‘知识产权’条款中关于乙方原有底层算法与专利的约定是否合规；\n"
        "2. 审查‘系统验收与付款’条款中是否存在单方无限期拖延或阻却付款的程序陷阱；\n"
        "3. 审查‘违约责任’条款：请检索出每日违约金比例，若延期 30 天，必须调用计算器精确计算产生的累计违约金与占合同总额比例，判断法律风险；\n"
        "4. 最终输出条理清晰、带原文引据和修改建议的专业审查报告。"
    )

    final_report = agent.audit(task)
    
    print("\n" + "=" * 60)
    print("📋 DocuMind-Agent 面对万字长篇合同交付的终极审查报告")
    print("=" * 60)
    print(final_report)
