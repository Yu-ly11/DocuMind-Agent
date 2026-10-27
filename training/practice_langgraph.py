r"""
================================================================================
DocuMind-Agent: 核心节点手写攻坚练习卷 (LangGraph Practice Exercise)
================================================================================
本练习卷用于彻底破除“只看不练、虚假理解”的幻觉。
你需要根据注释提示，亲手补齐以下 3 个核心关卡（共计约 50 行核心逻辑）：

关卡 1: 定义 LangGraph 全局强类型状态模式 (State Schema)
关卡 2: 手写 Critic 节点中的真实依据溯源检验逻辑 (Grounding Check)
关卡 3: 手写 LangGraph 条件路由决策函数 (Conditional Edge Router)

补齐后直接运行本脚本：
.\.venv\Scripts\python.exe practice_langgraph.py
如果全绿通过，说明你真正吃透了 LangGraph 状态机编排的灵魂！
================================================================================
"""

import os
import sys
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

# 适配 Windows 控制台输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ================================================================================
# 关卡 1: 定义状态模式 (State Schema)
# ================================================================================
# 【任务要求】：
# 补齐 MiniAuditState 的类型定义，确保包含以下 5 个关键字段：
# 1. contract_text: str (待审合同全文)
# 2. task_query: str (当前审查的问题)
# 3. retrieved_doc: str (检索到的条款原文)
# 4. is_grounded: bool (事实依据是否可溯源)
# 5. retry_count: int (重试轮数)
# 6. final_conclusion: Optional[str] (最终结论)
# ================================================================================

class MiniAuditState(TypedDict):
    contract_text: str
    task_query: str
    retrieved_doc: str
    # TODO: [请你补齐剩余 3 个字段的定义]
    is_grounded: bool
    retry_count: int
    final_conclusion: Optional[str]


# ================================================================================
# 模拟各专业节点业务函数
# ================================================================================
def retriever_node(state: MiniAuditState) -> Dict[str, Any]:
    """模拟检索节点：根据 query 抓取条款"""
    print(f"  🔍 [Retriever Node] 收到 Query: '{state['task_query']}' (当前重试轮数: {state['retry_count']})")
    # 模拟真实检索：如果 query 太模糊则返回无关内容；改写精确后返回真实违约条款
    if "违约金" in state["task_query"] or "每日" in state["task_query"]:
        matched = "【第七条】乙方逾期交付，每日须向甲方支付相当于合同总额 1.5% 的逾期违约金。"
    else:
        matched = "【第一条】双方本着诚信原则展开长期友好合作。"
    return {"retrieved_doc": matched}


def evaluator_node(state: MiniAuditState) -> Dict[str, Any]:
    """模拟合规评估节点：生成审查结论"""
    print(f"  ⚖️ [Evaluator Node] 正在评估检索内容: \"{state['retrieved_doc'][:30]}...\"")
    if "1.5%" in state["retrieved_doc"]:
        conclusion = "【高风险】发现每日违约金 1.5% 严重过高条款！"
    else:
        conclusion = "条款内容合规，未发现明显违规约定。"
    return {"final_conclusion": conclusion}


# ================================================================================
# 关卡 2: 手写 Critic 节点中的真实事实溯源核查 (Grounding Check)
# ================================================================================
# 【任务要求】：
# 1. 检查检索到的 `state["retrieved_doc"]` 是否真实存在于 `state["contract_text"]` 中；
# 2. 如果包含“违约金”核心字眼且确实在合同原文中，判定为 is_grounded = True；
# 3. 否则判定为 False，且将 retry_count 累加 1，并自动改写 task_query 为包含“每日违约金”的精准搜索词！
# ================================================================================

def critic_node(state: MiniAuditState) -> Dict[str, Any]:
    print("  🧐 [Critic Node] 正在进行反思检验 (Self-Reflection)...")
    doc = state.get("retrieved_doc", "")
    contract = state.get("contract_text", "")
    current_retry = state.get("retry_count", 0)

    # TODO: [请你补齐下方的核验与改写逻辑]
    # 提示：
    # in_contract = (doc in contract) if doc else False
    # has_core_info = ("违约金" in doc)
    #
    # 如果 in_contract and has_core_info:
    #     返回 {"is_grounded": True}
    # 否则:
    #     返回 {"is_grounded": False, "retry_count": current_retry + 1, "task_query": "检索合同每日逾期违约金比例"}

    in_contract = (doc in contract) if doc else False
    has_core_info = ("违约金" in doc)

    if in_contract and has_core_info:
        print("     ✅ 事实核查通过！")
        return {"is_grounded": True}
    else:
        new_retry = current_retry + 1
        print(f"     ⚠️ 事实核查未通过，触发第 {new_retry} 次改写重试！")
        return {
            "is_grounded": False,
            "retry_count": new_retry,
            "task_query": "检索合同每日逾期违约金比例"
        }


# ================================================================================
# 关卡 3: 手写 LangGraph 条件路由决策函数 (Conditional Edge Router)
# ================================================================================
# 【任务要求】：
# 根据 state 的状态，决定下一步走向哪一个节点：
# 1. 如果 state["is_grounded"] 为 True (检验合格) ➡️ 走向 "evaluator" (去生成结论)
# 2. 如果 state["is_grounded"] 为 False 且 state["retry_count"] < 3 (不合格但还有机会) ➡️ 走向 "retriever" (重搜)
# 3. 否则 (达到最大重试上限) ➡️ 强行走向 "evaluator" 兜底
# ================================================================================

def decide_next_step(state: MiniAuditState) -> str:
    # TODO: [请你补齐下方的条件分支判断]
    # 提示：返回字符串 "retriever" 或 "evaluator"
    if state.get("is_grounded", False):
        return "evaluator"
    elif state.get("retry_count", 0) < 3:
        return "retriever"
    else:
        return "evaluator"


# ================================================================================
# 组装状态机并执行自动化检验
# ================================================================================
def build_and_test_practice_graph():
    print("=" * 70)
    print("🛠️ 正在组装你的 LangGraph 手写练习状态机...")
    print("=" * 70)

    workflow = StateGraph(MiniAuditState)

    # 注册节点
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("evaluator", evaluator_node)

    # 编排基础有向边
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "critic")

    # 挂载你的手写条件边！
    workflow.add_conditional_edges(
        "critic",
        decide_next_step,
        {
            "retriever": "retriever",
            "evaluator": "evaluator"
        }
    )
    workflow.add_edge("evaluator", END)

    # 挂载检查点并编译
    app = workflow.compile(checkpointer=MemorySaver())

    # 构造一份带模糊提问的测试上下文
    mock_contract = (
        "【企业技术开发协议】\n"
        "【第一条】双方诚信合作。\n"
        "【第七条】乙方逾期交付，每日须向甲方支付相当于合同总额 1.5% 的逾期违约金。\n"
    )

    initial_state: MiniAuditState = {
        "contract_text": mock_contract,
        "task_query": "合同有什么风险吗？", # 故意给出模糊 Query，逼迫 Critic 触发第 1 轮改写拦截！
        "retrieved_doc": "",
        "is_grounded": False,
        "retry_count": 0,
        "final_conclusion": None
    }

    print("\n🚀 [自动化测试开始]：输入模糊 Query: '合同有什么风险吗？'...")
    config = {"configurable": {"thread_id": "practice_thread_001"}}
    final_output = app.invoke(initial_state, config=config)

    print("\n" + "=" * 70)
    print("📊 练习执行最终结果验证")
    print("=" * 70)
    print(f"最终结论: {final_output['final_conclusion']}")
    print(f"经历重试轮数: {final_output['retry_count']} 轮")
    print(f"最终改写 Query: '{final_output['task_query']}'")

    # 自动化断言验证
    assert final_output["is_grounded"] is True, "❌ 错误：is_grounded 应该为 True！"
    assert final_output["retry_count"] >= 1, "❌ 错误：模糊 Query 应该至少触发 1 次 Critic 改写重试！"
    assert "高风险" in final_output["final_conclusion"], "❌ 错误：最终结论未识别出违约金高风险！"

    print("\n🎉 恭喜！你手写的 LangGraph 状态机通过了全部单元测试与反思循环断言！")
    print("你已经彻底掌握了：有向有环图编排、Critic 事实核查、以及条件路由循环的底层机理！")


if __name__ == "__main__":
    build_and_test_practice_graph()
