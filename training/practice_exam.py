r"""
================================================================================
DocuMind-Agent: 核心架构闭卷实战练习题 (Industrial Coding Exam)
================================================================================
请补齐以下 3 道工业级核心算法题（本题库抽取自当前系统的三大硬核底层）。
全部补齐后，直接在控制台运行本脚本验证：
    .\.venv\Scripts\python.exe practice_exam.py

如果全部断言 ASSERTION PASSED，说明你对 RAG 融合、父子块去重与状态机循环了然于胸！
================================================================================
"""

import sys
from typing import List, Dict, Any, Tuple

# 强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 第一题：手写 RRF (Reciprocal Rank Fusion) 倒数排名融合算法
# ==============================================================================
# 【背景说明】：
# BM25 输出的是词频得分（比如 3.2、1.5），向量模型输出的是余弦距离（比如 0.85、0.62）。
# 两者量纲完全不同，绝不能直接相加！工业界标准解法是 RRF：
# RRF_Score(d) = Σ (1 / (k + rank_i(d)))
# 其中 rank_i 是该文档在第 i 个路召回榜单里的排名（从 1 开始），k 是平滑常数（标准取 60）。
#
# 【任务要求】：
# 补齐下面的 rrf_score_fusion 函数，返回按 RRF 得分从高到低排序的文档及得分元组列表。
# ==============================================================================

def rrf_score_fusion(
    bm25_ranked_docs: List[str], 
    vector_ranked_docs: List[str], 
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    计算两个排序列表的 RRF 融合分数并从高到低排序。
    :param bm25_ranked_docs: BM25 按相关性从高到低排序的文本列表
    :param vector_ranked_docs: 向量检索按相似度从高到低排序的文本列表
    :param k: 平滑因子，默认 60
    :return: 排序后的 [(doc, final_rrf_score), ...]
    """
    scores: Dict[str, float] = {}

    # TODO: [请你补齐核心代码]
    # 1. 遍历 bm25_ranked_docs，累加 1.0 / (k + rank)
    # 2. 遍历 vector_ranked_docs，累加 1.0 / (k + rank)
    # 3. 按最终分数降序排序并返回
    # -------------------------------------------------------------
    # YOUR CODE HERE
    pass
    # -------------------------------------------------------------
def rrf_score_fusion(bm25_ranked_docs,vector_ranked_docs,k=60):
    scores = {}
    for rank,doc in enumerate(bm25_ranked_docs,start=1):
        scores[doc]=scores.get(doc,0.0)+1.0/(k+rank)

    for rank,doc in enumerate(vector_ranked_docs,start=1):
        scores[doc]=scores.get(doc,0.0)+1.0/(k+rank)
    return sorted(scores.items(),key= lambda x: x[1],reverse=True) 


# ==============================================================================
# 第二题：Small-to-Big 父块回溯与拓扑去重算法
# ==============================================================================
# 【背景说明】：
# 子块（180字）命中度高，但丢失了上下文；父块（700字）上下文完整。
# 如果一次检索命中了 3 个子块，而其中有 2 个子块属于同一个父块，或者命中了相邻父块，
# 我们必须回溯其 parent_id，并按照父块在原文中的物理出现顺序进行去重。
# 绝不能让大模型看到重复或顺序错乱的上下文！
#
# 【任务要求】：
# 补齐 deduplicate_and_fetch_parents 函数：
# 接收命中的子块列表（包含 parent_id），从 parent_store 字典中提取对应父块，
# 按照 parent_id 的自然升序去重并拼接。
# ==============================================================================

def deduplicate_and_fetch_parents(
    retrieved_child_chunks: List[Dict[str, Any]], 
    parent_store: Dict[int, str]
) -> List[str]:
    """
    根据命中的子块，反查父块存储并按物理先后顺序去重。
    :param retrieved_child_chunks: 命中的子块字典列表，每个字典包含 {"child_id": str, "parent_id": int, "text": str}
    :param parent_store: 父块字典，格式为 {parent_id: parent_full_text}
    :return: 去重后按 parent_id 从小到大排序的父块文本列表
    """
    # TODO: [请你补齐核心代码]
    # 1. 提取所有命中子块里的 parent_id 并去重
    # 2. 按照 parent_id 从小到大升序排序
    # 3. 从 parent_store 中取出对应父块文本返回
    # -------------------------------------------------------------
    # YOUR CODE HERE
    pass
    # -------------------------------------------------------------
def deduplicate_and_fetch_parents(retrieved_child_chunks,parent_store):
    unique_parent_ids =sorted(list({chunk["parent_id"] for chunk in retrieved_child_chunks if "parent_id" in chunk}))
    return [parent_store[pid] for pid in unique_parent_ids if pid in parent_store]

# ==============================================================================
# 第三题：手写 Critic 事实核查与防死循环路由
# ==============================================================================
# 【背景说明】：
# 大模型经常“编造”不存在的合同行号或幻觉条款。
# 工业级 Critic 必须用确定的代码执行 Grounding Check：
# 提取的摘录证据 snippet 必须 100% 存在于 contract_full_text 中！
# 如果不在，说明大模型在脑补，必须驳回重试；但为了防止死循环，最多重试 3 次！
#
# 【任务要求】：
# 补齐 critic_grounding_evaluator 函数：
# 返回一个字典包含:
# - "is_grounded": bool (判断 snippet 是否在 contract_full_text 中)
# - "retry_count": int (若核验失败则重试次数 + 1，否则保持原样)
# - "next_action": str ("pass" 放行进入下一步评估; "retry" 驳回改写重试; "abort" 重试达到3次强制终止)
# ==============================================================================

def critic_grounding_evaluator(
    snippet: str, 
    contract_full_text: str, 
    current_retry_count: int, 
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    执行事实依据核验与循环路由决策
    """
    # TODO: [请你补齐核心代码]
    # 1. 检查 snippet 是否真实存在于 contract_full_text 中
    # 2. 若存在，返回 is_grounded=True, retry_count不变, next_action="pass"
    # 3. 若不存在：
    #    - 新重试次数 = current_retry_count + 1
    #    - 若新重试次数 >= max_retries: next_action = "abort"
    #    - 否则: next_action = "retry"
    # -------------------------------------------------------------
    # YOUR CODE HERE
    pass
    # -------------------------------------------------------------

def critic_grounding_evaluator(
    snippet: str, 
    contract_full_text: str, 
    current_retry_count: int, 
    max_retries: int = 3
):
    is_grounded = snippet in contract_full_text
    if is_grounded:
        return {
            "is_grounded": True,
            "retry_count": current_retry_count,
            "next_action": "pass"
        }
    new_retries = current_retry_count + 1
    next_action = "abort" if new_retries >= max_retries else "retry"
    return {
            "is_grounded": False,
            "retry_count": new_retries,
            "next_action": next_action
        }
# ==============================================================================
# 自动化检验单元测试套件 (Self-Checking Test Suite)
# ==============================================================================
def run_all_tests():
    print("=" * 70)
    print("🧪 正在启动实战练习自动化阅卷检验...")
    print("=" * 70)

    # ----------------- 测试第一题 -----------------
    bm25_test = ["条款A: 违约金", "条款B: 付款周期", "条款C: 验收标准"]
    vec_test = ["条款B: 付款周期", "条款A: 违约金", "条款D: 知识产权"]
    
    rrf_res = rrf_score_fusion(bm25_test, vec_test, k=60)
    assert rrf_res is not None, "❌ 第一题未实现：返回值为 None"
    assert len(rrf_res) == 4, f"❌ 第一题错误：融合文档数应为 4，实际为 {len(rrf_res)}"
    
    # 条款A: rank 1 (bm25) + rank 2 (vec) -> 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # 条款B: rank 2 (bm25) + rank 1 (vec) -> 1/62 + 1/61 = 0.032522
    top1_doc, top1_score = rrf_res[0]
    assert top1_score > 0.032, f"❌ 第一题分数计算错误: Top1 分数 {top1_score} 不符合预期"
    print("✅ [第一题: RRF 倒数排名融合算法] 验证通过！")

    # ----------------- 测试第二题 -----------------
    parent_db = {
        0: "【第一章 总则】双方诚信履约...",
        1: "【第二章 付款流程】甲方分三期支付...",
        2: "【第三章 违约责任】逾期每日支付 1.5%...",
        3: "【第四章 验收规范】系统终验流程..."
    }
    # 命中子块：有两个命中属于父块 2，一个属于父块 1，无序传入
    mock_children = [
        {"child_id": "c_2_1", "parent_id": 2, "text": "每日支付 1.5%"},
        {"child_id": "c_1_0", "parent_id": 1, "text": "甲方分三期支付"},
        {"child_id": "c_2_0", "parent_id": 2, "text": "逾期违约"}
    ]
    parents_res = deduplicate_and_fetch_parents(mock_children, parent_db)
    assert parents_res is not None, "❌ 第二题未实现：返回值为 None"
    assert len(parents_res) == 2, f"❌ 第二题去重失败：应当合并为 2 个父块，实际得到 {len(parents_res)}"
    assert "付款流程" in parents_res[0], "❌ 第二题拓扑顺序错误：应当按 parent_id 顺序，父块 1 应当在父块 2 前面！"
    print("✅ [第二题: Small-to-Big 父块回溯与拓扑去重] 验证通过！")

    # ----------------- 测试第三题 -----------------
    full_contract = "【第七条】乙方逾期交付，每日须向甲方支付合同总额 1.5% 的逾期违约金。"
    
    # 3.1 真实匹配通过测试
    real_snippet = "乙方逾期交付，每日须向甲方支付合同总额 1.5%"
    check_pass = critic_grounding_evaluator(real_snippet, full_contract, current_retry_count=0)
    assert check_pass is not None, "❌ 第三题未实现：返回值为 None"
    assert check_pass["is_grounded"] is True and check_pass["next_action"] == "pass", "❌ 第三题通过逻辑错误"

    # 3.2 幻觉重试测试
    hallucinated_snippet = "乙方如果不按时交货，就要赔偿甲方一百个亿。"
    check_retry = critic_grounding_evaluator(hallucinated_snippet, full_contract, current_retry_count=1)
    assert check_retry["is_grounded"] is False and check_retry["next_action"] == "retry" and check_retry["retry_count"] == 2, "❌ 第三题重试逻辑错误"

    # 3.3 超限熔断测试
    check_abort = critic_grounding_evaluator(hallucinated_snippet, full_contract, current_retry_count=2, max_retries=3)
    assert check_abort["next_action"] == "abort", "❌ 第三题达到上限未熔断"
    print("✅ [第三题: Critic 事实核查与防死循环路由] 验证通过！")

    print("\n" + "=" * 70)
    print("🎉 恭喜！你全部亲手完成了三大工业级算法算子，断言全部满分通过！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
