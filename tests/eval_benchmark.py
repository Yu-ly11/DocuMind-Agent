"""
DocuMind-Agent: RAG 工业级三路消融实验与量化评测基准流水线 (Ablation Benchmark)
权威验证三组消融实验：
1. 对照组 1：纯稠密向量检索 (Pure Dense Vector: BAAI/bge-m3)
2. 对照组 2：双路混合粗排检索 (BM25 + BGE-M3 + RRF 融合)
3. 对照组 3：双路混合 + BAAI/bge-reranker-v2-m3 深度交叉重排 (DocuMind 生产方案)

量化指标：
- Hit Rate@1: 黄金法务条目是否精准排在 Top 1
- Hit Rate@3: 黄金法务条目是否进入前三名候选池
- NDCG@3: 归一化折损累积增益 (衡量高相关条款排位的折扣权重)
- MRR: 平均倒数排名 (Mean Reciprocal Rank)
"""
import sys
import math
from pathlib import Path
from typing import List, Dict

# 适配 Windows 控制台 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.rag.hybrid_retriever import HybridRetriever

# ==========================================
# 1. 8 类高危法务陷阱黄金测试集 (Golden Dataset)
# ==========================================
GOLDEN_DATASET = [
    {
        "category": "知识产权侵占",
        "query": "服务商在签约前独立拥有的通用算法和专利归谁？",
        "expected_keyword": "底层背景技术侵夺条款"  # 第5.2条核心特征
    },
    {
        "category": "违约金超标",
        "query": "如果交付延误，每日违约金是多少？",
        "expected_keyword": "1.5%"  # 第7.1条核心特征
    },
    {
        "category": "解约惩罚陷阱",
        "query": "单笔最高的一次性惩罚性违约金比例是多少？",
        "expected_keyword": "一次性违约金"  # 第7.1条核心特征 (50%)
    },
    {
        "category": "验收阻却程序",
        "query": "客户收到系统后如果不表态，是否会自动视为验收合格？",
        "expected_keyword": "沉默认可"  # 第4.2条核心特征
    },
    {
        "category": "单方霸王约束",
        "query": "若客户未按约定付款，服务商能否暂停驻场开发？",
        "expected_keyword": "不得暂停或减缓驻场开发服务"  # 第7.2条核心特征
    },
    {
        "category": "无限连带保密",
        "query": "离职员工如果泄密，服务商要承担什么责任？",
        "expected_keyword": "离职满 5 年以内的前雇员"  # 第6.2条核心特征
    },
    {
        "category": "免责阻却陷阱",
        "query": "甲方在什么情形下可以单方顺延款项且不承担迟延违约金？",
        "expected_keyword": "特别支付阻却条款"  # 第3.3条核心特征 (抗辩权滥用)
    },
    {
        "category": "极限技术考核",
        "query": "系统灾备指标 RTO 和 RPO 考核门槛是多少？",
        "expected_keyword": "小于 15 分钟"  # 第4.1条核心特征
    }
]


def evaluate_pipeline(retriever: HybridRetriever, mode: str, verbose: bool = False) -> Dict[str, float]:
    """对特定检索模式进行客观数学跑分"""
    hit_at_1 = 0
    hit_at_3 = 0
    ndcg_list = []
    reciprocal_ranks = []

    for item in GOLDEN_DATASET:
        q = item["query"]
        target = item["expected_keyword"]
        
        # 执行特定模式检索
        hits = retriever.retrieve(q, top_k=3, mode=mode)
        retrieved_docs = [doc for doc, _ in hits]

        # 检查命中位次 (Rank)
        rank_found = 0
        for idx, doc in enumerate(retrieved_docs, start=1):
            if target in doc:
                rank_found = idx
                break

        if verbose:
            status_icon = "✅" if rank_found == 1 else ("🟡" if rank_found in (2, 3) else "❌")
            print(f"  {status_icon} [{item['category']}] Rank={rank_found} | Q: {q}")
            if rank_found == 0:
                print(f"     期望关键词: '{target}' 未出现在 Top 3 中！")

        # 1. 统计 Hit Rate 与 MRR
        if rank_found == 1:
            hit_at_1 += 1
            hit_at_3 += 1
            reciprocal_ranks.append(1.0)
            ndcg_list.append(1.0)
        elif rank_found in (2, 3):
            hit_at_3 += 1
            reciprocal_ranks.append(1.0 / rank_found)
            dcg = 1.0 / math.log2(rank_found + 1)
            ndcg_list.append(dcg)
        else:
            reciprocal_ranks.append(0.0)
            ndcg_list.append(0.0)

    total = len(GOLDEN_DATASET)
    return {
        "Hit_Rate@1": (hit_at_1 / total) * 100,
        "Hit_Rate@3": (hit_at_3 / total) * 100,
        "NDCG@3": sum(ndcg_list) / total if total > 0 else 0.0,
        "MRR": sum(reciprocal_ranks) / total if total > 0 else 0.0
    }


if __name__ == "__main__":
    contract_file = Path("tests/sample_contracts/large_scale_procurement_contract.md")
    with open(contract_file, "r", encoding="utf-8") as f:
        contract_text = f.read()

    print(f"[*] 载入万字大标的合同: {contract_file.name} (全文共 {len(contract_text)} 字)...")
    print(f"[*] 正在初始化两阶段检索基准引擎...")
    retriever = HybridRetriever(contract_text, chunk_size=350, overlap=50)

    print(f"[*] 权威 Golden Dataset 包含 {len(GOLDEN_DATASET)} 类核心法律风控盲测题。")
    print("\n" + "=" * 70)
    print("🚀 [1/3] 正在对【对照组 1：纯向量检索 (Pure Dense Vector)】进行自动化评测...")
    res_vec = evaluate_pipeline(retriever, mode="pure_vector", verbose=True)

    print("\n🚀 [2/3] 正在对【对照组 2：双路混合检索 (BM25 + Vector + RRF)】进行自动化评测...")
    res_rrf = evaluate_pipeline(retriever, mode="hybrid_rrf", verbose=True)

    print("\n🚀 [3/3] 正在对【对照组 3：双路混合 + BGE-Reranker 交叉精排】进行自动化评测...")
    res_rerank = evaluate_pipeline(retriever, mode="hybrid_rerank", verbose=True)

    print("\n" + "=" * 80)
    print("📊 DocuMind-Agent 检索消融实验核心指标对比榜单 (Ablation Benchmark)")
    print("=" * 80)
    print(f"{'评测指标':<16} | {'纯向量检索 (Baseline)':<20} | {'双路混合检索 (RRF)':<18} | {'混合+BGE交叉精排 (生产方案)':<22} | {'最终净提升'}")
    print("-" * 96)
    
    hr1_lift = res_rerank['Hit_Rate@1'] - res_vec['Hit_Rate@1']
    hr3_lift = res_rerank['Hit_Rate@3'] - res_vec['Hit_Rate@3']
    ndcg_lift = res_rerank['NDCG@3'] - res_vec['NDCG@3']
    mrr_lift = res_rerank['MRR'] - res_vec['MRR']

    print(f"{'Hit Rate@1':<18} | {res_vec['Hit_Rate@1']:>18.1f}% | {res_rrf['Hit_Rate@1']:>16.1f}% | {res_rerank['Hit_Rate@1']:>24.1f}% | {hr1_lift:>+9.1f}%")
    print(f"{'Hit Rate@3':<18} | {res_vec['Hit_Rate@3']:>18.1f}% | {res_rrf['Hit_Rate@3']:>16.1f}% | {res_rerank['Hit_Rate@3']:>24.1f}% | {hr3_lift:>+9.1f}%")
    print(f"{'NDCG@3':<20} | {res_vec['NDCG@3']:>20.4f} | {res_rrf['NDCG@3']:>18.4f} | {res_rerank['NDCG@3']:>26.4f} | {ndcg_lift:>+10.4f}")
    print(f"{'MRR':<21} | {res_vec['MRR']:>21.4f} | {res_rrf['MRR']:>19.4f} | {res_rerank['MRR']:>27.4f} | {mrr_lift:>+10.4f}")
    print("=" * 80)
