"""
DocuMind-Agent: 工业级 Cross-Encoder 二次交叉精排器 (Reranker)
基于 BAAI/bge-reranker-v2-m3 深度语义交叉注意力模型。
"""
import os
import requests
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()


class BGEReranker:
    """基于智源 BGE-Reranker-v2-m3 的高精度二次重排器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # 避免受本地代理影响
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        # 拼接 Rerank 专属端点
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        self.rerank_url = f"{base_url}/rerank"
        self.model_name = model_name

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        对粗排召回的一批候选文本进行深度交叉打分并重排
        :param query: 用户问题或审查专项
        :param documents: 粗排（BM25+向量）捞出来的候选块列表（通常 10~20 个）
        :param top_n: 最终精挑细选保留几个最核心的块
        :return: [(文档文本, 深度相关度得分 0~1), ...]
        """
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }

        response = requests.post(self.rerank_url, json=payload, headers=headers, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Reranker 服务调用失败: {response.status_code} - {response.text}")

        results = response.json().get("results", [])
        
        # 将重排结果按原文档和新打分重新组装
        reranked_docs = []
        for item in results:
            original_idx = item["index"]
            score = item["relevance_score"]
            reranked_docs.append((documents[original_idx], score))

        return reranked_docs


if __name__ == "__main__":
    # 极简单元测试
    reranker = BGEReranker()
    test_query = "服务商自有的底层框架和专利归谁？"
    candidate_docs = [
        "第一条：双方共同开发智慧供应链系统，工期两个月。",
        "第五条：乙方原有的底层架构、历史算法库及相关专利，本合同生效起无偿转让甲方排他所有。",
        "第八条：系统终验后享有24个月免费原厂运维保障。",
        "第四条：常规行业适用的沉默认可规则在此被排除，验收不受时间限制。"
    ]

    print(f"用户查询: '{test_query}'\n")
    print("--- 正在调用 BGE-Reranker-v2-m3 进行交叉注意力深度精排 ---")
    final_hits = reranker.rerank(test_query, candidate_docs, top_n=2)
    for rank, (doc, score) in enumerate(final_hits, start=1):
        print(f"\n[Top {rank}] (Cross-Encoder相关度打分: {score:.4f}):\n   {doc}")