"""
DocuMind-Agent: 工业级双阶段混合检索引擎 (Two-Stage Hybrid RAG)
架构升级：
1. 分词升级：jieba 工业精确分词 + 停用词过滤 (Stopwords)
2. 持久化升级：chromadb.PersistentClient 本地磁盘持久化 + 内容 MD5 指纹去重缓存
3. 粗排阶段：BM25 词频检索 + BAAI/bge-m3 语义向量检索 + RRF 倒数排名融合
4. 精排阶段：集成 BAAI/bge-reranker-v2-m3 深度交叉注意力二次重排
"""
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
import jieba
from dotenv import load_dotenv
from openai import OpenAI
from src.schemas import VerificationStatus

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.rag.chunking import recursive_semantic_chunking, parent_child_chunking, ParentChunk
    from src.rag.reranker import BGEReranker
except ModuleNotFoundError:
    from chunking import recursive_semantic_chunking, parent_child_chunking, ParentChunk
    from reranker import BGEReranker

load_dotenv()


# ==========================================
# 1. 工业级分词器与法律停用词过滤
# ==========================================
STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "及", "其", "或", "等", "与", "为", "对", "以", "按", "由",
    "，", "。", "；", "：", "“", "”", "（", "）", "、", "\n", " ", "\t"
}


def tokenize(text: str) -> List[str]:
    """
    工业级中文分词：
    1. 使用 jieba 精确分词；
    2. 过滤停用词与空白字符。
    """
    words = jieba.lcut(text.strip().lower())
    return [w for w in words if w not in STOP_WORDS and len(w.strip()) > 0]


# ==========================================
# 2. 打造专属中文 Embedding 翻译官 (BAAI/bge-m3)
# ==========================================
class BGE3EmbeddingFunction(EmbeddingFunction):
    """接入 BAAI/bge-m3 顶级中文向量模型"""
    def __init__(self):
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = "BAAI/bge-m3"

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB 会自动把文本清单传给这个方法"""
        response = self.client.embeddings.create(
            model=self.model,
            input=input
        )
        return [item.embedding for item in response.data]


# ==========================================
# 3. 双阶段持久化混合检索引擎 (HybridRetriever)
# ==========================================
class HybridRetriever:
    """具备本地磁盘持久化、MD5 缓存去重与父子块回溯能力的两阶段混合检索引擎"""

    def __init__(
        self, 
        full_text: str, 
        chunk_size: int = 300, 
        overlap: int = 50, 
        persist_dir: str = "chroma_db",
        use_parent_child: bool = False,
        sections: Optional[List[Any]] = None
    ):
        self.use_parent_child = use_parent_child
        self.full_text = full_text
        self.sections = sections
        self.last_retrieval_mode: str = "hybrid_rerank"
        self.parent_map: dict[str, str] = {}
        self.child_to_parent: dict[str, str] = {}

        # 1. 计算当前合同内容的 MD5 唯一数字指纹（前 12 位）
        prefix = "pc_" if use_parent_child else "doc_"
        self.doc_hash = hashlib.md5(full_text.encode("utf-8")).hexdigest()[:12]
        collection_name = f"{prefix}{self.doc_hash}"

        # 2. 改造为磁盘持久化客户端（落盘到本地 chroma_db 目录）
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = BGE3EmbeddingFunction()

        # 3. 检查该合同是否早已经入库落盘过？
        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        
        if collection_name in existing_collections:
            # 命中磁盘缓存！直接秒级复用，0 Token 浪费
            print(f"[*] [CACHE] 命中本地持久化向量缓存 (Collection: {collection_name})，跳过切分与 Embedding 生成！")
            self.collection = self.chroma_client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_fn
            )
            # 从已有的向量库中恢复文本切片
            data = self.collection.get()
            self.chunks = data["documents"]
            if use_parent_child and data.get("metadatas"):
                for doc, meta in zip(data["documents"], data["metadatas"]):
                    if meta and "parent_text" in meta:
                        self.child_to_parent[doc] = meta["parent_text"]
        else:
            # 🆕 首次遇到该合同：执行切分、计算向量并持久化落盘
            if use_parent_child:
                print(f"[*] 首次入库新合同，正在构建【父子块索引 (Small-to-Big)】(指纹: {self.doc_hash})...")
                parent_chunks = parent_child_chunking(
                    full_text,
                    parent_chunk_size=700,
                    parent_overlap=100,
                    child_chunk_size=chunk_size,
                    child_overlap=overlap
                )
                self.chunks = []
                chunk_metas = []
                for p in parent_chunks:
                    self.parent_map[p.parent_id] = p.text
                    for c in p.children:
                        self.chunks.append(c.text)
                        self.child_to_parent[c.text] = p.text
                        chunk_metas.append({
                            "parent_id": p.parent_id,
                            "child_id": c.child_id,
                            "parent_text": p.text
                        })

                if not self.chunks:
                    self.chunks = [full_text]
                    chunk_metas = [{"parent_id": "p0", "child_id": "c0", "parent_text": full_text}]

                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn
                )
                chunk_ids = [f"{self.doc_hash}_pc_{i+1}" for i in range(len(self.chunks))]
                self.collection.add(
                    documents=self.chunks,
                    metadatas=chunk_metas,
                    ids=chunk_ids
                )
                print(f"[*] 父子块持久化落盘成功！共管理 {len(parent_chunks)} 个父块，{len(self.chunks)} 个子切片。")
            else:
                print(f"[*] 首次入库新合同，正在切分并调用 BGE-M3 向量化 (指纹: {self.doc_hash})...")
                self.chunks = recursive_semantic_chunking(full_text, chunk_size=chunk_size, overlap=overlap)
                if not self.chunks:
                    self.chunks = [full_text]

                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn
                )
                chunk_ids = [f"{self.doc_hash}_{i+1}" for i in range(len(self.chunks))]
                self.collection.add(
                    documents=self.chunks,
                    ids=chunk_ids
                )
                print(f"[*] 持久化落盘成功！共入库 {len(self.chunks)} 个向量切片。")

        # 4. 构建 BM25 词频索引
        tokenized_corpus = [tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 5. 初始化交叉精排器
        self.reranker = BGEReranker()

    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        rrf_k: int = 60,
        use_rerank: bool = True,
        mode: str = "hybrid_rerank",
        return_parent: bool = False
    ) -> List[Tuple[str, float]]:
        """
        多模式检索执行函数（支持消融实验评估与父子块回溯）
        :param query: 搜索问题或审查主题
        :param top_k: 最终挑出的高纯度结果数量
        :param rrf_k: RRF 平滑常数 (默认 60)
        :param use_rerank: 兼容保留，当 mode="hybrid_rerank" 且 use_rerank=False 时自动退化为 hybrid_rrf
        :param mode: 检索模式，支持：
                     - 'pure_vector': 纯稠密向量检索 (ChromaDB BGE-M3)
                     - 'pure_bm25': 纯稀疏关键词检索 (BM25)
                     - 'hybrid_rrf': 双路粗排融合 (BM25 + BGE-M3 + RRF)
                     - 'hybrid_rerank': 粗排融合 + BGE-Reranker 交叉精排 (默认推荐)
        :param return_parent: 是否将命中的子块自动升维还原为完整父块上下文 (Small-to-Big)
        :return: [(文档文本, 最终置信得分), ...]
        """
        # 兼容兼容旧参数
        if not use_rerank and mode == "hybrid_rerank":
            mode = "hybrid_rrf"

        # --- 模式 1：纯向量检索 (Pure Vector) ---
        if mode == "pure_vector":
            vector_res = self.collection.query(query_texts=[query], n_results=top_k)
            docs = vector_res["documents"][0] if vector_res["documents"] else []
            dists = vector_res["distances"][0] if vector_res.get("distances") and vector_res["distances"] else [0.0] * len(docs)
            results = [(doc, round(1.0 / (1.0 + d), 4)) for doc, d in zip(docs, dists)]
            return self._resolve_parents(results, return_parent, top_k)

        # --- 模式 2：纯 BM25 检索 (Pure BM25) ---
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = [
            (doc, float(score)) for doc, score in sorted(zip(self.chunks, bm25_scores), key=lambda x: x[1], reverse=True)
        ]
        if mode == "pure_bm25":
            return self._resolve_parents(bm25_ranked[:top_k], return_parent, top_k)

        # --- 模式 3 & 4：双路召回 (BM25 + 向量) + RRF 融合 ---
        # 工业级漏斗设计：扩大初筛候选集至 40 个，给 Cross-Encoder 充足的精排重排序空间
        fetch_count = min(len(self.chunks), max(top_k * 10, 40))
        vector_res = self.collection.query(query_texts=[query], n_results=fetch_count)
        vector_docs = vector_res["documents"][0] if vector_res["documents"] else []

        rrf_scores = defaultdict(float)
        for rank, (doc, _) in enumerate(bm25_ranked[:fetch_count], start=1):
            rrf_scores[doc] += 1.0 / (rrf_k + rank)
        for rank, doc in enumerate(vector_docs, start=1):
            rrf_scores[doc] += 1.0 / (rrf_k + rank)

        rough_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_docs = [doc for doc, _ in rough_ranked[:fetch_count]]

        # 如果仅需混合检索 (hybrid_rrf)
        if mode == "hybrid_rrf" or not candidate_docs:
            return self._resolve_parents(rough_ranked[:top_k], return_parent, top_k)

        # --- 模式 4：混合检索 + BGE-Reranker 交叉精排 ---
        try:
            reranked = self.reranker.rerank(query, candidate_docs, top_n=top_k)
            self.last_retrieval_mode = "hybrid_rerank"
            return self._resolve_parents(reranked, return_parent, top_k)
        except Exception as e:
            print(f"[警告] Reranker 服务异常，自动降级为 RRF 粗排结果: {e}")
            self.last_retrieval_mode = "hybrid_rrf"
            return self._resolve_parents(rough_ranked[:top_k], return_parent, top_k)

    def _resolve_parents(
        self, 
        results: List[Tuple[str, float]], 
        return_parent: bool, 
        top_k: int
    ) -> List[Tuple[str, float]]:
        """若开启 return_parent，将子块映射回父块并自动去重"""
        if not return_parent or not self.child_to_parent:
            return results[:top_k]

        parent_results = []
        seen_parents = set()

        for chunk, score in results:
            parent_text = self.child_to_parent.get(chunk, chunk)
            if parent_text not in seen_parents:
                seen_parents.add(parent_text)
                parent_results.append((parent_text, score))

            if len(parent_results) >= top_k:
                break

        return parent_results

    def get_chunk_location(self, doc_text: str) -> dict[str, Any]:
        """
        根据引用的文本片段，在结构化章节与原始文档中精确计算物理行号与页码。
        优先使用 sections 做 0 误差物理行号与真实页码映射，消灭 8 行累积漂移；未定位到时返回 UNVERIFIED。
        """
        if not doc_text:
            return {
                "line_start": None, 
                "line_end": None, 
                "page": None, 
                "status": "unlocated",
                "verification_status": VerificationStatus.UNVERIFIED
            }

        search_snippet = doc_text.strip()[:40]
        fallback_snippet = search_snippet[:20] if len(search_snippet) > 20 else search_snippet

        # 方案 1: 若持有解析期产出的结构化章节 (ParsedSection)，实现绝对 0 误差定位与真实页码提取
        if self.sections:
            for sec in self.sections:
                sec_text = getattr(sec, "content", "")
                idx = sec_text.find(search_snippet)
                if idx == -1 and len(fallback_snippet) >= 15:
                    idx = sec_text.find(fallback_snippet)

                if idx != -1:
                    line_offset = sec_text[:idx].count("\n")
                    sec_line_start = getattr(sec, "line_start", 1) or 1
                    line_start = sec_line_start + line_offset
                    line_end = line_start + doc_text.strip().count("\n")
                    page = getattr(sec, "page_start", None)
                    return {
                        "line_start": line_start,
                        "line_end": line_end,
                        "page": page,
                        "status": "located",
                        "verification_status": VerificationStatus.VERIFIED
                    }

        # 方案 2: 降级在全文粗定位
        idx = self.full_text.find(search_snippet)
        if idx == -1 and len(fallback_snippet) >= 15:
            idx = self.full_text.find(fallback_snippet)

        if idx != -1:
            line_start = self.full_text[:idx].count("\n") + 1
            line_end = line_start + doc_text.strip().count("\n")
            return {
                "line_start": line_start,
                "line_end": line_end,
                "page": None,
                "status": "located",
                "verification_status": VerificationStatus.VERIFIED
            }

        return {
            "line_start": None,
            "line_end": None,
            "page": None,
            "status": "unlocated",
            "verification_status": VerificationStatus.UNVERIFIED
        }


if __name__ == "__main__":
    contract_file = Path("tests/sample_contracts/large_scale_procurement_contract.md")
    with open(contract_file, "r", encoding="utf-8") as f:
        contract_content = f.read()

    print(f"[*] 载入真实合同样本: {contract_file.name}，全文共 {len(contract_content)} 字")
    
    # 实例化双阶段检索引擎（观察第一次 vs 第二次的控制台反馈！）
    retriever = HybridRetriever(contract_content, chunk_size=350, overlap=50)

    # 检索实测
    test_q = "若甲方未按约定付款，乙方是否有权暂停服务？甲方是否承担滞纳金？"
    print(f"\n🔍 检索专项: '{test_q}'")
    hits = retriever.retrieve(test_q, top_k=2, use_rerank=True)
    for rank, (doc, score) in enumerate(hits, start=1):
        clean_text = doc.replace('\n', ' ')
        print(f"[Top {rank}] (Reranker 置信得分: {score:.4f}):\n   {clean_text}\n")
