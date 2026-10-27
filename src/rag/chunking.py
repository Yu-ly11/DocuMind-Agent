"""
DocuMind-Agent: 文本分块模块 (Chunking)
包含：
1. 固定窗口切块 (Fixed Size Chunking) - 基础对照组
2. 递归语义分块 (Recursive Semantic Chunking) - 工业级推荐实现
3. 父子块切分 (Parent-Child / Small-to-Big Chunking) - 解决“切太碎割裂上下文，切太大稀释向量特征”矛盾
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class ChildChunk(BaseModel):
    """用于微观高精度匹配的子块"""
    child_id: str = Field(..., description="子块唯一标识符，如 child_p1_1")
    parent_id: str = Field(..., description="关联的父块标识符，如 parent_1")
    text: str = Field(..., description="子块精细文本 (120~200 字)")


class ParentChunk(BaseModel):
    """用于宏观完整上下文还原的父块"""
    parent_id: str = Field(..., description="父块唯一标识符，如 parent_1")
    text: str = Field(..., description="完整宏观章节/法条上下文 (600~900 字)")
    children: List[ChildChunk] = Field(default_factory=list, description="所属子切片列表")


def fixed_size_chunking(text: str, chunk_size: int = 50, overlap: int = 10) -> List[str]:
    """
    固定窗口切块算法 (基础对照组)
    缺点：容易在句子中间、数字、专有名词处发生生硬截断，破坏语义。
    
    :param text: 要切分的长文本
    :param chunk_size: 每个块的字符长度
    :param overlap: 前后两块之间的重叠字符数
    :return: 切分好的文本块列表
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap！")
        
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
        start += (chunk_size - overlap)

    return chunks


def recursive_semantic_chunking(
    text: str,
    chunk_size: int = 50,
    overlap: int = 10,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    递归语义切块算法 (Recursive Semantic Chunking)
    原理：
    1. 优先按照自然层级分隔符（段落换行、句号、分号、逗号）切开文本；
    2. 如果某个切片仍然超过 chunk_size，递归降级到下一级分隔符细切；
    3. 最后将细小片段平滑合并，保持在 chunk_size 限制内，并在块与块之间保留 overlap 字符，
       保证句子和关键词的语义完整性。
    """
    if separators is None:
        # 优先级：双换行(段落) -> 单换行 -> 中英文句末标点 -> 分号 -> 逗号 -> 空格 -> 字符兜底
        separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""]

    def _split_text_with_separators(current_text: str, current_seps: List[str]) -> List[str]:
        if not current_text:
            return []
            
        if len(current_text) <= chunk_size or not current_seps:
            return [current_text]

        # 取出当前优先级最高的分隔符
        sep = current_seps[0]
        remaining_seps = current_seps[1:]

        if sep == "":
            # 已经没有更细的分隔符了，硬切兜底
            return [current_text[i : i + chunk_size] for i in range(0, len(current_text), chunk_size - overlap)]

        # 按当前分隔符分割
        if sep in current_text:
            splits = current_text.split(sep)
            reconstructed_splits = []
            for idx, s in enumerate(splits):
                if not s:
                    continue
                # 重新拼接上分隔符（最后一段不拼接），保留标点符号
                piece = s + sep if idx < len(splits) - 1 else s
                reconstructed_splits.append(piece)
        else:
            # 当前文本不含该分隔符，直接用更低级的分隔符尝试
            return _split_text_with_separators(current_text, remaining_seps)

        # 检查切出的片段：如果依然大于 chunk_size，递归向下继续切
        final_splits = []
        for piece in reconstructed_splits:
            if len(piece) > chunk_size and remaining_seps:
                final_splits.extend(_split_text_with_separators(piece, remaining_seps))
            else:
                final_splits.append(piece)

        return final_splits

    # 第一步：递归切解为较小原子语义片段
    raw_pieces = _split_text_with_separators(text, separators)

    # 第二步：将原子片段按 chunk_size 和 overlap 进行平滑组装合并
    final_chunks = []
    current_chunk = ""

    for piece in raw_pieces:
        if not current_chunk:
            current_chunk = piece
        elif len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece
        else:
            final_chunks.append(current_chunk)
            # 计算 overlap 部分，取当前块末尾的 overlap 个字符作为下一块的开头前缀
            overlap_prefix = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_prefix + piece

    if current_chunk:
        final_chunks.append(current_chunk)

    return final_chunks


def parent_child_chunking(
    text: str,
    parent_chunk_size: int = 700,
    parent_overlap: int = 100,
    child_chunk_size: int = 180,
    child_overlap: int = 30
) -> List[ParentChunk]:
    """
    工业级父子块切分 (Small-to-Big Retrieval)
    1. 先切出包含完整章节与免责前提的大块 (Parent Chunk: 600~900 字)；
    2. 在每个父块内部，进一步细切出高语义密度的子块 (Child Chunk: 120~200 字)；
    3. 子块用于在 ChromaDB 和 BM25 中高精度捕捉问题，父块用于回溯送给大模型，避免上下文残缺。
    """
    parent_texts = recursive_semantic_chunking(
        text, 
        chunk_size=parent_chunk_size, 
        overlap=parent_overlap
    )

    parent_chunks: List[ParentChunk] = []

    for p_idx, p_text in enumerate(parent_texts, start=1):
        parent_id = f"parent_{p_idx}"
        
        # 对当前父块进行细粒度子切分
        child_texts = recursive_semantic_chunking(
            p_text, 
            chunk_size=child_chunk_size, 
            overlap=child_overlap
        )

        children = [
            ChildChunk(
                child_id=f"child_{p_idx}_{c_idx}",
                parent_id=parent_id,
                text=c_text
            )
            for c_idx, c_text in enumerate(child_texts, start=1)
        ]

        parent_chunks.append(
            ParentChunk(
                parent_id=parent_id,
                text=p_text,
                children=children
            )
        )

    return parent_chunks


if __name__ == "__main__":
    # 测试用合同条款片段（包含数字、标点与多重责任界定）
    sample_text = (
        "【第五条 违约责任】\n"
        "1. 乙方若逾期交付，每日须向甲方支付违约金5000元人民币；若逾期超过30天，甲方有权单方面解除合同并全额没收履约保证金。\n"
        "2. 因地震、战争等不可抗力事件导致工程延误的，双方互不承担赔偿责任，由双方友好协商延期方案。"
    )

    # 确保在 Windows 控制台输出中文兼容
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 70)
    print("【待测试合同原文】")
    print(sample_text)
    print(f"(原文总字符数: {len(sample_text)} 字)")
    print("=" * 70)

    # 实验设置：单块最大 45 字，重叠 10 字
    CHUNK_SIZE = 45
    OVERLAP = 10

    print(f"\n--- [方案 A] 固定字符硬切块 (chunk_size={CHUNK_SIZE}, overlap={OVERLAP}) ---")
    fixed_chunks = fixed_size_chunking(sample_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    for idx, c in enumerate(fixed_chunks, start=1):
        clean_display = c.replace('\n', ' ')
        print(f"  块 {idx} (长 {len(c):2d} 字): {clean_display}")

    print(f"\n--- [方案 B] 递归语义切块 (chunk_size={CHUNK_SIZE}, overlap={OVERLAP}) ---")
    semantic_chunks = recursive_semantic_chunking(sample_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    for idx, c in enumerate(semantic_chunks, start=1):
        clean_display = c.replace('\n', ' ')
        print(f"  块 {idx} (长 {len(c):2d} 字): {clean_display}")

    print("\n[对比总结与面试要点]:")
    print("  - 方案 A: 在数字、句子中间粗暴断开，容易造成金额、责任主体的语义碎片化；")
    print("  - 方案 B: 优先尊重段落与句号边界，确保切片在向量空间中具备独立且完整的语义！")