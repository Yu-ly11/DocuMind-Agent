"""
DocuMind-Agent: 智能合同交互式流式问答引擎 (Interactive Streaming RAG Engine)
提供毫秒级检索、打字机逐字流式输出与 100% 物理行号证据链高亮。
"""

import os
import sys
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple, Optional
from dotenv import load_dotenv

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import get_llm_client, logger
from src.rag.hybrid_retriever import HybridRetriever
from src.schemas import format_line_range

load_dotenv()
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

CHAT_DB_PATH = Path("tmp_contracts") / "chat_memory.sqlite"


def init_chat_memory_db():
    """初始化 SQLite 会话记忆数据表"""
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_thread ON chat_memory(thread_id, ts)")


def save_chat_message(thread_id: str, role: str, content: str):
    """保存单条会话消息"""
    init_chat_memory_db()
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_memory (thread_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (thread_id, role, content, time.time())
        )


def load_chat_history(thread_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """读取指定会话的历史对话"""
    init_chat_memory_db()
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_memory WHERE thread_id = ? ORDER BY ts DESC LIMIT ?",
            (thread_id, limit)
        )
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


class ContractChatEngine:
    """企业级合同即问即答流式引擎（支持持久化记忆与物理行号证据高亮）"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.provider, self.client, self.model = get_llm_client()

    def query(
        self, 
        user_query: str, 
        history: List[Dict[str, str]] = None,
        top_k: int = 3,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        流式问答接口
        :param user_query: 用户提问
        :param history: 历史对话记录 [{"role": "user"/"assistant", "content": "..."}]
        :param top_k: 召回最相关的法条父块数量
        :param thread_id: 可选持久化会话 ID，提供跨进程与跨会话记忆 (与 session_id 等价)
        :param session_id: 会话标识别名
        :return: (token_stream_generator, citations_list)
        """
        active_thread_id = thread_id or session_id
        # 若未显式传入 history 但提供了 thread_id，从 SQLite 自动恢复记忆
        if history is None and active_thread_id:
            history = load_chat_history(active_thread_id, limit=6)
        # 1. 毫秒级双阶段混合检索 (BM25 + BGE-M3 + Rerank + Small-to-Big)
        hits = self.retriever.retrieve(
            user_query, 
            top_k=top_k, 
            mode="hybrid_rerank", 
            return_parent=True
        )

        # 2. 组装物理证据链与精准行号
        citations = []
        context_blocks = []
        for rank, (clause_text, score) in enumerate(hits, start=1):
            loc = self.retriever.get_chunk_location(clause_text)
            citations.append({
                "rank": rank,
                "score": round(score, 4),
                "line_start": loc["line_start"],
                "line_end": loc["line_end"],
                "page": loc.get("page"),
                "status": loc.get("status", "unlocated"),
                "text": clause_text
            })
            if loc["line_start"] is not None:
                loc_label = f"物理行号: {format_line_range(loc['line_start'], loc['line_end'])}"
            elif loc.get("page") is not None:
                loc_label = f"所在页码: 第 {loc['page']} 页"
            else:
                loc_label = "出处: 全文检索匹配"

            context_blocks.append(
                f"【参考法条依据 {rank} | {loc_label}】\n{clause_text}"
            )

        context_prompt = "\n\n".join(context_blocks)

        system_instruction = (
            "你是一名资深企业合同法务审查专家兼法务助手（DocuMind Contract Copilot）。\n"
            "你的任务是严格根据下方从合同中检索召回的真实条款依据，回答用户的法务与业务问题。\n\n"
            "【行为死命令】：\n"
            "1. 绝不脑补或编造不存在的条款！每个核心结论必须指明其出处的行号区间；\n"
            "2. 如果涉及违约金、赔偿、交付期等数值，必须明确列出条款约定的具体比例或金额；\n"
            "3. 如果条款中存在模糊陷阱、单方豁免或严重偏向某一方的霸王条款，请在回答末尾以【⚠️ 法务风控提醒】单独指出并提供防范建议；\n"
            "4. 回答语言客观、精练、专业，排版使用规范的 Markdown 格式。"
        )

        prompt_messages = [{"role": "system", "content": system_instruction}]

        # 注入最近历史对话（保持上下文连贯性，防御性剔除末尾可能重复的当前问题）
        if history:
            clean_history = list(history)
            if clean_history and clean_history[-1].get("content") == user_query:
                clean_history = clean_history[:-1]
            for msg in clean_history[-4:]:
                prompt_messages.append({"role": msg["role"], "content": msg["content"]})

        # 注入当前带凭证的问题
        prompt_messages.append({
            "role": "user",
            "content": f"【从当前合同中检索出的真实上下文依据】：\n{context_prompt}\n\n【用户问题】：\n{user_query}"
        })

        # 3. 呼叫大模型执行流式打字机生成
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt_messages,
            stream=True,
            temperature=0.2
        )

        def stream_generator():
            full_chunks = []
            for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices else ""
                if content:
                    full_chunks.append(content)
                    yield content

            # 若指定了会话 ID，生成完毕后自动持久化当前轮对话到 SQLite
            if active_thread_id:
                try:
                    save_chat_message(active_thread_id, "user", user_query)
                    save_chat_message(active_thread_id, "assistant", "".join(full_chunks))
                except Exception:
                    pass

        return stream_generator(), citations
