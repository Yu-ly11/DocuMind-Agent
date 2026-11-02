"""
DocuMind-Agent: 工业级轻量可观测性体系 (Observability & Tracing)
提供基于 UUID 的全链路 Trace ID、节点耗时瀑布流、Token 消耗统计与结构化 JSONL 本地持久化。
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

TRACE_DIR = Path("tmp_contracts") / "traces"


def new_trace_id() -> str:
    """生成全局唯一的 Trace ID"""
    return uuid.uuid4().hex


def extract_token_usage(response: Any) -> Dict[str, int]:
    """从大模型 Completion 响应对象中提取 Token 消耗"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return {
            "prompt_tokens": dumped.get("prompt_tokens", 0) or 0,
            "completion_tokens": dumped.get("completion_tokens", 0) or 0,
            "total_tokens": dumped.get("total_tokens", 0) or 0,
        }
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


class Tracer:
    """全链路追踪器：记录节点流转、耗时与 Token"""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or new_trace_id()
        self.records: List[Dict[str, Any]] = []
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = TRACE_DIR / f"{self.trace_id}.jsonl"

    def span(
        self, 
        node: str, 
        status: str = "ok", 
        duration_ms: float = 0.0, 
        tokens: Optional[Dict[str, int]] = None,
        **extra
    ) -> Dict[str, Any]:
        """记录单个节点的执行跨度 (Span)"""
        record = {
            "trace_id": self.trace_id,
            "ts": time.time(),
            "node": node,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "tokens": tokens or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            **extra
        }
        self.records.append(record)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return record

    def get_summary(self) -> Dict[str, Any]:
        """获取当前 Trace 的汇总统计"""
        total_duration = sum(r.get("duration_ms", 0.0) for r in self.records)
        total_prompt_tokens = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in self.records)
        total_completion_tokens = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in self.records)
        total_tokens = sum(r.get("tokens", {}).get("total_tokens", 0) for r in self.records)

        return {
            "trace_id": self.trace_id,
            "spans_count": len(self.records),
            "total_duration_ms": round(total_duration, 2),
            "total_tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens or (total_prompt_tokens + total_completion_tokens),
            },
            "nodes": [r.get("node") for r in self.records]
        }


def load_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """从磁盘加载指定 Trace ID 的 JSONL 追踪记录与聚合统计"""
    path = TRACE_DIR / f"{trace_id}.jsonl"
    if not path.exists():
        return None
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        total_duration_ms = sum(r.get("duration_ms", 0.0) for r in records)
        prompt_tokens = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in records)
        comp_tokens = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in records)
        return {
            "trace_id": trace_id,
            "spans_count": len(records),
            "total_duration_sec": round(total_duration_ms / 1000.0, 2),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": comp_tokens,
            "spans": [
                {
                    "node_name": r.get("node"),
                    "duration_ms": r.get("duration_ms", 0.0),
                    "total_tokens": r.get("tokens", {}).get("total_tokens", 0),
                    "status": r.get("status")
                }
                for r in records
            ],
            "raw_records": records
        }
    except Exception:
        return None
