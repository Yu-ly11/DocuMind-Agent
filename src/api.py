"""
DocuMind-Agent: 工业级合规审查微服务 (FastAPI + Server-Sent Events)
提供生产级 RESTful 与实时 SSE 状态机流式推送端点。
"""

import os
import sys
import json
import uuid
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Generator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.parser import DocumentParser, ScannedPDFError
from src.graph_agent import LangGraphComplianceAgent
from src.schemas import ContractAuditReport

# 初始化 FastAPI 应用
app = FastAPI(
    title="DocuMind-Agent Enterprise Compliance API",
    description="工业级企业合同合规审查微服务 —— 基于 LangGraph 状态机、Small-to-Big 双阶段 RAG 与确定性算子",
    version="1.0.0"
)

# 跨域资源共享配置 (支持前端前端框架如 React/Vue/Next.js 对接)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 请求/响应数据模型
# ==========================================
class AuditRequest(BaseModel):
    contract_text: str = Field(..., description="待审查合同纯文本全文")
    instruction: str = Field(
        default="请对本合同进行全方位合规审查，重点覆盖：知识产权归属、违约金比例合法性、系统验收与单方解除陷阱。",
        description="合规审查指令"
    )
    thread_id: Optional[str] = Field(default=None, description="会话持久化 Thread ID，便于 LangGraph 检查点恢复")
    use_parent_child: bool = Field(default=True, description="是否启用 Small-to-Big 父子块混合索引")


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    features: list[str]


# ==========================================
# 核心端点
# ==========================================

@app.get("/health", response_model=HealthResponse, tags=["系统监控"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["系统监控"])
async def health_check():
    """微服务健康探针接口"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="DocuMind-Agent",
        features=[
            "DocumentParser (High-fidelity geometry crop & table format)",
            "HybridRetriever (Small-to-Big BM25 + BGE-M3 + Rerank)",
            "ComplianceTools (Deterministic penalty calculation)",
            "LangGraph (5-node reflective state machine)",
            "Server-Sent Events (Real-time token & node streaming)"
        ]
    )


@app.post("/api/v1/audit/upload", tags=["文档解析"])
async def upload_contract(file: UploadFile = File(...)):
    """
    上传合同文件并执行高保真解析
    支持 .pdf 与 .md / .txt
    内置自适应页边距几何裁切与纯图片扫描件熔断拦截
    """
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in [".pdf", ".md", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: '{file_ext}'。系统仅支持 .pdf, .md, .txt 合同文件。"
        )

    # 写入系统临时安全文件
    temp_dir = Path(tempfile.gettempdir()) / "documind_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / f"{uuid.uuid4().hex}_{file.filename}"

    try:
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # 调用高保真解析引擎
        extracted_text = DocumentParser.extract_full_text(temp_file_path)
        total_lines = extracted_text.count("\n") + 1
        total_chars = len(extracted_text)

        return {
            "success": True,
            "filename": file.filename,
            "char_count": total_chars,
            "line_count": total_lines,
            "preview": extracted_text[:400] + ("..." if total_chars > 400 else ""),
            "full_text": extracted_text
        }
    except ScannedPDFError as spe:
        raise HTTPException(
            status_code=422,
            detail=f"【纯图片扫描件拦截】{str(spe)} 请提供可检索文本的数字版 PDF 或先接入 OCR 服务。"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档解析失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


@app.post("/api/v1/audit/sync", response_model=ContractAuditReport, tags=["审查业务"])
def run_sync_audit(req: AuditRequest):
    """
    同步审查接口 (适合后端批处理或微服务 RPC 调用)
    直接阻塞等待状态机执行完毕，返回 Pydantic 强类型合规大报告
    """
    if not req.contract_text.strip():
        raise HTTPException(status_code=400, detail="合同文本不能为空")

    thread_id = req.thread_id or f"sync_{uuid.uuid4().hex[:8]}"
    try:
        agent = LangGraphComplianceAgent(
            contract_content=req.contract_text,
            use_parent_child=req.use_parent_child
        )
        report = agent.audit(instruction=req.instruction, thread_id=thread_id)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"状态机审查执行异常: {str(e)}"
        )


@app.post("/api/v1/audit/stream", tags=["审查业务"])
async def run_stream_audit(req: AuditRequest):
    """
    SSE 流式审查接口 (生产级事件流驱动)
    通过 Server-Sent Events 将状态机的 5 个节点状态、决策分支、核验证据实时推送到前端客户端
    
    事件类型包含：
    - `node_step`: 某个节点完成执行，附带节点输出
    - `report_ready`: 最终报告生成
    - `done`: 状态图执行终止
    - `error`: 异常报警
    """
    if not req.contract_text.strip():
        raise HTTPException(status_code=400, detail="合同文本不能为空")

    thread_id = req.thread_id or f"stream_{uuid.uuid4().hex[:8]}"

    def event_generator() -> Generator[Dict[str, str], None, None]:
        try:
            # 1. 建立智能体
            yield {
                "event": "system",
                "data": json.dumps({
                    "stage": "init",
                    "message": "正在构建双阶段混合检索底座与状态机拓扑...",
                    "thread_id": thread_id
                }, ensure_ascii=False)
            }

            agent = LangGraphComplianceAgent(
                contract_content=req.contract_text,
                use_parent_child=req.use_parent_child
            )

            # 2. 流式遍历状态图更新事件
            for node_name, node_output in agent.stream_audit(req.instruction, thread_id=thread_id):
                # 针对不同节点提炼摘要信息供前端高亮显示
                summary_data: Dict[str, Any] = {
                    "node": node_name,
                    "thread_id": thread_id
                }

                if node_name == "task_planner":
                    summary_data["tasks"] = node_output.get("plan_tasks", [])
                    summary_data["message"] = f"成功拆解出 {len(summary_data['tasks'])} 个专项法务议题"
                elif node_name == "hybrid_retriever":
                    clauses = node_output.get("retrieved_clauses", [])
                    summary_data["retrieved_count"] = len(clauses)
                    summary_data["top1_score"] = clauses[0]["score"] if clauses else 0
                    summary_data["message"] = f"混合检索命中 {len(clauses)} 条法条依据"
                elif node_name == "penalty_calculator":
                    cert = node_output.get("penalty_cert")
                    summary_data["has_penalty_cert"] = cert is not None
                    summary_data["penalty_cert"] = cert
                    summary_data["message"] = "确定性风控算子完成法定上限核验" if cert else "未涉违约金计算，平滑跳过"
                elif node_name == "clause_evaluator":
                    issues = node_output.get("accumulated_issues", [])
                    summary_data["total_issues_so_far"] = len(issues)
                    summary_data["message"] = f"当前已累计识别 {len(issues)} 处潜在法律风险"
                elif node_name == "critic":
                    approved = node_output.get("critic_approved", True)
                    summary_data["approved"] = approved
                    summary_data["notes"] = node_output.get("critic_notes", "")
                    summary_data["retry_count"] = node_output.get("retry_count", 0)
                    summary_data["message"] = "反思检验通过，事实证据已溯源" if approved else "事实核查未通过，触发 Query 改写重试！"
                elif node_name == "report_aggregator":
                    summary_data["final_report"] = node_output.get("final_report")
                    summary_data["message"] = "合规审计终审报告生成完毕"

                yield {
                    "event": "node_step",
                    "data": json.dumps(summary_data, ensure_ascii=False)
                }

                # 若是终审报告节点，额外发送报告专有事件
                if node_name == "report_aggregator" and "final_report" in node_output:
                    yield {
                        "event": "report_ready",
                        "data": json.dumps(node_output["final_report"], ensure_ascii=False)
                    }

            # 3. 正常完成信号
            yield {
                "event": "done",
                "data": json.dumps({"status": "completed", "thread_id": thread_id}, ensure_ascii=False)
            }

        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


# ==========================================
# 调试与本地直接运行
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # 强制控制台 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("🚀 启动 DocuMind-Agent 生产级 FastAPI 微服务")
    print("📡 服务监听地址: http://127.0.0.1:8000")
    print("📖 OpenAPI 交互文档: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
