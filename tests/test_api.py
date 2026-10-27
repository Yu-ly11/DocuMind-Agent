"""
DocuMind-Agent: FastAPI 端点自动化测试套件
验证微服务健康探针、上传解析、扫描件异常熔断与请求校验。
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

venv_site = str(Path(root_dir) / ".venv" / "Lib" / "site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.api import app

client = TestClient(app)


def test_health_check():
    """测试健康检查探针"""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DocuMind-Agent"
    assert len(data["features"]) >= 4


def test_upload_markdown_file():
    """测试 Markdown 文件上传与物理行号解析"""
    sample_file = Path("tests/sample_contracts/sample_procurement_contract.md")
    assert sample_file.exists()

    with open(sample_file, "rb") as f:
        res = client.post(
            "/api/v1/audit/upload",
            files={"file": ("sample_procurement_contract.md", f, "text/markdown")}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "sample_procurement_contract.md"
    assert data["char_count"] > 100
    assert data["line_count"] > 10


def test_upload_scanned_pdf_rejection():
    """测试扫描件空文本拦截熔断 (应返回 422)"""
    scanned_file = Path("tests/sample_contracts/mock_scanned_contract.pdf")
    assert scanned_file.exists()

    with open(scanned_file, "rb") as f:
        res = client.post(
            "/api/v1/audit/upload",
            files={"file": ("mock_scanned_contract.pdf", f, "application/pdf")}
        )
    assert res.status_code == 422
    data = res.json()
    assert "纯图片扫描件拦截" in data["detail"]


def test_upload_invalid_file_format():
    """测试非合同格式拦截 (应返回 400)"""
    res = client.post(
        "/api/v1/audit/upload",
        files={"file": ("malicious.exe", b"test", "application/octet-stream")}
    )
    assert res.status_code == 400
    assert "不支持的文件格式" in res.json()["detail"]


def test_audit_empty_contract_validation():
    """测试审查请求参数校验 (空文本应拦截)"""
    res = client.post(
        "/api/v1/audit/sync",
        json={"contract_text": "   ", "instruction": "test"}
    )
    assert res.status_code == 400
    assert "合同文本不能为空" in res.json()["detail"]


if __name__ == "__main__":
    test_health_check()
    test_upload_markdown_file()
    test_upload_scanned_pdf_rejection()
    test_upload_invalid_file_format()
    test_audit_empty_contract_validation()
    print("ALL API TESTS PASSED!")
