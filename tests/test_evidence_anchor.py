"""
DocuMind-Agent: 证据链物理坐标精确锚定与 Fail-Closed 状态机测试
专门验证消灭行号漂移、真实页码对齐与未闭环安全闸门逻辑。
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

venv_site = str(Path(root_dir) / ".venv" / "Lib" / "site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.parser import DocumentParser
from src.rag.hybrid_retriever import HybridRetriever
from src.schemas import VerificationStatus


def test_markdown_zero_drift_line_anchor():
    """
    测试 1: 验证 Markdown 文件证据反查物理行号实现 100% 绝对 0 漂移
    彻底终结此前因 join('\\n\\n') 丢失空行导致的 -8 行累积偏差！
    """
    sample_file = Path("tests/sample_contracts/large_scale_procurement_contract.md")
    assert sample_file.exists(), "测试样本文件不存在"

    sections = DocumentParser.parse_file(sample_file)
    full_text = DocumentParser.extract_full_text(sample_file)

    # 注入 sections 的高保真检索引擎
    retriever = HybridRetriever(full_text, sections=sections)

    # 抽取前 12 个真实章节进行反查验证
    for sec in sections[:12]:
        snippet = sec.content.strip()[:35]
        loc = retriever.get_chunk_location(snippet)

        # 断言: 定位状态必须为 located
        assert loc["status"] == "located", f"章节 {sec.section_title} 应当精准定位"
        # 断言: 验证状态必须为 VERIFIED
        assert loc["verification_status"] == VerificationStatus.VERIFIED
        # 核心断言: 反查行号必须与真实物理起始行号绝对相等 (Δ == 0)
        delta = loc["line_start"] - sec.line_start
        assert delta == 0, f"章节 {sec.section_title} 出现行号漂移！真实行: {sec.line_start}, 反查行: {loc['line_start']}, 偏差: {delta}"

    print("✅ [测试 1 通过]: Markdown 证据链反查行号实现 100% 绝对 0 漂移验证！")


def test_fail_closed_unverified_anchor():
    """
    测试 2: 验证 Fail-Closed 安全设计
    对于模型脑补或篡改的虚假原文，绝不能误标为 VERIFIED 或默认放行！
    """
    sample_file = Path("tests/sample_contracts/large_scale_procurement_contract.md")
    sections = DocumentParser.parse_file(sample_file)
    full_text = DocumentParser.extract_full_text(sample_file)
    retriever = HybridRetriever(full_text, sections=sections)

    # 假造一个不存在的违约金虚假条款
    hallucinated_text = "乙方如果延期交付超过一天，必须赔偿甲方十个亿人民币违约金并承担刑事责任。"
    loc = retriever.get_chunk_location(hallucinated_text)

    # 断言: 必须为 unlocated，且状态必须为 UNVERIFIED
    assert loc["status"] == "unlocated"
    assert loc["line_start"] is None
    assert loc["verification_status"] == VerificationStatus.UNVERIFIED
    print("✅ [测试 2 通过]: 虚假证据准确拦截，Fail-Closed 安全状态机生效！")


if __name__ == "__main__":
    test_markdown_zero_drift_line_anchor()
    test_fail_closed_unverified_anchor()
    print("\n🎉 全部证据链物理坐标与安全闸门测试 100% PASS！")
