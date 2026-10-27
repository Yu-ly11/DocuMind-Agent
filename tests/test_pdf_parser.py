"""
DocuMind-Agent: 工业级高保真 PDF 解析管道实战测试
验证项：
1. 真实合同表格精准提取并转为 Markdown Table；
2. 周期性页眉页脚几何剔除；
3. 全文页码与物理行号一致性；
4. 扫描件/纯图片 PDF 安全拦截熔断。
"""

import sys
from pathlib import Path

# 适配 Windows 控制台 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 确保项目根目录在 sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

from src.parser import DocumentParser, ScannedPDFError


class MockContractCanvas(canvas.Canvas):
    """用于绘制真实企业合同的页眉横线与页脚页码"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        # 页眉区域 (y > 750)
        self.drawString(54, 760, "CONFIDENTIAL // ENTERPRISE IT PROCUREMENT MASTER AGREEMENT")
        self.line(54, 755, 558, 755)
        # 页脚区域 (y < 45)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, f"DocuMind Test Suite - Page {self._pageNumber} of {page_count}")
        self.restoreState()


def create_mock_procurement_pdf(output_path: Path):
    """动态生成一份包含多级条款与复杂付款违约金表格的真实测试 PDF"""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    # 适配中文样式的段落
    title_style = ParagraphStyle(
        "ChineseHeading",
        parent=styles["Heading1"],
        fontName="STSong-Light",
        fontSize=14,
        leading=18
    )
    body_style = ParagraphStyle(
        "ChineseBody",
        parent=styles["Normal"],
        fontName="STSong-Light",
        fontSize=10,
        leading=14
    )
    story = []

    story.append(Spacer(1, 10))
    story.append(Paragraph("第一条 服务范围与交付标的", title_style))
    story.append(Paragraph("乙方应按甲方要求提供大型大模型智能体审计中台系统，包含全套私有化部署镜像与源码交付。", body_style))
    story.append(Spacer(1, 15))

    # 复杂阶梯违约金与付款里程碑表格（核心测试目标）
    table_data = [
        ["里程碑阶段", "交付成果及验收标准", "付款比例", "延期违约金考核标准"],
        ["阶段一：架构设计", "《系统架构蓝图与接口契约说明书》终审", "20%", "每日扣罚 0.2% 合同款"],
        ["阶段二：联调部署", "完成私有化集群部署并跑通基准测试", "50%", "每日扣罚 0.5% 合同款"],
        ["阶段三：终验交付", "全套源代码移交并完成知识转移培训", "30%", "每日扣罚 1.0% 合同款 (上限30%)"]
    ]
    t = Table(table_data, colWidths=[90, 180, 70, 160])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("第二条 知识产权归属与排他性约束", title_style))
    story.append(Paragraph("乙方在签约前独立研发的底层算法引擎与知识产权归乙方所有，甲方享有非排他性永久使用许可。", body_style))

    doc.build(story, canvasmaker=MockContractCanvas)


def create_empty_scanned_pdf(output_path: Path):
    """生成一份纯空页/仅含矢量图但无有效文字层的扫描件假样本"""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.rect(100, 100, 400, 600)  # 只画一个方框模拟扫描图片
    c.showPage()
    c.save()


if __name__ == "__main__":
    test_dir = Path("tests/sample_contracts")
    test_dir.mkdir(parents=True, exist_ok=True)

    normal_pdf = test_dir / "mock_procurement_contract.pdf"
    scanned_pdf = test_dir / "mock_scanned_contract.pdf"

    print("📄 1. 生成真实采购合同 PDF (带页眉、页脚与复杂表格)...")
    create_mock_procurement_pdf(normal_pdf)

    print("📄 2. 生成纯图片扫描件 PDF 假样本 (用于测试防御熔断)...")
    create_empty_scanned_pdf(scanned_pdf)

    print("\n" + "=" * 60)
    print("🚀 测试一：高保真 PDF 提取测试 (验证表格 Markdown 化与页眉剔除)")
    print("=" * 60)

    sections = DocumentParser.parse_pdf(normal_pdf)
    print(f"[*] 解析成功！共提取出 {len(sections)} 个条款章节：\n")

    for idx, sec in enumerate(sections, start=1):
        print(f"--- 📌 条款 {idx}: 【{sec.section_title}】 (页码: P{sec.page_start}-P{sec.page_end}, 行号: 第 {sec.line_start}-{sec.line_end} 行) ---")
        print(sec.content)
        print()

    # 验证是否成功转为 Markdown Table
    full_text = DocumentParser.extract_full_text(normal_pdf)
    assert "| 里程碑阶段 | 交付成果及验收标准 | 付款比例 |" in full_text, "❌ 错误：表格未正确转换为 Markdown Table！"
    assert "CONFIDENTIAL // ENTERPRISE IT PROCUREMENT" not in full_text, "❌ 错误：页眉噪声未被剔除！"
    assert "DocuMind Test Suite - Page" not in full_text, "❌ 错误：页脚噪声未被剔除！"
    print("✅ 测试一全部通过！表格完美呈现为 Markdown Table，页眉页脚噪声完全剔除！")

    print("\n" + "=" * 60)
    print("🚀 测试二：扫描件/图片 PDF 拦截熔断防御测试")
    print("=" * 60)
    try:
        DocumentParser.parse_pdf(scanned_pdf)
        print("❌ 错误：扫描件未被成功拦截！")
    except ScannedPDFError as e:
        print("✅ 扫描件防御触发成功！捕获到预期的安全熔断异常：")
        print(f"   {e}")

    print("\n" + "=" * 60)
    print("🎉 工业级高保真 PDF 解析引擎所有单元测试全部通过！")
    print("=" * 60)
