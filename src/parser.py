"""
DocuMind-Agent: 工业级高保真文档解析器
支持 PDF、Markdown 与纯文本的结构化解析：
1. PDF 表格自动识别并转化为标准 Markdown Table（保留行列语义与对齐）；
2. 自适应几何裁切，剔除页眉、页脚与页码周期性噪声；
3. 空间切块排序（正文与表格精准分离，杜绝重复提取）；
4. 扫描件/影印件可提取性防御熔断（拒绝空文本欺骗）；
5. 全链路精确保留物理页码 (page) 与物理行号 (line_number)。
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class ScannedPDFError(Exception):
    """当 PDF 疑似为纯图片扫描件（无有效文本层）时抛出的安全防御异常"""
    pass


class DocumentLine(BaseModel):
    """单行文本元数据"""
    page: int = Field(default=1, description="所在页码 (从1开始)")
    line_number: int = Field(..., description="文档中的全局物理行号 (从1开始)")
    text: str = Field(..., description="单行文本内容")


class ParsedSection(BaseModel):
    """解析出的结构化章节/条款块"""
    section_title: str = Field(default="正文条款", description="条款章节标题")
    content: str = Field(..., description="条款全文内容")
    page_start: int = Field(default=1, description="起始页码")
    page_end: int = Field(default=1, description="结束页码")
    line_start: int = Field(..., description="起始行号")
    line_end: int = Field(..., description="结束行号")


class DocumentParser:
    """通用合同文档解析器"""

    # 常见合同条款标题正则（如：“第一条”、“第1.1节”、“一、”、“1. 合作范围”）
    CLAUSE_HEADING_PATTERN = re.compile(
        r"^(第[一二三四五六七八九十百千0-9]+[条章节款]|"
        r"[一二三四五六七八九十]+[、.]|"
        r"[\(（][一二三四五六七八九十0-9]+[\)）]|"
        r"\d+(\.\d+)*\s*[\.、\s]|\#{1,4}\s+)"
    )

    @classmethod
    def parse_file(cls, file_path: str | Path) -> List[ParsedSection]:
        """统一解析入口：根据后缀自动分发到对应解析器"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return cls.parse_pdf(path)
        elif suffix in (".md", ".txt"):
            return cls.parse_text_or_markdown(path)
        else:
            raise ValueError(f"暂不支持的文件格式: {suffix} (仅支持 .pdf, .md, .txt)")

    @classmethod
    def extract_full_text(cls, file_path: str | Path) -> str:
        """提取整篇文档的高保真清洗文本（按条款结构拼接）"""
        sections = cls.parse_file(file_path)
        return "\n\n".join(sec.content for sec in sections)

    @classmethod
    def parse_text_or_markdown(cls, file_path: str | Path) -> List[ParsedSection]:
        """解析 Markdown 或纯文本文件，保留物理行号"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()

        lines: List[DocumentLine] = [
            DocumentLine(page=1, line_number=idx + 1, text=line.rstrip("\r\n"))
            for idx, line in enumerate(raw_lines)
            if line.strip()
        ]
        return cls._segment_by_clauses(lines)

    @classmethod
    def parse_pdf(
        cls, 
        file_path: str | Path,
        header_margin: float = 40.0,
        footer_margin: float = 40.0,
        min_chars_per_page: float = 15.0
    ) -> List[ParsedSection]:
        """
        高保真解析 PDF：
        - 几何裁剪页眉与页脚；
        - 定位表格并转换为 Markdown 格式；
        - 表格与正文按 Y 轴空间有序重组；
        - 纯图片扫描件异常防御熔断。
        """
        import pdfplumber

        lines: List[DocumentLine] = []
        global_line_counter = 1
        total_extracted_chars = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise ValueError("PDF 文件为空，没有任何页面！")

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                page_h = float(page.height)
                page_w = float(page.width)

                # 1. 计算有效正文裁剪区域（剔除页眉与页脚几何区域）
                crop_top = header_margin if page_h > (header_margin + footer_margin + 50) else 0.0
                crop_bottom = (page_h - footer_margin) if page_h > (header_margin + footer_margin + 50) else page_h

                # 截取有效内容区
                content_area = page.crop((0, crop_top, page_w, crop_bottom))

                # 2. 定位页面内所有表格，按 Y 轴纵向自上而下排序
                tables = content_area.find_tables()
                tables_sorted = sorted(tables, key=lambda t: t.bbox[1])

                current_y = crop_top
                page_elements: List[str] = []

                for t in tables_sorted:
                    tb_x0, tb_y0, tb_x1, tb_y1 = t.bbox

                    # (1) 提取表格上方的纯正文段落
                    if tb_y0 > current_y + 3:
                        text_box = page.crop((0, current_y, page_w, tb_y0))
                        text_chunk = text_box.extract_text()
                        if text_chunk and text_chunk.strip():
                            page_elements.append(text_chunk.strip())

                    # (2) 提取表格二维数据并转为 Markdown Table
                    raw_table_data = t.extract()
                    md_table = cls._table_to_markdown(raw_table_data)
                    if md_table:
                        page_elements.append(md_table)

                    current_y = tb_y1

                # (3) 提取最后一个表格下方的剩余正文段落
                if current_y < crop_bottom - 3:
                    text_box = page.crop((0, current_y, page_w, crop_bottom))
                    text_chunk = text_box.extract_text()
                    if text_chunk and text_chunk.strip():
                        page_elements.append(text_chunk.strip())

                # 3. 将本页所有元素展开为带 line_number 的 DocumentLine
                for element in page_elements:
                    for raw_line in element.split("\n"):
                        clean_line = raw_line.rstrip()
                        if clean_line.strip():
                            lines.append(
                                DocumentLine(
                                    page=page_num,
                                    line_number=global_line_counter,
                                    text=clean_line
                                )
                            )
                            global_line_counter += 1
                            total_extracted_chars += len(clean_line.strip())

        # 4. 扫描件防御：可提取性检查与自动 OCR 智能降级转录
        avg_chars = total_extracted_chars / total_pages if total_pages > 0 else 0
        if avg_chars < min_chars_per_page:
            print(f"[*] 检测到纯图片扫描件 (平均每页仅 {avg_chars:.1f} 字符)，正在无缝启动 RapidOCR 引擎执行逐页图文转录...")
            try:
                return cls._parse_pdf_with_ocr(file_path)
            except Exception as ocr_err:
                raise ScannedPDFError(
                    f"安全拦截：该 PDF 全文共 {total_pages} 页，平均每页仅提取出 {avg_chars:.1f} 个有效字符！\n"
                    f"检测到该文件疑似为纯图片扫描件或无文字层的影印件，且 OCR 转录异常: {ocr_err}\n"
                    f"【处置建议】：请确认文件清晰度或环境具备 OCR 引擎支持。"
                )

        return cls._segment_by_clauses(lines)

    @classmethod
    def _parse_pdf_with_ocr(cls, file_path: str | Path, max_pages: int = 50) -> List[ParsedSection]:
        """
        工业级 OCR 降级管道：针对纯图片扫描件 PDF，自动启动 RapidOCR + PyMuPDF 逐页图文转录
        具备本地 MD5 高速磁盘缓存，避免重复跑 OCR 消耗算力。
        保留真实的物理页码 (page) 与物理行号 (line_number)
        """
        path_obj = Path(file_path)

        # 1. 计算文件 MD5 指纹
        hasher = hashlib.md5()
        with open(path_obj, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

        # 2. 检查本地高速磁盘缓存 (存在则 0.01 秒极速恢复)
        cache_dir = Path("tmp_contracts/.ocr_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{file_hash}.json"

        if cache_file.exists():
            print(f"[*] [OCR 磁盘缓存命中] 检测到该扫描件已完成识别 (指纹: {file_hash[:8]})，瞬间完成加载！")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                lines = [DocumentLine(**item) for item in cached_data]
                return cls._segment_by_clauses(lines)
            except Exception as e:
                print(f"[警告] 读取 OCR 缓存异常，重新执行识别: {e}")

        # 3. 未命中缓存，启动本地 RapidOCR 识别
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        doc = pymupdf.open(str(path_obj))
        lines: List[DocumentLine] = []
        global_line_counter = 1

        total_pages = min(len(doc), max_pages)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            print(f"[*] [OCR 识别中] 正在转录第 {page_num} / {total_pages} 页...")
            # 150 DPI 兼顾识别高精度与性能
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")

            ocr_res, _ = ocr(img_bytes)
            if ocr_res:
                raw_page_lines = [item[1].strip() for item in ocr_res if item[1].strip()]
                # 执行智能 OCR 文本流缝合 (解决竖排文字与断行问题)
                smoothed_page_lines = cls._smooth_ocr_lines(raw_page_lines)
                for clean_line in smoothed_page_lines:
                    lines.append(
                        DocumentLine(
                            page=page_num,
                            line_number=global_line_counter,
                            text=clean_line
                        )
                    )
                    global_line_counter += 1

        if not lines:
            raise ScannedPDFError("已启动 OCR 扫描，但该文件中未检测到任何可识别文字。")

        # 4. 将识别结果持久化缓存落盘，保证终身只识别一次
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump([l.model_dump() for l in lines], f, ensure_ascii=False, indent=2)
            print(f"[*] [OCR 缓存落盘] 已成功缓存 {len(lines)} 行转录结果至本地，下次秒级加载！")
        except Exception:
            pass

        return cls._segment_by_clauses(lines)

    @classmethod
    def _smooth_ocr_lines(cls, raw_lines: List[str]) -> List[str]:
        """
        OCR 后处理版面重构与断行平滑缝合器：
        1. 解决表格侧边栏、表头竖排单字断裂问题（如“指\\n导\\n教\\n师\\n意\\n见”合并为“指导教师意见”）；
        2. 解决同一自然段被 OCR 检测框横向截断的孤立短行问题；
        3. 清洗印章残留的零碎单字干扰。
        """
        clean_input = [l.strip() for l in raw_lines if l.strip()]
        if not clean_input:
            return []

        merged_pass1: List[str] = []
        vertical_buf: List[str] = []

        for line in clean_input:
            # 单字且不属于序号或单独标点，极大概率是竖排单元格
            if len(line) == 1 and not re.match(r"[0-9一二三四五六七八九十、.：:()（）√]", line):
                vertical_buf.append(line)
            else:
                if vertical_buf:
                    merged_pass1.append("".join(vertical_buf))
                    vertical_buf = []
                merged_pass1.append(line)

        if vertical_buf:
            merged_pass1.append("".join(vertical_buf))

        # 第二轮：段落非自然换行平滑缝合
        terminal_punct = ("。", "！", "？", "；", "：", ":", "”", "）", ")", "√")
        # 通用字段标签特征（例如"甲方："、"签署日期："等短键名，不参与跨行正文平滑缝合）
        field_label_pattern = re.compile(r"^[\u4e00-\u9fa5A-Za-z]{2,8}[：:]")

        stitched: List[str] = []
        for l in merged_pass1:
            # 如果上一行末尾没有终止标点，且不是章节标题/独立字段名标签，执行句子自然平滑缝合
            if (
                stitched
                and not stitched[-1].endswith(terminal_punct)
                and len(stitched[-1]) < 45
                and not cls.CLAUSE_HEADING_PATTERN.match(stitched[-1])
                and not cls.CLAUSE_HEADING_PATTERN.match(l)
                and not field_label_pattern.match(stitched[-1])
                and not field_label_pattern.match(l)
            ):
                stitched[-1] = stitched[-1] + l
            else:
                stitched.append(l)

        return stitched

    @staticmethod
    def _table_to_markdown(table_data: List[List[Any]]) -> str:
        """将 pdfplumber 提取的二维表格数据转为对齐良好的 Markdown 表格"""
        if not table_data or len(table_data) < 1:
            return ""

        # 清洗单元格换行与空格
        cleaned_rows: List[List[str]] = []
        for row in table_data:
            if not row:
                continue
            cleaned_row = [str(cell or "").replace("\n", " ").strip() for cell in row]
            # 过滤全空行
            if any(cleaned_row):
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return ""

        headers = cleaned_rows[0]
        col_count = len(headers)
        if col_count == 0:
            return ""

        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * col_count) + " |"
        ]

        for row in cleaned_rows[1:]:
            # 补齐不足列数的行
            padded_row = row + [""] * (col_count - len(row))
            md_lines.append("| " + " | ".join(padded_row[:col_count]) + " |")

        return "\n".join(md_lines)

    @classmethod
    def _segment_by_clauses(cls, lines: List[DocumentLine]) -> List[ParsedSection]:
        """将逐行文本按条款标题切分为结构化章节"""
        if not lines:
            return []

        sections: List[ParsedSection] = []
        current_title = "前言/合同签署主体"
        current_lines: List[DocumentLine] = []

        for item in lines:
            stripped = item.text.strip()
            # 如果匹配到条款标题且非超长段落，开启新 section
            if cls.CLAUSE_HEADING_PATTERN.match(stripped) and len(stripped) <= 60:
                if current_lines:
                    sections.append(cls._build_section(current_title, current_lines))
                    current_lines = []
                current_title = stripped

            current_lines.append(item)

        if current_lines:
            sections.append(cls._build_section(current_title, current_lines))

        return sections

    @staticmethod
    def _build_section(title: str, lines: List[DocumentLine]) -> ParsedSection:
        """组装单条 Section"""
        full_content = "\n".join(l.text for l in lines)
        return ParsedSection(
            section_title=title,
            content=full_content,
            page_start=lines[0].page,
            page_end=lines[-1].page,
            line_start=lines[0].line_number,
            line_end=lines[-1].line_number
        )
