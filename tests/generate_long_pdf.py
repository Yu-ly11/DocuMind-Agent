"""
生成 10~15 页长篇真实企业级合同 PDF (带页眉页脚与跨页表格)
用于严格检验 DocumentParser 的几何剔除、表格还原与多页行号定位能力。
"""

import sys
from pathlib import Path

# 强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 注册中文字体
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


class LongContractNumberedCanvas(canvas.Canvas):
    """真实企业合同画布：在每页顶部绘制机密页眉，在每页底部绘制页码与流水号"""
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
            self.draw_header_and_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_and_footer(self, total_pages):
        self.saveState()
        self.setFont("STSong-Light", 8)
        self.setFillColor(colors.HexColor("#666666"))

        # 页眉区域 (y = 810 pt，A4高度为 841.89 pt)
        self.drawString(50, 810, "【商业机密 · 绝密】华盛云联数据科技股份有限公司 · 数字化中台技术开发协议")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(50, 804, 545, 804)

        # 页脚区域 (y = 30 pt)
        self.line(50, 40, 545, 40)
        footer_text = f"第 {self._pageNumber} 页 / 共 {total_pages} 页  |  签署版本存证号: DX-2024-TECH-08892-SEC"
        self.drawRightString(545, 26, footer_text)
        self.restoreState()


def build_long_contract_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=55,      # 避开页眉区域 (55 pt > 38 pt)
        bottomMargin=55    # 避开页脚区域 (55 pt > 40 pt)
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="ContractTitle",
        fontName="STSong-Light",
        fontSize=16,
        leading=22,
        alignment=1, # 居中
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        name="ContractH1",
        fontName="STSong-Light",
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#0d47a1"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        name="ContractBody",
        fontName="STSong-Light",
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#262626"),
        firstLineIndent=18,
        spaceAfter=4
    )
    meta_style = ParagraphStyle(
        name="ContractMeta",
        fontName="STSong-Light",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#444444"),
        spaceAfter=3
    )

    story = []

    # 1. 标题与签署元信息
    story.append(Paragraph("企业级数字化中台与智能供应链系统技术开发与集成合同", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("合同编号：DX-2024-TECH-08892  ｜  签署日期：2024 年 03 月 15 日", meta_style))
    story.append(Paragraph("甲方（委托方 / 采购方）：华盛云联数据科技股份有限公司", meta_style))
    story.append(Paragraph("乙方（受托方 / 技术服务商）：创智数联智能软件系统工程（北京）有限公司", meta_style))
    story.append(Spacer(1, 15))

    # 2. 读取 Markdown 合同文本并结构化渲染为 Paragraph
    source_md = Path("tests/sample_contracts/real_judicial_software_dispute_contract.md")
    with open(source_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    import re

    def md_to_reportlab(txt: str) -> str:
        # 移除 markdown 中的加粗语法，避免 html 解析未闭合异常
        clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', txt)
        # 清理可能存在的孤立 & 符号
        clean = clean.replace('&', '&amp;')
        return clean

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean.startswith("# 企业级"):
            continue
        if line_clean.startswith("**合同编号") or line_clean.startswith("**甲方") or line_clean.startswith("**乙方"):
            continue

        if line_clean.startswith("### "):
            # 章节标题
            h_text = line_clean.replace("### ", "").replace("**", "")
            story.append(Spacer(1, 8))
            story.append(Paragraph(h_text, h1_style))
        elif line_clean.startswith("---"):
            story.append(Spacer(1, 6))
        elif line_clean.startswith("- "):
            # 列表项
            bullet_text = md_to_reportlab(line_clean[2:])
            story.append(Paragraph(f"• {bullet_text}", body_style))
        else:
            # 普通正文段落
            p_text = md_to_reportlab(line_clean)
            story.append(Paragraph(p_text, body_style))

    # 3. 注入关键表格：里程碑款项与验收考核表 (测试表格几何定位与 Markdown Table 转换)
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件一：项目里程碑付款与交付物考核明细表", h1_style))
    
    table_data = [
        ["款项期次", "付款比例", "金额 (元)", "达成条件与交付物", "逾期扣款标准"],
        ["第一期 (预付款)", "20%", "1,360,000.00", "签署合同并提交《项目实施规划方案》", "无"],
        ["第二期 (初验款)", "30%", "2,040,000.00", "准生产环境部署完毕，通过联合集成测试", "每日 1.0% (￥68,000/天)"],
        ["第三期 (终验款)", "40%", "2,720,000.00", "生产环境连续无故障运行90日，指标达标", "每日 1.0% (￥68,000/天)"],
        ["第四期 (质保金)", "10%", "680,000.00", "终验合格满12个月无遗留缺陷", "若有缺陷从质保金全额抵扣"]
    ]
    t_style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3f2fd")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (3, 1), (3, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90caf9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    milestone_table = Table(table_data, colWidths=[80, 55, 75, 175, 110], style=t_style)
    story.append(milestone_table)

    # 4. 注入第二个跨页表格：SLA 故障等级与索赔穿透标准表
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件二：SLA 服务等级标准与违约责任穿透矩阵", h1_style))
    sla_data = [
        ["故障等级", "定义指标", "响应恢复时限", "法律责任穿透判定"],
        ["P1 特大故障", "核心交易调度完全停摆，影响订单超1万笔", "30分钟响应，2小时内恢复", "直接穿透15%责任上限，承担无上限连带赔偿"],
        ["P2 重大故障", "单仓调度故障，降级为手工同步", "1小时响应，4小时内恢复", "当日扣除当月维保费50%，计入逾期天数"],
        ["P3 一般缺陷", "非关键报表统计延迟，主链路正常", "4小时响应，24小时内修复", "累计超过3次记为P2级故障"],
        ["P4 轻微问题", "交互UI文案错漏、字体排版问题", "24小时响应，下个版本修复", "不计入违约罚则"]
    ]
    sla_table = Table(sla_data, colWidths=[70, 160, 115, 150], style=t_style)
    story.append(sla_table)

    # 5. 注入附件三：数据安全与个人信息保护专项协议 (DPA)
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件三：数据安全合规与个人信息保护专项协议（DPA）", h1_style))
    dpa_clauses = [
        "1.1 本项目涉及甲方工业物联网传感器时序数据、货主货品交易记录及承运司机个人行踪轨迹等核心敏感数据。",
        "1.2 乙方承诺严格遵守《中华人民共和国数据安全法》、《中华人民共和国个人信息保护法》及《关键信息基础设施安全保护条例》。",
        "1.3 数据境内存储与出境禁止：乙方因系统测试、代码调试等任何原因采集或生成的全部业务数据，必须且仅能存储在甲方位于中华人民共和国境内的私有云基础设施中，严禁以任何直接或间接形式跨境传输或通过远程终端向境外服务器进行任何形式的同步备份。",
        "1.4 算法训练绝对禁止：乙方不得将甲方任何业务数据、交易流水、模型权重或知识图谱用于乙方自身通用基础大模型、公共商业算法库的训练、微调或测试，亦不得以任何匿名化、去标识化为借口主张对甲方业务数据享有衍生权利。",
        "1.5 违规穿透罚则：若乙方发生数据泄露或违规导出行为，甲方有权立即解除主协议并没收全部质保金，乙方须向甲方一次性支付违约金人民币 5,000,000.00 元（大写：人民币伍佰万元整），且此项赔偿责任不受主协议第八条责任上限之限制。"
    ]
    for c in dpa_clauses:
        story.append(Paragraph(c, body_style))

    # 6. 注入附件四：硬件基础设施配置与信创环境适配清单表
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件四：信创硬件基础设施与运行环境交付清单表", h1_style))
    hw_data = [
        ["设备类型", "规格配置要求", "操作系统/中间件", "数量", "验收考核指标"],
        ["核心调度服务器", "海光 7380 CPU (64核), 512GB DDR4, 4*3.84TB NVMe", "统信 UOS Server / 达梦 DM8", "8台", "连续压测72小时CPU使用率低于70%"],
        ["边缘网关节点", "飞腾 FT-2000/4, 32GB RAM, 1TB SSD", "Kylin Linux V10", "24台", "弱网环境离线断点续传成功率100%"],
        ["分布式存储集群", "2U 存储服务器, 12*16TB 企业级 SATA, 双万兆光口", "Ceph 18.2 企业优化版", "6台", "集群双副本高可用，单节点宕机0丢包"],
        ["安全隔离网闸", "国密二级认证, 双主机硬件物理隔离, 吞吐量>5Gbps", "定制安全微内核OS", "2台", "通过公安部第三研究所边界隔离检测"]
    ]
    hw_table = Table(hw_data, colWidths=[75, 150, 115, 45, 110], style=t_style)
    story.append(hw_table)

    # 7. 注入附件五：技术服务人员配置与人月投入核算表
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件五：驻场项目团队人员投入与资质核查表", h1_style))
    staff_data = [
        ["岗位角色", "最低资质要求", "规划投入人月", "月度服务费标准", "更换人员处罚约定"],
        ["项目总监/架构师", "10年以上分布式架构经验，主导过亿级调度系统", "6 人月", "￥65,000 / 人月", "未经甲方同意单方更换，罚款 50,000 元/人次"],
        ["高级算法专家", "应用数学或计算机博士，精通运筹优化与强化学习", "8 人月", "￥55,000 / 人月", "核心算法未达标乙方须免费追加人月"],
        ["后端微服务开发", "5年以上 Java/Go 开发经验，精通 Spring Cloud/K8s", "24 人月", "￥38,000 / 人月", "人员缺勤超过3日按双倍扣减人月服务费"],
        ["测试与安全工程师", "具备 CISP 或等保三级评测实战资质", "10 人月", "￥32,000 / 人月", "发现高危安全漏洞未修复不得撤场"]
    ]
    staff_table = Table(staff_data, colWidths=[80, 160, 65, 80, 110], style=t_style)
    story.append(staff_table)

    # 8. 注入附件六：开源软件成分合规审查与强传染性协议（GPL/AGPL）排查清单
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件六：引入开源软件组件合规声明与许可证排查矩阵", h1_style))
    oss_data = [
        ["开源组件名称", "当前引用版本", "开源许可证类型", "商业化分发传染性风险", "替代自研或隔离方案"],
        ["Linux Kernel 驱动模块", "v5.15 LTS", "GPL v2.0", "强传染性：动态链接可能导致上层业务代码被迫开源", "采用用户态 UIO/VFIO 架构隔离"],
        ["MongoDB 社区版连接驱动", "v4.4.2", "SSPL v1.0", "极端传染性：提供云服务必须开源整个管理运维代码", "全面平替为达梦关系型数据库或 PostgreSQL"],
        ["Elasticsearch 检索组件", "v7.10.2", "Apache 2.0", "无传染性：允许商业闭源修改，但须保留版权声明", "直接集成并保留原作者 NOTICE 声明文件"],
        ["FFmpeg 音视频转码库", "v6.0", "LGPL v2.1+", "弱传染性：若静态编译会导致整个二进制受传染", "必须采用动态链接库 (so/dll) 形式加载调用"],
        ["Neo4j 知识图谱引擎", "v4.4 社区版", "AGPL v3.0", "强网络传染性：任何云端网络调用均触发全网源码开源", "严禁引入生产环境，已责令乙方重构自研图遍历算法"]
    ]
    oss_table = Table(oss_data, colWidths=[90, 65, 75, 145, 120], style=t_style)
    story.append(oss_table)

    # 9. 注入附件七：企业数字化系统终验考核指标与全量压测验收标准细则
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件七：系统终验压力测试标准与非功能性指标（NFR）达标验收细则", h1_style))
    nfr_clauses = [
        "1.1 并发吞吐量指标（TPS）：在模拟 10,000 活跃用户并发下单与多仓调度场景下，系统集群综合事务处理能力（TPS）不得低于 5,000 笔/秒，且在连续 72 小时压力测试中错误率不得超过 0.001%。",
        "1.2 响应耗时指标（Latency）：核心读交易 P95 响应时间须在 80 毫秒以内，P99 响应时间在 200 毫秒以内；核心写交易（锁库、减保、分单）P95 在 150 毫秒以内，P99 在 300 毫秒以内。",
        "1.3 高可用容灾（RTO / RPO）：同城双活数据中心切换场景下，系统自动故障转移恢复时间目标（RTO）不得超过 60 秒，数据丢失恢复点目标（RPO）必须严格为零（RPO = 0）。",
        "1.4 容错降级与防雪崩机制：当外部第三方物流接口或银行清算通道响应超时（>3秒）时，系统必须在 50 毫秒内自动触发断路器（Circuit Breaker）熔断与限流降级，严禁导致内部微服务线程池耗尽或雪崩瘫痪。",
        "1.5 验收终审权与否决规则：若上述非功能性性能指标在终验测试中有任何一项连续两次未达标，甲方有权单方判定整体验收不合格，并按照主协议第七条向乙方全额追索赔偿。"
    ]
    for c in nfr_clauses:
        story.append(Paragraph(c, body_style))

    # 10. 注入附件八：源代码托管、物理交接封存与软件著作权转让公证备忘录
    story.append(Spacer(1, 15))
    story.append(Paragraph("附件八：源代码物理封存交接与软件著作权独家权属移交公证备忘录", h1_style))
    escrow_clauses = [
        "2.1 源代码物理光盘移交：终验合格之日起 3 个工作日内，乙方必须将本中台系统全部可无差错编译运行的代码工程、底层私有依赖包、自动化构建脚本（CI/CD Pipeline）、数据库建模文件（ERwin/PD）刻录至只读蓝光光盘（一式四份），双方代表签字并加盖公司公章后封存至指定商业银行保险箱。",
        "2.2 软件著作权变更登记：乙方承诺在收到第二期款项后 15 个工作日内，配合甲方向国家版权局提交本目标软件之计算机软件著作权登记变更申请，登记著作权人必须且仅能为甲方独家单独所有。",
        "2.3 核心技术人员脱密期与竞业限制承诺：乙方承诺直接参与本项目核心架构设计与算法研发的关键技术骨干人员（包括但不限于项目总监、算法负责人），在本合同终止后 24 个月内，严禁直接或间接加入与甲方存在同业竞争关系的企业，亦严禁以顾问、兼职等任何形式向甲方竞品提供技术指导；违者乙方须承担人民币 2,000,000.00 元连带违约金。"
    ]
    for c in escrow_clauses:
        story.append(Paragraph(c, body_style))

    # 构建长 PDF
    doc.build(story, canvasmaker=LongContractNumberedCanvas)
    print(f"✅ 成功生成企业级长篇实战 PDF: {output_path}")


if __name__ == "__main__":
    out_file = Path("tests/sample_contracts/real_judicial_software_dispute_contract.pdf")
    build_long_contract_pdf(out_file)
