"""
DocuMind-Agent: 核心数据契约模型 (Pydantic v2)
定义合同审查、条款追溯、风险定级与证据链的严格结构化输出规范。
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RiskLevel(str, Enum):
    """合规风险定级"""
    HIGH = "高风险"      # 存在严重法律隐患、单方霸王条款、重大经济损失风险
    MEDIUM = "中风险"    # 条款模糊、权责不对等、违约责任界定不清
    LOW = "低风险"       # 格式瑕疵、用词不规范或友好提示建议


class VerificationStatus(str, Enum):
    """证据链物理核验与基础设施状态（Fail-Closed 安全设计）"""
    VERIFIED = "verified"                                     # 100% 存在且物理坐标精准对齐
    UNVERIFIED = "unverified_needs_human_review"              # 未在原文闭环或坐标模糊，需法务人工核验
    INFRA_DEGRADED = "infra_degraded"                         # 检索精排基础设施降级/超时，置信度不可比


class ClauseEvidence(BaseModel):
    """条款原文证据链追溯（支持物理行号、页码与严格可信度状态）"""
    clause_title: str = Field(..., description="相关条款标题或章节名，例如：'第8条 违约赔偿责任'")
    original_text: str = Field(..., description="合同中违规或存在风险的原文逐字摘录")
    page_number: Optional[int] = Field(None, description="条款所在文档页码 (1-indexed，无页码概念时为 None)")
    line_start: Optional[int] = Field(None, description="条款所在起始物理行号")
    line_end: Optional[int] = Field(None, description="条款所在结束物理行号")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED, 
        description="证据物理核验状态 (Fail-Closed: 默认必须为 UNVERIFIED，需显式验证放行)"
    )


class ComplianceIssue(BaseModel):
    """单个合规/风控风险发现项"""
    issue_id: str = Field(..., description="风险项唯一标识，例如：'SEC-01', 'RISK-FIN-02'")
    title: str = Field(..., description="风险摘要，例如：'单方违约金比例超出法律保护上限'")
    risk_level: RiskLevel = Field(..., description="风险定级")
    category: str = Field(default="综合条款", description="风险分类：财务风控、违约责任、知识产权、保密协议、交付验收等")
    evidence: ClauseEvidence = Field(..., description="原文证据链引用")
    risk_analysis: str = Field(..., description="详细风险隐患剖析：为何违规、可能造成的法务或财务后果")
    suggestion: str = Field(..., description="业务处置建议与谈判策略")
    suggested_replacement: Optional[str] = Field(
        None, 
        description="修改后的建议条款标准法律表述（可直接用于替换原文）"
    )


class ContractAuditReport(BaseModel):
    """整份合同最终审查报告"""
    contract_name: str = Field(..., description="合同或文档标题")
    audit_summary: str = Field(..., description="审查总体结论与高管摘要 (Executive Summary)")
    overall_risk: RiskLevel = Field(..., description="合同综合风险定级")
    high_risk_count: int = Field(default=0, description="高风险项数量")
    medium_risk_count: int = Field(default=0, description="中风险项数量")
    low_risk_count: int = Field(default=0, description="低风险项数量")
    unverified_count: int = Field(default=0, description="未完全闭环证据项数量 (需法务人工核验)")
    needs_human_review: bool = Field(default=False, description="全篇合同是否存在未闭环证据项需要人工干预")
    issues: List[ComplianceIssue] = Field(default_factory=list, description="发现的所有风险问题清单")
    passed_items: List[str] = Field(default_factory=list, description="审查合格的条款模块摘要")
    audit_time: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="审查生成时间"
    )
