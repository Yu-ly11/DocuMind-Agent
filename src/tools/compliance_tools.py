"""
DocuMind-Agent: 智能体专用专业工具箱 (Tools)
包含违约金风控数学计算器与合同检索工具工厂函数。
"""
from typing import Dict, Any, Tuple, Callable, Optional
from src.rag.hybrid_retriever import HybridRetriever


def calc_penalty_risk(
    contract_total_amount: Optional[float] = None, 
    daily_rate: Optional[float] = None, 
    overdue_days: Optional[int] = 30
) -> Dict[str, Any]:
    """
    【工具 1：违约金风控数学计算器】
    根据合同总金额、日违约金比例和逾期天数，精确计算累计违约金，并依据民法典司法解释输出风控评估。
    拒绝编造默认值：若合同未约定金额或比例，显式返回数据不全状态，交由法务人工复核。
    
    :param contract_total_amount: 合同总金额（人民币元），若合同未载明可传 None
    :param daily_rate: 每日违约金比例（例如每日 1.5% 则传入 0.015），若未载明可传 None
    :param overdue_days: 预估或约定的逾期天数（默认 30 天）
    :return: 包含精确违约金、占合同总额商业比例、法律依据与司法调节风险提示
    """
    if contract_total_amount is None or daily_rate is None or contract_total_amount <= 0 or daily_rate <= 0:
        return {
            "status": "incomplete_data",
            "contract_total_amount": contract_total_amount,
            "daily_rate": daily_rate,
            "overdue_days": overdue_days,
            "calculated_total_penalty": None,
            "penalty_ratio_of_contract": None,
            "is_commercial_excessive": False,
            "legal_risk_warning": "【要素缺失需人工核查】合同条款未载明明确的合同总金额或日违约金费率，无法测算确定性金额。需法务线下核对违约金起算基数。"
        }

    days = overdue_days if (overdue_days is not None and overdue_days > 0) else 30
    total_penalty = round(contract_total_amount * daily_rate * days, 2)
    penalty_ratio = total_penalty / contract_total_amount
    is_commercial_excessive = penalty_ratio > 0.30

    return {
        "status": "success",
        "contract_total_amount": contract_total_amount,
        "daily_rate_percent": f"{daily_rate * 100:.2f}%",
        "overdue_days": days,
        "calculated_total_penalty": total_penalty,
        "penalty_ratio_of_contract": f"{penalty_ratio * 100:.2f}%",
        "is_commercial_excessive": is_commercial_excessive,
        "legal_reference": "《民法典》第585条及最高人民法院《合同编通则解释》第65条",
        "legal_risk_warning": (
            "【商业风险预警】累计违约金占合同总金额比例达 " f"{penalty_ratio * 100:.2f}%" "（已超30%商业警戒线）。"
            "法务要点提示：根据《民法典》第585条，司法实践中违约金调减的法定参照基准为守约方的【实际损失】（超过损失30%可申请法院调减）。"
            "当前约定比例显著过高，履约中极易被对方主张显失公平并申请司法酌减，建议设定累计违约金封顶上限（如合同总额的10%~20%）。"
            if is_commercial_excessive else 
            f"违约金占合同总额 {penalty_ratio * 100:.2f}%，处于常见商业可控区间。实际裁判中仍将结合履约实际损失进行个案权衡。"
        )
    }


# 计算器工具的 Schema 说明书常量
CALC_PENALTY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calc_penalty_risk",
        "description": "计算逾期违约金累计金额并给出民法典司法解释的风控提示。若合同未载明具体金额或比例，参数允许传 null，禁止臆造数据！",
        "parameters": {
            "type": "object",
            "properties": {
                "contract_total_amount": {"type": ["number", "null"], "description": "合同总金额(元)，若未载明传 null"},
                "daily_rate": {"type": ["number", "null"], "description": "每日违约金比例，如 1.5% 必须转换为 0.015，若未载明传 null"},
                "overdue_days": {"type": "integer", "description": "预估或约定的违约天数，默认 30"}
            },
            "required": []
        }
    }
}


def build_search_contract_tool(retriever: HybridRetriever) -> Tuple[Dict[str, Any], Callable[[str], str]]:
    """
    【工具 2 工厂：合同双路检索工具】
    绑定当前合同实例，生成检索工具的说明书与执行函数。
    """
    schema = {
        "type": "function",
        "function": {
            "name": "search_contract_clauses",
            "description": "在长篇合同中进行双路混合检索，查找与特定法律、财务、知识产权相关的条款原文与上下文",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的法律或业务关键词，如'知识产权归属'、'逾期违约金比例'"
                    }
                },
                "required": ["query"]
            }
        }
    }

    def execute(query: str) -> str:
        hits = retriever.retrieve(query, top_k=2)
        return "\n\n".join([f"[命中条款 (得分 {score:.4f})]: {doc}" for doc, score in hits])

    return schema, execute