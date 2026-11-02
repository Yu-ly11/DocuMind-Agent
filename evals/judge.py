"""
DocuMind-Agent: 答案层大模型评测员 (LLM-as-a-Judge)
评估指标：
1. 真实可信度 (Faithfulness): 模型回答中的每一条事实主张能否在检索上下文中找到依据。
2. 答案相关度 (Answer Relevance): 模型回答是否直接、完整回应了用户的法律与业务提问。
"""

import json
import re
from typing import Dict, Any


def _clean_and_parse_json(raw: str) -> Dict[str, Any]:
    """鲁棒的 JSON 解析器，自动剥离 Markdown 围栏与异常前后缀"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # 正则提取最外层花括号
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"score": 0.0, "reason": f"模型输出非标准 JSON: {raw[:100]}"}


def judge_faithfulness(client, model: str, context: str, answer: str) -> Dict[str, Any]:
    """
    判断答案中的每一条事实性陈述是否都能在上下文中找到依据 (0.0 - 1.0)
    score = 有依据的事实主张数量 / 答案包含的总事实主张数量
    """
    prompt = f"""你是极其严谨的企业法律合规评测专家。
给定【检索上下文依据】和【待测模型回答】，请逐条检验回答中提到的事实陈述、金额、费率、天数、权责约定是否都能在上下文中找到确凿依据。

输出严格 JSON 格式：
{{
  "score": 1.0, // 0.0 到 1.0 的浮点数
  "reason": "简明评估理由",
  "unsupported_claims": [] // 若存在未在上下文中支持的断言，在此列出
}}

【检索上下文依据】：
{context}

【待测模型回答】：
{answer}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=60
        )
        raw_text = resp.choices[0].message.content or ""
        parsed = _clean_and_parse_json(raw_text)
        if "score" not in parsed:
            parsed["score"] = 0.5
        parsed["score"] = float(parsed["score"])
        return parsed
    except Exception as e:
        return {"score": 0.0, "reason": f"Judge API 调用异常: {str(e)}", "unsupported_claims": []}


def judge_answer_relevance(client, model: str, question: str, answer: str) -> Dict[str, Any]:
    """
    判断答案是否直接、完整、切题地回答了用户问题 (0.0 - 1.0)
    """
    prompt = f"""你是极其严谨的企业法律合规评测专家。
给定【用户法务提问】和【待测模型回答】，请判断回答是否直接切中要害，是否完整解答了问题，是否有无意义废话或跑题现象。

输出严格 JSON 格式：
{{
  "score": 1.0, // 0.0 到 1.0 的浮点数
  "reason": "简明评估理由"
}}

【用户法务提问】：
{question}

【待测模型回答】：
{answer}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=60
        )
        raw_text = resp.choices[0].message.content or ""
        parsed = _clean_and_parse_json(raw_text)
        if "score" not in parsed:
            parsed["score"] = 0.5
        parsed["score"] = float(parsed["score"])
        return parsed
    except Exception as e:
        return {"score": 0.0, "reason": f"Judge API 调用异常: {str(e)}"}
