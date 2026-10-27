import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from openai import OpenAI
from dotenv import load_dotenv
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[
        {"role": "user", "content": "你好！请只回复我三个字：我准备好了"}
    ]
)

print("模型回复:", response.choices[0].message.content)

from pydantic import BaseModel,Field
from enum import Enum

class RiskLevel(str,Enum):
    HIGH="高风险"
    MEDIUM="中风险"
    LOW="低风险"

class RiskItem(BaseModel):
    title:str = Field(...,description="风险标题")
    risk_level:RiskLevel = Field(...,description="风险等级")
    line_number:int = Field(...,description="风险所在行号")
    advice: str = Field(..., description="修改建议")

# 试着手动创建一个合法的风险项打印看看
item = RiskItem(
    title="违约金过高",
    risk_level=RiskLevel.HIGH,
    line_number=25,
    advice="建议下调至10%以内"
)
print("Pydantic 验证通过的对象:", item)

import json
schema =json.dumps(RiskItem.model_json_schema(),ensure_ascii=False)
contract_clause = "【第18行】若乙方延迟交付1天，须赔偿甲方100万元，且甲方不负任何违约责任。"

prompt = f"""
请审查以下合同条款并找出风险：
{contract_clause}
【格式死命令】：
你必须且只能输出合法的 JSON 格式，严格符合以下说明书：
{schema}
"""
res = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"}
)

raw_json =res.choices[0].message.content

print ("\n--- 大模型吐出的原始 JSON 字符串 ---",raw_json)
final_risk = RiskItem.model_validate_json(raw_json)
print("\n--- Pydantic 最终转换成功的安全对象 ---")
print("标题:", final_risk.title)
print("风险级别:", final_risk.risk_level.value)
print("所在行号:", final_risk.line_number)
print("处置建议:", final_risk.advice)



from typing import List, Tuple
from rank_bm25 import BM25Okapi

def tokenize (text:str)->list[str]:
    return list(text.lower().strip())

class BM25Retriever:
    def __init__(self, corpus:List[str]):
        self.corpus = corpus
        tokenize_corpus = [tokenize(doc)for doc in corpus]
        self.bm25 = BM25Okapi(tokenize_corpus)

    def search(self,query:str, top_k:int=2)->List[Tuple[int,float]]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_results = sorted(
                    zip(self.corpus, scores),
                    key=lambda x: x[1],
                    reverse=True
                )
        return ranked_results[:top_k]

if __name__ == "__main__":
    # 假设这是我们切出来的 4 个合同文本块
    sample_chunks = [
        "第一条：双方本着诚信原则，共同开发 AI 知识库系统，工期为三个月。",
        "第二条：合同总金额为人民币五十万元整，分三期支付完毕。",
        "第三条：乙方若逾期交付，每日须支付违约金5000元；逾期超30天解除合同。",
        "第四条：双方对合作过程中获知的数据负有终身保密义务，不得外泄。"
    ]
    
    # 1. 建立检索器
    retriever = BM25Retriever(sample_chunks)
    
    # 2. 模拟搜索："逾期要赔多少钱违约金？"
    user_query = "逾期要赔多少钱违约金？"
    results = retriever.search(user_query, top_k=2)
    
    print(f"用户搜索问题: '{user_query}'\n")
    print("--- BM25 瞬间检索出的最相关条款 ---")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"Top {rank} (匹配得分: {score:.2f}):\n   {chunk}\n")




import chromadb

chromadb_client = chromadb.Client()

collection=chromadb_client.create_collection(name="contract_collecation")
collection.add(
    documents=[
        "双方共同开发 AI 知识库系统，工期为三个月。",
        "合同总金额为人民币五十万元整，分三期付清。",
        "乙方若逾期交付，每日须向甲方支付违约金5000元人民币。",
        "双方对合作过程中获知的数据负有终身保密义务。"
    ],
    ids=["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
    )

query_text = "迟到了要扣多少钱？"
results = collection.query(
    query_texts=[query_text],
    n_results=1
)
print(f"\n用户提问: '{query_text}'")
print("--- ChromaDB 向量语义检索命中的条款 ---")
print("匹配到的内容:", results["documents"][0][0])
print("对应的数据ID:", results["ids"][0][0])
print("空间语义距离 (Distance):", results["distances"][0][0])
bm25_res = retriever.search("逾期违约金多少", top_k=1)
print("BM25 精准命中的条款:", bm25_res[0][0])
print("BM25 得分:", bm25_res[0][1])

from collections import defaultdict

def rrf_fusion(bm25_ranks: list, vector_ranks: list, k: int = 60) -> list:
    """
    RRF 倒数排名融合算法
    :param bm25_ranks: BM25 检索排好序的文档列表（从第1名到最后一名）
    :param vector_ranks: 向量检索排好序的文档列表（从第1名到最后一名）
    :param k: 平滑常数，默认 60
    :return: 最终融合打分并排好序的榜单
    """
    rrf_scores = defaultdict(float)
    
    # 1. 计算每个文档在 BM25 榜单上的贡献分
    for rank, doc in enumerate(bm25_ranks, start=1):
        rrf_scores[doc] += 1.0 / (k + rank)
        
    # 2. 加上每个文档在 向量榜单 上的贡献分
    for rank, doc in enumerate(vector_ranks, start=1):
        rrf_scores[doc] += 1.0 / (k + rank)
        
    # 3. 按最终融合成绩从高到低排序交卷！
    final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return final_ranked


# 假设：
# BM25 榜单（词频狙击手）：把“违约金5000元”排在第 1 名，“工期三个月”排在第 2 名
bm25_list = ["第三条：违约金5000元", "第一条：工期三个月", "第二条：合同总额五十万"]

# 向量榜单（语义理解）：把“第三条：违约金5000元”排在第 2 名，“第二条：合同总额五十万”排在第 1 名
vector_list = ["第二条：合同总额五十万", "第三条：违约金5000元", "第一条：工期三个月"]

# 呼叫 RRF 融合裁判！
final_results = rrf_fusion(bm25_list, vector_list)

print("\n" + "="*50)
print("--- [RRF] 混合检索双路融合终极榜单 ---")
for rank, (doc, score) in enumerate(final_results, start=1):
    print(f"总榜第 {rank} 名 (RRF总分: {score:.5f}):\n   {doc}")



import json
from src.tools.compliance_tools import calc_penalty_risk

# ==========================================
# 步骤 1：把刚才写的计算器，做成大模型看得懂的“工具菜单”
# ==========================================
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calc_penalty_risk",
            "description": "根据合同总额、日违约金比例和逾期天数，精确计算累计违约金并评估法律超标风险",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_total_amount": {
                        "type": "number",
                        "description": "合同总金额，如 500000"
                    },
                    "daily_rate": {
                        "type": "number",
                        "description": "每日违约金比例，如 2% 必须换算成 0.02"
                    },
                    "overdue_days": {
                        "type": "integer",
                        "description": "逾期天数，如 30"
                    }
                },
                "required": ["contract_total_amount", "daily_rate", "overdue_days"]
            }
        }
    }
]

# ==========================================
# 步骤 2：用户提出一个复杂的审查问题，并发给大模型
# ==========================================
messages = [
    {
        "role": "user",
        "content": "合同总金额50万元，条款约定若乙方延迟，每天罚款2%，如果延期了30天，请帮我审查这个违约金合法合规吗？"
    }
]

print("\n" + "="*50)
print("[AGENT] 智能体启动：正在思考是否需要动用外部工具...")

first_response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=messages,
    tools=tools_schema,  # [CORE] 核心：把工具菜单递给大模型！
    tool_choice="auto"   # 让模型自主决定要不要用工具
)

response_message = first_response.choices[0].message

# ==========================================
# 步骤 3：检查大模型是否呼叫了工具！
# ==========================================
if response_message.tool_calls:
    tool_call = response_message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)
    
    print(f"\n[ACTION] 大模型自主决定呼叫工具: 【{func_name}】")
    print(f"[PARAM] 大模型自动从问题中提取的参数: {func_args}")
    
    # 本地 Python 执行真实的数学风控计算！
    if func_name == "calc_penalty_risk":
        tool_result = calc_penalty_risk(**func_args)
        print(f"[CALC] 本地计算结果: {tool_result}")

        # ==========================================
        # 步骤 4：把大模型的意图和本地执行结果，一起塞回对话历史！
        # ==========================================
        messages.append(response_message) # 记录模型的调用动作
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": json.dumps(tool_result, ensure_ascii=False) # 递交计算凭证
        })

        # ==========================================
        # 步骤 5：大模型根据真实凭证，输出最终审查定论！
        # ==========================================
        final_response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=messages
        )
        print("\n--- [REPORT] 智能体结合工具计算后的最终专业审查报告 ---")
        print(final_response.choices[0].message.content)