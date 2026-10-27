"""
测试脚本：验证环境依赖与大模型连通性
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    print("[1/2] 正在校验核心依赖库导入...")
    import openai
    import pydantic
    import pdfplumber
    import chromadb
    import rank_bm25
    import streamlit
    print("      所有核心依赖库导入成功！")

def test_llm_connection():
    print("[2/2] 正在测试大模型 API 连通性...")
    from src.config import get_llm_client
    provider, client, model = get_llm_client()
    print(f"      识别到 API 渠道: {provider}, 模型: {model}")

    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请回复'连接成功'这四个字。"}],
            max_tokens=20,
        )
        reply = response.choices[0].message.content.strip()
        print(f"      模型回复: {reply}")
    elif provider == "gemini":
        response = client.models.generate_content(
            model=model,
            contents="请回复'连接成功'这四个字。"
        )
        print(f"      模型回复: {response.text.strip()}")

    print(">>> 恭喜！开发环境与大模型 API 链路全部打通！<<<")

if __name__ == "__main__":
    try:
        test_imports()
        test_llm_connection()
    except Exception as e:
        print(f"\n[ERROR] 校验失败: {e}", file=sys.stderr)
        sys.exit(1)

