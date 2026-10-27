"""
DocuMind-Agent: 配置中心与模型 Client 管理
支持 OpenAI 兼容格式 (OpenAI, DeepSeek, Kimi, 智谱等) 以及 Google Gemini 原生 SDK。
"""

import os
from typing import Optional, Tuple, Any
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()


class Settings:
    """系统全局配置"""
    # 选项 A: OpenAI 兼容 API
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # 选项 B: Google Gemini 官方 API
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")


settings = Settings()


def get_llm_client() -> Tuple[str, Any, str]:
    """
    自适应获取可用的大模型客户端与模型名称。
    
    返回元组:
        (provider_type, client_instance, model_name)
        例如: ("openai", OpenAIClient, "deepseek-chat") 或 ("gemini", GeminiClient, "gemini-2.0-flash")
    """
    # 优先检测 OpenAI 兼容格式（支持 DeepSeek / Kimi / 智谱 等）
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here":
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        return "openai", client, settings.MODEL_NAME

    # 其次检测 Gemini 官方 SDK
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return "gemini", client, settings.GEMINI_MODEL_NAME

    raise ValueError(
        "未检测到有效的大模型 API 密钥！\n"
        "请检查项目根目录下的 `.env` 文件，确保填入了 `OPENAI_API_KEY` 或 `GEMINI_API_KEY`。"
    )
