# DocuMind-Agent 生产级容器构建 Dockerfile
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，禁止 Python 输出缓冲并固定 UTF-8 编码
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 安装系统基础依赖 (构建 C++ 扩展、libmagic 等可能需要的底层工具)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目代码与资源
COPY src/ ./src/
COPY app.py .
COPY tests/sample_contracts/ ./tests/sample_contracts/

# 创建持久化数据库目录与临时上传目录
RUN mkdir -p chroma_db /tmp/documind_uploads

# 暴露端口 (8000 为 FastAPI 服务，8501 为 Streamlit 界面)
EXPOSE 8000 8501

# 默认启动生产级 FastAPI 微服务
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
