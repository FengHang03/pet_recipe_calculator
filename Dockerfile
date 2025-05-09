# Dockerfile - 用于构建API服务的Docker镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=8000 \
    WORKERS=4 \
    RATE_LIMIT=100 \
    LOG_LEVEL=INFO

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY api.py .
COPY recipe_optimize.py .
COPY data_utils.py .
COPY .env.example .env

# 创建数据目录
RUN mkdir -p data

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安全：设置非root用户
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动API服务
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
