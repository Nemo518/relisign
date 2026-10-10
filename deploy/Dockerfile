# ============================================================
# Uni-Sign Web API Docker 镜像
#
# 构建:
#   docker build -f deploy/Dockerfile -t uni-sign-api:latest .
#
# 运行:
#   docker run --gpus all -p 8080:8080 \
#     -v /path/to/storage:/app/Uni-Sign-storage \
#     uni-sign-api:latest --mode balanced
# ============================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="Uni-Sign Team"
LABEL description="Uni-Sign 手语翻译 Web API 服务"

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN python -m pip install --upgrade pip setuptools wheel

# ---- 工作目录 ----
WORKDIR /app

# ---- 安装 Python 依赖 ----
# 先复制 requirements 利用 Docker 层缓存
COPY requirements.txt /app/requirements.txt
COPY deploy/requirements_deploy.txt /app/deploy/requirements_deploy.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -r /app/deploy/requirements_deploy.txt

# ---- 复制项目文件 ----
COPY . /app/

# ---- 环境变量 ----
ENV PYTHONUNBUFFERED=1
ENV UNISIGN_STORAGE_ROOT=/app/Uni-Sign-storage

# ---- 暴露端口 ----
EXPOSE 8080

# ---- 健康检查 ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# ---- 入口 ----
# 默认使用 balanced 模式；可通过 docker run 参数覆盖
ENTRYPOINT ["python", "-m", "deploy.server"]
CMD ["--mode", "balanced", "--host", "0.0.0.0", "--port", "8080"]
