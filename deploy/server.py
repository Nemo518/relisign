"""
FastAPI 推理服务 — 将 Uni-Sign 手语翻译模型部署为 REST API。

启动方式:
    python -m deploy.server                          # 默认 (balanced 模式, 0.0.0.0:8080)
    python -m deploy.server --mode fast --port 9000  # 快速模式, 端口 9000
    python -m deploy.server --mode minimal            # 最低显存模式

API 端点:
    POST /translate          — 上传姿态文件，返回翻译结果
    POST /translate/batch    — 批量翻译
    GET  /health             — 健康检查
    GET  /stats              — 服务统计
    GET  /docs               — Swagger 文档
"""
from __future__ import annotations

import gc
import io
import os
import pickle
import sys
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---- 应用生命周期 ----
_model_instance = None
_service_stats = {
    "total_requests": 0,
    "total_errors": 0,
    "start_time": time.time(),
    "last_request_time": None,
}


def get_model():
    global _model_instance
    if _model_instance is None:
        raise RuntimeError("模型尚未加载。请等待服务启动完成。")
    return _model_instance


@asynccontextmanager
async def lifespan(app):
    """FastAPI 生命周期：启动时加载模型，关闭时释放资源。"""
    global _model_instance

    import argparse

    # 解析命令行参数（在 uvicorn 启动时）
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rgb-support", action="store_true")
    known, _ = parser.parse_known_args()

    print(f"\n{'='*60}")
    print(f"  Uni-Sign 推理服务启动中...")
    print(f"  模式: {known.mode}")
    print(f"  地址: {known.host}:{known.port}")
    print(f"  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    from config_deploy import PRESETS
    from inference import SignLanguageInference

    config = PRESETS[known.mode]()
    if known.rgb_support:
        config.rgb_support = True

    config.host = known.host
    config.port = known.port

    _model_instance = SignLanguageInference(config)
    print("[server] 模型加载完成，服务就绪 ✓\n")

    yield  # 服务运行期间

    # 关闭
    print("[server] 正在关闭服务...")
    if _model_instance:
        _model_instance.unload()
    _model_instance = None
    gc.collect()
    print("[server] 服务已关闭")


# ---- FastAPI 应用 ----
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Uni-Sign 手语翻译 API",
    description="手语视频 → 自然语言文本翻译服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求/响应模型 ----
from pydantic import BaseModel


class TranslateResponse(BaseModel):
    success: bool
    text: str = ""
    time_ms: float = 0.0
    num_tokens: int = 0
    tokens_per_second: float = 0.0
    error: str = ""


class BatchTranslateResponse(BaseModel):
    success: bool
    results: List[TranslateResponse] = []
    total_time_ms: float = 0.0
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    uptime_seconds: float = 0.0
    total_requests: int = 0


# ---- API 端点 ----
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查。"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else ""
    vram_total = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_available
        else 0.0
    )
    vram_used = (
        torch.cuda.memory_allocated(0) / 1024**3 if gpu_available else 0.0
    )

    return HealthResponse(
        status="healthy" if _model_instance else "loading",
        model_loaded=_model_instance is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_total_gb=round(vram_total, 1),
        vram_used_gb=round(vram_used, 2),
        uptime_seconds=round(time.time() - _service_stats["start_time"], 1),
        total_requests=_service_stats["total_requests"],
    )


@app.get("/stats")
async def service_stats():
    """服务统计。"""
    return JSONResponse(content=_service_stats)


@app.post("/translate", response_model=TranslateResponse)
async def translate_pose(
    file: UploadFile = File(..., description="姿态 .pkl 文件"),
):
    """
    上传单个姿态文件并获取翻译结果。

    - **file**: 姿态 .pkl 文件（由 MMPose/RTMPose 提取）
    """
    _service_stats["total_requests"] += 1
    _service_stats["last_request_time"] = time.time()

    engine = get_model()

    try:
        # 读取上传文件
        contents = await file.read()
        pose_data = pickle.loads(contents)

        # 验证姿态数据格式
        if "keypoints" not in pose_data or "scores" not in pose_data:
            raise ValueError("姿态文件格式无效：缺少 'keypoints' 或 'scores' 字段")

        # 执行翻译
        result = engine.translate_pose_dict(pose_data)

        return TranslateResponse(
            success=True,
            text=result.text,
            time_ms=round(result.time_ms, 1),
            num_tokens=result.num_tokens,
            tokens_per_second=round(result.tokens_per_second, 1),
        )

    except Exception as e:
        _service_stats["total_errors"] += 1
        traceback.print_exc()
        return TranslateResponse(
            success=False,
            error=str(e),
        )


@app.post("/translate/batch", response_model=BatchTranslateResponse)
async def translate_batch(
    files: List[UploadFile] = File(..., description="多个姿态 .pkl 文件"),
):
    """
    批量上传姿态文件并获取翻译结果。

    - **files**: 最多 10 个姿态文件
    """
    _service_stats["total_requests"] += len(files)
    _service_stats["last_request_time"] = time.time()

    engine = get_model()

    if len(files) > 10:
        return BatchTranslateResponse(
            success=False,
            error=f"批量限制为 10 个文件，收到 {len(files)} 个",
        )

    results = []
    t0 = time.perf_counter()

    for file in files:
        try:
            contents = await file.read()
            pose_data = pickle.loads(contents)

            if "keypoints" not in pose_data or "scores" not in pose_data:
                results.append(
                    TranslateResponse(success=False, error="无效的姿态文件格式")
                )
                continue

            result = engine.translate_pose_dict(pose_data)
            results.append(
                TranslateResponse(
                    success=True,
                    text=result.text,
                    time_ms=round(result.time_ms, 1),
                    num_tokens=result.num_tokens,
                    tokens_per_second=round(result.tokens_per_second, 1),
                )
            )

        except Exception as e:
            _service_stats["total_errors"] += 1
            results.append(TranslateResponse(success=False, error=str(e)))

    total_time_ms = (time.perf_counter() - t0) * 1000

    return BatchTranslateResponse(
        success=True,
        results=results,
        total_time_ms=round(total_time_ms, 1),
    )


@app.post("/translate/raw", response_model=TranslateResponse)
async def translate_raw_json(data: Dict[str, Any]):
    """
    直接发送 JSON 格式的姿态数据（在线采集场景）。

    JSON 格式:
    ```json
    {
        "keypoints": [[[x1,y1], [x2,y2], ...], ...],
        "scores": [[s1, s2, ...], ...]
    }
    ```

    每帧包含 133 个关键点（全身 + 手部 + 面部）。
    """
    _service_stats["total_requests"] += 1
    _service_stats["last_request_time"] = time.time()

    engine = get_model()

    try:
        # 转换为 numpy 数组
        keypoints = [np.array(kp, dtype=np.float32) for kp in data["keypoints"]]
        scores = [np.array(sc, dtype=np.float32) for sc in data["scores"]]

        # 统一为 (1, 133, 2) 格式
        for i, (kp, sc) in enumerate(zip(keypoints, scores)):
            if kp.ndim == 2:
                keypoints[i] = kp[None]
            if sc.ndim == 1:
                scores[i] = sc[None]

        pose_data = {"keypoints": keypoints, "scores": scores}

        result = engine.translate_pose_dict(pose_data)

        return TranslateResponse(
            success=True,
            text=result.text,
            time_ms=round(result.time_ms, 1),
            num_tokens=result.num_tokens,
            tokens_per_second=round(result.tokens_per_second, 1),
        )

    except Exception as e:
        _service_stats["total_errors"] += 1
        traceback.print_exc()
        return TranslateResponse(success=False, error=str(e))


# ============================================================
# 直接启动
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign 推理服务")
    parser.add_argument("--mode", default="balanced", choices=["balanced", "fast", "minimal", "quality"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rgb-support", action="store_true", help="启用手部 RGB 分支")
    args = parser.parse_args()

    import uvicorn

    # 将参数传递给 lifespan
    sys.argv = [
        sys.argv[0],
        "--mode", args.mode,
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.rgb_support:
        sys.argv.append("--rgb-support")

    uvicorn.run(
        "deploy.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
