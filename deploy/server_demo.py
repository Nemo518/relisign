"""
Demo 模式服务器 — 展示完整部署架构，不需要真实模型权重。

模拟真实推理管线的所有层级：
   L1: API 网关层     — FastAPI 路由、CORS、请求验证
   L2: 预处理层       — 姿态数据解析、归一化、采样
   L3: 模型推理层     — Pose Encoder + mT5 Decoder（demo 用随机输出模拟）
   L4: 后处理层       — Token 解码、文本清洗
   L5: 监控层         — 健康检查、性能统计、日志

启动:
    python deploy/server_demo.py
    python deploy/server_demo.py --port 9000
"""
from __future__ import annotations

import os
import sys
import time
import random
import pickle
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 应用生命周期 ----
_service_stats: Dict[str, Any] = {}


def _init_stats():
    global _service_stats
    _service_stats = {
        "total_requests": 0,
        "total_errors": 0,
        "start_time": time.time(),
        "last_request_time": None,
        "recent_latencies": [],    # 最近 100 次延迟
        "mode": "demo (mock model)",
        "version": "1.0.0",
        "layers": {
            "api_gateway": "FastAPI + CORS + pydantic",
            "preprocessing": "姿态归一化 + 帧采样 + 子姿态提取",
            "model_inference": "PoseEncoder(GCN) → mT5 Decoder [DEMO MOCK]",
            "postprocessing": "Token decode + 文本清洗",
            "monitoring": "健康检查 + 延迟统计 + 错误追踪",
        },
    }


_init_stats()


@asynccontextmanager
async def lifespan(app):
    """FastAPI 生命周期。"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       Uni-Sign 手语翻译服务 — Demo 模式                   ║
╠══════════════════════════════════════════════════════════╣
║  状态: 运行中 (Mock Model)                               ║
║  说明: 模拟完整推理管线，不需要真实模型权重               ║
║  API:  http://localhost:{_PORT}/docs                     ║
║  健康: http://localhost:{_PORT}/health                   ║
╚══════════════════════════════════════════════════════════╝
""")
    yield
    print("[server] 服务已关闭")


# ---- FastAPI 应用 ----
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="Uni-Sign 手语翻译 API",
    description="""
## 手语视频 → 自然语言翻译服务

### 架构层级

| 层级 | 技术栈 | 说明 |
|------|--------|------|
| **API 网关** | FastAPI + CORS + Pydantic | 路由、验证、限流 |
| **预处理** | NumPy + 自定义采样器 | 姿态归一化、帧采样 |
| **模型推理** | GCN + Deformable Attention + mT5 | 端到端翻译 |
| **后处理** | Token Decode + 文本清洗 | 输出自然语言 |
| **监控** | 健康检查 + 延迟统计 | 服务可观测性 |

### Demo 模式说明
当前为 **Demo 模式**，使用 Mock 模型模拟推理过程，
展示完整的部署架构。切换为真实模型只需修改配置。
""",
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
class TranslateResponse(BaseModel):
    success: bool
    text: str = ""
    time_ms: float = 0.0
    num_tokens: int = 0
    tokens_per_second: float = 0.0
    # 显示各层耗时
    layer_timing: Dict[str, float] = field(default_factory=dict)
    error: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "text": "今天天气很好",
                "time_ms": 420.5,
                "num_tokens": 8,
                "tokens_per_second": 19.0,
                "layer_timing": {
                    "api_gateway_ms": 0.5,
                    "preprocessing_ms": 12.3,
                    "model_inference_ms": 405.2,
                    "postprocessing_ms": 2.5,
                },
                "error": "",
            }
        }


class BatchTranslateResponse(BaseModel):
    success: bool
    results: List[TranslateResponse] = []
    total_time_ms: float = 0.0
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    mode: str
    gpu_available: bool
    gpu_name: str = ""
    uptime_seconds: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    recent_requests: int = 0


class ArchitectureResponse(BaseModel):
    """展示部署架构的所有层级。"""
    layers: List[Dict[str, str]]
    data_flow: List[str]
    deployment_options: Dict[str, str]
    current_config: Dict[str, Any]


# ---- Mock 推理引擎（模拟真实管线） ----
class MockInferenceEngine:
    """
    Mock 推理引擎 — 模拟完整的 4 层推理管线。

    在真实部署中，这些层对应:
    - L1: API Gateway (FastAPI middleware)
    - L2: datasets.py → load_part_kp() + crop_scale()
    - L3: models.py → Uni_Sign.forward() + generate()
    - L4: mT5 tokenizer.decode()
    """

    # 模拟中文翻译结果
    DEMO_TEXTS = [
        "今天天气很好",
        "我想去超市买东西",
        "请帮我打开窗户",
        "谢谢你的帮助",
        "明天会下雨吗",
        "这本书很有趣",
        "我们一起去吃饭吧",
        "这是我的朋友",
        "好久不见",
        "祝你生日快乐",
    ]

    def __init__(self):
        self._call_count = 0

    def _simulate_preprocessing(self, pose_data: Dict) -> tuple:
        """L2: 模拟姿态预处理。"""
        time.sleep(random.uniform(0.005, 0.015))  # 模拟计算

        if "keypoints" not in pose_data or "scores" not in pose_data:
            raise ValueError("缺少 keypoints 或 scores 字段")

        kp = np.asarray(pose_data["keypoints"])
        sc = np.asarray(pose_data["scores"])

        num_frames = len(kp) if hasattr(kp, '__len__') else 1
        num_keypoints = kp.shape[-2] if kp.ndim >= 2 else 133

        return num_frames, num_keypoints, sc.mean() if sc.size > 0 else 0.0

    def _simulate_inference(self, num_frames: int, confidence: float) -> str:
        """L3: 模拟模型推理。"""
        # 模拟推理延迟（真实情况: INT8 ~400ms, NF4 ~550ms）
        base_latency = 0.35  # 秒
        frame_factor = max(num_frames, 1) / 128.0
        noise = random.uniform(-0.03, 0.03)
        delay = max(0.01, base_latency * frame_factor + noise)  # 确保非负
        time.sleep(delay)

        self._call_count += 1
        return random.choice(self.DEMO_TEXTS)

    def _simulate_postprocessing(self, raw_text: str) -> tuple:
        """L4: 模拟后处理（token decode + 文本清洗）。"""
        time.sleep(random.uniform(0.001, 0.003))
        num_tokens = len(raw_text)
        return raw_text.strip(), num_tokens

    def translate(self, pose_data: Dict) -> Dict:
        """执行完整的 4 层推理管线。"""
        timings = {}

        # L2: 预处理
        t0 = time.perf_counter()
        num_frames, num_kp, confidence = self._simulate_preprocessing(pose_data)
        timings["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # L3: 模型推理
        t0 = time.perf_counter()
        raw_text = self._simulate_inference(num_frames, confidence)
        timings["model_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # L4: 后处理
        t0 = time.perf_counter()
        text, num_tokens = self._simulate_postprocessing(raw_text)
        timings["postprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        total_ms = sum(timings.values())
        timings["api_gateway_ms"] = 0.5  # FastAPI 自身开销

        return {
            "text": text,
            "num_tokens": num_tokens,
            "total_ms": total_ms,
            "tokens_per_second": round(num_tokens / (total_ms / 1000), 1) if total_ms > 0 else 0,
            "layer_timing": timings,
            "metadata": {
                "frames": num_frames,
                "keypoints_per_frame": num_kp,
                "confidence": round(float(confidence), 3),
                "model_mode": "DEMO (Mock)",
            },
        }


# 全局引擎实例
_engine = MockInferenceEngine()

# 服务器端口（全局变量，启动时设置）
_PORT = 8080


# ---- API 端点 ----

@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 — 展示部署架构总览页面。"""
    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>Uni-Sign 手语翻译服务</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
            .container {{ max-width: 960px; margin: 0 auto; padding: 40px 20px; }}
            h1 {{ font-size: 2rem; color: #38bdf8; margin-bottom: 8px; }}
            .subtitle {{ color: #94a3b8; margin-bottom: 32px; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #334155; }}
            .card h2 {{ font-size: 1.2rem; color: #e2e8f0; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
            .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
            .badge-green {{ background: #065f46; color: #6ee7b7; }}
            .badge-blue {{ background: #1e3a5f; color: #93c5fd; }}
            .badge-yellow {{ background: #5c4a0a; color: #fde68a; }}
            .badge-purple {{ background: #3b1f5e; color: #c4b5fd; }}
            .layer-list {{ display: grid; gap: 12px; }}
            .layer {{ display: flex; align-items: center; gap: 16px; padding: 12px 16px; background: #0f172a; border-radius: 8px; border-left: 4px solid #38bdf8; }}
            .layer-num {{ width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.875rem; flex-shrink: 0; }}
            .layer-info {{ flex: 1; }}
            .layer-info strong {{ display: block; margin-bottom: 2px; }}
            .layer-info span {{ font-size: 0.85rem; color: #94a3b8; }}
            .layer-tech {{ font-size: 0.8rem; color: #64748b; flex-shrink: 0; }}
            .flow {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 16px 0; }}
            .flow-step {{ padding: 8px 16px; background: #1e293b; border-radius: 8px; font-size: 0.9rem; border: 1px solid #475569; }}
            .flow-arrow {{ color: #64748b; font-size: 1.2rem; }}
            .endpoints {{ display: grid; gap: 8px; }}
            .endpoint {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #0f172a; border-radius: 6px; font-family: monospace; font-size: 0.9rem; }}
            .method {{ font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; width: 52px; text-align: center; flex-shrink: 0; }}
            .method-get {{ background: #065f46; color: #6ee7b7; }}
            .method-post {{ background: #1e3a5f; color: #93c5fd; }}
            .status {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .stat {{ text-align: center; }}
            .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #38bdf8; }}
            .stat-label {{ font-size: 0.8rem; color: #94a3b8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤟 Uni-Sign 手语翻译服务</h1>
            <p class="subtitle">统一手语理解框架 — 多模态手语翻译 API</p>

            <!-- 状态卡片 -->
            <div class="card">
                <h2>📊 服务状态 <span class="badge badge-green">运行中</span></h2>
                <div class="status" id="status-panel">
                    <div class="stat"><div class="stat-value" id="stat-uptime">--</div><div class="stat-label">运行时间 (s)</div></div>
                    <div class="stat"><div class="stat-value" id="stat-requests">--</div><div class="stat-label">总请求数</div></div>
                    <div class="stat"><div class="stat-value" id="stat-latency">--</div><div class="stat-label">平均延迟 (ms)</div></div>
                    <div class="stat"><div class="stat-value" id="stat-mode">Demo</div><div class="stat-label">运行模式</div></div>
                </div>
            </div>

            <!-- 架构层级 -->
            <div class="card">
                <h2>🏗️ 部署架构层级</h2>
                <div class="layer-list">
                    <div class="layer">
                        <div class="layer-num" style="background:#3b82f6">1</div>
                        <div class="layer-info"><strong>API 网关层</strong><span>路由分发、请求验证、CORS、限流</span></div>
                        <div class="layer-tech">FastAPI + Pydantic</div>
                    </div>
                    <div class="layer">
                        <div class="layer-num" style="background:#8b5cf6">2</div>
                        <div class="layer-info"><strong>预处理层</strong><span>姿态归一化、帧采样、子姿态提取 (body/left/right/face)</span></div>
                        <div class="layer-tech">NumPy + 自定义采样器</div>
                    </div>
                    <div class="layer">
                        <div class="layer-num" style="background:#ec4899">3</div>
                        <div class="layer-info"><strong>模型推理层</strong><span>Pose Encoder (GCN) → mT5 Decoder 端到端翻译</span></div>
                        <div class="layer-tech">PyTorch + mT5 + GCN</div>
                    </div>
                    <div class="layer">
                        <div class="layer-num" style="background:#f59e0b">4</div>
                        <div class="layer-info"><strong>后处理层</strong><span>Token 解码、文本清洗、格式化输出</span></div>
                        <div class="layer-tech">SentencePiece + Tokenizer</div>
                    </div>
                    <div class="layer">
                        <div class="layer-num" style="background:#10b981">5</div>
                        <div class="layer-info"><strong>监控层</strong><span>健康检查、延迟统计、错误追踪、日志</span></div>
                        <div class="layer-tech">Health Check + Metrics</div>
                    </div>
                </div>
            </div>

            <!-- 数据流 -->
            <div class="card">
                <h2>📡 数据流</h2>
                <div class="flow">
                    <div class="flow-step">📹 姿态数据</div><span class="flow-arrow">→</span>
                    <div class="flow-step">🔍 预处理</div><span class="flow-arrow">→</span>
                    <div class="flow-step">🧠 模型推理</div><span class="flow-arrow">→</span>
                    <div class="flow-step">📝 文本输出</div><span class="flow-arrow">→</span>
                    <div class="flow-step">📊 监控统计</div>
                </div>
            </div>

            <!-- API 端点 -->
            <div class="card">
                <h2>🔌 API 端点</h2>
                <div class="endpoints">
                    <div class="endpoint"><span class="method method-get">GET</span><span>/health</span><span style="color:#94a3b8;margin-left:auto">健康检查 + GPU 状态</span></div>
                    <div class="endpoint"><span class="method method-get">GET</span><span>/stats</span><span style="color:#94a3b8;margin-left:auto">服务统计</span></div>
                    <div class="endpoint"><span class="method method-get">GET</span><span>/architecture</span><span style="color:#94a3b8;margin-left:auto">架构详情</span></div>
                    <div class="endpoint"><span class="method method-post">POST</span><span>/translate</span><span style="color:#94a3b8;margin-left:auto">上传 .pkl 文件翻译</span></div>
                    <div class="endpoint"><span class="method method-post">POST</span><span>/translate/batch</span><span style="color:#94a3b8;margin-left:auto">批量翻译 (max 10)</span></div>
                    <div class="endpoint"><span class="method method-post">POST</span><span>/translate/raw</span><span style="color:#94a3b8;margin-left:auto">JSON 在线姿态数据</span></div>
                    <div class="endpoint"><span class="method method-get">GET</span><span>/docs</span><span style="color:#94a3b8;margin-left:auto">Swagger 交互文档</span></div>
                </div>
            </div>

            <!-- 部署选项 -->
            <div class="card">
                <h2>🚀 部署选项</h2>
                <div class="layer-list">
                    <div class="layer" style="border-left-color:#3b82f6">
                        <div class="layer-num" style="background:#3b82f6">A</div>
                        <div class="layer-info"><strong>Web API 网站</strong><span>FastAPI + Docker + Nginx</span></div>
                        <div class="layer-tech"><span class="badge badge-blue">balanced</span> INT8 + beam=4</div>
                    </div>
                    <div class="layer" style="border-left-color:#10b981">
                        <div class="layer-num" style="background:#10b981">B</div>
                        <div class="layer-info"><strong>边缘设备 (Jetson)</strong><span>ONNX Runtime / TensorRT</span></div>
                        <div class="layer-tech"><span class="badge badge-yellow">edge_4gb</span> NF4 + greedy</div>
                    </div>
                    <div class="layer" style="border-left-color:#8b5cf6">
                        <div class="layer-num" style="background:#8b5cf6">C</div>
                        <div class="layer-info"><strong>Docker 容器化</strong><span>docker-compose up -d</span></div>
                        <div class="layer-tech"><span class="badge badge-purple">production</span> + Nginx</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // 自动刷新状态
            async function refreshStatus() {{
                try {{
                    const resp = await fetch('/health');
                    const data = await resp.json();
                    document.getElementById('stat-uptime').textContent = data.uptime_seconds.toFixed(0);
                    document.getElementById('stat-requests').textContent = data.total_requests;
                    document.getElementById('stat-latency').textContent = data.avg_latency_ms.toFixed(1);
                    document.getElementById('stat-mode').textContent = data.mode === 'demo (mock model)' ? 'Demo' : 'Live';
                }} catch(e) {{}}
            }}
            refreshStatus();
            setInterval(refreshStatus, 3000);
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查 — 返回服务状态和 GPU 信息。"""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"

    recent = _service_stats.get("recent_latencies", [])
    avg_latency = sum(recent) / len(recent) if recent else 0.0

    uptime = time.time() - _service_stats["start_time"]

    return HealthResponse(
        status="healthy",
        mode=_service_stats["mode"],
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        uptime_seconds=round(uptime, 1),
        total_requests=_service_stats["total_requests"],
        total_errors=_service_stats["total_errors"],
        avg_latency_ms=round(avg_latency, 1),
        recent_requests=len(recent),
    )


@app.get("/stats")
async def service_stats():
    """详细服务统计。"""
    recent = _service_stats.get("recent_latencies", [])
    return JSONResponse(content={
        **_service_stats,
        "recent_latency_count": len(recent),
        "avg_latency_ms": round(sum(recent) / len(recent), 1) if recent else 0,
        "min_latency_ms": round(min(recent), 1) if recent else 0,
        "max_latency_ms": round(max(recent), 1) if recent else 0,
    })


@app.get("/architecture", response_model=ArchitectureResponse)
async def architecture():
    """返回完整部署架构信息。"""
    return ArchitectureResponse(
        layers=[
            {"name": "API 网关层", "tech": "FastAPI + CORS + Pydantic", "desc": "路由分发、请求验证、CORS"},
            {"name": "预处理层", "tech": "NumPy + 自定义采样器", "desc": "姿态归一化、帧采样 (max_length=256)、子姿态提取 (body/left/right/face_all)"},
            {"name": "模型推理层", "tech": "PyTorch + mT5 + GCN", "desc": "Pose Encoder (4路GCN) → mT5 Decoder → 文本生成"},
            {"name": "后处理层", "tech": "SentencePiece Tokenizer", "desc": "Token decode、skip_special_tokens、文本清洗"},
            {"name": "监控层", "tech": "Health Check + Metrics", "desc": "健康检查、延迟统计、错误追踪、运行日志"},
        ],
        data_flow=[
            "姿态 .pkl 文件上传 → /translate",
            "姿态归一化 (crop_scale) → 子姿态提取 (load_part_kp)",
            "4路 GCN 编码 (body/left/right/face) → 时序池化",
            "mT5 Encoder → Cross-Attention → mT5 Decoder",
            "Beam Search (num_beams=4) → Token IDs",
            "SentencePiece Decode → 自然语言文本",
        ],
        deployment_options={
            "balanced": "INT8 量化 + beam=4, ~1.8GB (推荐 Web API)",
            "fast": "INT8 量化 + greedy, ~1.8GB (高并发)",
            "minimal": "NF4 量化 + greedy, ~1.2GB (边缘设备)",
            "quality": "FP16 + beam=4, ~2.2GB (质量优先)",
        },
        current_config={
            "mode": "demo",
            "host": "0.0.0.0",
            "port": _PORT,
            "gpu_available": __import__("torch").cuda.is_available(),
        },
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_pose(
    file: UploadFile = File(..., description="姿态 .pkl 文件 (MMPose/RTMPose 提取)"),
):
    """
    ## 上传姿态文件并获取翻译结果

    上传由 MMPose / RTMPose 提取的姿态 .pkl 文件，
    经过完整 4 层管线处理后返回中文翻译文本。

    ### 姿态文件格式
    - keypoints: 每帧关键点坐标 (N_persons, 133, 2)
    - scores: 每帧关键点置信度 (N_persons, 133)

    ### 处理流程
    1. API Gateway → 2. 预处理 → 3. 模型推理 → 4. 后处理
    """
    t_start = time.perf_counter()
    _service_stats["total_requests"] += 1
    _service_stats["last_request_time"] = time.time()

    # L1: API Gateway — 已在 FastAPI 层完成验证
    t_api = time.perf_counter()

    try:
        contents = await file.read()

        try:
            pose_data = pickle.loads(contents)
        except Exception:
            # 尝试作为 numpy 数组加载
            pose_data = np.load(io.BytesIO(contents), allow_pickle=True)
            if isinstance(pose_data, np.ndarray):
                pose_data = pose_data.item()

        if not isinstance(pose_data, dict) or "keypoints" not in pose_data:
            raise ValueError("姿态文件格式无效：需要包含 'keypoints' 和 'scores' 字段")

        # L2+L3+L4: 推理管线
        result = _engine.translate(pose_data)
        api_overhead = round((time.perf_counter() - t_api) * 1000, 1)
        result["layer_timing"]["api_gateway_ms"] = api_overhead

        total_ms = (time.perf_counter() - t_start) * 1000

        # 记录延迟
        _service_stats.setdefault("recent_latencies", []).append(total_ms)
        if len(_service_stats["recent_latencies"]) > 100:
            _service_stats["recent_latencies"] = _service_stats["recent_latencies"][-100:]

        print(f"[L1→L4] 翻译完成 | 文本: {result['text']} | 总延迟: {total_ms:.0f}ms | "
              f"各层: API={result['layer_timing']['api_gateway_ms']}ms "
              f"预处理={result['layer_timing']['preprocessing_ms']}ms "
              f"推理={result['layer_timing']['model_inference_ms']}ms "
              f"后处理={result['layer_timing']['postprocessing_ms']}ms")

        return TranslateResponse(
            success=True,
            text=result["text"],
            time_ms=round(total_ms, 1),
            num_tokens=result["num_tokens"],
            tokens_per_second=result["tokens_per_second"],
            layer_timing=result["layer_timing"],
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
    files: List[UploadFile] = File(..., description="多个姿态 .pkl 文件 (max 10)"),
):
    """批量翻译 — 最多 10 个文件。"""
    t_start = time.perf_counter()
    _service_stats["total_requests"] += len(files)

    if len(files) > 10:
        return BatchTranslateResponse(
            success=False,
            error=f"批量限制最多 10 个，收到 {len(files)}",
        )

    results = []
    for file in files:
        try:
            contents = await file.read()
            pose_data = pickle.loads(contents)
            r = _engine.translate(pose_data)
            results.append(TranslateResponse(
                success=True,
                text=r["text"],
                time_ms=r["total_ms"],
                num_tokens=r["num_tokens"],
                tokens_per_second=r["tokens_per_second"],
                layer_timing=r["layer_timing"],
            ))
        except Exception as e:
            _service_stats["total_errors"] += 1
            results.append(TranslateResponse(success=False, error=str(e)))

    total_ms = (time.perf_counter() - t_start) * 1000
    return BatchTranslateResponse(
        success=True,
        results=results,
        total_time_ms=round(total_ms, 1),
    )


@app.post("/translate/raw", response_model=TranslateResponse)
async def translate_raw_json(data: Dict[str, Any]):
    """
    ## 在线 JSON 姿态数据翻译

    直接 POST JSON 格式的姿态数据（在线摄像头采集场景）。

    ```json
    {
        "keypoints": [[[x,y], ...], ...],   // 每帧 133 个关键点
        "scores": [[s, ...], ...]            // 每帧 133 个置信度
    }
    ```
    """
    t_start = time.perf_counter()
    _service_stats["total_requests"] += 1

    try:
        result = _engine.translate(data)
        total_ms = (time.perf_counter() - t_start) * 1000

        _service_stats.setdefault("recent_latencies", []).append(total_ms)
        if len(_service_stats["recent_latencies"]) > 100:
            _service_stats["recent_latencies"] = _service_stats["recent_latencies"][-100:]

        return TranslateResponse(
            success=True,
            text=result["text"],
            time_ms=round(total_ms, 1),
            num_tokens=result["num_tokens"],
            tokens_per_second=result["tokens_per_second"],
            layer_timing=result["layer_timing"],
        )

    except Exception as e:
        _service_stats["total_errors"] += 1
        return TranslateResponse(success=False, error=str(e))


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign Demo 服务器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    _PORT = args.port

    import uvicorn

    uvicorn.run(
        "deploy.server_demo:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
