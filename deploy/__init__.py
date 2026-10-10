"""
Uni-Sign 部署模块 — 覆盖两大场景，不修改原有代码。

场景 1 — Web API 网站部署:
    from deploy import SignLanguageInference, preset_fast
    engine = SignLanguageInference(preset_fast())
    result = engine.translate_pose_file("pose.pkl")

场景 2 — 边缘设备部署:
    python deploy/run_edge.py --pose sample.pkl          # 自动检测模式
    python deploy/run_edge.py --mode edge_4gb --dir poses/  # 批量处理
    python deploy/run_edge.py --camera                      # 摄像头实时

模块清单:
    config_deploy.py      — 量化/生成/部署配置 + 7 种预设
    model_loader.py       — 量化模型加载 (FP16/INT8/INT4/NF4)
    inference.py          — 优化推理管线
    server.py             — FastAPI Web 推理服务
    benchmark.py          — 性能基准测试
    edge_export.py        — ONNX/TorchScript 边缘设备导出
    run_edge.py           — 边缘设备一键推理脚本
    Dockerfile            — Docker 镜像 (Web API 部署)
    docker-compose.yml    — 一键部署编排
    nginx.conf            — Nginx 反向代理配置
"""

from .config_deploy import (
    DeployConfig,
    GenerationConfig,
    QuantizationConfig,
    preset_balanced,
    preset_fast,
    preset_quality,
    preset_edge_4gb,
    preset_edge_2gb,
    preset_edge_cpu,
    preset_onnx_export,
    PRESETS,
)
from .inference import (
    SignLanguageInference,
    InferenceResult,
    BatchInferenceResult,
    quick_translate,
)
from .model_loader import load_model
from .benchmark import run_full_benchmark

__all__ = [
    # 配置
    "DeployConfig",
    "GenerationConfig",
    "QuantizationConfig",
    "PRESETS",
    # Web API / 笔记本 预设
    "preset_balanced",
    "preset_fast",
    "preset_quality",
    # 边缘设备 预设
    "preset_edge_4gb",
    "preset_edge_2gb",
    "preset_edge_cpu",
    # ONNX 导出
    "preset_onnx_export",
    # 推理
    "SignLanguageInference",
    "InferenceResult",
    "BatchInferenceResult",
    "quick_translate",
    # 模型加载
    "load_model",
    # 基准测试
    "run_full_benchmark",
]
