"""
部署配置文件 — 独立于项目原有 config.py，高内聚低耦合。
所有路径通过 paths.py 统一解析，不硬编码。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# 确保项目根目录在 sys.path 中（支持从任意位置运行）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    MT5_ROOT,
    OUTPUT_ROOT,
    PRETRAINED_ROOT,
    PROJECT_ROOT as _PROJECT_ROOT,
    resolve_storage_path,
)


# ============================================================
# 量化方案枚举
# ============================================================
@dataclass
class QuantizationConfig:
    """
    量化配置。

    mode:
        - "fp32"   — 原始精度，约 2.3GB（mT5-base），零精度损失
        - "fp16"   — 半精度，约 1.2GB，几乎零损失（推荐用于 >=8GB 显存）
        - "int8"   — 8-bit 量化，约 1.2GB，损失 < 0.5%（推荐）
        - "int4"   — 4-bit 量化，约 0.6GB，损失 1-2%（显存紧张时推荐）
        - "nf4"    — 4-bit NF4 量化（比 int4 精度更好），约 0.6GB
    """

    mode: Literal["fp32", "fp16", "int8", "int4", "nf4"] = "int8"

    # INT8 专用
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = True

    # INT4 / NF4 专用
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    @property
    def expected_memory_gb(self) -> float:
        """估算显存占用（含 pose encoder + EfficientNet-B0 + mT5）。"""
        base = {"fp32": 3.5, "fp16": 2.2, "int8": 1.8, "int4": 1.2, "nf4": 1.2}
        return base.get(self.mode, 2.2)

    @property
    def precision_loss(self) -> str:
        return {
            "fp32": "0%",
            "fp16": "<0.1%",
            "int8": "<0.5%",
            "int4": "1-3%",
            "nf4": "1-2%",
        }.get(self.mode, "未知")


# ============================================================
# 生成配置
# ============================================================
@dataclass
class GenerationConfig:
    """推理生成参数。"""

    max_new_tokens: int = 100
    num_beams: int = 4
    # 快速模式的 beams（num_beams=1 约 3-4x 加速）
    fast_mode: bool = False

    # 解码策略
    temperature: float = 1.0
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0

    def get_effective_beams(self) -> int:
        return 1 if self.fast_mode else self.num_beams


# ============================================================
# 部署配置
# ============================================================
@dataclass
class DeployConfig:
    """部署总配置。"""

    # ---- 设备 ----
    device: str = "cuda"  # "cuda" | "cpu"
    gpu_id: int = 0

    # ---- 量化 ----
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # ---- 生成 ----
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # ---- 模型路径 ----
    checkpoint_path: str = ""  # 空 = 使用默认路径
    mt5_path: str = ""         # 空 = 使用 paths.py 中的 MT5_ROOT

    # ---- RGB Support ----
    rgb_support: bool = False          # 是否开启手部 RGB 分支
    rgb_backbone_weights: str = ""     # EfficientNet-B0 权重路径

    # ---- 模型参数 ----
    hidden_dim: int = 256
    max_length: int = 256
    dataset: str = "CE_CSL"
    label_smoothing: float = 0.2

    # ---- 运行时优化 ----
    compile_pose_encoder: bool = True   # torch.compile 加速 pose encoder
    use_autocast: bool = True           # bf16/fp16 自动混合精度
    warmup_runs: int = 3                # 推理预热次数

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8080
    max_queue_size: int = 10

    def __post_init__(self):
        if not self.checkpoint_path:
            self.checkpoint_path = str(
                OUTPUT_ROOT / "cecsl_rgb_support_local" / "best_checkpoint.pth"
            )
        if not self.mt5_path:
            self.mt5_path = str(MT5_ROOT)
        if not self.rgb_backbone_weights:
            self.rgb_backbone_weights = str(
                PRETRAINED_ROOT / "efficientnet_b0_rwightman-7f5810bc.pth"
            )


# ============================================================
# 预设配置（开箱即用）
# ============================================================

# ---- Web API / 笔记本 GPU (RTX 4060/5060 8GB) ----
def preset_balanced() -> DeployConfig:
    """均衡模式：INT8 量化 + beam=4，适合 RTX 4060/5060 8GB。"""
    return DeployConfig(
        quantization=QuantizationConfig(mode="int8"),
        generation=GenerationConfig(num_beams=4, fast_mode=False),
    )


def preset_fast() -> DeployConfig:
    """快速模式：INT8 + greedy decode，适合高并发 Web API。"""
    return DeployConfig(
        quantization=QuantizationConfig(mode="int8"),
        generation=GenerationConfig(num_beams=1, fast_mode=True),
    )


def preset_quality() -> DeployConfig:
    """质量优先：FP16 + beam=4，适合 >=12GB 显存服务器。"""
    return DeployConfig(
        quantization=QuantizationConfig(mode="fp16"),
        generation=GenerationConfig(num_beams=4, fast_mode=False),
    )


# ---- 边缘设备 (Jetson Orin / 嵌入式 GPU / 低显存) ----
def preset_edge_4gb() -> DeployConfig:
    """边缘设备 4GB 显存：NF4 量化 + greedy + 短序列。"""
    return DeployConfig(
        quantization=QuantizationConfig(mode="nf4"),
        generation=GenerationConfig(
            num_beams=1, fast_mode=True, max_new_tokens=60
        ),
        max_length=128,
        compile_pose_encoder=True,
    )


def preset_edge_2gb() -> DeployConfig:
    """边缘设备 2GB 显存：NF4 量化 + 极致压缩 + CPU 混合。"""
    return DeployConfig(
        quantization=QuantizationConfig(
            mode="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        generation=GenerationConfig(
            num_beams=1, fast_mode=True, max_new_tokens=40
        ),
        max_length=96,
        compile_pose_encoder=True,
    )


def preset_edge_cpu() -> DeployConfig:
    """边缘设备纯 CPU 推理（无 GPU）：使用更小的 mT5-small 或 ONNX。"""
    return DeployConfig(
        device="cpu",
        quantization=QuantizationConfig(mode="fp32"),  # CPU 不支持 bnb 量化
        generation=GenerationConfig(
            num_beams=1, fast_mode=True, max_new_tokens=40
        ),
        max_length=64,
        compile_pose_encoder=False,
        use_autocast=False,
    )


# ---- ONNX / TensorRT 导出专用 ----
def preset_onnx_export() -> DeployConfig:
    """导出 ONNX 时使用：FP32 + 固定序列长度。"""
    return DeployConfig(
        quantization=QuantizationConfig(mode="fp32"),
        generation=GenerationConfig(num_beams=1, fast_mode=True),
        max_length=128,
        compile_pose_encoder=False,
        use_autocast=False,
    )


PRESETS = {
    # Web API / 笔记本
    "balanced": preset_balanced,
    "fast": preset_fast,
    "quality": preset_quality,
    # 边缘设备
    "edge_4gb": preset_edge_4gb,
    "edge_2gb": preset_edge_2gb,
    "edge_cpu": preset_edge_cpu,
    # 导出
    "onnx_export": preset_onnx_export,
}
