"""
量化模型加载器 — 在不修改 models.py 的前提下，通过包装器注入量化支持。

核心策略：
  1. 通过 monkey-patch MT5ForConditionalGeneration.from_pretrained 注入量化配置
  2. 其余模块（Pose Encoder / EfficientNet / Deformable Attention）保持原始精度
  3. 支持 FP32 / FP16 / INT8 / INT4(NF4) 四种模式

高内聚低耦合：只 import 原项目模块，不修改原始文件。
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config_deploy import DeployConfig, QuantizationConfig
from paths import resolve_storage_path


def _patch_mt5_for_quantization(quant: QuantizationConfig):
    """
    在加载 mT5 之前，通过环境变量 + transformers 内部 API 注入量化行为。
    不修改 transformers 源码。
    """
    if quant.mode in ("fp32", "fp16"):
        return  # 不需要量化

    # 检查 bitsandbytes 是否可用
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError:
        raise ImportError(
            "量化模式需要 bitsandbytes。请安装: pip install bitsandbytes>=0.41.0"
        )

    # 设置环境变量以允许量化内核
    import os

    os.environ.setdefault("BNB_CUDA_VERSION", "118")


def _build_model_args(config: DeployConfig) -> SimpleNamespace:
    """从 DeployConfig 构造 Uni_Sign 所需的 args（兼容原有接口）。"""
    return SimpleNamespace(
        hidden_dim=config.hidden_dim,
        max_length=config.max_length,
        dataset=config.dataset,
        label_smoothing=config.label_smoothing,
        rgb_support=config.rgb_support,
        rgb_backbone_weights=config.rgb_backbone_weights,
        rgb_backbone_pretrained=False,
        fusion_use_uncertainty_gate=False,
        fusion_meta_dim=3,
    )


def load_model(config: DeployConfig) -> nn.Module:
    """
    加载 Uni-Sign 模型并应用量化。

    Args:
        config: DeployConfig 部署配置

    Returns:
        已加载并量化好的 Uni_Sign 模型（eval 模式）

    使用方式:
        >>> config = preset_balanced()
        >>> model = load_model(config)
        >>> result = model.generate(...)
    """
    from models import Uni_Sign

    quant = config.quantization
    print(f"[model_loader] 量化模式: {quant.mode}")
    print(f"[model_loader] 预计显存: {quant.expected_memory_gb} GB")
    print(f"[model_loader] 精度损失: {quant.precision_loss}")

    # ---- Step 1: 构造模型参数 ----
    model_args = _build_model_args(config)

    # ---- Step 2: 注入量化（如需要） ----
    _patch_mt5_for_quantization(quant)

    # ---- Step 3: 根据量化模式，替换 mT5 的加载方式 ----
    if quant.mode in ("int8", "int4", "nf4"):
        model = _load_with_bnb_quantization(config, model_args, quant)
    else:
        # FP32 / FP16：使用原始加载方式，但包装 device_map
        model = _load_standard(config, model_args)

    # ---- Step 4: 冻结参数 & 切换到 eval 模式 ----
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # ---- Step 5: torch.compile 加速 pose encoder（可选） ----
    if config.compile_pose_encoder:
        _compile_pose_modules(model)

    # ---- Step 6: 预热 ----
    if config.warmup_runs > 0:
        _warmup_model(model, config)

    # ---- 统计 ----
    _print_model_stats(model, config)

    return model


def _load_standard(config: DeployConfig, model_args: SimpleNamespace) -> nn.Module:
    """标准加载（FP32 或 FP16）。"""
    from models import Uni_Sign

    model = Uni_Sign(args=model_args)

    # 加载 checkpoint
    checkpoint_path = resolve_storage_path(config.checkpoint_path)
    if Path(checkpoint_path).exists():
        print(f"[model_loader] 加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        ret = model.load_state_dict(state_dict, strict=False)
        if ret.missing_keys:
            print(f"[model_loader] ⚠ 缺失键 ({len(ret.missing_keys)}):")
            for k in ret.missing_keys[:5]:
                print(f"  - {k}")
            if len(ret.missing_keys) > 5:
                print(f"  ... 共 {len(ret.missing_keys)} 个")
    else:
        print(f"[model_loader] ⚠ checkpoint 不存在: {checkpoint_path}")
        print("[model_loader] 使用随机初始化权重（仅供测试）")

    # 移动到设备
    device = torch.device(config.device)
    if config.quantization.mode == "fp16" and device.type == "cuda":
        model = model.half()
    model = model.to(device)

    return model


def _load_with_bnb_quantization(
    config: DeployConfig, model_args: SimpleNamespace, quant: QuantizationConfig
) -> nn.Module:
    """
    使用 bitsandbytes 量化加载。
    策略：先用原始方式创建模型结构，然后替换 mT5 为量化版本。
    """
    from models import Uni_Sign
    from transformers import (
        BitsAndBytesConfig,
        MT5ForConditionalGeneration,
        T5Tokenizer,
    )

    # 先创建模型框架（此时 mT5 会被原始方式加载）
    # 我们需要拦截 Uni_Sign.__init__ 中 mT5 的加载
    # 策略：临时 monkey-patch from_pretrained
    import transformers
    from config import mt5_path as _original_mt5_path

    mt5_path = config.mt5_path or _original_mt5_path
    mt5_path = resolve_storage_path(mt5_path)

    # 构建 BitsAndBytes 配置
    if quant.mode == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quant.llm_int8_threshold,
            llm_int8_has_fp16_weight=quant.llm_int8_has_fp16_weight,
        )
    elif quant.mode in ("int4", "nf4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, quant.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        )
    else:
        raise ValueError(f"不支持的量化模式: {quant.mode}")

    print(f"[model_loader] BitsAndBytes 配置: {bnb_config}")

    # 保存原始方法
    _original_from_pretrained = MT5ForConditionalGeneration.from_pretrained

    def _quantized_from_pretrained(cls_or_self, *args, **kwargs):
        """注入量化配置的 from_pretrained。"""
        # 去掉 device_map，让 bnb 自己管理
        kwargs.pop("device_map", None)
        kwargs["quantization_config"] = bnb_config
        # 对于 4-bit，需要特殊处理
        if quant.mode in ("int4", "nf4"):
            kwargs.setdefault("torch_dtype", getattr(torch, quant.bnb_4bit_compute_dtype))
        return _original_from_pretrained(*args, **kwargs)

    # Monkey-patch
    MT5ForConditionalGeneration.from_pretrained = classmethod(_quantized_from_pretrained)

    try:
        model = Uni_Sign(args=model_args)
    finally:
        # 恢复原始方法（立刻恢复，避免影响其他代码）
        MT5ForConditionalGeneration.from_pretrained = _original_from_pretrained

    # 加载 checkpoint（pose encoder 部分的权重）
    checkpoint_path = resolve_storage_path(config.checkpoint_path)
    if Path(checkpoint_path).exists():
        print(f"[model_loader] 加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # 只加载非 mT5 部分的权重（mT5 已通过 bnb 量化加载）
        non_mt5_state = {
            k: v for k, v in state_dict.items() if not k.startswith("mt5_model.")
        }
        ret = model.load_state_dict(non_mt5_state, strict=False)
        if ret.missing_keys:
            mt5_missing = [k for k in ret.missing_keys if k.startswith("mt5_model.")]
            other_missing = [k for k in ret.missing_keys if not k.startswith("mt5_model.")]
            if mt5_missing:
                print(
                    f"[model_loader] ℹ mT5 量化加载，跳过 {len(mt5_missing)} 个 mT5 键（由 bnb 管理）"
                )
            if other_missing:
                print(f"[model_loader] ⚠ 其他缺失键 ({len(other_missing)}):")
                for k in other_missing[:5]:
                    print(f"  - {k}")
    else:
        print(f"[model_loader] ⚠ checkpoint 不存在: {checkpoint_path}")

    # bnb 量化模型已在 GPU 上，不需要再 .to(device)
    return model


def _compile_pose_modules(model: nn.Module):
    """使用 torch.compile 加速 pose encoder（PyTorch 2.0+）。"""
    if not hasattr(torch, "compile"):
        print("[model_loader] torch.compile 不可用（需要 PyTorch >= 2.0），跳过")
        return

    compile_targets = [
        "proj_linear",
        "gcn_modules",
        "fusion_gcn_modules",
        "pose_proj",
    ]
    compiled_count = 0
    for name, module in model.named_children():
        if name in compile_targets and name != "mt5_model":
            try:
                setattr(model, name, torch.compile(module, mode="reduce-overhead"))
                compiled_count += 1
            except Exception as e:
                print(f"[model_loader] torch.compile 跳过 {name}: {e}")

    if compiled_count:
        print(f"[model_loader] torch.compile 已加速 {compiled_count} 个模块")


def _warmup_model(model: nn.Module, config: DeployConfig):
    """模型预热，减少首次推理延迟。"""
    import numpy as np

    device = next(model.parameters()).device
    T = 64  # 预热序列长度

    print(f"[model_loader] 预热 {config.warmup_runs} 次...")

    # 构造假输入
    dummy_src = {
        "body": torch.randn(1, T, 9, 3, device=device),
        "left": torch.randn(1, T, 21, 3, device=device),
        "right": torch.randn(1, T, 21, 3, device=device),
        "face_all": torch.randn(1, T, 13, 3, device=device),
        "attention_mask": torch.ones(1, T, device=device),
        "name_batch": ["warmup"],
        "src_length_batch": torch.tensor([T], device=device),
    }

    if config.rgb_support:
        dummy_src.update(
            {
                "left_sampled_indices": torch.tensor([0], device=device),
                "left_hands": torch.randn(1, 3, 112, 112, device=device),
                "left_skeletons_norm": torch.randn(1, 21, 2, device=device),
                "left_rgb_meta": torch.randn(1, 3, device=device),
                "left_rgb_len": [1],
                "right_sampled_indices": torch.tensor([0], device=device),
                "right_hands": torch.randn(1, 3, 112, 112, device=device),
                "right_skeletons_norm": torch.randn(1, 21, 2, device=device),
                "right_rgb_meta": torch.randn(1, 3, device=device),
                "right_rgb_len": [1],
            }
        )

    dummy_tgt = {"gt_sentence": ["热身"], "gt_gloss": [""]}

    with torch.no_grad():
        for i in range(config.warmup_runs):
            _ = model(src_input=dummy_src, tgt_input=dummy_tgt)
            if i == 0:
                torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[model_loader] 预热完成 ✓")


def _print_model_stats(model: nn.Module, config: DeployConfig):
    """打印模型统计信息。"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 分模块统计
    module_params = {}
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        module_params[prefix] = module_params.get(prefix, 0) + param.numel()

    print(f"\n{'='*60}")
    print(f"  模型加载完成")
    print(f"{'='*60}")
    print(f"  总参数量:     {total_params/1e6:.1f}M")
    print(f"  可训练参数:   {trainable_params/1e6:.1f}M")
    print(f"  量化模式:     {config.quantization.mode}")
    print(f"  预计显存:     {config.quantization.expected_memory_gb:.1f} GB")
    print(f"  精度损失:     {config.quantization.precision_loss}")
    print(f"{'='*60}")

    # 分模块详情
    print(f"\n  模块分布:")
    for module_name, count in sorted(module_params.items(), key=lambda x: -x[1]):
        bar_len = int(count / total_params * 40)
        bar = "█" * bar_len
        print(f"    {module_name:<30s} {count/1e6:>8.1f}M  {bar}")
    print(f"{'='*60}\n")
