"""
边缘设备模型导出 — ONNX / TorchScript / TensorRT 格式导出。

目标：将 Uni-Sign 的 Pose Encoder（非 mT5 部分）和完整推理链路
导出为 ONNX，可在 Jetson、嵌入式设备等边缘端高效运行。

使用方式:
    python -m deploy.edge_export --format onnx
    python -m deploy.edge_export --format tensorrt --fp16
    python -m deploy.edge_export --format torchscript
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config_deploy import DeployConfig, preset_edge_4gb, preset_onnx_export
from .model_loader import load_model


# ============================================================
# Pose Encoder 导出（最核心的边缘加速部分）
# ============================================================
class PoseEncoderWrapper(nn.Module):
    """
    仅导出 Pose Encoder（GCN + 投影层），不包含 mT5。
    这是模型中唯一的自定义部分，mT5 可以直接用 ONNX Runtime 官方模型。
    """

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.proj_linear = full_model.proj_linear
        self.gcn_modules = full_model.gcn_modules
        self.fusion_gcn_modules = full_model.fusion_gcn_modules
        self.part_para = full_model.part_para
        self.pose_proj = full_model.pose_proj
        self.modes = full_model.modes

    def forward(
        self,
        body: torch.Tensor,       # [B, T, N_body, 3]
        left: torch.Tensor,       # [B, T, N_left, 3]
        right: torch.Tensor,      # [B, T, N_right, 3]
        face_all: torch.Tensor,   # [B, T, N_face, 3]
    ) -> torch.Tensor:
        """返回 pose embeddings [B, T, 768] — 可直接喂给 mT5 decoder。"""
        features = []
        body_feat = None

        inputs = {
            "body": body, "left": left, "right": right, "face_all": face_all,
        }

        for part in self.modes:
            proj_feat = self.proj_linear[part](inputs[part]).permute(0, 3, 1, 2)
            gcn_feat = self.gcn_modules[part](proj_feat)

            if part == "body":
                body_feat = gcn_feat
            elif part == "left":
                gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
            elif part == "right":
                gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
            elif part == "face_all":
                gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()

            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)
            features.append(pool_feat)

        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        return self.pose_proj(inputs_embeds)  # [B, T, 768]


def export_pose_encoder_onnx(
    model: nn.Module,
    output_path: str,
    seq_len: int = 128,
    opset_version: int = 14,
) -> str:
    """
    将 Pose Encoder 导出为 ONNX 格式。

    Args:
        model: 已加载的完整 Uni_Sign 模型
        output_path: 输出 .onnx 文件路径
        seq_len: 固定序列长度
        opset_version: ONNX opset 版本

    Returns:
        输出文件路径
    """
    wrapper = PoseEncoderWrapper(model)
    wrapper.eval()
    device = next(model.parameters()).device

    # 构造 dummy 输入
    dummy = {
        "body": torch.randn(1, seq_len, 9, 3, device=device),
        "left": torch.randn(1, seq_len, 21, 3, device=device),
        "right": torch.randn(1, seq_len, 21, 3, device=device),
        "face_all": torch.randn(1, seq_len, 13, 3, device=device),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[edge_export] 导出 Pose Encoder → ONNX (seq_len={seq_len})...")

    # 动态轴（支持可变 batch 和 seq_len）
    dynamic_axes = {
        "body": {0: "batch", 1: "seq_len"},
        "left": {0: "batch", 1: "seq_len"},
        "right": {0: "batch", 1: "seq_len"},
        "face_all": {0: "batch", 1: "seq_len"},
        "output": {0: "batch", 1: "seq_len"},
    }

    torch.onnx.export(
        wrapper,
        (dummy["body"], dummy["left"], dummy["right"], dummy["face_all"]),
        str(output_path),
        input_names=["body", "left", "right", "face_all"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"[edge_export] ✓ 导出完成: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


def export_pose_encoder_torchscript(
    model: nn.Module,
    output_path: str,
    seq_len: int = 128,
) -> str:
    """
    将 Pose Encoder 导出为 TorchScript (推荐用于边缘 PyTorch 环境)。
    """
    wrapper = PoseEncoderWrapper(model)
    wrapper.eval()
    device = next(model.parameters()).device

    dummy = (
        torch.randn(1, seq_len, 9, 3, device=device),
        torch.randn(1, seq_len, 21, 3, device=device),
        torch.randn(1, seq_len, 21, 3, device=device),
        torch.randn(1, seq_len, 13, 3, device=device),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[edge_export] 导出 Pose Encoder → TorchScript (seq_len={seq_len})...")

    traced = torch.jit.trace(wrapper, dummy, strict=False)
    traced.save(str(output_path))

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"[edge_export] ✓ 导出完成: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


# ============================================================
# 导出验证
# ============================================================
def verify_export(
    original_model: nn.Module,
    exported_path: str,
    seq_len: int = 64,
    fmt: str = "onnx",
) -> Dict[str, float]:
    """验证导出模型与原始模型输出的一致性。"""
    import onnxruntime as ort

    device = next(original_model.parameters()).device

    # 生成测试输入
    test_inputs = {
        "body": torch.randn(1, seq_len, 9, 3, device=device),
        "left": torch.randn(1, seq_len, 21, 3, device=device),
        "right": torch.randn(1, seq_len, 21, 3, device=device),
        "face_all": torch.randn(1, seq_len, 13, 3, device=device),
    }

    # 原始输出
    wrapper = PoseEncoderWrapper(original_model)
    wrapper.eval()
    with torch.no_grad():
        original_out = wrapper(
            test_inputs["body"],
            test_inputs["left"],
            test_inputs["right"],
            test_inputs["face_all"],
        )

    # ONNX 输出
    if fmt == "onnx":
        session = ort.InferenceSession(exported_path, providers=["CPUExecutionProvider"])
        onnx_inputs = {k: v.cpu().numpy() for k, v in test_inputs.items()}
        onnx_out = session.run(None, onnx_inputs)[0]
        diff = np.abs(original_out.cpu().numpy() - onnx_out)
    elif fmt == "torchscript":
        traced = torch.jit.load(exported_path)
        with torch.no_grad():
            ts_out = traced(
                test_inputs["body"],
                test_inputs["left"],
                test_inputs["right"],
                test_inputs["face_all"],
            )
        diff = (original_out - ts_out).abs()

    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"[edge_export] 验证结果 — max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    threshold = 1e-4
    if max_diff < threshold:
        print(f"[edge_export] ✓ 导出验证通过 (max_diff < {threshold})")
    else:
        print(f"[edge_export] ⚠ 差异较大，请检查 (max_diff={max_diff:.6f} > {threshold})")

    return {"max_diff": max_diff, "mean_diff": mean_diff}


# ============================================================
# 一键导出
# ============================================================
def export_all(
    output_dir: str = "",
    formats: Tuple[str, ...] = ("onnx", "torchscript"),
    seq_len: int = 128,
):
    """
    一键导出 Pose Encoder 到所有格式。

    Args:
        output_dir: 输出目录，默认 deploy/exported/
        formats: 导出格式 ("onnx", "torchscript")
        seq_len: 序列长度
    """
    if not output_dir:
        output_dir = str(PROJECT_ROOT / "deploy" / "exported")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Uni-Sign 边缘设备模型导出")
    print(f"  输出目录: {out}")
    print(f"  序列长度: {seq_len}")
    print(f"  格式: {formats}")
    print(f"{'='*60}\n")

    # 加载模型
    config = preset_onnx_export()
    config.max_length = seq_len
    model = load_model(config)
    model.eval()

    results = {}

    if "onnx" in formats:
        path = export_pose_encoder_onnx(model, str(out / "pose_encoder.onnx"), seq_len)
        try:
            results["onnx_verify"] = verify_export(model, path, min(seq_len, 64), "onnx")
        except ImportError:
            print("[edge_export] ⚠ onnxruntime 未安装，跳过验证")

    if "torchscript" in formats:
        path = export_pose_encoder_torchscript(
            model, str(out / "pose_encoder.pt"), seq_len
        )
        try:
            results["torchscript_verify"] = verify_export(
                model, path, min(seq_len, 64), "torchscript"
            )
        except Exception as e:
            print(f"[edge_export] ⚠ TorchScript 验证失败: {e}")

    print(f"\n[edge_export] 全部导出完成，文件在: {out}")
    return results


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign 边缘设备模型导出")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "all"],
        help="导出格式",
    )
    parser.add_argument("--output-dir", type=str, default="", help="输出目录")
    parser.add_argument("--seq-len", type=int, default=128, help="序列长度")
    parser.add_argument("--verify", action="store_true", help="导出后验证")
    args = parser.parse_args()

    formats_map = {
        "onnx": ("onnx",),
        "torchscript": ("torchscript",),
        "all": ("onnx", "torchscript"),
    }

    export_all(
        output_dir=args.output_dir,
        formats=formats_map[args.format],
        seq_len=args.seq_len,
    )
