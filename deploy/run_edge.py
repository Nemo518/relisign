#!/usr/bin/env python
"""
边缘设备一键推理脚本 — 简化版，适合嵌入式/Jetson 环境。

使用方式:
    # 默认（自动检测显存，选择最优模式）
    python deploy/run_edge.py --pose sample.pkl

    # 指定模式
    python deploy/run_edge.py --pose sample.pkl --mode edge_4gb

    # 从摄像头实时采集（需要 RTMPose）
    python deploy/run_edge.py --camera --mode edge_4gb

    # 批量处理目录
    python deploy/run_edge.py --dir poses/ --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 自动检测最优模式
# ============================================================
def auto_detect_mode() -> str:
    """根据可用显存自动选择最优模式。"""
    if not torch.cuda.is_available():
        print("[auto] 无 GPU 可用，使用 CPU 模式")
        return "edge_cpu"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[auto] 检测到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"[auto] 显存总量: {vram_gb:.1f} GB")

    if vram_gb >= 7:
        return "balanced"   # 8GB+ → INT8 + beam=4
    elif vram_gb >= 4:
        return "edge_4gb"   # 4-7GB → NF4
    elif vram_gb >= 2:
        return "edge_2gb"   # 2-4GB → NF4 极致
    else:
        return "edge_cpu"   # <2GB → CPU


# ============================================================
# 主逻辑
# ============================================================
def run_single(engine, pose_path: str, verbose: bool = True) -> dict:
    """处理单个文件。"""
    result = engine.translate_pose_file(pose_path)
    if verbose:
        print(f"  [{result.time_ms:6.0f}ms]  {result.text}")
    return {
        "file": pose_path,
        "text": result.text,
        "time_ms": result.time_ms,
        "tokens_per_second": result.tokens_per_second,
    }


def run_directory(engine, dir_path: str, glob_pattern: str = "*.pkl") -> List[dict]:
    """批量处理目录。"""
    dir_path = Path(dir_path)
    files = sorted(dir_path.rglob(glob_pattern))
    print(f"[edge] 找到 {len(files)} 个姿态文件")

    results = []
    total_time = 0.0
    for i, f in enumerate(files, 1):
        print(f"[{i:4d}/{len(files)}]", end=" ")
        r = run_single(engine, str(f), verbose=True)
        results.append(r)
        total_time += r["time_ms"]

    print(f"\n[edge] 处理完成: {len(files)} 个文件, 平均 {total_time/len(files):.0f}ms/个")
    return results


def run_camera(engine, camera_id: int = 0):
    """实时摄像头推理（需要 RTMPose / MediaPipe 提取姿态）。"""
    try:
        import cv2
    except ImportError:
        print("[edge] ⚠ OpenCV 未安装，无法使用摄像头")
        print("[edge] 请安装: pip install opencv-python")
        return

    print("[edge] 摄像头实时推理模式")
    print("[edge] 注意：此模式需要搭配姿态提取器使用")
    print("[edge] 参考项目 demo/ 目录中的 rtmlib 示例")
    print("[edge] ---")
    print("[edge] 按 'q' 退出")

    # TODO: 集成 RTMPose/MediaPipe 实时姿态提取
    # 当前为骨架代码，需配合 demo/ 中的姿态提取器

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[edge] ⚠ 无法打开摄像头")
        return

    # 此处需要集成姿态提取管线
    # 参考: demo/rtmlib-main/ 中的 RTMPose 实现
    print("[edge] 姿态提取管线需手动集成，请参考 demo/ 目录")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Uni-Sign 边缘设备推理")
    parser.add_argument("--pose", type=str, default="", help="单个姿态文件路径")
    parser.add_argument("--dir", type=str, default="", help="批量处理目录")
    parser.add_argument("--camera", action="store_true", help="摄像头实时模式")
    parser.add_argument("--mode", type=str, default="", help="推理模式 (空=自动检测)")
    parser.add_argument("--output", type=str, default="", help="结果 JSON 输出路径")
    parser.add_argument("--verbose", action="store_true", default=True, help="详细输出")
    args = parser.parse_args()

    # 自动检测模式
    mode = args.mode or auto_detect_mode()
    print(f"[edge] 推理模式: {mode}")

    # 加载引擎
    from .config_deploy import PRESETS
    from .inference import SignLanguageInference

    config = PRESETS[mode]()
    print(f"[edge] 加载模型...")
    engine = SignLanguageInference(config)
    print(f"[edge] 模型就绪 ✓\n")

    results = []

    try:
        if args.camera:
            run_camera(engine)
        elif args.dir:
            results = run_directory(engine, args.dir)
        elif args.pose:
            r = run_single(engine, args.pose, args.verbose)
            results = [r]
        else:
            print("[edge] 请指定 --pose、--dir 或 --camera")
            print("[edge] 示例: python deploy/run_edge.py --pose sample.pkl")
    finally:
        engine.unload()

    # 保存结果
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[edge] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
