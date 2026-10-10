"""
部署性能基准测试 — 对比不同量化模式下的模型大小、推理速度、显存占用。

使用方式:
    python -m deploy.benchmark                  # 对比所有模式
    python -m deploy.benchmark --mode int8      # 测试单一模式
    python -m deploy.benchmark --pose samples/  # 指定姿态文件目录
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 显存测量
# ============================================================
def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """获取 GPU 显存信息（单位 MB）。"""
    if not torch.cuda.is_available():
        return {"total_mb": 0, "allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}

    props = torch.cuda.get_device_properties(device_id)
    total = props.total_memory / (1024**2)
    allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**2)
    return {
        "total_mb": total,
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "free_mb": total - reserved,
    }


def peak_gpu_memory(device_id: int = 0) -> float:
    """峰值显存（MB）。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device_id) / (1024**2)


def reset_peak_memory(device_id: int = 0):
    """重置峰值显存统计。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)


# ============================================================
# 模型大小统计
# ============================================================
def measure_model_size_on_disk(checkpoint_path: str) -> Dict[str, float]:
    """测量 checkpoint 文件大小。"""
    path = Path(checkpoint_path)
    if not path.exists():
        return {"file_size_mb": 0, "exists": False}

    size_mb = path.stat().st_size / (1024**2)
    return {"file_size_mb": size_mb, "exists": True}


def measure_model_params(model: torch.nn.Module) -> Dict[str, Any]:
    """统计模型参数。（模型已在 GPU 上，所以通过 numel 而非 state_dict 文件）"""
    total = sum(p.numel() for p in model.parameters())
    dtype_counts = {}
    for p in model.parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + p.numel()

    # 估算大小
    bytes_per_param = {
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
    }
    estimated_mb = sum(
        count * bytes_per_param.get(dt, 2) for dt, count in dtype_counts.items()
    ) / (1024**2)

    return {
        "total_params_m": total / 1e6,
        "estimated_size_mb": estimated_mb,
        "dtype_distribution": dtype_counts,
    }


# ============================================================
# 推理速度基准
# ============================================================
def benchmark_inference_speed(
    engine,
    src_inputs: List[Dict[str, torch.Tensor]],
    num_runs: int = 20,
    warmup_runs: int = 5,
) -> Dict[str, float]:
    """
    测量推理速度。

    Returns:
        {
            "avg_time_ms": ...,
            "p50_time_ms": ...,
            "p95_time_ms": ...,
            "avg_tokens_per_sec": ...,
            "total_runs": ...,
        }
    """
    times_ms = []
    token_counts = []

    # 预热
    for i in range(warmup_runs):
        src = src_inputs[i % len(src_inputs)]
        _ = engine._model_forward(src)

    if engine.config.device == "cuda":
        torch.cuda.synchronize()

    # 正式测量
    for i in range(num_runs):
        src = src_inputs[i % len(src_inputs)]

        t0 = time.perf_counter()
        output_ids = engine._model_forward(src)
        if engine.config.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        times_ms.append(elapsed_ms)
        token_counts.append(output_ids.shape[1])

    times = np.array(times_ms)
    avg_tps = np.mean([c / (t / 1000) for c, t in zip(token_counts, times_ms)])

    return {
        "avg_time_ms": float(np.mean(times)),
        "p50_time_ms": float(np.percentile(times, 50)),
        "p95_time_ms": float(np.percentile(times, 95)),
        "min_time_ms": float(np.min(times)),
        "max_time_ms": float(np.max(times)),
        "avg_tokens_per_sec": float(avg_tps),
        "avg_tokens_generated": float(np.mean(token_counts)),
        "total_runs": num_runs,
    }


# ============================================================
# 综合基准
# ============================================================
def run_full_benchmark(
    mode: Optional[str] = None,
    pose_file: Optional[str] = None,
    num_speed_runs: int = 20,
) -> List[Dict[str, Any]]:
    """
    运行完整基准测试。

    Args:
        mode: 量化模式，None = 测试所有模式
        pose_file: 单个姿态文件路径
        num_speed_runs: 速度测试迭代次数

    Returns:
        每个模式的基准结果列表
    """
    from .config_deploy import DeployConfig, PRESETS, QuantizationConfig, GenerationConfig
    from .inference import SignLanguageInference

    modes = [mode] if mode else ["balanced", "fast", "minimal", "quality"]
    results = []

    # 准备测试数据
    if pose_file and Path(pose_file).exists():
        import pickle

        with open(pose_file, "rb") as f:
            test_pose = pickle.load(f)
    else:
        # 生成假数据
        print("[benchmark] 未提供真实姿态文件，使用随机数据测试速度")
        T = 128
        test_pose = {
            "keypoints": [np.random.randn(1, 133, 2) for _ in range(T)],
            "scores": [np.random.rand(1, 133) for _ in range(T)],
        }

    print(f"\n{'='*70}")
    print(f"  Uni-Sign 部署基准测试")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM 总量: {get_gpu_memory_info()['total_mb']:.0f} MB")
    print(f"{'='*70}\n")

    for mode_name in modes:
        print(f"{'─'*70}")
        print(f"  测试模式: {mode_name}")
        print(f"{'─'*70}")

        # 清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reset_peak_memory()

        mem_before = get_gpu_memory_info()

        # 加载模型
        t0 = time.perf_counter()
        config = PRESETS[mode_name]()
        config.warmup_runs = 3
        engine = SignLanguageInference(config)
        load_time = time.perf_counter() - t0

        mem_after_load = get_gpu_memory_info()
        peak_mem = peak_gpu_memory()

        # 参数量
        param_info = measure_model_params(engine.model)

        # 准备 src_input
        src_input = engine._preprocess_pose(test_pose, rgb_video_path=None)

        # 速度测试
        speed = benchmark_inference_speed(
            engine, [src_input], num_runs=num_speed_runs, warmup_runs=3
        )

        result = {
            "mode": mode_name,
            "quantization": config.quantization.mode,
            "num_beams": config.generation.num_beams,
            "load_time_sec": round(load_time, 2),
            "params_millions": round(param_info["total_params_m"], 1),
            "estimated_model_mb": round(param_info["estimated_size_mb"], 1),
            "vram_delta_mb": round(mem_after_load["allocated_mb"] - mem_before["allocated_mb"], 1),
            "peak_vram_mb": round(peak_mem, 1),
            "vram_remaining_mb": round(mem_after_load["free_mb"], 1),
            "avg_inference_ms": round(speed["avg_time_ms"], 1),
            "p95_inference_ms": round(speed["p95_time_ms"], 1),
            "tokens_per_second": round(speed["avg_tokens_per_sec"], 1),
        }
        results.append(result)

        # 打印
        print(f"    加载耗时:       {load_time:.1f}s")
        print(f"    参数量:         {param_info['total_params_m']:.1f}M")
        print(f"    显存增量:       {result['vram_delta_mb']:.0f} MB")
        print(f"    峰值显存:       {peak_mem:.0f} MB")
        print(f"    剩余显存:       {mem_after_load['free_mb']:.0f} MB")
        print(f"    平均推理速度:   {speed['avg_time_ms']:.0f} ms")
        print(f"    P95 推理速度:   {speed['p95_time_ms']:.0f} ms")
        print(f"    生成速度:       {speed['avg_tokens_per_sec']:.1f} tok/s")

        # 卸载模型
        engine.unload()

    # 汇总对比表
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  对比汇总")
        print(f"{'='*70}")
        header = (
            f"{'模式':<10s} {'量化':<6s} {'参数':>8s} {'显存':>8s} "
            f"{'平均延迟':>10s} {'速度':>12s}"
        )
        print(header)
        print("-" * 70)
        for r in results:
            print(
                f"{r['mode']:<10s} "
                f"{r['quantization']:<6s} "
                f"{r['params_millions']:>6.1f}M "
                f"{r['peak_vram_mb']:>6.0f}MB "
                f"{r['avg_inference_ms']:>8.0f}ms "
                f"{r['tokens_per_second']:>10.1f} tok/s"
            )
        print(f"{'='*70}\n")

    return results


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign 部署基准测试")
    parser.add_argument("--mode", type=str, default=None, help="量化模式 (balanced/fast/minimal/quality)")
    parser.add_argument("--pose", type=str, default=None, help="姿态文件路径 (.pkl)")
    parser.add_argument("--runs", type=int, default=20, help="速度测试迭代次数")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    results = run_full_benchmark(
        mode=args.mode,
        pose_file=args.pose,
        num_speed_runs=args.runs,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已保存: {args.output}")
