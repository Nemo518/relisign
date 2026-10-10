"""
优化推理管线 — 端到端手语翻译推理。

支持三种输入模式：
  1. pose_file  — 从 .pkl 文件读取姿态数据
  2. pose_dict  — 直接传入字典格式的姿态数据（在线采集场景）
  3. rgb_video  — 同时使用 RGB 视频 + 姿态（需 rgb_support=True）

高内聚低耦合：只依赖 deploy/ 内模块 + 项目原有 datasets.py 中的
数据处理函数（load_part_kp 等），不修改原始文件。
"""
from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config_deploy import DeployConfig, GenerationConfig
from .model_loader import load_model


# ============================================================
# 推理结果
# ============================================================
@dataclass
class InferenceResult:
    """单次推理结果。"""

    text: str                           # 预测文本
    tokens: Optional[torch.Tensor] = None  # 原始 token IDs
    time_ms: float = 0.0                # 推理耗时（毫秒）
    num_tokens: int = 0                 # 生成 token 数
    tokens_per_second: float = 0.0      # 生成速度

    def __repr__(self):
        return (
            f"InferenceResult(text={self.text!r}, "
            f"time={self.time_ms:.0f}ms, "
            f"speed={self.tokens_per_second:.1f} tok/s)"
        )


@dataclass
class BatchInferenceResult:
    """批量推理结果。"""

    results: List[InferenceResult] = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.time_ms for r in self.results) / len(self.results)

    @property
    def avg_tokens_per_second(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.tokens_per_second for r in self.results) / len(self.results)

    def __repr__(self):
        return (
            f"BatchInferenceResult(n={len(self.results)}, "
            f"avg_time={self.avg_time_ms:.0f}ms, "
            f"avg_speed={self.avg_tokens_per_second:.1f} tok/s)"
        )


# ============================================================
# 推理引擎
# ============================================================
class SignLanguageInference:
    """
    手语翻译推理引擎。

    使用方式:
        >>> config = preset_balanced()
        >>> engine = SignLanguageInference(config)
        >>> result = engine.translate_pose_file("/path/to/pose.pkl")
        >>> print(result.text)
    """

    def __init__(self, config: Optional[DeployConfig] = None):
        if config is None:
            from config_deploy import preset_balanced

            config = preset_balanced()

        self.config = config
        self.device = torch.device(config.device)
        self._model = None
        self._compiled = False

    # ---- 属性 ----
    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_model(self.config)
        return self._model

    @property
    def tokenizer(self):
        return self.model.mt5_tokenizer

    # ---- 核心推理方法 ----
    def translate_pose_file(self, pose_path: str) -> InferenceResult:
        """从 .pkl 文件加载姿态并翻译。"""
        with open(pose_path, "rb") as f:
            pose_data = pickle.load(f)
        return self.translate_pose_dict(pose_data)

    def translate_pose_dict(
        self,
        pose_data: Dict[str, Any],
        rgb_video_path: Optional[str] = None,
    ) -> InferenceResult:
        """
        从姿态字典翻译（支持在线采集数据）。

        Args:
            pose_data: 姿态字典，需包含:
                - 'keypoints': list of ndarray, 每帧关键点 (1, 133, 2) 或 (N, 133, 2)
                - 'scores': list of ndarray, 每帧置信度 (1, 133) 或 (N, 133)
            rgb_video_path: 可选，RGB 视频路径（开启 rgb_support 时使用）

        Returns:
            InferenceResult
        """
        t0 = time.perf_counter()

        # Step 1: 数据预处理（采样 + 子姿态提取）
        src_input = self._preprocess_pose(pose_data, rgb_video_path)

        # Step 2: 模型前向
        output_ids = self._model_forward(src_input)

        # Step 3: 解码
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        num_tokens = output_ids.shape[1]
        tps = num_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        return InferenceResult(
            text=text,
            tokens=output_ids,
            time_ms=elapsed_ms,
            num_tokens=num_tokens,
            tokens_per_second=tps,
        )

    def translate_batch_pose_files(
        self,
        pose_paths: List[str],
    ) -> BatchInferenceResult:
        """批量翻译多个姿态文件。"""
        results = []
        for path in pose_paths:
            result = self.translate_pose_file(path)
            results.append(result)
        return BatchInferenceResult(results=results)

    # ---- 内部方法 ----
    def _preprocess_pose(
        self,
        pose_data: Dict[str, Any],
        rgb_video_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """将原始姿态数据转换为模型输入格式。"""
        from datasets import load_part_kp
        from datasets import load_support_rgb_dict as _load_rgb
        from datasets import build_uniform_indices

        max_length = self.config.max_length

        # ---- 处理多人检测（取置信度最高的人） ----
        skeletons = []
        confs = []
        for kp, sc in zip(pose_data["keypoints"], pose_data["scores"]):
            kp_arr = np.asarray(kp)
            sc_arr = np.asarray(sc)
            if sc_arr.ndim == 2 and sc_arr.shape[0] > 1:
                best_idx = int(np.argmax(sc_arr.mean(axis=1)))
                sc_arr = sc_arr[best_idx : best_idx + 1]
                kp_arr = kp_arr[best_idx : best_idx + 1]
            elif sc_arr.ndim == 1:
                sc_arr = sc_arr[None]
                kp_arr = kp_arr[None]
            skeletons.append(kp_arr)
            confs.append(sc_arr)

        # ---- 帧采样 ----
        duration = len(skeletons)
        if duration > max_length:
            indices = build_uniform_indices(duration, max_length)
        else:
            indices = np.arange(duration)

        skeletons_sampled = [skeletons[i] for i in indices]
        confs_sampled = [confs[i] for i in indices]

        # ---- 提取子姿态 ----
        kps_with_scores = load_part_kp(skeletons_sampled, confs_sampled, force_ok=True)

        # ---- 构造 src_input ----
        src_input = {}
        for part in ["body", "left", "right", "face_all"]:
            tensor = kps_with_scores[part]  # [T, N, 3]
            src_input[part] = tensor.unsqueeze(0)  # [1, T, N, 3]

        # attention mask
        T = src_input["body"].shape[1]
        mask_gen = torch.ones(1, T) + 7
        src_input["attention_mask"] = (mask_gen != 0).long()
        src_input["name_batch"] = ["online"]
        src_input["src_length_batch"] = torch.tensor([T])

        # ---- RGB Support ----
        if self.config.rgb_support and rgb_video_path:
            from torchvision import transforms

            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

            # 需要把 index 转换回原始的
            tmp_indices = np.array(indices)
            rgb_dict = _load_rgb(
                tmp_indices,
                skeletons_sampled,
                confs_sampled,
                rgb_video_path,
                data_transform,
                sample_mode="confidence_topk",
            )

            for key in [
                "left_sampled_indices",
                "left_hands",
                "left_skeletons_norm",
                "left_rgb_meta",
                "right_sampled_indices",
                "right_hands",
                "right_skeletons_norm",
                "right_rgb_meta",
            ]:
                src_input[key] = rgb_dict[key]

            src_input["left_rgb_len"] = [len(rgb_dict["left_sampled_indices"])]
            src_input["right_rgb_len"] = [len(rgb_dict["right_sampled_indices"])]
        elif self.config.rgb_support:
            # 无 RGB 视频时填充空数据（pose-only 回退）
            from datasets import _fill_empty_hand_dict

            empty = {}
            _fill_empty_hand_dict(empty)
            for key, value in empty.items():
                src_input[key] = value
            src_input["left_rgb_len"] = [1]
            src_input["right_rgb_len"] = [1]

        return src_input

    def _model_forward(self, src_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行模型前向 + 生成。"""
        gen_cfg = self.config.generation
        use_autocast = (
            self.config.use_autocast
            and self.device.type == "cuda"
        )

        # 移动到设备
        for key, value in src_input.items():
            if isinstance(value, torch.Tensor):
                src_input[key] = value.to(self.device)

        # 空 tgt_input（仅用于触发 encoder）
        tgt_input = {
            "gt_sentence": [""],
            "gt_gloss": [""],
        }

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stack_out = self.model(src_input, tgt_input)
            else:
                stack_out = self.model(src_input, tgt_input)

            output_ids = self.model.generate(
                stack_out,
                max_new_tokens=gen_cfg.max_new_tokens,
                num_beams=gen_cfg.get_effective_beams(),
            )

        return output_ids

    # ---- 资源管理 ----
    def clear_cache(self):
        """清空 GPU 缓存。"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def unload(self):
        """卸载模型释放显存。"""
        import gc

        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# 便捷函数
# ============================================================
def quick_translate(
    pose_path: str,
    mode: str = "balanced",
) -> str:
    """
    一行翻译。

    Args:
        pose_path: 姿态 .pkl 文件路径
        mode: 预设模式 ("balanced" | "fast" | "minimal" | "quality")

    Returns:
        翻译后的中文文本
    """
    from config_deploy import PRESETS

    config = PRESETS[mode]()
    engine = SignLanguageInference(config)
    result = engine.translate_pose_file(pose_path)
    engine.unload()
    return result.text
