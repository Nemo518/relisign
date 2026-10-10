from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .data_service import build_alignment_dataframe
except ImportError:  # pragma: no cover - script mode fallback
    from data_service import build_alignment_dataframe

from paths import PRETRAINED_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import S2T_Dataset_online
from models import Uni_Sign
import utils as utils


@dataclass
class HookBundle:
    activations: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    cross_attention_calls: list[torch.Tensor]
    handles: list[Any]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _build_args(
    rgb_support: bool,
    max_length: int = 256,
    rgb_frame_sampling: str = "confidence_topk",
    fusion_use_uncertainty_gate: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=rgb_support,
        rgb_backbone_weights=str(PRETRAINED_ROOT / "efficientnet_b0_rwightman-7f5810bc.pth"),
        rgb_backbone_pretrained=False,
        rgb_frame_sampling=rgb_frame_sampling,
        fusion_use_uncertainty_gate=fusion_use_uncertainty_gate,
        fusion_meta_dim=3,
        max_length=max_length,
        hidden_dim=256,
        label_smoothing=0.2,
    )


def _load_pose_extractor() -> Any:
    module_path = PROJECT_ROOT / "demo" / "online_inference.py"
    spec = importlib.util.spec_from_file_location("online_inference_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load demo/online_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pose_extraction


def ensure_pose_payload(video_path: Path, existing_pose_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if existing_pose_payload is not None:
        return existing_pose_payload
    pose_extraction = _load_pose_extractor()
    return pose_extraction(str(video_path))


def _move_src_input(src_input: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, dtype=torch.float32 if value.dtype.is_floating_point else value.dtype)
        else:
            moved[key] = value
    return moved


def _load_checkpoint_flexible(model: Uni_Sign, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    filtered_state = {}

    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value

    model.load_state_dict(filtered_state, strict=False)
    return checkpoint


def _prepare_batch(
    video_path: Path,
    pose_payload: dict[str, Any],
    rgb_support: bool,
    max_length: int,
    rgb_frame_sampling: str,
    fusion_use_uncertainty_gate: bool,
) -> tuple[dict[str, Any], dict[str, Any], SimpleNamespace]:
    args = _build_args(
        rgb_support=rgb_support,
        max_length=max_length,
        rgb_frame_sampling=rgb_frame_sampling,
        fusion_use_uncertainty_gate=fusion_use_uncertainty_gate,
    )
    dataset = S2T_Dataset_online(args=args)
    dataset.rgb_data = str(video_path)
    dataset.pose_data = pose_payload
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, shuffle=False)
    src_input, tgt_input = next(iter(dataloader))
    return src_input, tgt_input, args


def _register_hooks(model: Uni_Sign) -> HookBundle:
    bundle = HookBundle(activations={}, gradients={}, cross_attention_calls=[], handles=[])

    for part in model.modes:
        def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor, part_name: str = part) -> None:
            bundle.activations[part_name] = output

        def backward_hook(_module: torch.nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[torch.Tensor, ...], part_name: str = part) -> None:
            if grad_output and grad_output[0] is not None:
                bundle.gradients[part_name] = grad_output[0]

        bundle.handles.append(model.fusion_gcn_modules[part].register_forward_hook(forward_hook))
        bundle.handles.append(model.fusion_gcn_modules[part].register_full_backward_hook(backward_hook))

    if getattr(model.args, "rgb_support", False):
        def cross_attn_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                bundle.cross_attention_calls.append(output[1].detach().cpu())

        bundle.handles.append(model.fusion_pose_rgb_DA.cross_attn.register_forward_hook(cross_attn_hook))

    return bundle


def _compute_gradcam_series(activation: torch.Tensor, gradient: torch.Tensor, valid_length: int) -> np.ndarray:
    activation = activation.detach().float()
    gradient = gradient.detach().float()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1)).mean(dim=-1)
    cam = cam[0, :valid_length]
    cam = cam.cpu().numpy()
    if cam.size == 0:
        return cam
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-6:
        return np.zeros_like(cam)
    return (cam - cam_min) / (cam_max - cam_min)


def _decode_prediction(model: Uni_Sign, stack_out: dict[str, Any]) -> str:
    output = model.generate(stack_out, max_new_tokens=100, num_beams=4)
    prediction = model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)
    return prediction[0].strip() if prediction else ""


def _encoder_attention_frame(
    model: Uni_Sign,
    stack_out: dict[str, Any],
    src_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder_outputs = model.mt5_model.encoder(
        inputs_embeds=stack_out["inputs_embeds"].detach(),
        attention_mask=stack_out["attention_mask"].detach(),
        output_attentions=True,
        return_dict=True,
    )
    attn = encoder_outputs.attentions[-1].detach().float().mean(dim=1)[0]
    prefix_len = attn.shape[0] - src_length
    sign_attn = attn[prefix_len:, prefix_len:][:src_length, :src_length].cpu().numpy()
    frame_importance = (sign_attn.mean(axis=0) + sign_attn.mean(axis=1)) / 2.0

    importance_df = pd.DataFrame(
        {
            "frame": np.arange(len(frame_importance)),
            "encoder_attention": frame_importance,
        }
    )

    matrix_rows = []
    for row_idx in range(sign_attn.shape[0]):
        for col_idx in range(sign_attn.shape[1]):
            matrix_rows.append({"source_frame": row_idx, "target_frame": col_idx, "attention": float(sign_attn[row_idx, col_idx])})
    return importance_df, pd.DataFrame(matrix_rows)


def _rgb_cross_attention_frame(calls: list[torch.Tensor]) -> pd.DataFrame:
    rows = []
    branch_names = ["left", "right"]
    for branch_index, call in enumerate(calls[:2]):
        weights = call.float().mean(dim=1).mean(dim=0).cpu().numpy()
        if weights.size != 16:
            continue
        grid = weights.reshape(4, 4)
        for y in range(4):
            for x in range(4):
                rows.append(
                    {
                        "branch": branch_names[branch_index],
                        "patch_x": x,
                        "patch_y": y,
                        "attention": float(grid[y, x]),
                    }
                )
    return pd.DataFrame(rows)


def analyze_video_xai(
    video_path: Path,
    checkpoint_path: Path,
    rgb_support: bool,
    reference_text: str | None = None,
    pose_payload: dict[str, Any] | None = None,
    device: str | None = None,
    max_length: int = 256,
    rgb_frame_sampling: str = "confidence_topk",
    fusion_use_uncertainty_gate: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    utils.set_seed(seed)
    resolved_pose_payload = ensure_pose_payload(video_path, existing_pose_payload=pose_payload)
    src_input, tgt_input, args = _prepare_batch(
        video_path=video_path,
        pose_payload=resolved_pose_payload,
        rgb_support=rgb_support,
        max_length=max_length,
        rgb_frame_sampling=rgb_frame_sampling,
        fusion_use_uncertainty_gate=fusion_use_uncertainty_gate,
    )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # RTX 5060 (Blackwell sm_120) requires PyTorch 2.7+ — older builds lack kernel images
    if resolved_device.type == "cuda":
        try:
            torch.empty(1, device=resolved_device) + 1
        except RuntimeError:
            resolved_device = torch.device("cpu")
    model = Uni_Sign(args=args)
    _load_checkpoint_flexible(model, checkpoint_path)
    model = model.to(resolved_device)
    model.eval()

    src_input = _move_src_input(src_input, resolved_device)
    seed_sentence = reference_text or "手语翻译"
    tgt_input["gt_sentence"] = [seed_sentence]
    tgt_input["gt_gloss"] = [""]

    with torch.no_grad():
        seed_stack_out = model(src_input, tgt_input)
        prediction = _decode_prediction(model, seed_stack_out)

    target_sentence = prediction or reference_text or seed_sentence
    tgt_input["gt_sentence"] = [target_sentence]

    hook_bundle = _register_hooks(model)
    try:
        model.zero_grad(set_to_none=True)
        stack_out = model(src_input, tgt_input)
        loss = stack_out["loss"]
        loss.backward()
        src_length = int(src_input["src_length_batch"][0].item())

        cam_rows = {"frame": np.arange(src_length)}
        for part in model.modes:
            activation = hook_bundle.activations.get(part)
            gradient = hook_bundle.gradients.get(part)
            if activation is None or gradient is None:
                cam_rows[part] = np.zeros(src_length, dtype=np.float32)
                continue
            cam_rows[part] = _compute_gradcam_series(activation, gradient, src_length)

        gradcam_df = pd.DataFrame(cam_rows)
        part_columns = [part for part in model.modes if part in gradcam_df.columns]
        gradcam_df["gradcam_combined"] = gradcam_df[part_columns].mean(axis=1)

        attention_df, attention_matrix_df = _encoder_attention_frame(model, stack_out, src_length)
        merged_xai_df = attention_df.merge(gradcam_df, on="frame", how="left")
        merged_xai_df["encoder_attention_norm"] = merged_xai_df["encoder_attention"] / max(1e-6, float(merged_xai_df["encoder_attention"].max()))
        merged_xai_df["xai_score"] = 0.55 * merged_xai_df["gradcam_combined"] + 0.45 * merged_xai_df["encoder_attention_norm"]
        top_frames_df = merged_xai_df.sort_values("xai_score", ascending=False).head(6).reset_index(drop=True)

        part_summary_rows = []
        for part in model.modes:
            series = merged_xai_df[part]
            part_summary_rows.append(
                {
                    "part": part,
                    "mean_importance": float(series.mean()),
                    "peak_importance": float(series.max()),
                }
            )
        part_summary_df = pd.DataFrame(part_summary_rows).sort_values("peak_importance", ascending=False)
        rgb_cross_df = _rgb_cross_attention_frame(hook_bundle.cross_attention_calls)
        alignment_df = build_alignment_dataframe(reference_text or target_sentence, prediction or "")

        return {
            "prediction": prediction,
            "target_sentence": target_sentence,
            "loss": float(loss.detach().cpu().item()),
            "xai_timeline_df": merged_xai_df,
            "attention_matrix_df": attention_matrix_df,
            "rgb_cross_df": rgb_cross_df,
            "top_frames_df": top_frames_df,
            "part_summary_df": part_summary_df,
            "alignment_df": alignment_df,
            "pose_payload": resolved_pose_payload,
        }
    finally:
        hook_bundle.close()
