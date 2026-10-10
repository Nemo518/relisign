"""
GFSLT-VLP Simplified Baseline Model.

This is a simplified reimplementation of the GFSLT-VLP architecture for
comparison with Uni-Sign on CE-CSL. The model uses a simpler pose encoder
(per-part MLP + mean pooling + Transformer encoder) instead of Uni-Sign's
specialized part-based ST-GCN encoder.

Reference: GFSLT-VLP (CVPR 2022) - https://github.com/zhoubenjia/GFSLT-VLP
"""

import math

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import mt5_path


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class GFSLT_VLP_Simplified(nn.Module):
    """
    Simplified GFSLT-VLP model using:
    - Per-part MLP projection (no ST-GCN)
    - Mean pooling over keypoints per part
    - Transformer encoder for temporal modeling
    - mT5 decoder for text generation
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Keypoint dimensions per part: (num_keypoints, coords)
        # body: 9 kp, left_hand: 21 kp, right_hand: 21 kp, face_all: 18 kp
        self.kp_dims = {
            "body": 9 * 3,
            "left": 21 * 3,
            "right": 21 * 3,
            "face_all": 18 * 3,
        }
        self.modes = ["body", "left", "right", "face_all"]

        hidden_dim = getattr(args, "hidden_dim", 256)

        # Per-part MLP projections (simpler than ST-GCN in Uni-Sign)
        self.part_proj = nn.ModuleDict()
        for mode in self.modes:
            self.part_proj[mode] = nn.Sequential(
                nn.Linear(self.kp_dims[mode], 64),
                nn.GELU(),
                nn.Linear(64, 64),
            )
        # Share left/right projections
        self.part_proj["left"] = self.part_proj["right"]

        # Merge 4 * 64 = 256 → hidden_dim
        self.merge_proj = nn.Linear(64 * len(self.modes), hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=512)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=4
        )

        # Bridge to mT5 hidden size (768 for mt5-small/base)
        self.bridge_proj = nn.Linear(hidden_dim, 768)

        # Language setting
        if "CSL" in getattr(self.args, "dataset", ""):
            self.lang = "Chinese"
        else:
            self.lang = "English"

        # mT5 decoder
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_pose(self, src_input):
        """
        Encode pose keypoints using simple MLP + Transformer encoder.

        Args:
            src_input: dict with keys 'body', 'left', 'right', 'face_all'
                       each (B, T, V, 3), plus 'attention_mask' (B, T)
        Returns:
            encoded features (B, T, 768)
        """
        part_features = []
        for mode in self.modes:
            x = src_input[mode]  # (B, T, V, 3)
            B, T, V, C = x.shape
            x = x.reshape(B, T, V * C).float()  # (B, T, V*3)
            x = self.part_proj[mode](x)  # (B, T, 64)
            part_features.append(x)

        x = torch.cat(part_features, dim=-1)  # (B, T, 256)
        x = self.merge_proj(x)  # (B, T, hidden_dim)
        x = self.pos_encoding(x)

        # Build padding mask from attention_mask
        attn_mask = src_input.get("attention_mask", None)
        if attn_mask is not None:
            src_key_padding_mask = (attn_mask == 0)  # (B, T)
        else:
            src_key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.bridge_proj(x)  # (B, T, 768)
        return x

    def forward(self, src_input, tgt_input):
        inputs_embeds = self.encode_pose(src_input)

        bs = len(tgt_input["gt_sentence"])
        prefix_token = self.mt5_tokenizer(
            [f"Translate sign language video to {self.lang}: "] * bs,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(inputs_embeds.device)

        prefix_embeds = self.mt5_model.encoder.embed_tokens(
            prefix_token["input_ids"]
        )
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        pose_attn = src_input.get(
            "attention_mask",
            torch.ones(bs, inputs_embeds.size(1) - prefix_embeds.size(1)).to(
                inputs_embeds.device
            ),
        )
        attention_mask = torch.cat(
            [prefix_token["attention_mask"].to(inputs_embeds.device), pose_attn], dim=1
        )

        tgt_tokenizer = self.mt5_tokenizer(
            tgt_input["gt_sentence"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
        )
        labels = tgt_tokenizer["input_ids"].to(inputs_embeds.device)
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100

        out = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        label = labels.reshape(-1)
        logits = out["logits"].reshape(-1, out["logits"].shape[-1])
        loss_fct = torch.nn.CrossEntropyLoss(
            label_smoothing=getattr(self.args, "label_smoothing", 0.2),
            ignore_index=-100,
        )
        loss = loss_fct(logits, label.to(logits.device, non_blocking=True))

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams):
        return self.mt5_model.generate(
            inputs_embeds=pre_compute_item["inputs_embeds"],
            attention_mask=pre_compute_item["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
