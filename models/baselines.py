"""
models/baselines.py
───────────────────
Eight baseline architectures evaluated alongside PI-CTBA-Net
on the CARE benchmark.

Models:
  1. CNN1D
  2. LSTMModel
  3. BiLSTMModel
  4. TCNModel
  5. TransformerModel
  6. CNNLSTMModel
  7. CNNBiLSTMModel
  8. CNNTBAttModel  (ablation baseline — same arch as PI-CTBA-Net, no physics loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import weight_norm


# ── 1. 1D-CNN ────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """Three convolutional blocks → global avg pool → MLP head. (54 K params)"""

    def __init__(self, num_features: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x.permute(0, 2, 1))).squeeze(-1)


# ── 2. LSTM ──────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Two-layer LSTM → last hidden state → MLP. (78 K params)"""

    def __init__(self, num_features: int = 12, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 48), nn.ReLU(inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


# ── 3. Bi-LSTM ───────────────────────────────────────────────────────────────

class BiLSTMModel(nn.Module):
    """Two-layer BiLSTM → concat final states → MLP. (134 K params)"""

    def __init__(self, num_features: int = 12, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, (h, _) = self.lstm(x)
        # Concatenate last forward + last backward hidden state
        out = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(out).squeeze(-1)


# ── 4. TCN ───────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel,
                                           padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel,
                                           padding=pad, dilation=dilation))
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x)[..., :-self.pad])
        out = self.drop(out)
        out = self.relu(self.conv2(out)[..., :-self.pad])
        out = self.drop(out)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Four dilated causal TCN blocks → adaptive avg pool → MLP. (211 K params)"""

    def __init__(self, num_features: int = 12, channels: int = 64):
        super().__init__()
        dilations = [1, 2, 4, 8]
        blocks = []
        in_ch = num_features
        for d in dilations:
            blocks.append(_TCNBlock(in_ch, channels, kernel=3, dilation=d))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.tcn(x.permute(0, 2, 1))).squeeze(-1)


# ── 5. Transformer ───────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    """Linear projection → 2-layer Transformer → mean pool → MLP. (274 K params)
    No positional encoding (permutation-invariant baseline)."""

    def __init__(self, num_features: int = 12, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(self.proj(x))    # (B, T, d_model)
        return self.head(h.mean(dim=1)).squeeze(-1)


# ── 6. CNN-LSTM ──────────────────────────────────────────────────────────────

class CNNLSTMModel(nn.Module):
    """Two-block CNN → single-layer LSTM (64 units) → MLP. (310 K params)"""

    def __init__(self, num_features: int = 12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (hn, _) = self.lstm(h)
        return self.head(hn[-1]).squeeze(-1)


# ── 7. CNN-BiLSTM ────────────────────────────────────────────────────────────

class CNNBiLSTMModel(nn.Module):
    """Two-block CNN → BiLSTM (64 units) → MLP. (517 K params)"""

    def __init__(self, num_features: int = 12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (hn, _) = self.lstm(h)
        out = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.head(out).squeeze(-1)


# ── 8. CNN-TB-Att (Ablation baseline) ───────────────────────────────────────

class CNNTBAttModel(nn.Module):
    """
    Ablation baseline: same architecture as PI-CTBA-Net but with
    4-head Transformer and BiLSTM (64 units) instead of 8-head / 128 units.
    Trained with BCE loss only — no physics auxiliary losses. (557 K params)
    """

    def __init__(self, num_features: int = 12):
        super().__init__()
        from models.pi_ctba_net import (
            MultiScaleCNNEncoder, TransformerEncoder,
            FeatureFusionBiLSTM, MultiHeadAttentionPooling,
        )
        self.cnn  = MultiScaleCNNEncoder(num_features, 64, 128)
        self.tr   = TransformerEncoder(num_features, d_model=128, nhead=4,
                                        num_layers=2, dim_feedforward=256)
        self.fuse = FeatureFusionBiLSTM(128, 128, hidden_size=64)
        self.pool = MultiHeadAttentionPooling(
            embed_dim=128, nhead=4, mlp_hidden=[128, 64, 32])

    def forward(self, x: Tensor) -> Tensor:
        h_cnn = self.cnn(x.permute(0, 2, 1))
        h_tr  = self.tr(x)
        h_bi  = self.fuse(h_cnn, h_tr)
        return self.pool(h_bi)


# ── Registry ─────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "1D-CNN":       CNN1D,
    "LSTM":         LSTMModel,
    "Bi-LSTM":      BiLSTMModel,
    "TCN":          TCNModel,
    "Transformer":  TransformerModel,
    "CNN-LSTM":     CNNLSTMModel,
    "CNN-BiLSTM":   CNNBiLSTMModel,
    "CNN-TB-Att":   CNNTBAttModel,
}


def build_baseline(name: str, num_features: int = 12) -> nn.Module:
    """Instantiate a baseline model by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](num_features=num_features)
