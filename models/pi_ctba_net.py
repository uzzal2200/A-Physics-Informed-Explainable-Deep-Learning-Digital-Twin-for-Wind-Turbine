"""
models/pi_ctba_net.py
─────────────────────
PI-CTBA-Net: Physics-Informed CNN-Transformer-BiLSTM-Attention Network.

Architecture (four sequential blocks):
  Block 1 — Multi-Scale CNN Encoder
  Block 2 — Transformer Encoder (no positional encoding)
  Block 3 — Feature Fusion + Bidirectional LSTM
  Block 4 — Multi-Head Attention Pooling + MLP Classifier

Total trainable parameters: 650,689

Reference:
  Debnath et al. (2026), Mathematics (MDPI).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MultiScaleCNNEncoder(nn.Module):
    """
    Block 1 — Two-stage 1-D convolutional encoder.

    Conv1(k=3, 64) → BN → ReLU → MaxPool
    Conv2(k=5, 128) → BN → ReLU → MaxPool

    Input  : (B, F, T)   [channels-first for Conv1d]
    Output : (B, 128, T')
    """

    def __init__(self, in_channels: int, filters_1: int = 64, filters_2: int = 128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, filters_1, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters_1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(filters_1, filters_2, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters_2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F, T)
        x = self.block1(x)
        x = self.block2(x)
        return x   # (B, 128, T//4)


class TransformerEncoder(nn.Module):
    """
    Block 2 — Transformer global self-attention encoder.

    No positional encoding; temporal ordering is delegated to BiLSTM.

    Input  : (B, T, F)
    Output : (B, T, d_model)
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, F)
        x = self.proj(x)           # (B, T, d_model)
        return self.encoder(x)     # (B, T, d_model)


class FeatureFusionBiLSTM(nn.Module):
    """
    Block 3 — Feature fusion + Bidirectional LSTM.

    Fuses CNN output (down-sampled) and Transformer output (every 4th token),
    then passes through a BiLSTM to model sequential dependencies.

    Output : (B, T', 2*hidden)
    """

    def __init__(
        self,
        cnn_channels: int = 128,
        transformer_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        fuse_in = cnn_channels + transformer_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fuse_in),
            nn.ReLU(inplace=True),
        )
        self.bilstm = nn.LSTM(
            input_size=fuse_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, h_cnn: Tensor, h_tr: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h_cnn : (B, 128, T//4)  — CNN output (channels-first)
        h_tr  : (B, T, 128)     — Transformer output (time-first)
        """
        # Align sequence lengths: take every 4th token from Transformer
        h_tr_sub = h_tr[:, ::4, :]            # (B, T//4, 128)
        h_cnn_t  = h_cnn.permute(0, 2, 1)    # (B, T//4, 128)

        # Ensure lengths match (trim if needed)
        min_len = min(h_cnn_t.size(1), h_tr_sub.size(1))
        h_cnn_t  = h_cnn_t[:, :min_len, :]
        h_tr_sub = h_tr_sub[:, :min_len, :]

        fused = torch.cat([h_cnn_t, h_tr_sub], dim=-1)  # (B, T', 256)
        fused = self.fusion(fused)
        out, _ = self.bilstm(fused)            # (B, T', 2*hidden)
        return out


class MultiHeadAttentionPooling(nn.Module):
    """
    Block 4 — Multi-head self-attention pooling + MLP classifier.

    Aggregates the BiLSTM sequence into a fixed-length representation,
    then projects through an MLP head to a scalar logit.
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int = 8,
        mlp_hidden: list = None,
        mlp_dropout: list = None,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [128, 64, 32]
        if mlp_dropout is None:
            mlp_dropout = [0.3, 0.2, 0.0]

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            batch_first=True,
        )

        # Build MLP head
        layers = []
        in_dim = embed_dim
        for i, (out_dim, drop) in enumerate(zip(mlp_hidden, mlp_dropout)):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if drop > 0:
                layers.append(nn.Dropout(drop))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, T', embed_dim)

        Returns
        -------
        logit : (B,)  — raw scalar logit (sigmoid applied at inference)
        """
        attn_out, _ = self.mha(x, x, x)       # (B, T', embed_dim)
        pooled = attn_out.mean(dim=1)          # (B, embed_dim)
        logit  = self.mlp(pooled).squeeze(-1)  # (B,)
        return logit


class PICtbaNet(nn.Module):
    """
    PI-CTBA-Net: Physics-Informed CNN-Transformer-BiLSTM-Attention Network.

    Parameters
    ----------
    num_features : int
        Number of input features F (default 12 = 9 physics + 3 temporal).
    config : dict
        Model sub-dict from config.yaml.

    Usage
    -----
    >>> model = PICtbaNet(num_features=12, config=cfg["model"])
    >>> logit = model(x)              # (B,)  during training
    >>> prob  = torch.sigmoid(logit)  # (B,)  at inference
    """

    def __init__(self, num_features: int = 12, config: dict = None):
        super().__init__()
        cfg = config or {}

        f1   = cfg.get("cnn_filters_1", 64)
        f2   = cfg.get("cnn_filters_2", 128)
        d    = cfg.get("d_model", 128)
        nh   = cfg.get("nhead", 8)
        nl   = cfg.get("num_transformer_layers", 2)
        dff  = cfg.get("dim_feedforward", 256)
        tdrop= cfg.get("transformer_dropout", 0.1)
        bhid = cfg.get("bilstm_hidden", 128)
        bl   = cfg.get("bilstm_layers", 1)
        bdrop= cfg.get("bilstm_dropout", 0.1)
        mha_h= cfg.get("mha_heads", 8)
        mlp_h= cfg.get("mlp_hidden", [128, 64, 32])
        mlp_d= cfg.get("mlp_dropout", [0.3, 0.2])

        # Blocks
        self.cnn_encoder = MultiScaleCNNEncoder(num_features, f1, f2)
        self.tr_encoder  = TransformerEncoder(num_features, d, nh, nl, dff, tdrop)
        self.fusion_lstm = FeatureFusionBiLSTM(f2, d, bhid, bl, bdrop)
        self.attn_pool   = MultiHeadAttentionPooling(
            embed_dim=2 * bhid,
            nhead=mha_h,
            mlp_hidden=mlp_h,
            mlp_dropout=mlp_d,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)  — batch of windows

        Returns
        -------
        logit : (B,)  — raw logit (no sigmoid; use BCEWithLogitsLoss)
        """
        # Block 1 — CNN (needs channels-first)
        h_cnn = self.cnn_encoder(x.permute(0, 2, 1))   # (B, 128, T//4)

        # Block 2 — Transformer
        h_tr  = self.tr_encoder(x)                      # (B, T, d_model)

        # Block 3 — Fusion + BiLSTM
        h_bi  = self.fusion_lstm(h_cnn, h_tr)           # (B, T', 2*hidden)

        # Block 4 — Attention pooling + MLP
        logit = self.attn_pool(h_bi)                    # (B,)

        return logit

    def predict_proba(self, x: Tensor) -> Tensor:
        """Inference: returns anomaly probability in [0, 1]."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
