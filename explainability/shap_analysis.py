"""
explainability/shap_analysis.py
────────────────────────────────
GradientSHAP explainability analysis for PI-CTBA-Net.

Produces:
  - Global feature importance (mean |SHAP|)
  - SHAP dependence plots per feature
  - Attention weight heatmaps (from Block 4)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "wind_speed", "active_power", "reactive_power",
    "ambient_temp", "grid_frequency", "rotor_speed",
    "gearbox_oil_temp", "generator_temp", "pitch_angle",
    "hour", "day", "month",
]


class GradientSHAPAnalyzer:
    """
    GradientSHAP attribution for sequence models.

    Uses captum.attr.GradientShap if available,
    otherwise falls back to integrated gradients.

    Parameters
    ----------
    model  : nn.Module — trained PI-CTBA-Net
    device : torch.device
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.eval()
        self.device = device

    def compute_shap_values(
        self,
        X_background: np.ndarray,   # (N_bg, T, F)
        X_test:       np.ndarray,   # (N_test, T, F)
        seed: int = 42,
    ) -> np.ndarray:
        """
        Compute GradientSHAP attributions.

        Returns
        -------
        shap_values : (N_test, T, F)  — per-timestep per-feature attributions
        """
        try:
            from captum.attr import GradientShap
            torch.manual_seed(seed)

            bg = torch.from_numpy(X_background).float().to(self.device)
            x  = torch.from_numpy(X_test).float().to(self.device)

            def forward_fn(inp: Tensor) -> Tensor:
                return torch.sigmoid(self.model(inp))

            gs = GradientShap(forward_fn)
            attrs = gs.attribute(
                x,
                baselines=bg,
                n_samples=50,
                stdevs=0.09,
            )
            return attrs.cpu().detach().numpy()

        except ImportError:
            logger.warning(
                "captum not installed. Falling back to gradient × input."
            )
            return self._gradient_x_input(X_test)

    def global_feature_importance(
        self, shap_values: np.ndarray
    ) -> np.ndarray:
        """
        Mean |SHAP| across timesteps and test samples.

        Returns
        -------
        importance : (F,) — one value per feature
        """
        # shap_values: (N, T, F)
        return np.abs(shap_values).mean(axis=(0, 1))   # (F,)

    def _gradient_x_input(self, X_test: np.ndarray) -> np.ndarray:
        """Gradient × Input attribution as a fallback."""
        x = torch.from_numpy(X_test).float().to(self.device)
        x.requires_grad_(True)

        self.model.zero_grad()
        out = torch.sigmoid(self.model(x))
        out.sum().backward()

        return (x.grad * x).detach().cpu().numpy()

    def plot_global_importance(
        self,
        importance: np.ndarray,
        save_path: Optional[str] = None,
        feature_names: List[str] = None,
    ):
        """Bar chart of global feature importance."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Skipping plot.")
            return

        names = feature_names or FEATURE_NAMES
        order = np.argsort(importance)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            [names[i] for i in order],
            importance[order],
            color="#2196F3",
        )
        for bar, val in zip(bars, importance[order]):
            ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.5f}", va="center", fontsize=8)

        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("GradientSHAP Global Feature Importance — PI-CTBA-Net")
        plt.tight_layout()

        if save_path:
            os.makedirs(Path(save_path).parent, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        plt.show()


class AttentionVisualizer:
    """
    Extract and visualise multi-head attention weights from Block 4.

    Registers a forward hook on the MHA layer of PI-CTBA-Net.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attn_weights = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Hook into the attention pooling MHA layer."""
        def hook_fn(module, input, output):
            # output = (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2:
                self._attn_weights = output[1].detach().cpu()

        # Navigate to attn_pool.mha
        if hasattr(self.model, "attn_pool") and \
           hasattr(self.model.attn_pool, "mha"):
            self._hook_handle = self.model.attn_pool.mha.register_forward_hook(
                hook_fn
            )
        else:
            logger.warning("Could not locate attn_pool.mha — hook not registered.")

    def get_attention_weights(
        self,
        x: np.ndarray,
        device: torch.device,
    ) -> Optional[np.ndarray]:
        """
        Run a forward pass and return the attention weight matrix.

        Returns
        -------
        weights : (B, nhead, T', T') or None
        """
        self.model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(x).float().to(device)
            _ = self.model(inp)
        return self._attn_weights.numpy() if self._attn_weights is not None else None

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

    def plot_attention_map(
        self,
        weights: np.ndarray,
        title: str = "Attention Weight Map",
        save_path: Optional[str] = None,
    ):
        """Plot mean attention map averaged over heads and batch."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available.")
            return

        mean_map = weights.mean(axis=(0, 1))   # (T', T')

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(mean_map, aspect="auto", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Key timestep")
        ax.set_ylabel("Query timestep")
        plt.tight_layout()

        if save_path:
            os.makedirs(Path(save_path).parent, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        plt.show()
