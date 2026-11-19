"""
training/trainer.py
───────────────────
Training loop for PI-CTBA-Net and all baseline models.

Features:
  - Cosine annealing LR schedule
  - Gradient clipping
  - Early stopping (patience on validation loss)
  - Best-checkpoint saving
  - Per-epoch loss logging (all physics sub-losses)

Reviewer additions (revision APEN-D-26-18228):
  - lambda_override parameter in Trainer.__init__() for physics weight
    sensitivity analysis (Section 5.6, Tables 25–26).
  - raw_physics parameter to swap in PhysicsInformedLossRawUnit, which
    de-normalizes inputs before computing L1 (Betz) and L3 (Newton's
    cooling) in physical units (Section 5.5, Table 24).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.physics_loss import (
    PhysicsInformedLoss,
    PhysicsInformedLossRawUnit,
    WEIGHT_CONFIGS,
    NormStats,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer supporting both PI-CTBA-Net (physics loss)
    and baseline models (BCE loss only).

    Parameters
    ----------
    model         : nn.Module
    config        : full config dict
    device        : torch.device
    use_physics   : bool — if False, uses BCE loss only (for baselines)
    lambda_override : str | dict | None
        Weight preset name from WEIGHT_CONFIGS (e.g. "no_physics",
        "half", "original", "double", "equal"), or a custom dict
        {"betz": ..., "tsr": ..., "thermal": ...,
         "power_factor": ..., "pitch": ...}.
        When None, lambdas are read from config["physics_loss"].
        Used for weight sensitivity analysis (Section 5.6, Tables 25–26).
    raw_physics   : bool
        If True, L1 (Betz power curve) and L3 (Newton's cooling) are
        computed on de-normalized physical units; L2/L4/L5 remain
        z-score normalized (Section 5.5, Table 24, ablation).
        Requires config["norm_stats"] to be present.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        use_physics: bool = True,
        lambda_override: "str | dict | None" = None,
        raw_physics: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.train_cfg = config["training"]
        self.use_physics = use_physics

        # ── Loss function ─────────────────────────────────────────────────
        if use_physics:
            pl_cfg = config["physics_loss"]

            # Resolve lambda values: preset > override dict > config
            if lambda_override is None:
                lambdas = {
                    "betz":         pl_cfg["lambda_betz"],
                    "tsr":          pl_cfg["lambda_tsr"],
                    "thermal":      pl_cfg["lambda_thermal"],
                    "power_factor": pl_cfg["lambda_power_factor"],
                    "pitch":        pl_cfg["lambda_pitch"],
                }
            elif isinstance(lambda_override, str):
                if lambda_override not in WEIGHT_CONFIGS:
                    raise ValueError(
                        f"Unknown weight preset '{lambda_override}'. "
                        f"Choose from: {list(WEIGHT_CONFIGS.keys())}"
                    )
                lambdas = dict(WEIGHT_CONFIGS[lambda_override])
                logger.info(f"Using weight preset '{lambda_override}': {lambdas}")
            else:
                # Custom dict passed directly
                lambdas = dict(lambda_override)
                logger.info(f"Using custom lambda dict: {lambdas}")

            pos_weight = self.train_cfg["pos_weight"]
            eps = pl_cfg["epsilon"]

            if raw_physics:
                # Build NormStats from config (required for de-normalization)
                ns_cfg = config.get("norm_stats", {})
                norm_stats = NormStats(
                    wind_speed_mu=ns_cfg["wind_speed_mu"],
                    wind_speed_sigma=ns_cfg["wind_speed_sigma"],
                    active_power_mu=ns_cfg["active_power_mu"],
                    active_power_sigma=ns_cfg["active_power_sigma"],
                    gearbox_oil_temp_mu=ns_cfg["gearbox_oil_temp_mu"],
                    gearbox_oil_temp_sigma=ns_cfg["gearbox_oil_temp_sigma"],
                    ambient_temp_mu=ns_cfg["ambient_temp_mu"],
                    ambient_temp_sigma=ns_cfg["ambient_temp_sigma"],
                )
                self.criterion = PhysicsInformedLossRawUnit(
                    lambdas=lambdas,
                    norm_stats=norm_stats,
                    pos_weight=pos_weight,
                    eps=eps,
                ).to(device)
                logger.info("Using raw-unit physics loss (L1/L3 de-normalized).")
            else:
                self.criterion = PhysicsInformedLoss(
                    lambdas=lambdas,
                    pos_weight=pos_weight,
                    eps=eps,
                ).to(device)

        else:
            pw = torch.tensor([self.train_cfg["pos_weight"]]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimiser = Adam(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # ── LR scheduler ──────────────────────────────────────────────────
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=self.train_cfg["max_epochs"],
            eta_min=1e-6,
        )

        # ── Bookkeeping ───────────────────────────────────────────────────
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "lr": []
        }
        self.best_val_loss = float("inf")
        self.stale_epochs = 0
        self.best_checkpoint_path: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "pi_ctba_net",
    ) -> Dict[str, list]:
        """
        Run training loop.

        Returns training history dict.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        max_epochs = self.train_cfg["max_epochs"]
        patience   = self.train_cfg["patience"]
        grad_clip  = self.train_cfg["grad_clip_norm"]

        logger.info(
            f"Starting training: {model_name} | "
            f"device={self.device} | "
            f"max_epochs={max_epochs} | patience={patience}"
        )

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, grad_clip)
            val_loss   = self._val_epoch(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                f"time={elapsed:.1f}s"
            )

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.stale_epochs = 0
                path = Path(checkpoint_dir) / f"{model_name}_best.pt"
                torch.save(self.model.state_dict(), path)
                self.best_checkpoint_path = str(path)
                logger.info(f"  ✓ Best checkpoint saved → {path}")
            else:
                self.stale_epochs += 1
                if self.stale_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        # Save training history
        history_path = Path(checkpoint_dir) / f"{model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_best(self):
        """Load the best checkpoint weights into the model."""
        if self.best_checkpoint_path is None:
            raise RuntimeError("No checkpoint saved yet. Call fit() first.")
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_path, map_location=self.device)
        )
        logger.info(f"Loaded best checkpoint: {self.best_checkpoint_path}")

    # ── Private ───────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, grad_clip: float) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimiser.zero_grad()
            logits = self.model(x)   # (B,)

            if self.use_physics:
                loss, _ = self.criterion(logits, y, x)
            else:
                loss = self.criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimiser.step()

            total_loss += loss.item() * len(y)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)

            if self.use_physics:
                loss, _ = self.criterion(logits, y, x)
            else:
                loss = self.criterion(logits, y)

            total_loss += loss.item() * len(y)

        return total_loss / len(loader.dataset)