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
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.physics_loss import PhysicsInformedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer supporting both PI-CTBA-Net (physics loss)
    and baseline models (BCE loss only).

    Parameters
    ----------
    model      : nn.Module
    config     : full config dict
    device     : torch.device
    use_physics: bool — if False, uses BCE loss only (for baselines)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        use_physics: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.train_cfg = config["training"]
        self.use_physics = use_physics

        # Loss function
        if use_physics:
            pl_cfg = config["physics_loss"]
            lambdas = {
                "betz":         pl_cfg["lambda_betz"],
                "tsr":          pl_cfg["lambda_tsr"],
                "thermal":      pl_cfg["lambda_thermal"],
                "power_factor": pl_cfg["lambda_power_factor"],
                "pitch":        pl_cfg["lambda_pitch"],
            }
            self.criterion = PhysicsInformedLoss(
                lambdas=lambdas,
                pos_weight=self.train_cfg["pos_weight"],
                eps=pl_cfg["epsilon"],
            ).to(device)
        else:
            pw = torch.tensor([self.train_cfg["pos_weight"]]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Optimiser
        self.optimiser = Adam(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # LR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=self.train_cfg["max_epochs"],
            eta_min=1e-6,
        )

        # Bookkeeping
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

        logger.info(f"Starting training: {model_name} | "
                    f"device={self.device} | "
                    f"max_epochs={max_epochs} | patience={patience}")

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
