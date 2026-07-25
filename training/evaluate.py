"""
training/evaluate.py
────────────────────
Evaluation utilities for PI-CTBA-Net and baselines.

Metrics:
  Accuracy, Precision, Recall, F1, AUC-ROC, MCC, FAR,
  Brier Score, ECE, McNemar's test, Bootstrap 95% CI
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference over a DataLoader.

    Returns
    -------
    y_true  : (N,) int
    y_prob  : (N,) float  — predicted anomaly probability
    y_pred  : (N,) int    — hard labels at given threshold
    """
    model.eval()
    all_prob, all_true = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        all_prob.append(prob)
        all_true.append(y.numpy())

    y_prob = np.concatenate(all_prob)
    y_true = np.concatenate(all_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    return y_true, y_prob, y_pred


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Returns
    -------
    dict with keys:
      accuracy, precision, recall, f1, auc_roc, mcc,
      far, brier_score, ece, tp, fp, fn, tn
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    ece = _expected_calibration_error(y_true, y_prob, n_bins)

    return {
        "accuracy":    accuracy_score(y_true, y_pred) * 100,
        "precision":   precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall":      recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1":          f1_score(y_true, y_pred, zero_division=0) * 100,
        "auc_roc":     roc_auc_score(y_true, y_prob),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "far":         far,
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece":         ece,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE across B equal-width probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    McNemar's test (with continuity correction).

    b = model correct, baseline wrong
    c = model wrong,   baseline correct
    χ² = (|b - c| - 1)² / (b + c)
    """
    b = int(((y_pred_model == y_true) & (y_pred_baseline != y_true)).sum())
    c = int(((y_pred_model != y_true) & (y_pred_baseline == y_true)).sum())

    if (b + c) == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_value),
        "b": b,
        "c": c,
        "significant": bool(p_value < alpha),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap 95% confidence intervals for F1 and AUC-ROC.

    Returns dict with {'f1': (lo, hi), 'auc': (lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    f1_scores, auc_scores = [], []

    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        y_t = y_true[idx]
        y_p = y_prob[idx]
        y_hat = (y_p >= 0.5).astype(int)

        if len(np.unique(y_t)) < 2:
            continue

        f1_scores.append(f1_score(y_t, y_hat, zero_division=0) * 100)
        auc_scores.append(roc_auc_score(y_t, y_p))

    alpha = (1 - ci) / 2
    return {
        "f1":  (float(np.percentile(f1_scores, alpha * 100)),
                float(np.percentile(f1_scores, (1 - alpha) * 100))),
        "auc": (float(np.percentile(auc_scores, alpha * 100)),
                float(np.percentile(auc_scores, (1 - alpha) * 100))),
    }


# ── Full evaluation pipeline ─────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict:
    """
    Full evaluation: metrics + bootstrap CIs.
    """
    y_true, y_prob, y_pred = get_predictions(model, test_loader, device, threshold)
    metrics = compute_metrics(y_true, y_prob, y_pred)
    ci = bootstrap_ci(y_true, y_prob, n_bootstrap, seed=bootstrap_seed)

    return {
        "metrics": metrics,
        "bootstrap_ci": ci,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
