"""
training/evaluate.py
────────────────────
Evaluation utilities for PI-CTBA-Net and baselines.

Metrics:
  Accuracy, Precision, Recall, F1, AUC-ROC, MCC, FAR,
  Brier Score, ECE, McNemar's test, Bootstrap 95% CI

Reviewer-added functions (revision APEN-D-26-18228):
  compute_lead_time()       — mean/median warning lead time vs FAR at
                              three decision thresholds (Sec. 4.1.2,
                              Table 16).
  compute_operational_far() — translates sample-level FAR to per-turbine-
                              per-day and fleet-level operational estimates
                              (Sec. 4.1, Table 15).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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

    FAR (False Alarm Rate) = FP / (FP + TN) × 100.
    For PI-CTBA-Net the corrected test-set FAR is 2.22% (Table 14),
    not 3.04% as reported in the initial submission — the correction
    is noted in the response to Reviewer 3, Comment 4.

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


# ── Operational FAR translation (Section 4.1, Table 15) ─────────────────────

def compute_operational_far(
    far_pct: float,
    window_size_steps: int = 144,
    stride_steps: int = 72,
    step_duration_min: int = 10,
    n_turbines: int = 1,
) -> Dict[str, float]:
    """
    Translate a sample-level False Alarm Rate into operational terms
    (per-turbine-per-day and fleet-level), as reported in Table 15
    (Section 4.1, Reviewer 1 Comment 6).

    Parameters
    ----------
    far_pct : float
        Sample-level FAR in percent (e.g. 2.22).
    window_size_steps : int
        Number of SCADA steps per window (default 144 = 24 h at 10 min).
    stride_steps : int
        Stride in SCADA steps (default 72 = 12 h).
    step_duration_min : int
        Duration of one SCADA step in minutes (default 10).
    n_turbines : int
        Fleet size for fleet-level calculation.

    Returns
    -------
    dict with:
        windows_per_turbine_per_day : float
        fa_per_turbine_per_day      : float
        days_between_false_alarms   : float
        fa_fleet_per_day            : float  (if n_turbines > 1)
    """
    window_duration_min = window_size_steps * step_duration_min
    day_min = 24 * 60

    # Number of non-overlapping windows in one day
    windows_per_day = day_min / (stride_steps * step_duration_min)

    fa_per_turbine_per_day = windows_per_day * (far_pct / 100.0)
    days_between_fa = 1.0 / fa_per_turbine_per_day if fa_per_turbine_per_day > 0 else float("inf")

    result = {
        "far_pct":                      far_pct,
        "windows_per_turbine_per_day":  windows_per_day,
        "fa_per_turbine_per_day":       round(fa_per_turbine_per_day, 4),
        "days_between_false_alarms":    round(days_between_fa, 1),
    }
    if n_turbines > 1:
        result["fa_fleet_per_day"] = round(fa_per_turbine_per_day * n_turbines, 2)

    return result


# ── Early-warning lead time (Section 4.1.2, Table 16) ───────────────────────

def compute_lead_time(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    event_ids: np.ndarray,
    thresholds: List[float] = None,
    stride_steps: int = 72,
    step_duration_min: int = 10,
) -> Dict[float, Dict[str, float]]:
    """
    Compute mean and median warning lead time for each decision threshold
    and report the trade-off against FAR (Table 16, Section 4.1.2,
    Reviewer 1 Comment 7).

    Lead time for event e at threshold τ:
        t_lead(e, τ) = time from first window where P(anomaly) ≥ τ
                       to the confirmed anomaly onset of event e.

    Parameters
    ----------
    y_true : (N,) int
        Ground-truth window labels (1 = anomaly).
    y_prob : (N,) float
        Model anomaly probability per window.
    event_ids : (N,) array-like
        Event identifier for each window (used to group windows by event
        and find onset time).  Pass integer indices or string event IDs.
    thresholds : list of float
        Decision thresholds to evaluate (default [0.3, 0.5, 0.7]).
    stride_steps : int
        Stride in SCADA steps between consecutive windows.
    step_duration_min : int
        Duration of one SCADA step in minutes (default 10).

    Returns
    -------
    results : dict keyed by threshold, each value a dict with:
        mean_lead_time_steps   : float
        median_lead_time_steps : float
        mean_lead_time_hours   : float
        median_lead_time_hours : float
        far_pct                : float
        n_events_detected      : int
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    event_ids = np.asarray(event_ids)
    results: Dict[float, Dict[str, float]] = {}

    for tau in thresholds:
        y_pred_tau = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_tau).ravel()
        far_pct = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0

        # Find anomaly events and compute lead time per event
        lead_times_steps: List[float] = []
        unique_events = np.unique(event_ids)

        for eid in unique_events:
            mask = event_ids == eid
            ev_true = y_true[mask]
            ev_prob = y_prob[mask]

            # Skip purely normal events
            if ev_true.max() == 0:
                continue

            # Anomaly onset: first window in this event labelled anomaly
            onset_idx = int(np.argmax(ev_true))

            # First alert: first window where model fires at threshold τ
            # (must come before or at onset to count as early warning)
            alert_indices = np.where(ev_prob >= tau)[0]
            if len(alert_indices) == 0:
                continue  # missed detection at this threshold

            first_alert = int(alert_indices[0])

            # Lead time in windows (positive = early warning)
            lead_windows = onset_idx - first_alert
            lead_times_steps.append(lead_windows * stride_steps)

        if len(lead_times_steps) == 0:
            results[tau] = {
                "mean_lead_time_steps":   float("nan"),
                "median_lead_time_steps": float("nan"),
                "mean_lead_time_hours":   float("nan"),
                "median_lead_time_hours": float("nan"),
                "far_pct":                far_pct,
                "n_events_detected":      0,
            }
            continue

        lt = np.array(lead_times_steps, dtype=float)
        step_to_hour = step_duration_min / 60.0

        results[tau] = {
            "mean_lead_time_steps":   float(lt.mean()),
            "median_lead_time_steps": float(np.median(lt)),
            "mean_lead_time_hours":   float(lt.mean() * step_to_hour),
            "median_lead_time_hours": float(np.median(lt) * step_to_hour),
            "far_pct":                far_pct,
            "n_events_detected":      len(lead_times_steps),
        }

    return results


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