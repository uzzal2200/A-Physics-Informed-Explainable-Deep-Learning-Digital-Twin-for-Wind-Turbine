"""
models/physics_loss.py
──────────────────────
Five physics-informed auxiliary loss terms for PI-CTBA-Net.

L1  Betz power curve consistency      (λ₁ = 0.10)
L2  Tip speed ratio (TSR) stability   (λ₂ = 0.05)
L3  Newton's law of cooling           (λ₃ = 0.05)
L4  Electrical power factor           (λ₄ = 0.05)
L5  Pitch angle smoothness            (λ₅ = 0.05)

All losses operate on z-score normalised inputs (per-farm) unless the
*raw-unit* variant is used (see PhysicsInformedLossRawUnit).

Reviewer-added content (revision APEN-D-26-18228):
  - WEIGHT_CONFIGS  : five λ presets for sensitivity analysis (Sec. 5.6,
                      Table 25).  Keys: 'no_physics', 'half', 'original',
                      'double', 'equal'.
  - NormStats       : dataclass carrying per-feature μ/σ for de-
                      normalisation inside raw-unit losses.
  - BetzPowerCurveLossRaw  : L1 computed on physical units (Sec. 5.5).
  - NewtonCoolingLossRaw   : L3 computed on physical units (Sec. 5.5).
  - PhysicsInformedLossRawUnit : combined loss using raw-unit L1 and L3,
                      normalised L2/L4/L5 (Table 24).

Reference:
  Debnath et al. (2026), "A Physics-Informed Explainable Deep Learning
  Digital Twin Component for Wind Turbine Anomaly Detection, Condition
  Monitoring, and Maintenance Decision Support", Applied Energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ── Feature index mapping (matches ALL_MODEL_FEATURES order) ─────────────────
IDX = {
    "wind_speed":       0,
    "active_power":     1,
    "reactive_power":   2,
    "ambient_temp":     3,
    "grid_frequency":   4,
    "rotor_speed":      5,
    "gearbox_oil_temp": 6,
    "generator_temp":   7,
    "pitch_angle":      8,
    "hour":             9,
    "day":             10,
    "month":           11,
}

# ── Sensitivity analysis weight configurations (Section 5.6, Table 25) ───────
WEIGHT_CONFIGS: Dict[str, Dict[str, float]] = {
    # No physics auxiliary losses (BCE only)
    "no_physics": {
        "betz": 0.00, "tsr": 0.00, "thermal": 0.00,
        "power_factor": 0.00, "pitch": 0.00,
    },
    # Half the manuscript weights
    "half": {
        "betz": 0.05, "tsr": 0.025, "thermal": 0.025,
        "power_factor": 0.025, "pitch": 0.025,
    },
    # Manuscript original weights (Table 8)
    "original": {
        "betz": 0.10, "tsr": 0.05, "thermal": 0.05,
        "power_factor": 0.05, "pitch": 0.05,
    },
    # Double the manuscript weights
    "double": {
        "betz": 0.20, "tsr": 0.10, "thermal": 0.10,
        "power_factor": 0.10, "pitch": 0.10,
    },
    # Equal weights for all five terms
    "equal": {
        "betz": 0.06, "tsr": 0.06, "thermal": 0.06,
        "power_factor": 0.06, "pitch": 0.06,
    },
}


# ── Normalisation statistics for raw-unit ablation (Section 5.5) ─────────────

@dataclass
class NormStats:
    """
    Per-feature means and standard deviations from per-farm z-score
    normalisation.  Pass to PhysicsInformedLossRawUnit so that L1
    (Betz) and L3 (Newton's cooling) can be evaluated on physical units.

    Attributes
    ----------
    wind_speed_mu, wind_speed_sigma  : float — m/s
    active_power_mu, active_power_sigma : float — kW
    gearbox_oil_temp_mu, gearbox_oil_temp_sigma : float — °C
    ambient_temp_mu, ambient_temp_sigma : float — °C
    """
    wind_speed_mu:        float = 0.0
    wind_speed_sigma:     float = 1.0
    active_power_mu:      float = 0.0
    active_power_sigma:   float = 1.0
    gearbox_oil_temp_mu:  float = 0.0
    gearbox_oil_temp_sigma: float = 1.0
    ambient_temp_mu:      float = 0.0
    ambient_temp_sigma:   float = 1.0

    def denorm_wind_speed(self, v: Tensor) -> Tensor:
        return v * self.wind_speed_sigma + self.wind_speed_mu

    def denorm_active_power(self, p: Tensor) -> Tensor:
        return p * self.active_power_sigma + self.active_power_mu

    def denorm_gearbox_temp(self, t: Tensor) -> Tensor:
        return t * self.gearbox_oil_temp_sigma + self.gearbox_oil_temp_mu

    def denorm_ambient_temp(self, t: Tensor) -> Tensor:
        return t * self.ambient_temp_sigma + self.ambient_temp_mu


# ── Normalised-input physics losses (original formulation) ───────────────────

class BetzPowerCurveLoss(nn.Module):
    """
    L1 — Betz power curve consistency (normalised-input variant).

    Enforces the cubic relationship P ∝ V³:
        L1 = mean( (P̃ - Ṽ³)² )
    where P̃ = P/σ(P) and Ṽ³ = V³/σ(V³) are scale-normalised within
    each batch window.

    Note: computed on z-score normalised inputs; enforces relative
    proportionality rather than absolute Betz compliance.  For the
    raw-unit formulation see BetzPowerCurveLossRaw.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, T, F)"""
        P = x[:, :, IDX["active_power"]]
        V = x[:, :, IDX["wind_speed"]]

        V3 = V ** 3

        sigma_P  = P.std(dim=1, keepdim=True).clamp(min=self.eps)
        sigma_V3 = V3.std(dim=1, keepdim=True).clamp(min=self.eps)

        P_tilde  = P  / sigma_P
        V3_tilde = V3 / sigma_V3

        return ((P_tilde - V3_tilde) ** 2).mean()


class TSRStabilityLoss(nn.Module):
    """
    L2 — Tip speed ratio (TSR) stability.

    Penalises high temporal variance in the TSR proxy ω_r / |V|:
        L2 = E[ Var_t( ω_r(t) / (|V(t)| + ε) ) ]

    High variance → rotor control instability.
    TSR is dimensionless; this loss is identical in normalised and raw
    units, so no raw-unit variant is needed.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        omega = x[:, :, IDX["rotor_speed"]]
        V     = x[:, :, IDX["wind_speed"]]

        tsr_proxy = omega / (V.abs() + self.eps)
        return tsr_proxy.var(dim=1).mean()


class NewtonCoolingLoss(nn.Module):
    """
    L3 — Newton's law of cooling consistency (normalised-input variant).

    Enforces stationarity of ΔT = T_gearbox − T_ambient:
        L3 = mean( (ΔTᵢ − mean(ΔT))² )

    A rising ΔT above the steady-state baseline signals thermal runaway.
    For the raw-unit formulation see NewtonCoolingLossRaw.
    """

    def forward(self, x: Tensor) -> Tensor:
        T_gb  = x[:, :, IDX["gearbox_oil_temp"]]
        T_amb = x[:, :, IDX["ambient_temp"]]

        delta_T      = T_gb - T_amb
        delta_T_mean = delta_T.mean(dim=1, keepdim=True)

        return ((delta_T - delta_T_mean) ** 2).mean()


class PowerFactorLoss(nn.Module):
    """
    L4 — Electrical power factor constraint.

    Grid codes require cos φ ≥ 0.9.
        L4 = mean( (|P| / √(P² + Q² + ε) − 0.9)² )

    PF = P/S is dimensionless; this loss is identical in normalised and
    raw units, so no raw-unit variant is needed.
    """

    def __init__(self, target_pf: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.target_pf = target_pf
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        P = x[:, :, IDX["active_power"]]
        Q = x[:, :, IDX["reactive_power"]]

        S  = (P ** 2 + Q ** 2 + self.eps).sqrt()
        PF = P.abs() / S

        return ((PF - self.target_pf) ** 2).mean()


class PitchSmoothnessLoss(nn.Module):
    """
    L5 — Pitch angle smoothness.

    Penalises rapid pitch angle changes (actuator faults):
        L5 = mean_t( (β(t) − β(t-1))² )

    Pitch change rate is dimensionless under z-score scaling; no raw-unit
    variant is needed.
    """

    def forward(self, x: Tensor) -> Tensor:
        beta = x[:, :, IDX["pitch_angle"]]
        diff = beta[:, 1:] - beta[:, :-1]
        return (diff ** 2).mean()


# ── Raw-unit physics losses (Section 5.5 ablation) ───────────────────────────

class BetzPowerCurveLossRaw(nn.Module):
    """
    L1 — Betz power curve consistency (raw-unit variant, Section 5.5).

    De-normalises V and P using stored per-farm statistics, then enforces
    the cubic relationship on physical units:
        L1_raw = mean( (P̃_phys − Ṽ³_phys)² )
    where P̃_phys = P_phys / σ(P_phys) and Ṽ³_phys = V³_phys / σ(V³_phys).

    Cross-farm scale mismatch risk: Farm A/B/C differ substantially in
    rated capacity (86/257/957 features and different power ratings), so
    this loss term may carry systematically larger magnitude for higher-
    capacity farms.  See Section 5.5 discussion.

    Parameters
    ----------
    norm_stats : NormStats
        Per-farm normalisation statistics.
    eps : float
        Numerical stability constant.
    """

    def __init__(self, norm_stats: NormStats, eps: float = 1e-6):
        super().__init__()
        self.ns  = norm_stats
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, T, F) — z-score normalised"""
        P_phys = self.ns.denorm_active_power(x[:, :, IDX["active_power"]])
        V_phys = self.ns.denorm_wind_speed(x[:, :, IDX["wind_speed"]])

        V3_phys = V_phys ** 3

        sigma_P  = P_phys.std(dim=1, keepdim=True).clamp(min=self.eps)
        sigma_V3 = V3_phys.std(dim=1, keepdim=True).clamp(min=self.eps)

        P_tilde  = P_phys  / sigma_P
        V3_tilde = V3_phys / sigma_V3

        return ((P_tilde - V3_tilde) ** 2).mean()


class NewtonCoolingLossRaw(nn.Module):
    """
    L3 — Newton's law of cooling (raw-unit variant, Section 5.5).

    De-normalises T_gearbox and T_ambient using stored per-farm statistics,
    then enforces thermal stationarity on physical temperature units (°C):
        L3_raw = mean( (ΔT_phys,i − mean(ΔT_phys))² )

    Parameters
    ----------
    norm_stats : NormStats
        Per-farm normalisation statistics.
    """

    def __init__(self, norm_stats: NormStats):
        super().__init__()
        self.ns = norm_stats

    def forward(self, x: Tensor) -> Tensor:
        T_gb_phys  = self.ns.denorm_gearbox_temp(x[:, :, IDX["gearbox_oil_temp"]])
        T_amb_phys = self.ns.denorm_ambient_temp(x[:, :, IDX["ambient_temp"]])

        delta_T      = T_gb_phys - T_amb_phys
        delta_T_mean = delta_T.mean(dim=1, keepdim=True)

        return ((delta_T - delta_T_mean) ** 2).mean()


# ── Combined physics-informed losses ─────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed auxiliary loss (normalised-input variant).

    L_total = L_BCE + Σⱼ λⱼ Lⱼ

    Parameters
    ----------
    lambdas : dict
        Loss weights keyed by
        {'betz', 'tsr', 'thermal', 'power_factor', 'pitch'}.
        Use WEIGHT_CONFIGS['original'] for the manuscript values.
        When lambdas['betz'] == lambdas['tsr'] == … == 0.0, reduces to
        BCE-only training (the 'no_physics' baseline in Table 25).
    pos_weight : float
        BCE positive-class weight (N_normal / N_anomaly ≈ 1.105,
        see Table 8 and Section 3.7.1).
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        lambdas: dict,
        pos_weight: float = 1.105,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.lambda_betz         = lambdas.get("betz",         0.10)
        self.lambda_tsr          = lambdas.get("tsr",          0.05)
        self.lambda_thermal      = lambdas.get("thermal",      0.05)
        self.lambda_power_factor = lambdas.get("power_factor", 0.05)
        self.lambda_pitch        = lambdas.get("pitch",        0.05)

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        self.L1 = BetzPowerCurveLoss(eps)
        self.L2 = TSRStabilityLoss(eps)
        self.L3 = NewtonCoolingLoss()
        self.L4 = PowerFactorLoss(eps=eps)
        self.L5 = PitchSmoothnessLoss()

    def forward(
        self,
        logits: Tensor,   # (B,)  raw classifier output
        y:      Tensor,   # (B,)  binary labels
        x:      Tensor,   # (B, T, F) input windows
    ) -> Tuple[Tensor, dict]:
        """
        Returns
        -------
        total_loss : Tensor (scalar)
        loss_dict  : dict with individual loss values for logging.
        """
        l_bce = self.bce(logits, y)
        l1    = self.L1(x)
        l2    = self.L2(x)
        l3    = self.L3(x)
        l4    = self.L4(x)
        l5    = self.L5(x)

        total = (
            l_bce
            + self.lambda_betz         * l1
            + self.lambda_tsr          * l2
            + self.lambda_thermal      * l3
            + self.lambda_power_factor * l4
            + self.lambda_pitch        * l5
        )

        return total, {
            "loss_total":        total.item(),
            "loss_bce":          l_bce.item(),
            "loss_betz":         l1.item(),
            "loss_tsr":          l2.item(),
            "loss_thermal":      l3.item(),
            "loss_power_factor": l4.item(),
            "loss_pitch":        l5.item(),
        }


class PhysicsInformedLossRawUnit(nn.Module):
    """
    Combined physics-informed auxiliary loss — raw-unit L1/L3 variant
    (Section 5.5 ablation, Table 24).

    L2, L4, L5 are dimensionless and unchanged.
    L1 (Betz) and L3 (Newton's cooling) are computed after de-normalising
    the relevant features to physical units using norm_stats.

    Parameters
    ----------
    lambdas : dict   — same keys as PhysicsInformedLoss.
    norm_stats : NormStats — per-farm normalisation statistics.
    pos_weight : float
    eps : float
    """

    def __init__(
        self,
        lambdas: dict,
        norm_stats: NormStats,
        pos_weight: float = 1.105,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.lambda_betz         = lambdas.get("betz",         0.10)
        self.lambda_tsr          = lambdas.get("tsr",          0.05)
        self.lambda_thermal      = lambdas.get("thermal",      0.05)
        self.lambda_power_factor = lambdas.get("power_factor", 0.05)
        self.lambda_pitch        = lambdas.get("pitch",        0.05)

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # Raw-unit variants for L1 and L3
        self.L1 = BetzPowerCurveLossRaw(norm_stats, eps)
        self.L3 = NewtonCoolingLossRaw(norm_stats)

        # Dimensionless — identical in normalised and raw units
        self.L2 = TSRStabilityLoss(eps)
        self.L4 = PowerFactorLoss(eps=eps)
        self.L5 = PitchSmoothnessLoss()

    def forward(
        self,
        logits: Tensor,
        y:      Tensor,
        x:      Tensor,
    ) -> Tuple[Tensor, dict]:
        l_bce = self.bce(logits, y)
        l1    = self.L1(x)
        l2    = self.L2(x)
        l3    = self.L3(x)
        l4    = self.L4(x)
        l5    = self.L5(x)

        total = (
            l_bce
            + self.lambda_betz         * l1
            + self.lambda_tsr          * l2
            + self.lambda_thermal      * l3
            + self.lambda_power_factor * l4
            + self.lambda_pitch        * l5
        )

        return total, {
            "loss_total":        total.item(),
            "loss_bce":          l_bce.item(),
            "loss_betz_raw":     l1.item(),
            "loss_tsr":          l2.item(),
            "loss_thermal_raw":  l3.item(),
            "loss_power_factor": l4.item(),
            "loss_pitch":        l5.item(),
        }