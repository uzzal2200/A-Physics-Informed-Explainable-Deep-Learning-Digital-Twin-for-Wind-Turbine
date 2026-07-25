"""
models/physics_loss.py
──────────────────────
Five physics-informed auxiliary loss terms for PI-CTBA-Net.

L1  Betz power curve consistency      (λ₁ = 0.10)
L2  Tip speed ratio (TSR) stability   (λ₂ = 0.05)
L3  Newton's law of cooling           (λ₃ = 0.05)
L4  Electrical power factor           (λ₄ = 0.05)
L5  Pitch angle smoothness            (λ₅ = 0.05)

All losses operate on z-score normalised inputs (per-farm).

Reference:
  Debnath et al. (2026), "A Physics-Informed Explainable Deep Learning
  Digital Twin for Wind Turbine System Identification, Condition
  Monitoring, and Predictive Maintenance", Mathematics (MDPI).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Feature index mapping (matches ALL_MODEL_FEATURES order)
IDX = {
    "wind_speed":      0,
    "active_power":    1,
    "reactive_power":  2,
    "ambient_temp":    3,
    "grid_frequency":  4,
    "rotor_speed":     5,
    "gearbox_oil_temp":6,
    "generator_temp":  7,
    "pitch_angle":     8,
    "hour":            9,
    "day":            10,
    "month":          11,
}


class BetzPowerCurveLoss(nn.Module):
    """
    L1 — Betz power curve consistency.

    Enforces the cubic relationship P ∝ V³:
        L1 = mean( (P̃ - Ṽ³)² )
    where P̃ = P/σ(P) and Ṽ³ = V³/σ(V³) are scale-normalised.

    Note: computed on z-score normalised inputs; enforces relative
    proportionality rather than absolute Betz compliance.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, F)
            Batch of windows.
        """
        P = x[:, :, IDX["active_power"]]    # (B, T)
        V = x[:, :, IDX["wind_speed"]]      # (B, T)

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
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        omega = x[:, :, IDX["rotor_speed"]]   # (B, T)
        V     = x[:, :, IDX["wind_speed"]]    # (B, T)

        tsr_proxy = omega / (V.abs() + self.eps)   # (B, T)
        # Variance over timesteps, mean over batch
        return tsr_proxy.var(dim=1).mean()


class NewtonCoolingLoss(nn.Module):
    """
    L3 — Newton's law of cooling consistency.

    Enforces stationarity of ΔT = T_gearbox - T_ambient:
        L3 = mean( (ΔTᵢ - mean(ΔT))² )

    A rising ΔT above the steady-state baseline signals thermal runaway.
    """

    def forward(self, x: Tensor) -> Tensor:
        T_gb  = x[:, :, IDX["gearbox_oil_temp"]]   # (B, T)
        T_amb = x[:, :, IDX["ambient_temp"]]        # (B, T)

        delta_T      = T_gb - T_amb                 # (B, T)
        delta_T_mean = delta_T.mean(dim=1, keepdim=True)

        return ((delta_T - delta_T_mean) ** 2).mean()


class PowerFactorLoss(nn.Module):
    """
    L4 — Electrical power factor constraint.

    Grid codes require cos φ ≥ 0.9.
        L4 = mean( (|P| / √(P² + Q² + ε) − 0.9)² )

    Note: symmetric squared-error form; one-sided hinge recommended
    for future work: max(0, 0.9 - PF)².
    """

    def __init__(self, target_pf: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.target_pf = target_pf
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        P = x[:, :, IDX["active_power"]]     # (B, T)
        Q = x[:, :, IDX["reactive_power"]]   # (B, T)

        S  = (P ** 2 + Q ** 2 + self.eps).sqrt()
        PF = P.abs() / S

        return ((PF - self.target_pf) ** 2).mean()


class PitchSmoothnessLoss(nn.Module):
    """
    L5 — Pitch angle smoothness.

    Penalises rapid pitch angle changes (actuator faults):
        L5 = mean_t( (β(t) - β(t-1))² )
    """

    def forward(self, x: Tensor) -> Tensor:
        beta = x[:, :, IDX["pitch_angle"]]   # (B, T)
        diff = beta[:, 1:] - beta[:, :-1]    # (B, T-1)
        return (diff ** 2).mean()


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed auxiliary loss.

    L_total = L_BCE + Σⱼ λⱼ Lⱼ

    Parameters
    ----------
    lambdas : dict
        Loss weights keyed by
        {'betz', 'tsr', 'thermal', 'power_factor', 'pitch'}.
    pos_weight : float
        BCE positive class weight (N_normal / N_anomaly ≈ 1.105).
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
        l_bce     = self.bce(logits, y)
        l1        = self.L1(x)
        l2        = self.L2(x)
        l3        = self.L3(x)
        l4        = self.L4(x)
        l5        = self.L5(x)

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


# Type hint alias
from typing import Tuple   # noqa: E402 (used above)
