"""
scripts/evaluate_all.py
────────────────────────
Load saved checkpoints for all nine models and produce the
full set of evaluation tables from the revised manuscript
(APEN-D-26-18228).

Tables produced
---------------
Table 14 (originally Table 8):
    Main results — optimistic baseline (window-level split).
Table 11:
    Event-level split results — strict leakage-free evaluation
    (Reviewer 2 Comment 2, Section 3.3.2).
Table 15:
    Operational FAR — per-turbine-per-day false alarm rate
    (Reviewer 1 Comment 6, Section 4.1).
Table 16:
    Early-warning lead time at thresholds 0.3 / 0.5 / 0.7
    (Reviewer 1 Comment 7, Section 4.1.2).
Table 24:
    Raw-unit physics loss ablation — ΔF1 for L1/L3 de-normalized
    (Reviewer 3 Comment 7, Section 5.5).
Tables 25–26:
    Physics weight sensitivity analysis — 5 lambda presets,
    F1 and GradientSHAP top-3 feature rankings
    (Reviewer 3 Comment 8, Section 5.6).
McNemar CSV:
    PI-CTBA-Net vs all baselines, χ² and p-values.

Usage
-----
python scripts/evaluate_all.py --config config/config.yaml

# Skip the event-level re-evaluation (much faster, omits Table 11)
python scripts/evaluate_all.py --skip_event_level

# Skip weight sensitivity (omits Tables 25–26)
python scripts/evaluate_all.py --skip_sensitivity
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import (
    CAREPreprocessor,
    make_dataloaders,
    load_config,
    compute_class_distribution,
)
from models.pi_ctba_net import PICtbaNet
from models.baselines import build_baseline, BASELINE_REGISTRY
from models.physics_loss import WEIGHT_CONFIGS, NormStats, PhysicsInformedLossRawUnit
from training.trainer import Trainer
from training.evaluate import (
    evaluate_model,
    get_predictions,
    mcnemar_test,
    compute_operational_far,
    compute_lead_time,
)
from utils.utils import set_seed, get_device, setup_logging, save_results

logger = logging.getLogger(__name__)

ALL_MODELS = list(BASELINE_REGISTRY.keys()) + ["pi_ctba_net"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(
    name: str,
    cfg: dict,
    device: torch.device,
    ckpt_suffix: str = "",
) -> torch.nn.Module:
    """
    Load a model from its best checkpoint.
    ckpt_suffix lets weight-preset runs use separate checkpoints,
    e.g. 'pi_ctba_net_no_physics_best.pt'.
    """
    num_feat = cfg["model"]["num_features"]
    ckpt_dir = Path(cfg["data"]["checkpoints_dir"])
    ckpt_name = f"{name}{ckpt_suffix}_best.pt"
    ckpt_path = ckpt_dir / ckpt_name

    if name == "pi_ctba_net":
        model = PICtbaNet(num_features=num_feat, config=cfg["model"])
    else:
        model = build_baseline(name, num_features=num_feat)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"  Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(
            f"  Checkpoint not found: {ckpt_path}. "
            "Evaluating with random weights — results will be meaningless."
        )
    return model.to(device)


def eval_row(
    name: str,
    cfg: dict,
    device: torch.device,
    loaders: dict,
    eval_cfg: dict,
    ckpt_suffix: str = "",
    split_label: str = "optimistic",
) -> tuple[dict, dict]:
    """
    Evaluate one model; return (metrics_row_dict, predictions_dict).
    """
    model = load_model(name, cfg, device, ckpt_suffix)
    res = evaluate_model(
        model, loaders["test"], device,
        threshold=eval_cfg["threshold"],
        n_bootstrap=eval_cfg["bootstrap_n"],
        bootstrap_seed=eval_cfg.get("bootstrap_seed", 42),
    )
    m  = res["metrics"]
    ci = res["bootstrap_ci"]

    row = {
        "Model":       name,
        "Split":       split_label,
        "Acc (%)":     round(m["accuracy"],  2),
        "Prec (%)":    round(m["precision"], 2),
        "Rec (%)":     round(m["recall"],    2),
        "F1 (%)":      round(m["f1"],        2),
        "AUC":         round(m["auc_roc"],   4),
        "MCC":         round(m["mcc"],       4),
        "FAR (%)":     round(m["far"],       2),
        "Brier":       round(m["brier_score"], 4),
        "ECE":         round(m["ece"],       4),
        "F1 CI":       f"[{ci['f1'][0]:.2f}, {ci['f1'][1]:.2f}]",
        "AUC CI":      f"[{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]",
    }
    preds = {
        "y_true": res["y_true"],
        "y_pred": res["y_pred"],
        "y_prob": res["y_prob"],
    }
    return row, preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--skip_event_level", action="store_true",
        help="Skip event-level re-evaluation (Table 11). Saves time.",
    )
    parser.add_argument(
        "--skip_sensitivity", action="store_true",
        help="Skip weight sensitivity analysis (Tables 25–26).",
    )
    parser.add_argument(
        "--skip_raw_physics", action="store_true",
        help="Skip raw-unit physics ablation (Table 24).",
    )
    args = parser.parse_args()

    setup_logging(log_file="results/evaluate_all.log")
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    eval_cfg = cfg["evaluation"]
    Path("results").mkdir(exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 14 — Optimistic baseline (window-level split, original protocol)
    # NOTE: labelled "optimistic baseline" per Reviewer 2 Comment 2 (the
    #       original window-level split has ~21% cross-window overlap).
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("TABLE 14 — Optimistic baseline (window-level split)")
    logger.info("=" * 65)

    preprocessor = CAREPreprocessor(cfg)
    df_opt, splits_opt = preprocessor.run()
    loaders_opt = make_dataloaders(splits_opt, cfg["training"]["batch_size"])

    rows_opt, preds_opt = [], {}
    for name in ALL_MODELS:
        logger.info(f"  {name}")
        row, preds = eval_row(
            name, cfg, device, loaders_opt, eval_cfg,
            split_label="optimistic (window-level)",
        )
        rows_opt.append(row)
        preds_opt[name] = preds

    df_t14 = pd.DataFrame(rows_opt)
    logger.info("\n" + df_t14.to_string(index=False))
    df_t14.to_csv("results/table14_optimistic_results.csv", index=False)
    logger.info("→ results/table14_optimistic_results.csv")

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 11 — Strict event-level split
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_event_level:
        logger.info("=" * 65)
        logger.info("TABLE 11 — Event-level split (leakage-free)")
        logger.info("=" * 65)

        df_ev, splits_ev = preprocessor.run_event_level()
        loaders_ev = make_dataloaders(splits_ev, cfg["training"]["batch_size"])

        # Class distribution
        dist = compute_class_distribution(df_ev)
        logger.info(f"Class distribution (event-level):\n{dist}")

        rows_ev, preds_ev = [], {}
        for name in ALL_MODELS:
            logger.info(f"  {name}")
            row, preds = eval_row(
                name, cfg, device, loaders_ev, eval_cfg,
                ckpt_suffix="_event_level",
                split_label="event-level",
            )
            rows_ev.append(row)
            preds_ev[name] = preds

        df_t11 = pd.DataFrame(rows_ev)
        logger.info("\n" + df_t11.to_string(index=False))
        df_t11.to_csv("results/table11_event_level_results.csv", index=False)
        logger.info("→ results/table11_event_level_results.csv")
    else:
        logger.info("Skipping Table 11 (--skip_event_level).")
        preds_ev = None

    # ════════════════════════════════════════════════════════════════════════
    # McNemar's test — PI-CTBA-Net vs all baselines (both splits)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("McNemar's Test — PI-CTBA-Net vs baselines")
    logger.info("=" * 65)

    alpha = eval_cfg.get("mcnemar_alpha", 0.05)

    def _mcnemar_table(preds: dict, label: str) -> pd.DataFrame:
        pi_true = preds["pi_ctba_net"]["y_true"]
        pi_pred = preds["pi_ctba_net"]["y_pred"]
        rows = []
        for name in BASELINE_REGISTRY.keys():
            if name not in preds:
                continue
            r = mcnemar_test(pi_true, pi_pred, preds[name]["y_pred"], alpha)
            rows.append({
                "Baseline":    name,
                "Split":       label,
                "b":           r["b"],
                "c":           r["c"],
                "chi2":        round(r["chi2"], 3),
                "p_value":     round(r["p_value"], 4),
                "Significant": r["significant"],
            })
        return pd.DataFrame(rows)

    df_mc_opt = _mcnemar_table(preds_opt, "optimistic")
    logger.info("\nOptimistic split:\n" + df_mc_opt.to_string(index=False))

    mc_frames = [df_mc_opt]
    if preds_ev:
        df_mc_ev = _mcnemar_table(preds_ev, "event-level")
        logger.info("\nEvent-level split:\n" + df_mc_ev.to_string(index=False))
        mc_frames.append(df_mc_ev)

    df_mc_all = pd.concat(mc_frames, ignore_index=True)
    df_mc_all.to_csv("results/mcnemar_results.csv", index=False)
    logger.info("→ results/mcnemar_results.csv")

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 15 — Operational FAR (per-turbine-per-day)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("TABLE 15 — Operational FAR")
    logger.info("=" * 65)

    op_far_rows = []
    for row in rows_opt:
        op = compute_operational_far(
            far_pct=row["FAR (%)"],
            window_size_steps=cfg["data"].get("window_size", 144),
            stride_steps=cfg["data"].get("stride", 72),
        )
        op_far_rows.append({
            "Model":                        row["Model"],
            "FAR (%)":                      row["FAR (%)"],
            "Windows/turbine/day":          op["windows_per_turbine_per_day"],
            "FA/turbine/day":               op["fa_per_turbine_per_day"],
            "Days between false alarms":    op["days_between_false_alarms"],
        })

    df_t15 = pd.DataFrame(op_far_rows)
    logger.info("\n" + df_t15.to_string(index=False))
    df_t15.to_csv("results/table15_operational_far.csv", index=False)
    logger.info("→ results/table15_operational_far.csv")

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 16 — Early-warning lead time
    # ════════════════════════════════════════════════════════════════════════
    test_meta = splits_opt.get("test_meta", {})
    if "event_ids" in test_meta:
        logger.info("=" * 65)
        logger.info("TABLE 16 — Early-warning lead time (PI-CTBA-Net)")
        logger.info("=" * 65)

        lt = compute_lead_time(
            y_true=preds_opt["pi_ctba_net"]["y_true"],
            y_prob=preds_opt["pi_ctba_net"]["y_prob"],
            event_ids=test_meta["event_ids"],
            thresholds=[0.3, 0.5, 0.7],
            stride_steps=cfg["data"].get("stride", 72),
        )
        lt_rows = []
        for tau, v in lt.items():
            lt_rows.append({
                "Threshold":           tau,
                "Mean lead time (h)":  round(v["mean_lead_time_hours"], 2),
                "Median lead time (h)":round(v["median_lead_time_hours"], 2),
                "FAR (%)":             round(v["far_pct"], 2),
                "Events detected":     v["n_events_detected"],
            })
        df_t16 = pd.DataFrame(lt_rows)
        logger.info("\n" + df_t16.to_string(index=False))
        df_t16.to_csv("results/table16_lead_time.csv", index=False)
        logger.info("→ results/table16_lead_time.csv")
    else:
        logger.info(
            "Skipping Table 16: no event_ids in test_meta. "
            "Run with preprocessing that propagates event metadata."
        )

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 24 — Raw-unit physics loss ablation (PI-CTBA-Net only)
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_raw_physics and "norm_stats" in cfg:
        logger.info("=" * 65)
        logger.info("TABLE 24 — Raw-unit physics ablation")
        logger.info("=" * 65)

        # Load PI-CTBA-Net with raw-physics checkpoint
        row_raw, _ = eval_row(
            "pi_ctba_net", cfg, device, loaders_opt, eval_cfg,
            ckpt_suffix="_raw_physics",
            split_label="raw-unit physics (ablation)",
        )
        # Original result for comparison
        row_orig = next(r for r in rows_opt if r["Model"] == "pi_ctba_net")

        df_t24 = pd.DataFrame([row_orig, row_raw])
        delta_f1 = round(row_raw["F1 (%)"] - row_orig["F1 (%)"], 2)
        logger.info("\n" + df_t24.to_string(index=False))
        logger.info(f"  ΔF1 (raw-unit − z-score): {delta_f1:+.2f} pp")
        df_t24.to_csv("results/table24_raw_unit_ablation.csv", index=False)
        logger.info("→ results/table24_raw_unit_ablation.csv")
    else:
        if args.skip_raw_physics:
            logger.info("Skipping Table 24 (--skip_raw_physics).")
        else:
            logger.info("Skipping Table 24: 'norm_stats' not in config.")

    # ════════════════════════════════════════════════════════════════════════
    # TABLES 25–26 — Physics weight sensitivity analysis
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_sensitivity:
        logger.info("=" * 65)
        logger.info("TABLES 25–26 — Physics weight sensitivity")
        logger.info("=" * 65)

        sensitivity_rows = []
        for preset_name, lambdas in WEIGHT_CONFIGS.items():
            logger.info(f"  Preset: {preset_name}  lambdas={lambdas}")
            row_s, _ = eval_row(
                "pi_ctba_net", cfg, device, loaders_opt, eval_cfg,
                ckpt_suffix=f"_{preset_name}",
                split_label=f"weight-{preset_name}",
            )
            sensitivity_rows.append({
                "Preset":          preset_name,
                "λ_betz":          lambdas["betz"],
                "λ_tsr":           lambdas["tsr"],
                "λ_thermal":       lambdas["thermal"],
                "λ_power_factor":  lambdas["power_factor"],
                "λ_pitch":         lambdas["pitch"],
                "F1 (%)":          row_s["F1 (%)"],
                "F1 CI":           row_s["F1 CI"],
                "AUC":             row_s["AUC"],
                "FAR (%)":         row_s["FAR (%)"],
                "MCC":             row_s["MCC"],
            })

        df_t25 = pd.DataFrame(sensitivity_rows)
        logger.info("\nTable 25 — Weight sensitivity metrics:\n" + df_t25.to_string(index=False))
        df_t25.to_csv("results/table25_weight_sensitivity.csv", index=False)
        logger.info("→ results/table25_weight_sensitivity.csv")

        # Table 26 note: GradientSHAP top-3 rankings are populated
        # by shap_analysis.py using the trained checkpoints for each preset.
        # This script records the metric columns; the SHAP columns are
        # merged in post-processing.
        logger.info(
            "Table 26 (SHAP top-3 rankings) requires shap_analysis.py "
            "to be run separately for each weight preset checkpoint."
        )
    else:
        logger.info("Skipping Tables 25–26 (--skip_sensitivity).")

    logger.info("\n✓ evaluate_all complete — see results/ for all CSVs.")


if __name__ == "__main__":
    main()