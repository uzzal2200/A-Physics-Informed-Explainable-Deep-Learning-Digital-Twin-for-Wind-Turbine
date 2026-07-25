"""
scripts/evaluate_all.py
────────────────────────
Load saved checkpoints for all nine models and produce
Table 8 (main results) + McNemar's test comparisons.

Usage
-----
python scripts/evaluate_all.py --config config/config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import CAREPreprocessor, make_dataloaders, load_config
from models.pi_ctba_net import PICtbaNet
from models.baselines import build_baseline, BASELINE_REGISTRY
from training.evaluate import evaluate_model, mcnemar_test, get_predictions
from utils.utils import set_seed, get_device, setup_logging, save_results


ALL_MODELS = list(BASELINE_REGISTRY.keys()) + ["pi_ctba_net"]


def load_model(name: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    num_feat = cfg["model"]["num_features"]
    ckpt_dir = Path(cfg["data"]["checkpoints_dir"])
    ckpt_path = ckpt_dir / f"{name}_best.pt"

    if name == "pi_ctba_net":
        model = PICtbaNet(num_features=num_feat, config=cfg["model"])
    else:
        model = build_baseline(name, num_features=num_feat)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logging.getLogger(__name__).info(f"Loaded: {ckpt_path}")
    else:
        logging.getLogger(__name__).warning(
            f"Checkpoint not found: {ckpt_path}. Evaluating with random weights."
        )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    setup_logging(log_file="results/evaluate_all.log")
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])

    # ── Preprocess ───────────────────────────────────────────────────────
    preprocessor = CAREPreprocessor(cfg)
    _, splits = preprocessor.run()
    loaders = make_dataloaders(splits, cfg["training"]["batch_size"])

    eval_cfg = cfg["evaluation"]
    rows = []
    predictions = {}

    # ── Evaluate each model ──────────────────────────────────────────────
    for name in ALL_MODELS:
        logger.info(f"Evaluating: {name}")
        model = load_model(name, cfg, device)
        res = evaluate_model(
            model, loaders["test"], device,
            threshold=eval_cfg["threshold"],
            n_bootstrap=eval_cfg["bootstrap_n"],
        )
        m = res["metrics"]
        ci = res["bootstrap_ci"]
        predictions[name] = {
            "y_true": res["y_true"],
            "y_pred": res["y_pred"],
        }
        rows.append({
            "Model":     name,
            "Acc (%)":   round(m["accuracy"],  2),
            "Prec (%)":  round(m["precision"], 2),
            "Rec (%)":   round(m["recall"],    2),
            "F1 (%)":    round(m["f1"],        2),
            "AUC":       round(m["auc_roc"],   4),
            "MCC":       round(m["mcc"],       4),
            "FAR (%)":   round(m["far"],       2),
            "F1 CI":     f"[{ci['f1'][0]:.2f}, {ci['f1'][1]:.2f}]",
            "AUC CI":    f"[{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]",
        })

    # ── Table 8 ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    logger.info("\n" + df.to_string(index=False))
    df.to_csv("results/table8_main_results.csv", index=False)
    logger.info("Table 8 saved: results/table8_main_results.csv")

    # ── McNemar's test: PI-CTBA-Net vs all baselines ─────────────────────
    pi_true = predictions["pi_ctba_net"]["y_true"]
    pi_pred = predictions["pi_ctba_net"]["y_pred"]

    mcnemar_rows = []
    for name in BASELINE_REGISTRY.keys():
        if name not in predictions:
            continue
        bl_pred = predictions[name]["y_pred"]
        result = mcnemar_test(pi_true, pi_pred, bl_pred, eval_cfg["mcnemar_alpha"])
        mcnemar_rows.append({
            "Baseline": name,
            "b":        result["b"],
            "c":        result["c"],
            "chi2":     round(result["chi2"], 2),
            "p_value":  result["p_value"],
            "Significant": result["significant"],
        })

    df_mc = pd.DataFrame(mcnemar_rows)
    logger.info("\nMcNemar's Test Results:\n" + df_mc.to_string(index=False))
    df_mc.to_csv("results/mcnemar_results.csv", index=False)
    logger.info("McNemar results saved: results/mcnemar_results.csv")


if __name__ == "__main__":
    main()
