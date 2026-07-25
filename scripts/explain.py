"""
scripts/explain.py
──────────────────
GradientSHAP + Attention weight analysis for PI-CTBA-Net.

Usage
-----
python scripts/explain.py --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import CAREPreprocessor, load_config
from models.pi_ctba_net import PICtbaNet
from explainability.shap_analysis import GradientSHAPAnalyzer, AttentionVisualizer
from utils.utils import set_seed, get_device, setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n_background", type=int, default=200)
    parser.add_argument("--n_test",       type=int, default=500)
    args = parser.parse_args()

    setup_logging(log_file="results/explain.log")
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])

    # ── Load data ────────────────────────────────────────────────────────
    preprocessor = CAREPreprocessor(cfg)
    _, splits = preprocessor.run()

    X_bg   = splits["X_train"][:args.n_background]
    X_test = splits["X_test"][:args.n_test]

    # ── Load model ───────────────────────────────────────────────────────
    ckpt = Path(cfg["data"]["checkpoints_dir"]) / "pi_ctba_net_best.pt"
    model = PICtbaNet(num_features=cfg["model"]["num_features"], config=cfg["model"])
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    logger.info(f"Model loaded from {ckpt}")

    # ── GradientSHAP ─────────────────────────────────────────────────────
    logger.info("Computing GradientSHAP values …")
    analyzer = GradientSHAPAnalyzer(model, device)
    shap_vals = analyzer.compute_shap_values(X_bg, X_test,
                                              seed=cfg["training"]["seed"])

    importance = analyzer.global_feature_importance(shap_vals)
    logger.info("\nGlobal Feature Importance (mean |SHAP|):")
    from data.preprocessing import PHYSICS_FEATURES, TEMPORAL_FEATURES
    all_names = PHYSICS_FEATURES + TEMPORAL_FEATURES
    for name, val in sorted(zip(all_names, importance),
                             key=lambda x: -x[1]):
        logger.info(f"  {name:25s}: {val:.5f}")

    analyzer.plot_global_importance(
        importance,
        save_path="figures/shap_global_importance.png",
        feature_names=all_names,
    )
    np.save("results/shap_values.npy", shap_vals)
    logger.info("SHAP values saved: results/shap_values.npy")

    # ── Attention weights ─────────────────────────────────────────────────
    logger.info("Extracting attention weights …")
    viz = AttentionVisualizer(model)
    attn = viz.get_attention_weights(X_test[:64], device)
    if attn is not None:
        viz.plot_attention_map(attn, save_path="figures/attention_map.png")
        np.save("results/attention_weights.npy", attn)
        logger.info("Attention weights saved: results/attention_weights.npy")
    viz.remove_hook()


if __name__ == "__main__":
    main()
