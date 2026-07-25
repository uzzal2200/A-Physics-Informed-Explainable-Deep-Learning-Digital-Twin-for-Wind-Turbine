"""
scripts/train.py
────────────────
Train PI-CTBA-Net (or any baseline) on the CARE benchmark.

Usage
-----
# Train PI-CTBA-Net (physics loss)
python scripts/train.py --model pi_ctba_net

# Train a baseline
python scripts/train.py --model 1D-CNN
python scripts/train.py --model CNN-TB-Att

# Custom config
python scripts/train.py --model pi_ctba_net --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import CAREPreprocessor, make_dataloaders, load_config
from models.pi_ctba_net import PICtbaNet
from models.baselines import build_baseline, BASELINE_REGISTRY
from training.trainer import Trainer
from training.evaluate import evaluate_model, mcnemar_test
from utils.utils import set_seed, get_device, setup_logging, save_results, print_model_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PI-CTBA-Net or baselines on CARE benchmark."
    )
    parser.add_argument(
        "--model", type=str, default="pi_ctba_net",
        help=f"Model name. Options: pi_ctba_net, {', '.join(BASELINE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device: 'cuda' or 'cpu'."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=f"results/{args.model}_train.log")
    logger = logging.getLogger(__name__)

    # Config
    cfg = load_config(args.config)
    seed = args.seed or cfg["training"]["seed"]
    set_seed(seed)

    device_str = args.device or cfg["training"]["device"]
    device = get_device(device_str)

    # ── 1. Preprocess data ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Step 1: CARE Dataset Preprocessing")
    logger.info("=" * 60)
    preprocessor = CAREPreprocessor(cfg)
    _, splits = preprocessor.run()

    loaders = make_dataloaders(
        splits,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    # ── 2. Build model ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Step 2: Building Model — {args.model}")
    logger.info("=" * 60)

    num_feat = cfg["model"]["num_features"]
    use_physics = False

    if args.model == "pi_ctba_net":
        model = PICtbaNet(num_features=num_feat, config=cfg["model"])
        use_physics = True
    else:
        model = build_baseline(args.model, num_features=num_feat)

    print_model_summary(model, args.model)

    # ── 3. Train ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Step 3: Training")
    logger.info("=" * 60)

    trainer = Trainer(model, cfg, device, use_physics=use_physics)
    history = trainer.fit(
        loaders["train"], loaders["val"],
        checkpoint_dir=cfg["data"]["checkpoints_dir"],
        model_name=args.model,
    )

    # ── 4. Evaluate on test set ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Step 4: Test Set Evaluation")
    logger.info("=" * 60)

    trainer.load_best()
    eval_cfg = cfg["evaluation"]
    results = evaluate_model(
        trainer.model,
        loaders["test"],
        device,
        threshold=eval_cfg["threshold"],
        n_bootstrap=eval_cfg["bootstrap_n"],
        bootstrap_seed=eval_cfg["bootstrap_seed"],
    )

    metrics = results["metrics"]
    ci      = results["bootstrap_ci"]

    logger.info(f"\n{'─'*50}")
    logger.info(f"  Results — {args.model}")
    logger.info(f"{'─'*50}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    logger.info(f"  Precision : {metrics['precision']:.2f}%")
    logger.info(f"  Recall    : {metrics['recall']:.2f}%")
    logger.info(f"  F1        : {metrics['f1']:.2f}%  "
                f"95% CI [{ci['f1'][0]:.2f}, {ci['f1'][1]:.2f}]")
    logger.info(f"  AUC-ROC   : {metrics['auc_roc']:.4f}  "
                f"95% CI [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]")
    logger.info(f"  MCC       : {metrics['mcc']:.4f}")
    logger.info(f"  FAR       : {metrics['far']:.2f}%")
    logger.info(f"  Brier     : {metrics['brier_score']:.4f}")
    logger.info(f"  ECE       : {metrics['ece']:.4f}")
    logger.info(f"  TP={metrics['tp']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TN={metrics['tn']}")
    logger.info(f"{'─'*50}\n")

    # ── 5. Save results ──────────────────────────────────────────────────
    out_path = Path(cfg["data"]["results_dir"]) / f"{args.model}_results.json"
    save_results(
        {
            "model": args.model,
            "seed": seed,
            "metrics": metrics,
            "bootstrap_ci": ci,
            "history": history,
        },
        str(out_path),
    )
    logger.info(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()
