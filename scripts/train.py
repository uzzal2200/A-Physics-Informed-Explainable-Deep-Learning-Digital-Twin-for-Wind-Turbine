"""
scripts/train.py
────────────────
Train PI-CTBA-Net (or any baseline) on the CARE benchmark.

Usage
-----
# Train PI-CTBA-Net with original window-level split (optimistic baseline)
python scripts/train.py --model pi_ctba_net

# Train with strict event-level split (Reviewer 2 Comment 2)
python scripts/train.py --model pi_ctba_net --event_level_split

# Train with a physics weight preset (sensitivity analysis, Sec. 5.6)
python scripts/train.py --model pi_ctba_net --weight_config no_physics
python scripts/train.py --model pi_ctba_net --weight_config half
python scripts/train.py --model pi_ctba_net --weight_config double
python scripts/train.py --model pi_ctba_net --weight_config equal

# Train with raw-unit physics loss (ablation, Sec. 5.5)
python scripts/train.py --model pi_ctba_net --raw_physics

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

from data.preprocessing import (
    CAREPreprocessor,
    make_dataloaders,
    load_config,
    compute_class_distribution,
)
from models.pi_ctba_net import PICtbaNet
from models.baselines import build_baseline, BASELINE_REGISTRY
from models.physics_loss import WEIGHT_CONFIGS
from training.trainer import Trainer
from training.evaluate import (
    evaluate_model,
    compute_operational_far,
    compute_lead_time,
    mcnemar_test,
)
from utils.utils import (
    set_seed,
    get_device,
    setup_logging,
    save_results,
    print_model_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PI-CTBA-Net or baselines on CARE benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="pi_ctba_net",
        help=f"Model name. Options: pi_ctba_net, {', '.join(BASELINE_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device: 'cuda' or 'cpu'.",
    )

    # ── Reviewer additions ────────────────────────────────────────────────
    parser.add_argument(
        "--event_level_split", action="store_true", default=False,
        help=(
            "Use strict event-level train/val/test split instead of "
            "window-level split (Reviewer 2 Comment 2, Sec. 3.3.2). "
            "Eliminates cross-window data leakage (~21%% overlap in original). "
            "Results are labelled 'event-level' vs 'optimistic baseline'."
        ),
    )
    parser.add_argument(
        "--weight_config", type=str, default=None,
        choices=list(WEIGHT_CONFIGS.keys()) + [None],
        help=(
            "Physics weight preset for sensitivity analysis (Sec. 5.6, Tables 25–26). "
            "Options: " + ", ".join(WEIGHT_CONFIGS.keys()) + ". "
            "Overrides config physics_loss lambdas when set. "
            "Ignored for baseline models."
        ),
    )
    parser.add_argument(
        "--raw_physics", action="store_true", default=False,
        help=(
            "Use raw-unit physics loss variant: L1 (Betz) and L3 (Newton's "
            "cooling) are computed after de-normalizing inputs to physical "
            "units (Sec. 5.5, Table 24 ablation). Requires norm_stats in "
            "config. Ignored for baseline models."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Tag run name with experimental settings
    run_tag = args.model
    if args.event_level_split:
        run_tag += "_event_level"
    if args.weight_config:
        run_tag += f"_{args.weight_config}"
    if args.raw_physics:
        run_tag += "_raw_physics"

    setup_logging(log_file=f"results/{run_tag}_train.log")
    logger = logging.getLogger(__name__)

    # Config
    cfg = load_config(args.config)
    seed = args.seed or cfg["training"]["seed"]
    set_seed(seed)

    device_str = args.device or cfg["training"]["device"]
    device = get_device(device_str)

    split_mode = "event-level" if args.event_level_split else "optimistic baseline (window-level)"
    logger.info(f"Run: {run_tag} | split={split_mode} | seed={seed}")

    # ── 1. Preprocess data ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Step 1: CARE Dataset Preprocessing")
    logger.info("=" * 60)

    preprocessor = CAREPreprocessor(cfg)

    if args.event_level_split:
        logger.info("Using strict event-level split (leakage-free).")
        df, splits = preprocessor.run_event_level()
    else:
        logger.info("Using original window-level split (optimistic baseline).")
        df, splits = preprocessor.run()

    # Log class distribution (Table 10)
    dist = compute_class_distribution(df)
    logger.info(f"Class distribution:\n{dist}")

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

    trainer = Trainer(
        model,
        cfg,
        device,
        use_physics=use_physics,
        lambda_override=args.weight_config if use_physics else None,
        raw_physics=args.raw_physics if use_physics else False,
    )

    history = trainer.fit(
        loaders["train"],
        loaders["val"],
        checkpoint_dir=cfg["data"]["checkpoints_dir"],
        model_name=run_tag,
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

    logger.info(f"\n{'─'*54}")
    logger.info(f"  Results — {run_tag}  [{split_mode}]")
    logger.info(f"{'─'*54}")
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
    logger.info(f"{'─'*54}\n")

    # ── 5. Operational FAR (Table 15) ────────────────────────────────────
    op_far = compute_operational_far(
        far_pct=metrics["far"],
        window_size_steps=cfg["data"].get("window_size", 144),
        stride_steps=cfg["data"].get("stride", 72),
    )
    logger.info(
        f"Operational FAR (Table 15): "
        f"{op_far['fa_per_turbine_per_day']:.4f} false alarms/turbine/day  "
        f"(≈1 per {op_far['days_between_false_alarms']:.1f} days)"
    )

    # ── 6. Early-warning lead time (Table 16) ────────────────────────────
    # Only computed when test set has event_id metadata
    if "event_ids" in splits.get("test_meta", {}):
        lead_times = compute_lead_time(
            y_true=results["y_true"],
            y_prob=results["y_prob"],
            event_ids=splits["test_meta"]["event_ids"],
            thresholds=[0.3, 0.5, 0.7],
            stride_steps=cfg["data"].get("stride", 72),
        )
        logger.info("Early-warning lead time (Table 16):")
        for tau, lt in lead_times.items():
            logger.info(
                f"  τ={tau}: mean={lt['mean_lead_time_hours']:.1f}h  "
                f"median={lt['median_lead_time_hours']:.1f}h  "
                f"FAR={lt['far_pct']:.2f}%  "
                f"detected={lt['n_events_detected']}"
            )
    else:
        lead_times = {}

    # ── 7. Save all results ──────────────────────────────────────────────
    out_path = Path(cfg["data"]["results_dir"]) / f"{run_tag}_results.json"
    save_results(
        {
            "run_tag":       run_tag,
            "model":         args.model,
            "split_mode":    split_mode,
            "weight_config": args.weight_config,
            "raw_physics":   args.raw_physics,
            "seed":          seed,
            "metrics":       metrics,
            "bootstrap_ci":  ci,
            "operational_far": op_far,
            "lead_times":    {str(k): v for k, v in lead_times.items()},
            "history":       history,
        },
        str(out_path),
    )
    logger.info(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()