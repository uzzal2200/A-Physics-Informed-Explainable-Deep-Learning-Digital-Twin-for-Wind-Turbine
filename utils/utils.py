"""
utils/utils.py
──────────────
General utility functions for PI-CTBA-Net experiments.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "cuda") -> torch.device:
    """Return available device (falls back to CPU if CUDA unavailable)."""
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger(__name__)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).info("Using CPU.")
    return device


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure root logger."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: Dict[str, Any], path: str):
    """Save results dict to JSON."""
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serialiser)
    logging.getLogger(__name__).info(f"Results saved: {path}")


def _json_serialiser(obj):
    """Handle numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} not JSON serialisable")


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """Print parameter count summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"{'='*50}\n")
