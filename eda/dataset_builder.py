"""
eda/dataset_builder.py
──────────────────────
Builds the unified main_dataset.csv from raw CARE event CSVs.

Based on: build_main_dataset_v5.ipynb

Steps:
  1. Load event labels from event_info.csv
  2. Load each event CSV → rename sensors → unified features
  3. Fill missing timestamps (10-min grid)
  4. Z-score normalise per farm
  5. Add temporal features (hour, day, month)
  6. Concatenate all farms → save main_dataset.csv

Usage
-----
  from eda.dataset_builder import CAREDatasetBuilder
  builder = CAREDatasetBuilder(root="data/CARE")
  df = builder.build(save_path="data/main_dataset.csv")
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature map (verified by direct CSV inspection) ──────────────────────────
FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "Farm_A": {
        "wind_speed":          "wind_speed_3_avg",
        "active_power":        "power_30_avg",
        "reactive_power":      "react_power_27_avg",
        "ambient_temperature": "sensor_0_avg",
        "grid_frequency":      "sensor_26_avg",
        "rotor_speed":         "sensor_52_avg",
        "gearbox_oil_temp":    "sensor_12_avg",
        "generator_temp":      "sensor_15_avg",
        "pitch_angle":         "sensor_5_avg",
    },
    # ⚠️ Farm B: generator_temp = sensor_32_avg
    # (sensor_199 does NOT exist in Farm B; max sensor index is 57)
    "Farm_B": {
        "wind_speed":          "wind_speed_59_avg",
        "active_power":        "power_62_avg",
        "reactive_power":      "react_power_11_avg",
        "ambient_temperature": "sensor_8_avg",
        "grid_frequency":      "sensor_23_avg",
        "rotor_speed":         "sensor_25_avg",
        "gearbox_oil_temp":    "sensor_39_avg",
        "generator_temp":      "sensor_32_avg",
        "pitch_angle":         "sensor_10_avg",
    },
    "Farm_C": {
        "wind_speed":          "wind_speed_235_avg",
        "active_power":        "power_6_avg",
        "reactive_power":      "react_power_119_avg",
        "ambient_temperature": "sensor_7_avg",
        "grid_frequency":      "sensor_47_avg",
        "rotor_speed":         "sensor_144_avg",
        "gearbox_oil_temp":    "sensor_186_avg",
        "generator_temp":      "sensor_199_avg",
        "pitch_angle":         "sensor_103_avg",
    },
}

FEATURE_COLS: List[str] = [
    "wind_speed", "active_power", "reactive_power",
    "ambient_temperature", "grid_frequency", "rotor_speed",
    "gearbox_oil_temp", "generator_temp", "pitch_angle",
]

THERMAL_COLS: List[str] = [
    "ambient_temperature", "gearbox_oil_temp", "generator_temp"
]


class CAREDatasetBuilder:
    """
    End-to-end builder for the CARE unified dataset.

    Parameters
    ----------
    root : str
        Path to CARE root directory containing Farm_A/, Farm_B/, Farm_C/.
    """

    def __init__(self, root: str = "data/CARE"):
        self.root = Path(root)

    def build(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Build the unified dataset.

        Returns
        -------
        df : pd.DataFrame  shape (5_273_872, 18)
        """
        farm_frames = []
        for farm in ["Farm_A", "Farm_B", "Farm_C"]:
            logger.info(f"\nProcessing {farm} …")
            df_farm = self._process_farm(farm)
            farm_frames.append(df_farm)

        df = pd.concat(farm_frames, ignore_index=True)
        df = df.sort_values(["farm", "event_id", "time_stamp"]).reset_index(drop=True)

        assert df[FEATURE_COLS].isnull().sum().sum() == 0, "Null values found!"
        logger.info(f"\n✅ Unified dataset ready: {df.shape}")
        logger.info(f"   Rows per farm:\n{df['farm'].value_counts().to_string()}")
        logger.info(f"   Label dist:\n{df['event_label'].value_counts().to_string()}")

        if save_path:
            os.makedirs(Path(save_path).parent, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"   Saved → {save_path}")

        return df

    # ── Private helpers ───────────────────────────────────────────────────

    def _process_farm(self, farm: str) -> pd.DataFrame:
        farm_dir  = self.root / farm
        label_map = self._load_event_labels(farm_dir)
        csv_files = sorted((farm_dir / "datasets").glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {farm_dir}/datasets")

        events, skipped = [], 0
        for csv_path in csv_files:
            event_id = csv_path.stem
            try:
                ev = self._load_one_event(csv_path, farm, event_id, label_map)
                events.append(ev)
            except (ValueError, KeyError) as e:
                logger.warning(f"  [SKIP] {csv_path.name}: {e}")
                skipped += 1

        farm_df = pd.concat(events, ignore_index=True)
        farm_df = self._fill_missing_timestamps(farm_df)
        farm_df = self._zscore_normalise(farm_df)
        farm_df = self._add_temporal_features(farm_df)

        logger.info(
            f"  {farm}: {len(csv_files)-skipped}/{len(csv_files)} events "
            f"→ {len(farm_df):,} rows  (skipped: {skipped})"
        )
        return farm_df

    @staticmethod
    def _load_event_labels(farm_dir: Path) -> Dict[str, str]:
        """Parse event_info.csv → {event_id: 'anomaly'|'normal'}."""
        path = farm_dir / "event_info.csv"
        if not path.exists():
            logger.warning(f"event_info.csv not found: {path}")
            return {}
        df = pd.read_csv(path, sep=";")
        return dict(zip(df["event_id"].astype(str), df["event_label"].str.lower()))

    @staticmethod
    def _load_one_event(
        csv_path: Path,
        farm: str,
        event_id: str,
        label_map: Dict[str, str],
    ) -> pd.DataFrame:
        """Load one event CSV → select mapped features."""
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        fmap = FEATURE_MAP[farm]

        missing = [v for v in fmap.values() if v not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        out = pd.DataFrame()
        if "time_stamp" in df.columns:
            out["time_stamp"] = pd.to_datetime(df["time_stamp"])
        if "asset_id" in df.columns:
            out["asset_id"] = df["asset_id"]
        if "status_type_id" in df.columns:
            out["status_type_id"] = df["status_type_id"]
        if "train_test" in df.columns:
            out["train_test"] = df["train_test"]

        for feat, col in fmap.items():
            out[feat] = df[col]

        out["event_id"]    = event_id
        out["farm"]        = farm
        out["event_label"] = label_map.get(str(event_id), "unknown")
        return out

    @staticmethod
    def _fill_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct full 10-min grid per event with domain-aware imputation."""
        meta_cols  = ["asset_id", "event_id", "farm", "event_label",
                      "status_type_id", "train_test"]
        other_cols = [c for c in FEATURE_COLS if c not in THERMAL_COLS]

        filled = []
        for eid, g in df.groupby("event_id", sort=False):
            g = g.sort_values("time_stamp").set_index("time_stamp")
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq="10min")
            g = g.reindex(full_idx)

            # Forward-fill metadata
            for col in meta_cols:
                if col in g.columns:
                    g[col] = g[col].ffill().bfill()

            # Temperature: ffill → bfill → 0
            for col in THERMAL_COLS:
                if col in g.columns:
                    g[col] = g[col].ffill().bfill().fillna(0)

            # Other features: zero fill
            for col in other_cols:
                if col in g.columns:
                    g[col] = g[col].fillna(0)

            g.index.name = "time_stamp"
            filled.append(g.reset_index())

        return pd.concat(filled, ignore_index=True)

    @staticmethod
    def _zscore_normalise(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        """In-place per-farm z-score normalisation."""
        for col in FEATURE_COLS:
            mu    = df[col].mean()
            sigma = df[col].std()
            if sigma == 0 or np.isnan(sigma):
                df[col] = 0.0
            else:
                df[col] = (df[col] - mu) / sigma
        return df

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add hour, day, month from time_stamp."""
        if "time_stamp" in df.columns:
            ts = pd.to_datetime(df["time_stamp"])
            df["hour"]  = ts.dt.hour
            df["day"]   = ts.dt.day
            df["month"] = ts.dt.month
        else:
            df["hour"] = df["day"] = df["month"] = 0
        return df
