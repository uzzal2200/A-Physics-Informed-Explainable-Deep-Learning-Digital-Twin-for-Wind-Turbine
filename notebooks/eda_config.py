"""
notebooks/eda_config.py
────────────────────────
Shared constants, colour palette, feature metadata
for all EDA modules.

Based on: notebook1_EDA_updated.ipynb
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
os.makedirs("tables",  exist_ok=True)

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "wind_speed", "active_power", "reactive_power",
    "ambient_temperature", "grid_frequency", "rotor_speed",
    "gearbox_oil_temp", "generator_temp", "pitch_angle",
]

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]

# ── Units for axis labels ─────────────────────────────────────────────────────
UNITS = {
    "wind_speed":          "m/s (norm)",
    "active_power":        "kW (norm)",
    "reactive_power":      "kVAr (norm)",
    "ambient_temperature": "°C (norm)",
    "grid_frequency":      "Hz (norm)",
    "rotor_speed":         "rpm (norm)",
    "gearbox_oil_temp":    "°C (norm)",
    "generator_temp":      "°C (norm)",
    "pitch_angle":         "deg (norm)",
    "hour":  "hour",
    "day":   "day",
    "month": "month",
}

# ── Colour palette ────────────────────────────────────────────────────────────
C_NORMAL  = "#2196F3"   # blue
C_ANOMALY = "#F44336"   # red
C_FARM_A  = "#2196F3"   # blue
C_FARM_B  = "#4CAF50"   # green
C_FARM_C  = "#FF9800"   # orange

FARM_COLORS = {
    "Wind Farm A": C_FARM_A,
    "Wind Farm B": C_FARM_B,
    "Wind Farm C": C_FARM_C,
}

FEAT_COLORS = [
    "#E53935", "#FB8C00", "#FDD835", "#43A047", "#00ACC1",
    "#1E88E5", "#8E24AA", "#D81B60", "#6D4C41",
]

HATCHES = ["/", "\\", "x", "-", "+", "o", "*", "//", ".."]

STATUS_MAP = {
    0: "Normal Op", 1: "Derated", 2: "Idling",
    3: "Service",   4: "Downtime", 5: "Other",
}
STATUS_COLORS = [
    "#4CAF50", "#FFC107", "#2196F3",
    "#9C27B0", "#F44336", "#607D8B",
]

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.4,
})


def load_dataset(path: str = "data/main_dataset.csv") -> pd.DataFrame:
    """Load the preprocessed main dataset."""
    df = pd.read_csv(path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Farms : {df['farm'].value_counts().to_dict()}")
    print(f"Labels: {df['event_label'].value_counts().to_dict()}")
    return df


def save_fig(fig: plt.Figure, name: str):
    """Save figure to figures/ directory."""
    path = f"figures/{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
