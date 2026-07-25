"""
eda/eda_plots.py
────────────────
Publication-ready EDA figures for the PI-CTBA-Net paper.

Based on: notebook1_EDA_updated.ipynb

Figures produced (matching paper):
  Fig 1  — Dataset Overview (4 subplots)
  Fig 2  — Feature Histograms with KDE (3×3)
  Fig 3  — Box Plots (3×3)
  Fig 4  — Correlation Heatmap (12×12)
  Fig 5  — Power Curve Validation (Betz Law)
  Fig 6  — Physics Constraint Validation (4 subplots)
  Fig 7  — Temporal Patterns (diurnal + seasonal)
  Fig 8  — Sample Time Series
  Fig 9  — Violin Plots Normal vs Anomaly (3×3)
  Fig 10 — Feature Discriminative Power (statistical tests)
  Fig 11 — Cross-Farm KDE Distributions (3×3)
  Fig 12 — Cross-Farm Anomaly Analysis
  Fig 13 — Data Quality (missing/zero rates)
  Fig 14 — Radar Chart (feature variability)
  Fig 15 — t-SNE (2D projection)
  Fig 16 — Binned Power Curve per Farm ±1σ
  Fig 17 — TSR Proxy Distribution per Farm

Usage
-----
  from eda.eda_plots import EDAPlotter
  plotter = EDAPlotter(df, save_dir="figures/EDA")
  plotter.run_all()           # generate all figures
  plotter.figure_overview()   # generate one specific figure
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, ks_2samp, ttest_ind

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "wind_speed", "active_power", "reactive_power",
    "ambient_temperature", "grid_frequency", "rotor_speed",
    "gearbox_oil_temp", "generator_temp", "pitch_angle",
]

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
}

C_NORMAL  = "#2196F3"
C_ANOMALY = "#F44336"
C_FARM_A  = "#2196F3"
C_FARM_B  = "#4CAF50"
C_FARM_C  = "#FF9800"
FARM_COLORS = {"Farm_A": C_FARM_A, "Farm_B": C_FARM_B, "Farm_C": C_FARM_C}
FARMS = ["Farm_A", "Farm_B", "Farm_C"]
FARM_LABELS = {"Farm_A": "Wind Farm A", "Farm_B": "Wind Farm B", "Farm_C": "Wind Farm C"}


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
        "axes.grid": True, "grid.alpha": 0.4,
    })


class EDAPlotter:
    """
    Generates all EDA figures for the PI-CTBA-Net paper.

    Parameters
    ----------
    df       : pd.DataFrame  — unified CARE dataset (main_dataset.csv)
    save_dir : str           — directory for saving figures
    show     : bool          — whether to display figures (False for scripts)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        save_dir: str = "figures/EDA",
        show: bool = False,
    ):
        self.df = df
        self.save_dir = Path(save_dir)
        self.show = show
        os.makedirs(self.save_dir, exist_ok=True)
        _setup_style()

    def run_all(self):
        """Generate all figures in sequence."""
        logger.info("Generating all EDA figures …")
        methods = [
            self.figure_overview,
            self.figure_feature_histograms,
            self.figure_boxplots,
            self.figure_correlation_heatmap,
            self.figure_power_curve,
            self.figure_physics_validation,
            self.figure_temporal_patterns,
            self.figure_timeseries_sample,
            self.figure_violin_plots,
            self.figure_discriminative_power,
            self.figure_crossfarm_kde,
            self.figure_crossfarm_anomaly,
            self.figure_data_quality,
            self.figure_tsne,
            self.figure_binned_power_curve,
            self.figure_tsr_distribution,
        ]
        for method in methods:
            try:
                method()
            except Exception as e:
                logger.warning(f"  [{method.__name__}] failed: {e}")
        logger.info("All figures complete.")

    # ── Figure 1 — Dataset Overview ──────────────────────────────────────────

    def figure_overview(self):
        """Figure 1: Dataset Overview (rows, events, label dist, status)."""
        df = self.df
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        farms      = FARMS
        farm_rows  = [df[df["farm"] == f].shape[0] for f in farms]
        farm_cols  = [C_FARM_A, C_FARM_B, C_FARM_C]
        farm_names = [FARM_LABELS[f] for f in farms]

        # (a) Rows per farm
        ax = axes[0]
        bars = ax.bar(farm_names, farm_rows, color=farm_cols, edgecolor="white")
        for bar, val in zip(bars, farm_rows):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 15000,
                    f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("(a) Rows per Farm")
        ax.set_ylabel("Number of Rows")
        ax.tick_params(axis="x", rotation=15)

        # (b) Events per farm
        anom = [df[(df["farm"]==f) & (df["event_label"]=="anomaly")]["event_id"].nunique() for f in farms]
        norm = [df[(df["farm"]==f) & (df["event_label"]=="normal")]["event_id"].nunique() for f in farms]
        ax = axes[1]
        ax.bar(farm_names, norm,  label="Normal",  color=C_NORMAL,  alpha=0.8)
        ax.bar(farm_names, anom,  label="Anomaly", color=C_ANOMALY, alpha=0.8, bottom=norm)
        ax.set_title("(b) Events per Farm")
        ax.set_ylabel("Event Count")
        ax.legend()
        ax.tick_params(axis="x", rotation=15)

        # (c) Label distribution
        ax = axes[2]
        lc = df["event_label"].value_counts()
        ax.pie(lc.values, labels=lc.index,
               colors=[C_NORMAL, C_ANOMALY],
               autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("(c) Label Distribution")

        # (d) Status type
        ax = axes[3]
        status_labels = {0: "Normal\nOp", 1: "Derated", 2: "Idling",
                         3: "Service", 4: "Downtime", 5: "Other"}
        sc = df["status_type_id"].value_counts().sort_index()
        colors_s = ["#4CAF50","#FFC107","#2196F3","#9C27B0","#F44336","#607D8B"]
        ax.bar([status_labels.get(int(i), str(i)) for i in sc.index],
               sc.values, color=colors_s[: len(sc)], edgecolor="white")
        ax.set_title("(d) Operational Status")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=10)

        plt.tight_layout()
        self._save_fig(fig, "figure1_dataset_overview.png")

    # ── Figure 2 — Feature Histograms ───────────────────────────────────────

    def figure_feature_histograms(self):
        """Figure 2: Feature distributions with KDE overlay (3×3)."""
        df = self.df
        fig, axes = plt.subplots(3, 3, figsize=(16, 13))

        for ax, feat in zip(axes.flat, FEATURE_COLS):
            data = df[feat].dropna()
            ax.hist(data, bins=60, color="steelblue", alpha=0.6,
                    density=True, label="Histogram")
            xmin, xmax = np.percentile(data, [0.5, 99.5])
            xs = np.linspace(xmin, xmax, 200)
            try:
                kde = gaussian_kde(data.sample(min(50_000, len(data))))
                ax.plot(xs, kde(xs), color=C_ANOMALY, lw=2, label="KDE")
            except Exception:
                pass
            ax.axvline(data.mean(),   color="blue",  ls="--", lw=1.5,
                       label=f"Mean={data.mean():.2f}")
            ax.axvline(data.median(), color="green", ls="--", lw=1.5,
                       label=f"Median={data.median():.2f}")
            ax.set_title(feat.replace("_", " ").title())
            ax.set_xlabel(UNITS[feat])
            ax.legend(fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, "figure2_feature_histograms.png")

    # ── Figure 3 — Box Plots ─────────────────────────────────────────────────

    def figure_boxplots(self):
        """Figure 3: Box plots — outlier analysis (3×3)."""
        df = self.df
        fig, axes = plt.subplots(3, 3, figsize=(16, 13))

        for ax, feat in zip(axes.flat, FEATURE_COLS):
            sample = df[feat].dropna().sample(min(100_000, len(df)))
            ax.boxplot(
                sample, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#4CAF50", alpha=0.7),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker=".", markersize=1, alpha=0.3, color="gray"),
            )
            ax.set_title(feat.replace("_", " ").title())
            ax.set_xlabel(UNITS[feat])

        plt.tight_layout()
        self._save_fig(fig, "figure3_boxplots.png")

    # ── Figure 4 — Correlation Heatmap ──────────────────────────────────────

    def figure_correlation_heatmap(self):
        """Figure 4: Feature correlation matrix (12×12)."""
        corr_cols = FEATURE_COLS + ["hour", "day", "month"]
        corr = self.df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size": 8},
        )
        plt.tight_layout()
        self._save_fig(fig, "figure4_correlation_heatmap.png")

    # ── Figure 5 — Power Curve ───────────────────────────────────────────────

    def figure_power_curve(self):
        """Figure 5: Power curve validation — P ∝ V³ (Betz law)."""
        df = self.df
        sample = df[df["status_type_id"] == 0].sample(min(80_000, len(df)))
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # (a) All farms
        ax = axes[0]
        for farm, col in FARM_COLORS.items():
            s = sample[sample["farm"] == farm]
            ax.scatter(s["wind_speed"], s["active_power"],
                       c=col, s=3, alpha=0.15, label=FARM_LABELS[farm])
        V = np.linspace(sample["wind_speed"].min(), sample["wind_speed"].max(), 200)
        P_theory = np.clip(
            V ** 3 / (V ** 3).max() * sample["active_power"].max(),
            None, sample["active_power"].max()
        )
        ax.plot(V, P_theory, "k-", lw=2.5, label="Theoretical P ∝ V³", zorder=5)
        ax.set_xlabel("Wind Speed (normalized)")
        ax.set_ylabel("Active Power (normalized)")
        ax.set_title("(a) Power Curve — All Farms")
        ax.legend()
        ax.text(0.05, 0.92, "Betz Limit: Cₚ ≤ 0.593",
                transform=ax.transAxes,
                bbox=dict(facecolor="lightyellow", edgecolor="orange", boxstyle="round"))

        # (b) Per farm
        ax = axes[1]
        for farm, col in FARM_COLORS.items():
            s = sample[sample["farm"] == farm]
            ax.scatter(s["wind_speed"], s["active_power"],
                       c=col, s=3, alpha=0.2, label=FARM_LABELS[farm])
        corr_val = sample["wind_speed"].corr(sample["active_power"])
        ax.set_title(f"(b) Wind Speed vs Power  (r = {corr_val:.3f})")
        ax.set_xlabel("Wind Speed (norm)")
        ax.set_ylabel("Active Power (norm)")
        ax.legend()

        plt.tight_layout()
        self._save_fig(fig, "figure5_power_curve.png")

    # ── Figure 6 — Physics Validation ───────────────────────────────────────

    def figure_physics_validation(self):
        """Figure 6: Physics constraint validation (Cp, TSR, thermal, PF)."""
        df   = self.df
        s    = df.sample(min(200_000, len(df)))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # (a) Power coefficient Cp
        Cp = (s["active_power"] / (s["wind_speed"] ** 3 + 1e-6)).clip(-2, 2)
        ax = axes[0, 0]
        ax.hist(Cp, bins=60, color="#9C27B0", alpha=0.7, density=True)
        ax.axvline(0.593, color="red", lw=2, ls="--", label="Betz Limit Cₚ=0.593")
        ax.set_title("(a) Power Coefficient Cₚ Distribution")
        ax.set_xlabel("Cₚ (proxy)")
        ax.set_ylabel("Density")
        ax.legend()
        pct = (Cp <= 0.593).mean() * 100
        ax.text(0.60, 0.85, f"{pct:.1f}% ≤ Betz Limit",
                transform=ax.transAxes, color="darkred", fontweight="bold")

        # (b) TSR proxy
        TSR = (s["rotor_speed"] / (s["wind_speed"].abs() + 1e-6)).clip(-5, 5)
        ax  = axes[0, 1]
        ax.hist(TSR, bins=60, color="#FF9800", alpha=0.7, density=True)
        ax.axvspan(-0.5, 0.5, alpha=0.15, color="green", label="Optimal TSR zone")
        ax.set_title("(b) Tip Speed Ratio (TSR) Distribution")
        ax.set_xlabel("TSR (normalized proxy)")
        ax.set_ylabel("Density")
        ax.legend()

        # (c) Thermal dynamics
        ax = axes[1, 0]
        s2 = s.sample(min(30_000, len(s)))
        ax.scatter(s2["ambient_temperature"], s2["gearbox_oil_temp"],
                   c=C_ANOMALY, s=2, alpha=0.15)
        slope, intercept, r, _, _ = stats.linregress(
            s2["ambient_temperature"], s2["gearbox_oil_temp"])
        xs = np.linspace(s2["ambient_temperature"].min(),
                         s2["ambient_temperature"].max(), 100)
        ax.plot(xs, slope * xs + intercept, "k-", lw=2, label=f"Linear fit r={r:.3f}")
        ax.set_title("(c) Thermal: Gearbox Temp vs Ambient Temp")
        ax.set_xlabel("Ambient Temperature (norm)")
        ax.set_ylabel("Gearbox Oil Temp (norm)")
        ax.legend()

        # (d) Power factor
        denom = np.sqrt(s["active_power"] ** 2 + s["reactive_power"] ** 2 + 1e-6)
        PF    = (s["active_power"] / denom).clip(0, 1)
        ax    = axes[1, 1]
        ax.hist(PF, bins=60, color=C_NORMAL, alpha=0.7, density=True)
        ax.axvline(0.9, color="red", lw=2, ls="--", label="Grid code PF ≥ 0.9")
        ax.set_title("(d) Power Factor cos(φ) = P / √(P²+Q²)")
        ax.set_xlabel("Power Factor")
        ax.set_ylabel("Density")
        ax.legend()
        pct_pf = (PF >= 0.9).mean() * 100
        ax.text(0.05, 0.85, f"{pct_pf:.1f}% meet PF ≥ 0.9",
                transform=ax.transAxes, color="darkred", fontweight="bold")

        plt.tight_layout()
        self._save_fig(fig, "figure6_physics_validation.png")

    # ── Figure 7 — Temporal Patterns ─────────────────────────────────────────

    def figure_temporal_patterns(self):
        """Figure 7: Diurnal and seasonal patterns."""
        df = self.df
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        feats  = ["wind_speed", "active_power", "rotor_speed"]
        groups = ["hour", "month"]
        labels = ["Hourly (Diurnal)", "Monthly (Seasonal)"]

        for col_i, feat in enumerate(feats):
            for row_i, (grp, lbl) in enumerate(zip(groups, labels)):
                ax    = axes[row_i, col_i]
                gdata = df.groupby(grp)[feat].agg(["mean", "std"])
                x     = gdata.index
                color = [C_FARM_A, C_FARM_B, C_FARM_C][col_i]
                ax.plot(x, gdata["mean"], color=color, lw=2.5)
                ax.fill_between(x,
                    gdata["mean"] - gdata["std"],
                    gdata["mean"] + gdata["std"],
                    alpha=0.2, color=color)
                ax.set_xlabel("Hour of Day" if grp == "hour" else "Month")
                ax.set_ylabel(feat.replace("_", " ").title())
                ax.set_title(f"{lbl}: {feat.replace('_',' ').title()}")

        plt.tight_layout()
        self._save_fig(fig, "figure7_temporal_patterns.png")

    # ── Figure 8 — Time Series Sample ────────────────────────────────────────

    def figure_timeseries_sample(self):
        """Figure 8: Sample time series per farm."""
        df   = self.df
        fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
        feats = ["wind_speed", "active_power", "rotor_speed"]

        for i, feat in enumerate(feats):
            ax = axes[i]
            for farm, fc in FARM_COLORS.items():
                ev = df[df["farm"] == farm]["event_id"].iloc[0]
                ev_data = df[(df["farm"] == farm) & (df["event_id"] == ev)].head(200)
                ax.plot(range(len(ev_data)), ev_data[feat],
                        color=fc, lw=1.2, alpha=0.9, label=FARM_LABELS[farm])
            ax.set_ylabel(feat.replace("_", " ").title() + " (norm)")
            ax.legend(loc="upper right", fontsize=9)

        axes[-1].set_xlabel("Timestep (10-min intervals)")
        plt.tight_layout()
        self._save_fig(fig, "figure8_timeseries_sample.png")

    # ── Figure 9 — Violin Plots ──────────────────────────────────────────────

    def figure_violin_plots(self):
        """Figure 9: Normal vs Anomaly violin plots (3×3)."""
        df     = self.df
        sample = df.sample(min(200_000, len(df)))
        fig, axes = plt.subplots(3, 3, figsize=(16, 13))

        for ax, feat in zip(axes.flat, FEATURE_COLS):
            data_n = sample[sample["event_label"] == "normal"][feat].dropna()
            data_a = sample[sample["event_label"] == "anomaly"][feat].dropna()
            parts = ax.violinplot(
                [data_n.sample(min(20_000, len(data_n))),
                 data_a.sample(min(20_000, len(data_a)))],
                positions=[0, 1], showmedians=True, showextrema=True,
            )
            for pc, c in zip(parts["bodies"], [C_NORMAL, C_ANOMALY]):
                pc.set_facecolor(c)
                pc.set_alpha(0.7)
            for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
                if part in parts:
                    parts[part].set_color("black")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Normal", "Anomaly"])
            ax.set_title(feat.replace("_", " ").title())
            ax.set_ylabel(UNITS[feat])

        plt.tight_layout()
        self._save_fig(fig, "figure9_violin_plots.png")

    # ── Figure 10 — Feature Discriminative Power ─────────────────────────────

    def figure_discriminative_power(self):
        """Figure 10: Feature discriminative power (t-test + KS test)."""
        df     = self.df
        sample = df.sample(min(300_000, len(df)))
        results = []

        for feat in FEATURE_COLS:
            n = sample[sample["event_label"] == "normal"][feat].dropna().sample(10_000)
            a = sample[sample["event_label"] == "anomaly"][feat].dropna().sample(10_000)
            t_stat, t_pval = ttest_ind(n, a)
            ks_stat, ks_pval = ks_2samp(n, a)
            results.append({
                "Feature":       feat,
                "Mean_Diff":     abs(n.mean() - a.mean()),
                "t_stat":        round(t_stat, 3),
                "t_pval":        round(t_pval, 5),
                "ks_stat":       round(ks_stat, 3),
                "ks_pval":       round(ks_pval, 5),
                "Significant":   (t_pval < 0.05) and (ks_pval < 0.05),
            })

        stat_df = pd.DataFrame(results).sort_values("Mean_Diff", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_colors = ["#4CAF50" if s else C_ANOMALY for s in stat_df["Significant"]]
        bars = ax.barh(stat_df["Feature"].tolist(), stat_df["Mean_Diff"].tolist(),
                       color=bar_colors, edgecolor="white", height=0.6)
        for bar, v in zip(bars, stat_df["Mean_Diff"]):
            ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=10)
        ax.set_xlabel("Mean Absolute Difference (normalized)")
        p1 = mpatches.Patch(color="#4CAF50", label="Statistically significant (p<0.05)")
        p2 = mpatches.Patch(color=C_ANOMALY, label="Not significant")
        ax.legend(handles=[p1, p2])

        plt.tight_layout()
        self._save_fig(fig, "figure10_feature_discriminative_power.png")
        logger.info(f"\n{stat_df[['Feature','Mean_Diff','Significant']].to_string(index=False)}")

    # ── Figure 11 — Cross-Farm KDE ───────────────────────────────────────────

    def figure_crossfarm_kde(self):
        """Figure 11: Feature distributions across farms (KDE)."""
        df     = self.df
        sample = df.sample(min(200_000, len(df)))
        fig, axes = plt.subplots(3, 3, figsize=(16, 13))

        for ax, feat in zip(axes.flat, FEATURE_COLS):
            for farm, col in FARM_COLORS.items():
                d = sample[sample["farm"] == farm][feat].dropna()
                d.plot.kde(ax=ax, color=col, lw=2, label=FARM_LABELS[farm])
            ax.set_title(feat.replace("_", " ").title())
            ax.set_xlabel(UNITS[feat])
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, "figure11_crossfarm_kde.png")

    # ── Figure 12 — Cross-Farm Anomaly ──────────────────────────────────────

    def figure_crossfarm_anomaly(self):
        """Figure 12: Cross-farm anomaly rate + feature mean heatmap."""
        df    = self.df
        farms = FARMS
        farm_names = [FARM_LABELS[f] for f in farms]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Anomaly rate
        ax    = axes[0]
        rates = {FARM_LABELS[f]: df[df["farm"] == f]["event_label"].eq("anomaly").mean() * 100
                 for f in farms}
        ax.bar(rates.keys(), rates.values(),
               color=[C_FARM_A, C_FARM_B, C_FARM_C], edgecolor="white")
        for i, (k, v) in enumerate(rates.items()):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
        ax.set_title("(a) Anomaly Rate per Farm (%)")
        ax.set_ylabel("Anomaly Row Rate (%)")
        ax.tick_params(axis="x", rotation=10)

        # Feature mean heatmap
        ax         = axes[1]
        farm_means = df.groupby("farm")[FEATURE_COLS].mean()
        farm_means = farm_means.reindex(farms)
        im = ax.imshow(farm_means.T.values, aspect="auto", cmap="RdYlGn",
                       vmin=-1, vmax=1)
        ax.set_xticks(range(len(farms)))
        ax.set_xticklabels(["Farm A", "Farm B", "Farm C"])
        ax.set_yticks(range(len(FEATURE_COLS)))
        ax.set_yticklabels([f.replace("_", " ") for f in FEATURE_COLS])
        ax.set_title("(b) Feature Means per Farm (z-score)")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        self._save_fig(fig, "figure12_crossfarm_anomaly.png")

    # ── Figure 13 — Data Quality ─────────────────────────────────────────────

    def figure_data_quality(self):
        """Figure 13: Missing values, zero rates, event completeness."""
        df    = self.df
        farms = FARMS
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # (a) Null rates
        null_rates = df[FEATURE_COLS].isnull().sum() / len(df) * 100
        ax = axes[0]
        ax.barh(FEATURE_COLS, null_rates.values, color=C_ANOMALY, alpha=0.8)
        ax.set_title("(a) Null Value Rate (%)")
        ax.set_xlabel("Rate (%)")
        ax.set_xlim(0, max(null_rates.max() + 0.01, 0.1))

        # (b) Zero rates per farm
        zero_rates = {FARM_LABELS[f]: (df[df["farm"] == f][FEATURE_COLS] == 0).mean() * 100
                      for f in farms}
        zdf = pd.DataFrame(zero_rates, index=FEATURE_COLS)
        ax  = axes[1]
        zdf.plot(kind="bar", ax=ax, color=[C_FARM_A, C_FARM_B, C_FARM_C],
                 alpha=0.85, edgecolor="white")
        ax.set_title("(b) Zero Value Rate (%) per Farm")
        ax.set_ylabel("Rate (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper right")

        # (c) Event completeness
        ax          = axes[2]
        event_comp  = df.groupby("event_id")["time_stamp"].count()
        ax.hist(event_comp.values, bins=30, color=C_NORMAL, alpha=0.8, edgecolor="white")
        ax.set_title("(c) Rows per Event (Completeness)")
        ax.set_xlabel("Row Count per Event")
        ax.set_ylabel("Event Count")
        ax.axvline(event_comp.mean(), color="red", ls="--", lw=2,
                   label=f"Mean={event_comp.mean():.0f}")
        ax.legend()

        plt.tight_layout()
        self._save_fig(fig, "figure13_data_quality.png")

    # ── Figure 14 — t-SNE ────────────────────────────────────────────────────

    def figure_tsne(self, n_samples: int = 5000):
        """Figure 14: 2D t-SNE projection."""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("scikit-learn not available. Skipping t-SNE.")
            return

        df     = self.df.sample(min(n_samples, len(self.df)), random_state=42)
        X      = df[FEATURE_COLS].fillna(0).values
        labels = df["event_label"].values
        farms  = df["farm"].values

        tsne  = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d  = tsne.fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # (a) by label
        ax = axes[0]
        for lbl, col in [("normal", C_NORMAL), ("anomaly", C_ANOMALY)]:
            mask = labels == lbl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, s=5,
                       alpha=0.4, label=lbl.title())
        ax.set_title("(a) Coloured by Label")
        ax.legend()

        # (b) by farm
        ax = axes[1]
        for farm, col in FARM_COLORS.items():
            mask = farms == farm
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, s=5,
                       alpha=0.4, label=FARM_LABELS[farm])
        ax.set_title("(b) Coloured by Farm")
        ax.legend()

        plt.tight_layout()
        self._save_fig(fig, "figure14_tsne.png")

    # ── Figure 15 — Binned Power Curve ───────────────────────────────────────

    def figure_binned_power_curve(self, n_bins: int = 30):
        """Figure 15: Binned mean power curve ±1σ per farm (Figure 6 in paper)."""
        df    = self.df[self.df["status_type_id"] == 0]
        farms = FARMS
        fig, axes = plt.subplots(1, len(farms), figsize=(18, 6))

        for ax, farm in zip(axes, farms):
            s      = df[df["farm"] == farm]
            bins   = pd.cut(s["wind_speed"], bins=n_bins)
            binned = s.groupby(bins, observed=True)["active_power"].agg(["mean", "std"])
            x      = [iv.mid for iv in binned.index]

            ax.plot(x, binned["mean"], color=FARM_COLORS[farm], lw=2.5)
            ax.fill_between(
                x,
                binned["mean"] - binned["std"],
                binned["mean"] + binned["std"],
                alpha=0.25, color=FARM_COLORS[farm],
            )
            ax.set_title(FARM_LABELS[farm])
            ax.set_xlabel("Wind Speed Bin (norm)")
            ax.set_ylabel("Mean Active Power (norm)")

        plt.tight_layout()
        self._save_fig(fig, "figureA_binned_power_curve.png")

    # ── Figure 16 — TSR Distribution ─────────────────────────────────────────

    def figure_tsr_distribution(self):
        """Figure 16: TSR proxy distribution per farm — normal vs anomaly."""
        df    = self.df
        farms = FARMS
        fig, axes = plt.subplots(1, len(farms), figsize=(18, 5))

        for ax, farm in zip(axes, farms):
            s   = df[df["farm"] == farm].sample(min(50_000, len(df)))
            tsr = (s["rotor_speed"] / (s["wind_speed"].abs() + 1e-6)).clip(-10, 10)
            s   = s.copy()
            s["tsr"] = tsr

            for lbl, col in [("normal", C_NORMAL), ("anomaly", C_ANOMALY)]:
                d = s[s["event_label"] == lbl]["tsr"].dropna()
                d.plot.kde(ax=ax, color=col, lw=2, label=lbl.title())

            ax.axvspan(5, 8, alpha=0.12, color="green", label="Optimal TSR zone")
            ax.set_title(FARM_LABELS[farm])
            ax.set_xlabel("TSR proxy (normalised)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)

        plt.tight_layout()
        self._save_fig(fig, "figure_tsr_per_farm.png")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_fig(self, fig: plt.Figure, filename: str):
        path = self.save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)
        logger.info(f"  Saved: {path}")
