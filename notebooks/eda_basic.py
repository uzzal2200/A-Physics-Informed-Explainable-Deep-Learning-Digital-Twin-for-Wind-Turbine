"""
notebooks/eda_basic.py
───────────────────────
Basic EDA: dataset overview, feature distributions,
correlation, temporal patterns, violin plots,
statistical tests, cross-farm KDE, data quality.

Figures produced (matches paper):
  Fig 1  — Dataset Overview (rows, events, labels, anomaly rate)
  Fig 2  — Feature Histograms 3×3 with KDE overlay
  Fig 3  — Feature Box Plots 3×3
  Fig 4  — Correlation Heatmap 12×12
  Fig 7  — Temporal Patterns (diurnal + seasonal)
  Fig 8  — Time Series Sample
  Fig 9  — Violin Plots — Normal vs Anomaly
  Fig 10 — Statistical Tests (t-test + KS)
  Fig 11 — Cross-Farm KDE Distributions
  Fig 12 — Cross-Farm Anomaly Rate
  Fig 13 — Data Quality (null rates, zero rates, completeness)
  Fig 14 — Radar Chart — Feature Variability by Label
  Fig 15 — Operational Status Distribution
  Fig 18 — Feature Variability (std comparison)
  Table 3 — Dataset Statistics
  Table 4 — Feature Statistics

Based on: notebook1_EDA_updated.ipynb
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind

from notebooks.eda_config import (
    C_ANOMALY, C_FARM_A, C_FARM_B, C_FARM_C, C_NORMAL,
    FARM_COLORS, FARMS, FEAT_COLORS, FEATURE_COLS, HATCHES,
    STATUS_COLORS, STATUS_MAP, UNITS, save_fig,
)

warnings.filterwarnings("ignore")


# ── Figure 1 — Dataset Overview ───────────────────────────────────────────────

def plot_dataset_overview(df: pd.DataFrame):
    """Four-panel dataset overview."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    farm_rows = [df[df["farm"] == f].shape[0] for f in FARMS]
    farm_cols = [C_FARM_A, C_FARM_B, C_FARM_C]

    # (a) Rows per farm
    ax = axes[0]
    bars = ax.bar(FARMS, farm_rows, color=farm_cols, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, farm_rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("(a) Rows per Farm")
    ax.set_ylabel("Number of Rows")
    ax.tick_params(axis="x", rotation=15)

    # (b) Events per farm (stacked anomaly/normal)
    anom = [df[(df["farm"] == f) & (df["event_label"] == "anomaly")]["event_id"].nunique() for f in FARMS]
    norm = [df[(df["farm"] == f) & (df["event_label"] == "normal")]["event_id"].nunique() for f in FARMS]
    ax = axes[1]
    ax.bar(FARMS, norm, label="Normal",  color=C_NORMAL,  alpha=0.8)
    ax.bar(FARMS, anom, bottom=norm, label="Anomaly", color=C_ANOMALY, alpha=0.8)
    ax.set_title("(b) Events per Farm"); ax.set_ylabel("Event Count")
    ax.legend(); ax.tick_params(axis="x", rotation=15)

    # (c) Anomaly rate per farm
    rates = [df[df["farm"] == f]["event_label"].eq("anomaly").mean() * 100 for f in FARMS]
    ax = axes[2]
    ax.bar(FARMS, rates, color=farm_cols, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_title("(c) Anomaly Row Rate (%)"); ax.set_ylabel("Rate (%)")
    ax.tick_params(axis="x", rotation=15)

    # (d) Turbines per farm
    turbines = [df[df["farm"] == f]["asset_id"].nunique() for f in FARMS]
    ax = axes[3]
    ax.bar(FARMS, turbines, color=farm_cols, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(turbines):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    ax.set_title("(d) Turbines per Farm"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig(fig, "figure1_dataset_overview")


# ── Table 3 & 4 ───────────────────────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame):
    """Print Table 3 (farm stats) and Table 4 (feature stats)."""
    stats = df.groupby("farm").agg(
        rows=("event_label", "count"),
        events=("event_id", "nunique"),
        anomaly_events=("event_label", lambda x: (x == "anomaly").any()),
        turbines=("asset_id", "nunique"),
        anomaly_rate=("event_label", lambda x: round((x == "anomaly").mean() * 100, 1)),
    ).reset_index()
    print("\n=== Table 3 — Dataset Statistics ===")
    print(stats.to_string(index=False))
    stats.to_csv("tables/table3_dataset_stats.csv", index=False)

    desc = df[FEATURE_COLS].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    desc.loc["skewness"] = df[FEATURE_COLS].skew()
    desc.loc["kurtosis"] = df[FEATURE_COLS].kurtosis()
    print("\n=== Table 4 — Feature Statistics ===")
    print(desc.round(4))
    desc.to_csv("tables/table4_feature_statistics.csv")


# ── Figure 2 — Feature Histograms ────────────────────────────────────────────

def plot_feature_histograms(df: pd.DataFrame):
    """3×3 feature histograms with KDE overlay."""
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, feat in zip(axes.flat, FEATURE_COLS):
        data = df[feat].dropna()
        ax.hist(data, bins=60, color="steelblue", alpha=0.6, density=True, label="Histogram")
        xmin, xmax = np.percentile(data, [0.5, 99.5])
        xs = np.linspace(xmin, xmax, 200)
        try:
            kde = gaussian_kde(data.sample(min(50000, len(data))))
            ax.plot(xs, kde(xs), color=C_ANOMALY, lw=2, label="KDE")
        except Exception:
            pass
        ax.axvline(data.mean(),   color="blue",  ls="--", lw=1.5, label=f"Mean={data.mean():.2f}")
        ax.axvline(data.median(), color="green", ls="--", lw=1.5, label=f"Median={data.median():.2f}")
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel(UNITS[feat])
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "figure2_feature_histograms")


# ── Figure 3 — Box Plots ──────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame):
    """3×3 feature box plots."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, feat in zip(axes.flat, FEATURE_COLS):
        sample = df[feat].dropna().sample(min(100_000, len(df)))
        ax.boxplot(sample, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#4CAF50", alpha=0.7),
                   medianprops=dict(color="red", linewidth=2),
                   flierprops=dict(marker=".", markersize=1, alpha=0.3, color="gray"),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel(UNITS[feat])
    plt.tight_layout()
    save_fig(fig, "figure3_boxplots")


# ── Figure 4 — Correlation Heatmap ───────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """12×12 correlation heatmap."""
    corr_cols = FEATURE_COLS + ["hour", "day", "month"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, linecolor="white",
                ax=ax, annot_kws={"size": 8})
    plt.tight_layout()
    save_fig(fig, "figure4_correlation_heatmap")

    corr_pairs = corr.unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0].abs().sort_values(ascending=False)
    print("\nTop 8 feature correlations:")
    print(corr_pairs.head(8).round(3))


# ── Figure 7 — Temporal Patterns ─────────────────────────────────────────────

def plot_temporal_patterns(df: pd.DataFrame):
    """Diurnal and seasonal patterns (2×3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    feats  = ["wind_speed", "active_power", "rotor_speed"]
    groups = ["hour", "month"]
    labels = ["Hourly (Diurnal)", "Monthly (Seasonal)"]

    for col_i, feat in enumerate(feats):
        for row_i, grp in enumerate(groups):
            ax = axes[row_i, col_i]
            gdata = df.groupby(grp)[feat].agg(["mean", "std"])
            x = gdata.index
            col = list(FARM_COLORS.values())[col_i % 3]
            ax.plot(x, gdata["mean"], color=col, lw=2.5)
            ax.fill_between(x, gdata["mean"] - gdata["std"],
                            gdata["mean"] + gdata["std"], alpha=0.2, color=col)
            ax.set_xlabel("Hour of Day" if grp == "hour" else "Month")
            ax.set_ylabel(feat.replace("_", " ").title())
            ax.set_title(f"{labels[row_i]}: {feat.replace('_', ' ').title()}")

    plt.tight_layout()
    save_fig(fig, "figure7_temporal_patterns")


# ── Figure 8 — Time Series Sample ────────────────────────────────────────────

def plot_timeseries_sample(df: pd.DataFrame):
    """Sample time series for one event per farm."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    feats  = ["wind_speed", "active_power", "rotor_speed"]
    colors = [C_FARM_A, C_FARM_B, C_FARM_C]

    for i, feat in enumerate(feats):
        ax = axes[i]
        for farm, fc in FARM_COLORS.items():
            ev      = df[df["farm"] == farm]["event_id"].iloc[0]
            ev_data = df[(df["farm"] == farm) & (df["event_id"] == ev)].head(200)
            ax.plot(range(len(ev_data)), ev_data[feat], color=fc, lw=1.2,
                    alpha=0.9, label=farm)
        ax.set_ylabel(feat.replace("_", " ").title() + " (norm)")
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Timestep (10-min intervals)")
    plt.tight_layout()
    save_fig(fig, "figure8_timeseries_sample")


# ── Figure 9 — Violin Plots ───────────────────────────────────────────────────

def plot_violin_plots(df: pd.DataFrame):
    """Violin plots: Normal vs Anomaly per feature."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    sample_v = df.sample(min(200_000, len(df)))

    for ax, feat in zip(axes.flat, FEATURE_COLS):
        data_n = sample_v[sample_v["event_label"] == "normal"][feat].dropna()
        data_a = sample_v[sample_v["event_label"] == "anomaly"][feat].dropna()
        parts  = ax.violinplot(
            [data_n.sample(min(20_000, len(data_n))),
             data_a.sample(min(20_000, len(data_a)))],
            positions=[0, 1], showmedians=True, showextrema=True,
        )
        for pc, c in zip(parts["bodies"], [C_NORMAL, C_ANOMALY]):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            if part in parts:
                parts[part].set_color("black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_title(feat.replace("_", " ").title())
        ax.set_ylabel(UNITS[feat])

    plt.tight_layout()
    save_fig(fig, "figure9_violin_plots")


# ── Figure 10 — Statistical Tests ────────────────────────────────────────────

def plot_statistical_tests(df: pd.DataFrame):
    """t-test + KS test discriminative power."""
    results = []
    sample_s = df.sample(min(300_000, len(df)))

    for feat in FEATURE_COLS:
        n = sample_s[sample_s["event_label"] == "normal"][feat].dropna().sample(10_000)
        a = sample_s[sample_s["event_label"] == "anomaly"][feat].dropna().sample(10_000)
        t_stat, t_pval = ttest_ind(n, a)
        ks_stat, ks_pval = ks_2samp(n, a)
        results.append({
            "Feature": feat,
            "Mean_Normal":  round(n.mean(), 4),
            "Mean_Anomaly": round(a.mean(), 4),
            "Mean_Diff":    round(abs(n.mean() - a.mean()), 4),
            "t_stat":  round(t_stat,   3),
            "t_pval":  round(t_pval,   5),
            "ks_stat": round(ks_stat,  3),
            "ks_pval": round(ks_pval,  5),
            "Significant": (t_pval < 0.05) and (ks_pval < 0.05),
        })

    stat_df = pd.DataFrame(results).sort_values("Mean_Diff", ascending=False)
    stat_df.to_csv("tables/statistical_tests.csv", index=False)
    print("\nStatistical tests saved: tables/statistical_tests.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = ["#4CAF50" if s else "#F44336" for s in stat_df["Significant"]]
    bars = ax.barh(stat_df["Feature"].tolist(), stat_df["Mean_Diff"].tolist(),
                   color=bar_colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, stat_df["Mean_Diff"]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |Anomaly − Normal| (z-score)")
    ax.set_title("Feature Discriminative Power (t-test + KS significant = green)")
    plt.tight_layout()
    save_fig(fig, "figure10_statistical_tests")


# ── Figure 11 — Cross-Farm KDE ────────────────────────────────────────────────

def plot_crossfarm_kde(df: pd.DataFrame):
    """3×3 KDE per feature, coloured by farm."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    sample_c = df.sample(min(200_000, len(df)))

    for ax, feat in zip(axes.flat, FEATURE_COLS):
        for farm, col in FARM_COLORS.items():
            d = sample_c[sample_c["farm"] == farm][feat].dropna()
            d.plot.kde(ax=ax, color=col, lw=2, label=farm)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel(UNITS[feat]); ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "figure11_crossfarm_kde")


# ── Figure 12 — Cross-Farm Anomaly Rate ──────────────────────────────────────

def plot_crossfarm_anomaly(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) anomaly rate
    rates = {farm: df[df["farm"] == farm]["event_label"].eq("anomaly").mean() * 100
             for farm in FARMS}
    ax = axes[0]
    ax.bar(rates.keys(), rates.values(),
           color=[C_FARM_A, C_FARM_B, C_FARM_C], edgecolor="white", linewidth=1.5)
    for i, (k, v) in enumerate(rates.items()):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_title("(a) Anomaly Rate per Farm (%)")
    ax.set_ylabel("Anomaly Row Rate (%)")
    ax.tick_params(axis="x", rotation=10)

    # (b) feature means heatmap
    farm_means = df.groupby("farm")[FEATURE_COLS].mean()
    ax = axes[1]
    im = ax.imshow(farm_means.T.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FARMS)))
    ax.set_xticklabels(["Farm A", "Farm B", "Farm C"])
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels([f.replace("_", " ") for f in FEATURE_COLS])
    ax.set_title("(b) Feature Means per Farm (z-score)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_fig(fig, "figure12_crossfarm_anomaly")


# ── Figure 13 — Data Quality ──────────────────────────────────────────────────

def plot_data_quality(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) null rates
    null_rates = df[FEATURE_COLS].isnull().sum() / len(df) * 100
    ax = axes[0]
    ax.barh(FEATURE_COLS, null_rates.values, color="#F44336", alpha=0.8)
    ax.set_title("(a) Null Value Rate (%)")
    ax.set_xlabel("Rate (%)")
    ax.set_xlim(0, max(null_rates.max() + 0.01, 0.1))

    # (b) zero rates per farm
    zero_df = pd.DataFrame(
        {farm: (df[df["farm"] == farm][FEATURE_COLS] == 0).sum() /
               len(df[df["farm"] == farm]) * 100
         for farm in FARMS},
        index=FEATURE_COLS,
    )
    ax = axes[1]
    zero_df.plot(kind="bar", ax=ax, color=[C_FARM_A, C_FARM_B, C_FARM_C],
                 alpha=0.85, edgecolor="white")
    ax.set_title("(b) Zero Value Rate (%) per Farm")
    ax.set_ylabel("Rate (%)"); ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right")

    # (c) event completeness
    event_comp = df.groupby("event_id")["time_stamp"].count()
    ax = axes[2]
    ax.hist(event_comp.values, bins=30, color=C_NORMAL, alpha=0.8, edgecolor="white")
    ax.set_title("(c) Rows per Event (Completeness)")
    ax.set_xlabel("Rows per event"); ax.set_ylabel("Count")

    plt.tight_layout()
    save_fig(fig, "figure13_data_quality")


# ── Figure 14 — Radar Chart ───────────────────────────────────────────────────

def plot_radar_chart(df: pd.DataFrame):
    labels = [f.replace("_", "\n").title() for f in FEATURE_COLS]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for label, color in [("normal", C_NORMAL), ("anomaly", C_ANOMALY)]:
        vals = df[df["event_label"] == label][FEATURE_COLS].std().values.tolist()
        max_val = max(vals) + 1e-9
        vals = [v / max_val for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, lw=2.5, label=label.title(), markersize=5)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
    plt.tight_layout()
    save_fig(fig, "figure14_radar_chart")


# ── Figure 15 — Status Distribution ──────────────────────────────────────────

def plot_status_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, farm in zip(axes, FARMS):
        sc      = df[df["farm"] == farm]["status_type_id"].value_counts().sort_index()
        labels_ = [STATUS_MAP.get(int(k), str(k)) for k in sc.index]
        ax.pie(sc.values, labels=labels_,
               colors=STATUS_COLORS[: len(sc)], autopct="%1.1f%%",
               startangle=90, pctdistance=0.8,
               wedgeprops=dict(edgecolor="white", linewidth=1.5))
        ax.set_title(farm)
    plt.tight_layout()
    save_fig(fig, "figure15_status_distribution")


# ── Figure 18 — Feature Variability ──────────────────────────────────────────

def plot_feature_variability(df: pd.DataFrame):
    s_n = df[df["event_label"] == "normal"][FEATURE_COLS].std()
    s_a = df[df["event_label"] == "anomaly"][FEATURE_COLS].std()
    x   = np.arange(len(FEATURE_COLS)); w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    for i, (feat, fc, ht) in enumerate(zip(FEATURE_COLS, FEAT_COLORS, HATCHES)):
        b_n = ax.bar(i - w / 2, s_n[feat], w, color=fc, alpha=0.55,
                     edgecolor="white", hatch=ht, zorder=3)
        b_a = ax.bar(i + w / 2, s_a[feat], w, color=fc, alpha=0.95,
                     edgecolor="white", hatch=ht, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in FEATURE_COLS], fontsize=8)
    ax.set_ylabel("Standard Deviation (z-score)")
    ax.set_title("(a) Feature Std — Normal (light) vs Anomaly (dark)")
    legend_patches = [
        mpatches.Patch(color="gray", alpha=0.4, label="Normal"),
        mpatches.Patch(color="gray", alpha=0.9, label="Anomaly"),
    ]
    ax.legend(handles=legend_patches)

    # (b) variance ratio
    ratio = (s_a / (s_n + 1e-9)).sort_values(ascending=True)
    ax = axes[1]
    bars = ax.barh(ratio.index.str.replace("_", " "), ratio.values,
                   color=[FEAT_COLORS[FEATURE_COLS.index(f)] for f in ratio.index])
    ax.axvline(1.0, color="black", ls="--", lw=1.5, label="Ratio = 1")
    ax.set_xlabel("Anomaly Std / Normal Std")
    ax.set_title("(b) Variability Ratio (Anomaly / Normal)")
    ax.legend()

    plt.tight_layout()
    save_fig(fig, "figure18_feature_variability")
