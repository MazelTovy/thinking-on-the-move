"""Visualize the 7-cluster demographic profile of DTBK workers.

Reads cluster_attributes.csv (pre-aggregated per-cluster weighted means) and
produces four artifacts:

  1. bars_per_feature_7clusters.png  — 15-panel bar chart (one per feature,
     7 cluster bars each) matching the old 5-cluster style.
  2. heatmap_zscored.png             — feature × cluster heatmap, z-scored
     across clusters, RdBu_r. Compact comparison.
  3. radar_per_cluster.png           — 7 overlaid radar plots on normalised
     features. Captures each cluster's "personality".
  4. category_mix_by_cluster.png     — stacked bar of POI category share per
     cluster (Full-Service / Limited-Service / Snack).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = [
    ("median_household_income",                       "Median HH income ($)"),
    ("unemployment_rate",                             "Unemployment rate (%)"),
    ("labor_force_rate",                              "Labor force rate"),
    ("rented_rate",                                   "Rented rate"),
    ("white_share",                                   "White share"),
    ("black_share",                                   "Black share"),
    ("asian_share",                                   "Asian share"),
    ("some_college_less_than_1_year_share",           "Some college <1y share"),
    ("some_college_1_or_more_years_no_degree_share",  "Some college ≥1y no deg"),
    ("associates_degree_share",                       "Associates degree share"),
    ("bachelors_degree_share",                        "Bachelors degree share"),
    ("masters_degree_share",                          "Masters degree share"),
    ("professional_school_degree_share",              "Professional school share"),
    ("doctorate_degree_share",                        "Doctorate share"),
]

POI_CATS = [
    ("poi_Full-Service_Restaurants", "Full-Service"),
    ("poi_Limited-Service_Restaurants", "Limited-Service"),
    ("poi_Snack_and_Nonalcoholic_Beverage_Bars", "Snack & Beverage"),
]

# Explicit cluster palette (tab10-ish), 7 clusters.
PALETTE = ["#1f77b4", "#2ca02c", "#8c564b", "#7f7f7f", "#17becf",
           "#e377c2", "#ff7f0e"]


def plot_bars(df, out):
    n = len(FEATURES)
    cols = 4
    rows = math.ceil(n / cols)
    # Panels enlarged ~4× (2× previous) so 2×-larger fonts don't crowd
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 11.0, rows * 8.5))
    clusters = df["demo_cluster_pure"].astype(int).tolist()
    colors = [PALETTE[i - 1] for i in clusters]
    for i, (col, label) in enumerate(FEATURES):
        ax = axes[i // cols, i % cols]
        vals = df[col].values
        ax.bar([str(c) for c in clusters], vals, color=colors,
               edgecolor="black", linewidth=2.4)
        ax.set_title(label, fontsize=44, fontweight="bold", pad=20)
        ax.set_xlabel("Cluster", fontsize=36, fontweight="bold", labelpad=14)
        ax.set_ylabel("Weighted mean", fontsize=36, fontweight="bold", labelpad=14)
        ax.tick_params(axis="both", labelsize=32, width=3.2, length=10)
        for s in ax.spines.values():
            s.set_linewidth(3.2)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=2.2)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("DTBK workers — 7-cluster demographic profile (weighted means)",
                 fontsize=60, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df, out):
    clusters = df["demo_cluster_pure"].astype(int).tolist()
    col_names = [label for _, label in FEATURES]
    X = df[[c for c, _ in FEATURES]].values.astype(float)
    # Z-score across clusters (each feature compared to its own cluster mean/std)
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(col_names) + 1))
    vmax = float(np.max(np.abs(Z)))
    im = ax.imshow(Z.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"C{c}" for c in clusters])
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels(col_names)
    ax.set_title("Cluster × feature  (z-scored across clusters; red = above-avg)")
    # Cell annotations
    for i in range(len(col_names)):
        for j in range(len(clusters)):
            ax.text(j, i, f"{Z[j, i]:+.1f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(Z[j, i]) < 1.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="z-score")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar(df, out):
    cols = [c for c, _ in FEATURES]
    labels = [lbl for _, lbl in FEATURES]
    X = df[cols].values.astype(float)
    # Normalise each feature to [0, 1] so axes are comparable
    mn = X.min(axis=0); mx = X.max(axis=0)
    rng = np.where(mx - mn > 0, mx - mn, 1.0)
    Xn = (X - mn) / rng

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    K = X.shape[0]
    cols_grid = 4
    rows_grid = math.ceil(K / cols_grid)
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 3.5, rows_grid * 3.5),
                             subplot_kw=dict(polar=True))
    clusters = df["demo_cluster_pure"].astype(int).tolist()
    for k, cid in enumerate(clusters):
        ax = axes[k // cols_grid, k % cols_grid]
        vals = Xn[k].tolist() + [Xn[k][0]]
        ax.plot(angles, vals, color=PALETTE[cid - 1], linewidth=2)
        ax.fill(angles, vals, color=PALETTE[cid - 1], alpha=0.20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([lbl[:18] for lbl in labels], fontsize=7)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["", "0.5", ""], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"C{cid}", fontsize=11, pad=15)
        ax.tick_params(axis="x", pad=8)

    # Last panel: overlay of all 7
    ax = axes[(K) // cols_grid, (K) % cols_grid]
    for k, cid in enumerate(clusters):
        vals = Xn[k].tolist() + [Xn[k][0]]
        ax.plot(angles, vals, color=PALETTE[cid - 1], linewidth=1.5, label=f"C{cid}")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl[:14] for lbl in labels], fontsize=6)
    ax.set_yticks([])
    ax.set_title("All clusters overlaid", fontsize=11, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)

    # Hide unused subplots
    for j in range(K + 1, rows_grid * cols_grid):
        axes[j // cols_grid, j % cols_grid].axis("off")

    fig.suptitle("DTBK 7-cluster radar profiles  (each feature min-max scaled)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_mix(df, out):
    cats = [c for c, _ in POI_CATS]
    labels = [l for _, l in POI_CATS]
    clusters = df["demo_cluster_pure"].astype(int).tolist()

    if not all(c in df.columns for c in cats):
        print(f"  [skip] missing POI category columns in cluster_attributes.csv")
        return

    X = df[cats].values.astype(float)
    # Row-normalise to shares
    X = X / X.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(clusters))
    cat_palette = ["#4c72b0", "#dd8452", "#55a868"]
    for i, (vals, lbl, col) in enumerate(zip(X.T, labels, cat_palette)):
        ax.bar([f"C{c}" for c in clusters], vals, bottom=bottom,
               color=col, edgecolor="white", label=lbl, linewidth=0.5)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.05:
                ax.text(j, b + v / 2, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Category share")
    ax.set_title("POI category mix per cluster (trip-weighted)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attrs", default="cluster_attributes.csv")
    p.add_argument("--out_dir", default="cluster_viz_7")
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.attrs).sort_values("demo_cluster_pure").reset_index(drop=True)
    print(f"Loaded {len(df)} clusters, {df.shape[1]} cols")
    print(f"Clusters: {df['demo_cluster_pure'].tolist()}")

    plot_bars(df, out / "bars_per_feature_7clusters.png")
    plot_heatmap(df, out / "heatmap_zscored.png")
    plot_radar(df, out / "radar_per_cluster.png")
    plot_category_mix(df, out / "category_mix_by_cluster.png")
    for f in out.iterdir():
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
