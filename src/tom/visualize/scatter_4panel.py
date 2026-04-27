#!/usr/bin/env python3
"""
15_scatter_4panel.py — 4-panel temporal scatter (linear + log views)

Adapted from /scratch/sx2490/mobility/tokyo_poi_prediction/scripts/12_scatter_4panel.py
for NYC Metro experiment outputs.

  Top row:    linear scale
  Bottom row: log-log scale, clipped to focus on the informative range
  Left:       in-sample (2021)
  Right:      out-of-sample (2022)

POI status (stable / new / disappeared) is computed from the actual visit sets
recorded in the prediction files (true_poi_id field).
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

BASE = "/scratch/sx2490/econai/nyc_metro"


def build_mass_streaming(path):
    """Stream JSONL and build actual + predicted distributions without holding all records."""
    actual, predicted = Counter(), Counter()
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            t = p.get("true_poi_id") or ""
            if t:
                actual[t] += 1
            nc = p.get("next_choice") or ""
            if nc:
                predicted[nc] += 1
    at = max(sum(actual.values()), 1)
    pt = max(sum(predicted.values()), 1)
    return (
        {k: v / at for k, v in actual.items()},
        {k: v / pt for k, v in predicted.items()},
        n,
    )


def get_status(train_actual, test_actual):
    """Classify POIs as stable / new / disappeared from actual visit sets."""
    train_set = set(train_actual.keys())
    test_set = set(test_actual.keys())
    stable = train_set & test_set
    new = test_set - train_set
    disapp = train_set - test_set
    return stable, new, disapp


def draw_panel(ax, actual, pred, stable, new_set, disapp_set, title, log_mode=False):
    all_pois = set(actual) | set(pred)

    LOG_LO = 2e-6
    LOG_HI = 5e-2

    sx, sy, nx, ny, dx, dy = [], [], [], [], [], []
    for pid in all_pois:
        a = actual.get(pid, 0)
        p = pred.get(pid, 0)
        if log_mode:
            if a == 0 and p == 0:
                continue
            a = max(a, LOG_LO * 0.7)
            p = max(p, LOG_LO * 0.7)
        if pid in new_set:
            nx.append(a); ny.append(p)
        elif pid in disapp_set:
            dx.append(a); dy.append(p)
        else:
            sx.append(a); sy.append(p)

    ax.scatter(sx, sy, s=5, alpha=0.3, c="#4C72B0", label=f"Stable ({len(sx):,})",
               zorder=3, rasterized=True)
    if nx:
        ax.scatter(nx, ny, s=14, alpha=0.6, c="#C44E52", marker="^",
                   label=f"New ({len(nx):,})", zorder=4)
    if dx:
        ax.scatter(dx, dy, s=14, alpha=0.6, c="#DD8452", marker="v",
                   label=f"Disappeared ({len(dx):,})", zorder=4)

    if log_mode:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(LOG_LO, LOG_HI)
        ax.set_ylim(LOG_LO, LOG_HI)
        ax.plot([LOG_LO, LOG_HI], [LOG_LO, LOG_HI], "k--", alpha=0.3, lw=1)
        ax.set_xlabel("Actual visit mass (log)")
        ax.set_ylabel("Predicted visit mass (log)")
    else:
        hi = max(max(actual.values(), default=0.001),
                 max(pred.values(), default=0.001)) * 1.1
        ax.plot([0, hi], [0, hi], "k--", alpha=0.3, lw=1)
        ax.set_xlim(-hi * 0.02, hi)
        ax.set_ylim(-hi * 0.02, hi)
        ax.set_xlabel("Actual visit mass")
        ax.set_ylabel("Predicted visit mass")

    # Annotate top POIs by actual mass
    n_labels = 12 if not log_mode else 18
    top = sorted(actual.items(), key=lambda x: -x[1])[:n_labels]
    for pid, _ in top:
        name = pid.split("|")[0][:18]
        px = actual.get(pid, LOG_LO if log_mode else 0)
        py = pred.get(pid, LOG_LO if log_mode else 0)
        if log_mode:
            px = max(px, LOG_LO * 0.7)
            py = max(py, LOG_LO * 0.7)
        ax.annotate(name, (px, py), fontsize=5.5, alpha=0.85,
                    xytext=(4, 3), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              alpha=0.8, lw=0))

    ax.set_title(title)
    ax.legend(loc="upper left", markerscale=1.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp01")
    parser.add_argument("--pred_train", default=None)
    parser.add_argument("--pred_test", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    exp_dir = f"{BASE}/experiments/{args.exp}"
    pred_train = args.pred_train or f"{exp_dir}/pred_insample_2021.jsonl"
    pred_test = args.pred_test or f"{exp_dir}/pred_oos_2022.jsonl"
    out_dir = args.out_dir or f"{exp_dir}/eval/figures"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Streaming predictions from {pred_train}...")
    is_actual, is_pred, n_is = build_mass_streaming(pred_train)
    print(f"  {n_is:,} predictions, {len(is_actual):,} unique actual POIs, {len(is_pred):,} unique predicted POIs")

    print(f"Streaming predictions from {pred_test}...")
    oos_actual, oos_pred, n_oos = build_mass_streaming(pred_test)
    print(f"  {n_oos:,} predictions, {len(oos_actual):,} unique actual POIs, {len(oos_pred):,} unique predicted POIs")

    stable, new, disapp = get_status(is_actual, oos_actual)
    print(f"  Stable: {len(stable):,}, New: {len(new):,}, Disappeared: {len(disapp):,}")

    print("Drawing 4-panel figure...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    draw_panel(axes[0, 0], is_actual, is_pred, stable, set(), disapp,
               f"In-Sample 2021 ({n_is:,} preds)", log_mode=False)
    draw_panel(axes[0, 1], oos_actual, oos_pred, stable, new, set(),
               f"OOS 2022 ({n_oos:,} preds)", log_mode=False)
    draw_panel(axes[1, 0], is_actual, is_pred, stable, set(), disapp,
               "In-Sample 2021 — Log Scale", log_mode=True)
    draw_panel(axes[1, 1], oos_actual, oos_pred, stable, new, set(),
               "OOS 2022 — Log Scale", log_mode=True)

    fig.suptitle(f"NYC Metro [{args.exp}] — Actual vs Predicted Visit Mass",
                 fontsize=14, y=1.00)
    fig.tight_layout()

    out_png = f"{out_dir}/temporal_scatter_4panel.png"
    out_pdf = f"{out_dir}/temporal_scatter_4panel.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
