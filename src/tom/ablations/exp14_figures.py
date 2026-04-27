#!/usr/bin/env python3
"""27_exp14_figures.py — Paper-ready figures for Exp 14 tail slicing.

Generates three grouped bar charts comparing LoRA vs lgb_rank Top-1 on
exp02_wcbg OOS 2022, broken out by:
  - polygon_type of truth POI
  - cluster_freq bucket of truth POI
  - cluster size tier (top-10 / mid / small)

Outputs to experiments/exp02_wcbg/eval/figures/tail_slice_*.png|.pdf.
"""
import json
import os
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "/scratch/sx2490/econai/nyc_metro/experiments/exp02_wcbg"
FIG_DIR = f"{BASE}/eval/figures"

# Consistent colors across all three figures
LORA_COLOR = "#E65100"  # burnt orange
LGB_COLOR = "#1565C0"   # dark blue


def load_preds_streaming(path):
    """Yield (key, next_choice, top5_pois) tuples without loading everything."""
    for line in open(path):
        r = json.loads(line)
        yield (
            (r.get("cuebiq_id"), r.get("stop_ts"), r.get("true_poi_id")),
            r.get("next_choice"),
            [x.get("poi") for x in (r.get("top_5") or [])],
        )


def build_truth_meta(records_path):
    """From records, map truth POI → (polygon_type, cluster_freq, cluster_id)."""
    truth_meta = {}
    cluster_sizes = Counter()
    for line in open(records_path):
        r = json.loads(line)
        truth = r.get("true_poi_id")
        cluster = r.get("cluster")
        cluster_sizes[cluster] += 1
        if truth in truth_meta:
            continue
        for c in (r.get("candidates") or []):
            if c.get("poi_id") == truth:
                truth_meta[truth] = {
                    "polygon_type": c.get("polygon_type", "") or "unknown",
                    "cluster_freq": int(c.get("cluster_freq", 0) or 0),
                }
                break
    return truth_meta, cluster_sizes


def freq_bucket(f):
    if f == 0:     return "Q0\n(=0)"
    if f <= 10:    return "Q1\n(1–10)"
    if f <= 50:    return "Q2\n(11–50)"
    if f <= 200:   return "Q3\n(51–200)"
    if f <= 1000:  return "Q4\n(201–1000)"
    return "Q5\n(>1000)"


FREQ_ORDER = ["Q0\n(=0)", "Q1\n(1–10)", "Q2\n(11–50)",
              "Q3\n(51–200)", "Q4\n(201–1000)", "Q5\n(>1000)"]


def aggregate_slices(records_path, lora_pred, lgb_pred):
    truth_meta, cluster_sizes = build_truth_meta(records_path)
    print(f"  {len(truth_meta):,} unique truth POIs, {len(cluster_sizes)} clusters")

    # Stream pred files and merge on key
    lora_by_key = {}
    for key, nc, t5 in load_preds_streaming(lora_pred):
        lora_by_key[key] = (nc, t5)
    print(f"  loaded {len(lora_by_key):,} LoRA preds")

    # For lgb we can merge as we stream (keep only intersection)
    poly_buckets = defaultdict(lambda: {"n": 0, "lora_t1": 0, "lgb_t1": 0})
    freq_buckets = defaultdict(lambda: {"n": 0, "lora_t1": 0, "lgb_t1": 0})
    cluster_buckets = defaultdict(lambda: {"n": 0, "lora_t1": 0, "lgb_t1": 0})

    # Cluster tier by size
    by_size = cluster_sizes.most_common()
    cluster_tier = {}
    for i, (cid, _) in enumerate(by_size):
        if   i < 10: tier = "top-10 largest"
        elif i < 20: tier = "mid (rank 11–20)"
        else:        tier = "small (rank 21–30)"
        cluster_tier[cid] = tier

    # Stream lgb + merge truth
    n_merged = 0
    for key, lgb_nc, _ in load_preds_streaming(lgb_pred):
        if key not in lora_by_key:
            continue
        n_merged += 1
        truth = key[2]
        meta = truth_meta.get(truth) or {}
        lora_nc, _ = lora_by_key[key]
        lo_hit = 1 if lora_nc == truth else 0
        lg_hit = 1 if lgb_nc == truth else 0

        # polygon_type
        pt = meta.get("polygon_type", "unknown")
        poly_buckets[pt]["n"] += 1
        poly_buckets[pt]["lora_t1"] += lo_hit
        poly_buckets[pt]["lgb_t1"]  += lg_hit

        # cluster_freq
        fb = freq_bucket(meta.get("cluster_freq", 0))
        freq_buckets[fb]["n"] += 1
        freq_buckets[fb]["lora_t1"] += lo_hit
        freq_buckets[fb]["lgb_t1"]  += lg_hit

        # cluster tier — use the visit's cluster, not truth POI's
        # but we don't have cluster in key... we need to look it up
        # Actually the key (cuebiq_id, stop_ts, true_poi_id) doesn't carry cluster.
        # We need to add it. Let me restream records to map key→cluster.
        # Simpler: add cluster to key during load.

    print(f"  merged {n_merged:,} visits on key")
    return poly_buckets, freq_buckets, cluster_tier, cluster_sizes


def aggregate_cluster_slice(records_path, lora_pred, lgb_pred, cluster_tier):
    """Second-pass aggregation for cluster-size tier, because the pred files
    do not carry cluster_id in the join key."""
    # Load cluster_id per visit
    visit_cluster = {}
    for line in open(records_path):
        r = json.loads(line)
        key = (r.get("cuebiq_id"), r.get("stop_ts"), r.get("true_poi_id"))
        visit_cluster[key] = r.get("cluster")

    # Load LoRA
    lora_by_key = {}
    for key, nc, _ in load_preds_streaming(lora_pred):
        lora_by_key[key] = nc

    buckets = defaultdict(lambda: {"n": 0, "lora_t1": 0, "lgb_t1": 0})
    for key, lgb_nc, _ in load_preds_streaming(lgb_pred):
        if key not in lora_by_key or key not in visit_cluster:
            continue
        cid = visit_cluster[key]
        tier = cluster_tier.get(cid, "unknown")
        truth = key[2]
        buckets[tier]["n"] += 1
        if lora_by_key[key] == truth: buckets[tier]["lora_t1"] += 1
        if lgb_nc == truth:           buckets[tier]["lgb_t1"]  += 1
    return buckets


def plot_grouped_bars(buckets, order, title_en, xlabel,
                      out_path, sort_by_delta=False, highlight_extremes=False):
    names = [k for k in order if k in buckets and buckets[k]["n"] > 0]
    if sort_by_delta:
        names.sort(key=lambda k: (buckets[k]["lora_t1"] - buckets[k]["lgb_t1"]) / buckets[k]["n"])
    n_groups = len(names)

    lora_vals = [buckets[k]["lora_t1"] / buckets[k]["n"] for k in names]
    lgb_vals  = [buckets[k]["lgb_t1"]  / buckets[k]["n"] for k in names]
    ns        = [buckets[k]["n"] for k in names]
    deltas    = [(l - g) * 100 for l, g in zip(lora_vals, lgb_vals)]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    x = list(range(n_groups))
    width = 0.38
    ax.bar([i - width/2 for i in x], lora_vals, width,
           color=LORA_COLOR, label="LoRA-Qwen7B", edgecolor="white", linewidth=0.5)
    ax.bar([i + width/2 for i in x], lgb_vals, width,
           color=LGB_COLOR,  label="LambdaRank (lgb)", edgecolor="white", linewidth=0.5)

    # Annotate values above bars
    for i, (lv, gv) in enumerate(zip(lora_vals, lgb_vals)):
        ax.text(i - width/2, lv + 0.005, f"{lv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, gv + 0.005, f"{gv:.3f}", ha="center", va="bottom", fontsize=8)

    # Delta labels above each group
    ymax_bars = max(max(lora_vals), max(lgb_vals))
    delta_y = ymax_bars + 0.06
    for i, d in enumerate(deltas):
        color = "#C62828" if d < 0 else ("#2E7D32" if d >= 2.0 else "#616161")
        sign = "+" if d >= 0 else ""
        weight = "bold" if abs(d) >= 2.0 else "normal"
        ax.text(i, delta_y, f"Δ = {sign}{d:.2f}pp", ha="center", va="bottom",
                fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(x)
    # Inline sample size in xtick label to avoid overlap with axis ticks
    ax.set_xticklabels(
        [f"{nm}\n(n={ni:,})" for nm, ni in zip(names, ns)],
        fontsize=9,
    )
    ax.set_ylabel("Top-1 Accuracy (OOS 2022)", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylim(0, delta_y + 0.03)
    ax.set_title(title_en, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}.png / .pdf")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    records = f"{BASE}/infer_records/records_oos_2022.jsonl"
    lora_pred = f"{BASE}/pred_oos_2022.jsonl"
    lgb_pred = f"{BASE}/baselines/lgb_rank/pred_oos_2022.jsonl"

    print("=== Aggregating polygon + freq slices ===")
    poly, freq, cluster_tier, cluster_sizes = aggregate_slices(records, lora_pred, lgb_pred)

    # Figure 1: polygon_type (sorted by delta, ascending)
    plot_grouped_bars(
        poly,
        order=["FALLBACK", "SHARED_BUILDING", "OWNED", "SHARED_DISTINCT"],
        title_en="LoRA advantage scales with POI-name informativeness",
        xlabel="Truth POI polygon type (curation quality, low → high)",
        out_path=f"{FIG_DIR}/tail_slice_polygon_oos_2022",
    )

    # Figure 2: cluster_freq bucket (Q0..Q5 in order)
    plot_grouped_bars(
        freq,
        order=FREQ_ORDER,
        title_en="LoRA advantage peaks at mid-popularity POIs, not the tail",
        xlabel="Truth POI cluster-frequency bucket",
        out_path=f"{FIG_DIR}/tail_slice_cluster_freq_oos_2022",
    )

    # Figure 3: cluster size tier
    print("=== Aggregating cluster-size tier slice ===")
    cluster_buckets = aggregate_cluster_slice(records, lora_pred, lgb_pred, cluster_tier)
    plot_grouped_bars(
        cluster_buckets,
        order=["top-10 largest", "mid (rank 11–20)", "small (rank 21–30)"],
        title_en="LoRA advantage is flat across cluster size — persona reasoning is not the driver",
        xlabel="Demographic cluster size tier",
        out_path=f"{FIG_DIR}/tail_slice_cluster_size_oos_2022",
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
