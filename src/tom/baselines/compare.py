#!/usr/bin/env python3
"""Summarize all baseline metrics + LoRA into one compact table.

Usage: python3 compare_baselines.py [exp_name]  (default: exp02_wcbg)
"""
import json
import os
import sys

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp02_wcbg"
BASE = f"/scratch/sx2490/econai/nyc_metro/experiments/{EXP}"
MODELS = [(f"LoRA ({EXP})", f"{BASE}/eval/metrics.json")]
for d in ["frequency", "gravity", "huff", "mnl_grid", "mnl_rich",
          "deep_gravity", "lgb_rank", "xgb_rank"]:
    MODELS.append((d, f"{BASE}/baselines/{d}/eval/metrics.json"))


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  -  "


def rank_keys(s):
    return {
        "spearman": s.get("spearman"),
        "js": s.get("js_divergence"),
        "ndcg10": (s.get("ndcg") or {}).get("10"),
        "ndcg50": (s.get("ndcg") or {}).get("50"),
    }


def print_split(split):
    print(f"\n========== {split} ==========")
    print(f"{'model':<16} {'spear':>7} {'JS':>7} {'ndcg10':>7} {'ndcg50':>7}")
    for name, path in MODELS:
        m = json.load(open(path))
        k = rank_keys(m.get(split, {}))
        print(f"{name:<16} {fmt(k['spearman']):>7} {fmt(k['js']):>7} "
              f"{fmt(k['ndcg10']):>7} {fmt(k['ndcg50']):>7}")


def print_visit_accuracy():
    print(f"\n========== visit_accuracy ==========")
    print(f"{'model':<16} {'2021 top1':>10} {'2021 top5':>10} {'2022 top1':>10} {'2022 top5':>10}")
    for name, path in MODELS:
        m = json.load(open(path))
        va = m.get("visit_accuracy", {})
        v21 = va.get("in_sample_2021_val", {})
        v22 = va.get("oos_2022", {})
        print(f"{name:<16} {v21.get('top1_accuracy', '-'):>10.4f} "
              f"{v21.get('top5_accuracy', '-'):>10.4f} "
              f"{v22.get('top1_accuracy', '-'):>10.4f} "
              f"{v22.get('top5_accuracy', '-'):>10.4f}")


def print_fit_params():
    print(f"\n========== fit parameters ==========")
    for name, path in MODELS:
        pred_path = path.replace("/eval/metrics.json", "/pred_insample_2021.jsonl")
        if not os.path.exists(pred_path) or "LoRA" in name:
            continue
        with open(pred_path) as f:
            r = json.loads(f.readline())
        a = r.get("baseline_alpha")
        b = r.get("baseline_beta")
        fit = r.get("baseline_fit")
        if a is not None:
            nll = fit.get("nll") if fit else None
            print(f"  {name:<12} alpha={a}  beta={b}  nll={fmt(nll)}  n_fit={fit.get('n_fit') if fit else '-'}")
        elif fit:
            print(f"  {name:<12} {fit}")


if __name__ == "__main__":
    for split in ["in_sample_2021_val", "oos_2022", "oos_2022_stable"]:
        print_split(split)
    print_visit_accuracy()
    print_fit_params()
