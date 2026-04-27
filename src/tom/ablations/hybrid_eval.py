#!/usr/bin/env python3
"""24_hybrid_eval.py — Visit-level + aggregate metrics for hybrid predictions.

Computes the same metrics as 14_temporal_eval.py but reads a `top_k` field
(variable length K; defaults to 10 for hybrid) instead of `top_5`. Also
computes a "rank of truth" distribution across visits for sanity.

Inputs:
  --pred        hybrid pred JSONL with `top_k` or `top_5`
  --records     infer_records/records_oos_2022.jsonl (for truth verification)
  --out_json    metrics.json path
  --name        label for the run (printed in header)
"""
import argparse
import json
import math
import sys
from collections import defaultdict


def load_actual(records_path):
    """Build {poi_id: count} of actual 2022 visits for aggregate eval."""
    actual = defaultdict(int)
    n = 0
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            truth = r.get("true_poi_id")
            if truth:
                actual[truth] += 1
                n += 1
    return dict(actual), n


def eval_pred(pred_path, actual, name):
    n = n_top1 = n_top5 = n_top10 = 0
    pos_dist = defaultdict(int)  # rank of truth among top_k
    predicted = defaultdict(float)  # aggregate predicted mass per POI

    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            truth = r.get("true_poi_id")
            if not truth:
                continue
            n += 1

            # LoRA's `next_choice` is its authoritative top-1 (separate from
            # top_5 ordering — the two disagree ~69% of the time in LoRA's
            # outputs because the LLM generates next_choice and top_5 as
            # independent JSON fields). Using next_choice matches
            # 14_temporal_eval.py's convention.
            next_choice = r.get("next_choice")
            topk = r.get("top_k") or r.get("top_5") or []
            items = [(x.get("poi"), float(x.get("prob", 0))) for x in topk]
            pois = [p for p, _ in items]

            if next_choice == truth:
                n_top1 += 1
            if truth in pois[:5]:
                n_top5 += 1
            if truth in pois[:10]:
                n_top10 += 1

            try:
                rank_of_truth = pois.index(truth) + 1
                pos_dist[rank_of_truth] += 1
            except ValueError:
                pos_dist[0] += 1  # not in top_k

            for poi, prob in items:
                predicted[poi] += prob

    # Aggregate metrics: NDCG, Spearman, JS
    all_pois = sorted(set(actual) | set(predicted))
    try:
        import numpy as np
        from scipy.stats import spearmanr
    except ImportError:
        print("WARN: numpy/scipy not available; aggregate metrics skipped", file=sys.stderr)
        return {
            "name": name, "n_pred": n,
            "top1": n_top1 / max(n, 1),
            "top5": n_top5 / max(n, 1),
            "top10": n_top10 / max(n, 1),
            "truth_rank_hist": dict(sorted(pos_dist.items())),
        }

    act = np.array([actual.get(p, 0.0) for p in all_pois], dtype=float)
    pred = np.array([predicted.get(p, 0.0) for p in all_pois], dtype=float)
    act_norm = act / (act.sum() or 1.0)
    pred_norm = pred / (pred.sum() or 1.0)

    mask = act > 0
    spear = float(spearmanr(act[mask], pred[mask])[0]) if mask.sum() >= 2 else float("nan")

    m = (act_norm + pred_norm) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        js = float(
            0.5 * np.where(act_norm > 0, act_norm * np.log(act_norm / np.where(m > 0, m, 1e-300)), 0).sum()
          + 0.5 * np.where(pred_norm > 0, pred_norm * np.log(pred_norm / np.where(m > 0, m, 1e-300)), 0).sum()
        )

    sorted_actual = sorted(actual, key=actual.get, reverse=True)
    sorted_pred = sorted(predicted, key=predicted.get, reverse=True)
    ndcg = {}
    topk_recall = {}
    for k in [10, 20, 50]:
        top_pred = sorted_pred[:k]
        top_actual = sorted_actual[:k]
        dcg = sum(actual.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(top_pred))
        idcg = sum(actual.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(top_actual))
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
        overlap = len(set(top_actual) & set(top_pred))
        topk_recall[k] = overlap / max(len(top_actual), 1)

    return {
        "name": name,
        "n_pred": n,
        "top1": n_top1 / max(n, 1),
        "top5": n_top5 / max(n, 1),
        "top10": n_top10 / max(n, 1),
        "spearman": spear,
        "js_divergence": js,
        "ndcg_10": ndcg[10],
        "ndcg_20": ndcg[20],
        "ndcg_50": ndcg[50],
        "recall_at_10": topk_recall[10],
        "truth_rank_hist_top10": {k: pos_dist[k] for k in range(1, 11) if pos_dist[k]},
        "n_truth_not_in_topk": pos_dist.get(0, 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()

    print(f"Loading actual distribution from {args.records}")
    actual, n_visits = load_actual(args.records)
    print(f"  {n_visits:,} visits across {len(actual):,} unique truth POIs")

    print(f"Scoring {args.pred} ({args.name})")
    m = eval_pred(args.pred, actual, args.name)
    import os
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(m, f, indent=2)

    print("\n=== Results ===")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:<22} {v}")
        else:
            print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
