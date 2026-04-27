#!/usr/bin/env python3
"""23_hybrid_decode.py — Combine LoRA top-5 with lgb tail ranks 6–30.

Rationale: LoRA wins visit-level top-1/top-5 but loses aggregate NDCG@10
because it emits mass only for its top-5, leaving popular POIs at ranks 6+
with zero predicted mass. LambdaMART assigns a score to all 30 candidates.
This hybrid keeps LoRA's top-5 verbatim and fills positions 6–30 by ranking
the remaining 25 candidates by the lgb score.

Input:
  --lora_pred   LoRA pred_*.jsonl (must have top_5, candidates)
  --lgb_pred    lgb_rank pred_*.jsonl (must have candidate_scores after fix)
Output:
  --out         hybrid pred_*.jsonl with new field `top_k` containing 10
                entries {poi, prob}; top_5 kept for backward-compat eval.

Prob assignment:
  Positions 1–5: LoRA's top_5 probs (sum to ≈1.0 if LoRA's output is
  valid). We keep them unscaled.
  Positions 6–10: softmax over lgb scores of the 5 best non-LoRA
  candidates, scaled by `tail_mass` (default 0.3) so the aggregate
  distribution contains meaningful mass on positions 6–10 without
  overwhelming LoRA's top-5 decision.
"""
import argparse
import json
import math
import os


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


def load_pred(path, keys):
    """Load a pred JSONL into a dict keyed by (cuebiq_id, stop_ts, true_poi_id)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            k = tuple(r.get(x) for x in keys)
            out[k] = r
    return out


def hybrid_record(lora_rec, lgb_rec, tail_mass):
    lora_top5 = lora_rec.get("top_5") or []
    lora_pois = [item["poi"] for item in lora_top5]
    lora_set = set(lora_pois)

    # lgb candidate_scores: [{"poi": ..., "score": float, "prob": float}, ...]
    lgb_scores = lgb_rec.get("candidate_scores") or []
    # Filter out POIs already in LoRA's top-5
    rest = [x for x in lgb_scores if x["poi"] not in lora_set]
    # Rank remaining 25 by lgb score desc
    rest.sort(key=lambda x: x["score"], reverse=True)
    # Take top-5 of rest → positions 6-10
    tail_top5 = rest[:5]

    if not tail_top5:
        return lora_rec  # degenerate (shouldn't happen with 30 candidates)

    # Softmax over tail scores, scaled by tail_mass
    tail_scores = [x["score"] for x in tail_top5]
    tail_sm = softmax(tail_scores)
    tail_probs = [p * tail_mass for p in tail_sm]

    # Build top_k field (10 entries)
    top_k = []
    for item in lora_top5:
        top_k.append({"poi": item["poi"], "prob": float(item["prob"])})
    for x, p in zip(tail_top5, tail_probs):
        top_k.append({"poi": x["poi"], "prob": round(float(p), 4)})

    out = dict(lora_rec)
    out["top_k"] = top_k
    out["hybrid_source"] = "lora_top5+lgb_tail5"
    out["hybrid_tail_mass"] = tail_mass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_pred", required=True)
    p.add_argument("--lgb_pred", required=True,
                   help="lgb_rank pred with candidate_scores field")
    p.add_argument("--out", required=True)
    p.add_argument("--tail_mass", type=float, default=0.3,
                   help="Fraction of mass budget assigned to lgb positions 6-10")
    args = p.parse_args()

    print(f"Loading LoRA pred: {args.lora_pred}")
    print(f"Loading lgb pred : {args.lgb_pred}")
    print(f"tail_mass = {args.tail_mass}")

    keys = ("cuebiq_id", "stop_ts", "true_poi_id")
    # Stream both files to keep memory low
    print("Streaming merge...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_in, n_out, n_missing_lgb = 0, 0, 0

    # First pass: load lgb into dict (smaller, each record has candidate_scores)
    lgb_by_key = {}
    with open(args.lgb_pred) as f:
        for line in f:
            r = json.loads(line)
            k = tuple(r.get(x) for x in keys)
            # Strip heavy fields to save memory
            lgb_by_key[k] = {"candidate_scores": r.get("candidate_scores")}
    print(f"  lgb records: {len(lgb_by_key):,}")

    with open(args.lora_pred) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n_in += 1
            k = tuple(r.get(x) for x in keys)
            lgb_rec = lgb_by_key.get(k)
            if not lgb_rec or not lgb_rec.get("candidate_scores"):
                n_missing_lgb += 1
                out = r  # fall back to LoRA alone (no hybrid field)
            else:
                out = hybrid_record(r, lgb_rec, args.tail_mass)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            if n_in % 20000 == 0:
                print(f"  {n_in:,} lines processed")

    print(f"done: {n_in:,} in → {n_out:,} out  (missing lgb: {n_missing_lgb:,})")


if __name__ == "__main__":
    main()
