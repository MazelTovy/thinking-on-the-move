#!/usr/bin/env python3
"""19_tail_slicing.py — Slice OOS 2022 performance by POI/cluster/category rarity.

Question: where does LoRA beat lgb_rank most? Hypothesis: on the tail.

Reads pred_*.jsonl for LoRA and lgb_rank on the same records, groups visits
by:
  - truth POI cluster_freq quintile (tail vs blockbuster)
  - truth POI sub_category rarity (top-15 vs rest)
  - user's demographic cluster size

and reports top1/top5 for each group.
"""
import json, sys, argparse, os
from collections import defaultdict, Counter


def load_preds(path):
    by_id = {}
    n = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r.get('cuebiq_id'), r.get('stop_ts'), r.get('true_poi_id'))
            by_id[key] = r
            n += 1
    return by_id, n


def slice_metrics(records, slice_fn, labels=None):
    """Compute top1/top5 per slice. slice_fn(rec) -> bucket name."""
    buckets = defaultdict(lambda: {'n': 0, 'top1': 0, 'top5': 0})
    for r in records:
        b = slice_fn(r)
        truth = r.get('true_poi_id')
        next_c = r.get('next_choice')
        top5 = [x.get('poi') for x in (r.get('top_5') or [])]
        buckets[b]['n'] += 1
        if next_c == truth:
            buckets[b]['top1'] += 1
        if truth in top5:
            buckets[b]['top5'] += 1
    rows = []
    for k, v in buckets.items():
        if v['n'] == 0: continue
        rows.append((k, v['n'], v['top1']/v['n'], v['top5']/v['n']))
    rows.sort(key=lambda x: (labels.index(x[0]) if labels and x[0] in labels else 99, -x[1]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="records_oos_2022.jsonl (for candidate metadata)")
    p.add_argument("--lora_pred", required=True)
    p.add_argument("--lgb_pred", required=True)
    p.add_argument("--xgb_pred", default="")
    args = p.parse_args()

    print("Loading preds...")
    lora, n_lora = load_preds(args.lora_pred)
    lgb,  n_lgb  = load_preds(args.lgb_pred)
    xgb,  n_xgb  = load_preds(args.xgb_pred) if args.xgb_pred else ({}, 0)
    print(f"  LoRA: {n_lora:,}  lgb: {n_lgb:,}  xgb: {n_xgb:,}")

    # Build POI metadata from records: truth_poi → cluster_freq, sub_category
    print("Loading records for POI metadata...")
    truth_meta = {}
    cluster_sizes = Counter()
    subcat_counts = Counter()
    with open(args.records) as f:
        for line in f:
            rec = json.loads(line)
            truth = rec.get('true_poi_id')
            cluster_sizes[rec.get('cluster')] += 1
            for c in (rec.get('candidates') or []):
                if c['poi_id'] == truth:
                    truth_meta[truth] = {
                        'cluster_freq': c.get('cluster_freq', 0),
                        'sub_category': c.get('sub_category', ''),
                        'polygon_type': c.get('polygon_type', ''),
                    }
                    subcat_counts[c.get('sub_category', '')] += 1
                    break
    top15_cats = set([c for c, _ in subcat_counts.most_common(15)])
    print(f"  {len(truth_meta):,} truth POIs; {len(cluster_sizes)} clusters")

    def enrich(recs):
        out = []
        for r in recs.values():
            truth = r.get('true_poi_id')
            meta = truth_meta.get(truth) or {}
            r2 = dict(r)
            r2['_truth_cluster_freq'] = meta.get('cluster_freq', 0)
            r2['_truth_subcat'] = meta.get('sub_category', '')
            r2['_truth_polygon'] = meta.get('polygon_type', '')
            r2['_cluster_size'] = cluster_sizes[r.get('cluster')]
            out.append(r2)
        return out

    lora_recs = enrich(lora)
    lgb_recs  = enrich(lgb)
    xgb_recs  = enrich(xgb) if xgb else []

    # ── slice 1: truth POI cluster_freq quintile ───────────────────────────
    def freq_bucket(r):
        f = r['_truth_cluster_freq']
        if f == 0:     return 'Q0 (freq=0)'
        if f <= 10:    return 'Q1 (1-10)'
        if f <= 50:    return 'Q2 (11-50)'
        if f <= 200:   return 'Q3 (51-200)'
        if f <= 1000:  return 'Q4 (201-1000)'
        return 'Q5 (>1000)'

    order_freq = ['Q0 (freq=0)','Q1 (1-10)','Q2 (11-50)','Q3 (51-200)','Q4 (201-1000)','Q5 (>1000)']

    print("\n" + "="*76)
    print(" Slice 1: OOS 2022 top1/top5 by truth POI cluster_freq bucket")
    print("="*76)
    print(f"{'bucket':<18}{'n':>8}  {'LoRA t1':>8}{'lgb t1':>8}{'Δ':>8}  {'LoRA t5':>8}{'lgb t5':>8}{'Δ':>8}")
    print('-'*76)
    lora_b = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lora_recs, freq_bucket, order_freq))
    lgb_b  = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lgb_recs,  freq_bucket, order_freq))
    for b in order_freq:
        if b not in lora_b: continue
        n, lt1, lt5 = lora_b[b]; _, gt1, gt5 = lgb_b.get(b, (0,0,0))
        print(f"{b:<18}{n:>8}  {lt1:>8.4f}{gt1:>8.4f}{(lt1-gt1)*100:>+7.2f}pp  "
              f"{lt5:>8.4f}{gt5:>8.4f}{(lt5-gt5)*100:>+7.2f}pp")

    # ── slice 2: truth sub_category top-15 vs rest ─────────────────────────
    def subcat_bucket(r):
        return 'top-15 (in lgb features)' if r['_truth_subcat'] in top15_cats else 'rare (OOV for lgb)'
    order_sc = ['top-15 (in lgb features)', 'rare (OOV for lgb)']
    print("\n" + "="*76)
    print(" Slice 2: OOS 2022 top1/top5 by truth sub_category rarity")
    print("="*76)
    print(f"{'bucket':<28}{'n':>8}  {'LoRA t1':>8}{'lgb t1':>8}{'Δ':>8}  {'LoRA t5':>8}{'lgb t5':>8}{'Δ':>8}")
    print('-'*76)
    lora_b = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lora_recs, subcat_bucket, order_sc))
    lgb_b  = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lgb_recs,  subcat_bucket, order_sc))
    for b in order_sc:
        if b not in lora_b: continue
        n, lt1, lt5 = lora_b[b]; _, gt1, gt5 = lgb_b.get(b, (0,0,0))
        print(f"{b:<28}{n:>8}  {lt1:>8.4f}{gt1:>8.4f}{(lt1-gt1)*100:>+7.2f}pp  "
              f"{lt5:>8.4f}{gt5:>8.4f}{(lt5-gt5)*100:>+7.2f}pp")

    # ── slice 3: cluster size (how many 2022 visits came from this cluster) ─
    sizes_sorted = sorted(cluster_sizes.items(), key=lambda x: -x[1])
    cluster_tier = {}
    for i, (c, _) in enumerate(sizes_sorted):
        if   i < 10: cluster_tier[c] = 'top-10 largest'
        elif i < 20: cluster_tier[c] = 'mid (rank 11-20)'
        else:        cluster_tier[c] = 'small (rank 21-30)'
    def cluster_bucket(r): return cluster_tier.get(r.get('cluster'), 'unknown')
    order_cl = ['top-10 largest','mid (rank 11-20)','small (rank 21-30)']
    print("\n" + "="*76)
    print(" Slice 3: OOS 2022 top1/top5 by cluster size tier")
    print("="*76)
    print(f"{'bucket':<22}{'n':>8}  {'LoRA t1':>8}{'lgb t1':>8}{'Δ':>8}  {'LoRA t5':>8}{'lgb t5':>8}{'Δ':>8}")
    print('-'*76)
    lora_b = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lora_recs, cluster_bucket, order_cl))
    lgb_b  = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lgb_recs,  cluster_bucket, order_cl))
    for b in order_cl:
        if b not in lora_b: continue
        n, lt1, lt5 = lora_b[b]; _, gt1, gt5 = lgb_b.get(b, (0,0,0))
        print(f"{b:<22}{n:>8}  {lt1:>8.4f}{gt1:>8.4f}{(lt1-gt1)*100:>+7.2f}pp  "
              f"{lt5:>8.4f}{gt5:>8.4f}{(lt5-gt5)*100:>+7.2f}pp")

    # ── slice 4: truth polygon type (data-quality view) ─────────────────────
    def poly_bucket(r): return r['_truth_polygon'] or 'unknown'
    order_pol = ['OWNED','SHARED_DISTINCT','SHARED_BUILDING','FALLBACK']
    print("\n" + "="*76)
    print(" Slice 4: OOS 2022 top1/top5 by truth polygon_type")
    print("="*76)
    print(f"{'bucket':<22}{'n':>8}  {'LoRA t1':>8}{'lgb t1':>8}{'Δ':>8}  {'LoRA t5':>8}{'lgb t5':>8}{'Δ':>8}")
    print('-'*76)
    lora_b = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lora_recs, poly_bucket, order_pol))
    lgb_b  = dict((b, (n, t1, t5)) for b, n, t1, t5 in slice_metrics(lgb_recs,  poly_bucket, order_pol))
    for b in order_pol:
        if b not in lora_b: continue
        n, lt1, lt5 = lora_b[b]; _, gt1, gt5 = lgb_b.get(b, (0,0,0))
        print(f"{b:<22}{n:>8}  {lt1:>8.4f}{gt1:>8.4f}{(lt1-gt1)*100:>+7.2f}pp  "
              f"{lt5:>8.4f}{gt5:>8.4f}{(lt5-gt5)*100:>+7.2f}pp")


if __name__ == "__main__":
    main()
