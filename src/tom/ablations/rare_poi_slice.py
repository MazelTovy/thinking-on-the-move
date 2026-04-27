#!/usr/bin/env python3
"""25_rare_poi_slice.py — LoRA vs lgb_rank on globally-rare POIs.

Previous tail slicing (19_tail_slicing.py) bucketed by the truth POI's
cluster_freq for the user's cluster. That confounds two things:
  (i) the POI is globally unpopular (true cold-tail) and
  (ii) the POI is popular overall but not in this user's cluster.
Here we use the GLOBAL freq — sum of cluster_freq across all 30 clusters
— from the 2021 training-prior pool to identify POIs that are rare for
every cluster. These are where LoRA's name-semantic signal is most
likely to beat lgb's popularity prior (which has nothing to lean on).
"""
import argparse
import json
import os
from collections import defaultdict


def build_global_freq(cbg_poi_dir):
    """Aggregate cluster_freq across all per-cluster pool CSVs."""
    import csv, re
    pat = re.compile(r"^cluster_(\d+)_pois\.csv$")
    gf = defaultdict(float)
    for fname in sorted(os.listdir(cbg_poi_dir)):
        m = pat.match(fname)
        if not m: continue
        with open(f"{cbg_poi_dir}/{fname}") as f:
            for row in csv.DictReader(f):
                gf[row["poi_id"]] += float(row.get("cluster_freq", 0) or 0)
    return gf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--lora_pred", required=True)
    p.add_argument("--lgb_pred", required=True)
    p.add_argument("--cbg_poi_dir", required=True)
    args = p.parse_args()

    print("Building global POI freq table...")
    gf = build_global_freq(args.cbg_poi_dir)
    print(f"  {len(gf):,} POIs with global freq")

    # Map truth -> global freq via streaming the records once
    truth_gf = {}
    with open(args.records) as f:
        for line in f:
            r = json.loads(line)
            tid = r.get("true_poi_id")
            if tid and tid not in truth_gf:
                truth_gf[tid] = gf.get(tid, 0.0)
    print(f"  {len(truth_gf):,} unique truth POIs in records")

    def bucket(f):
        if f == 0:      return "Q0 (global_freq=0)"
        if f <= 50:     return "Q1 (1-50)"
        if f <= 200:    return "Q2 (51-200)"
        if f <= 1000:   return "Q3 (201-1000)"
        if f <= 5000:   return "Q4 (1001-5000)"
        return "Q5 (>5000)"
    order = ["Q0 (global_freq=0)","Q1 (1-50)","Q2 (51-200)",
             "Q3 (201-1000)","Q4 (1001-5000)","Q5 (>5000)"]

    # Stream both pred files aligned by key
    def load(path):
        d = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                k = (r.get("cuebiq_id"), r.get("stop_ts"), r.get("true_poi_id"))
                d[k] = (r.get("next_choice"),
                        [x.get("poi") for x in (r.get("top_5") or [])])
        return d

    print("Loading LoRA pred...")
    lora = load(args.lora_pred)
    print(f"  {len(lora):,} records")
    print("Loading lgb pred...")
    lgb  = load(args.lgb_pred)
    print(f"  {len(lgb):,} records")

    buckets = defaultdict(lambda: {"n":0, "lora_t1":0, "lgb_t1":0,
                                    "lora_t5":0, "lgb_t5":0})
    for key, lo in lora.items():
        gb = lgb.get(key)
        if not gb: continue
        truth = key[2]
        g_freq = truth_gf.get(truth, 0.0)
        b = bucket(g_freq)
        buckets[b]["n"] += 1
        l_next, l_top5 = lo
        g_next, g_top5 = gb
        if l_next == truth: buckets[b]["lora_t1"] += 1
        if g_next == truth: buckets[b]["lgb_t1"]  += 1
        if truth in l_top5: buckets[b]["lora_t5"] += 1
        if truth in g_top5: buckets[b]["lgb_t5"]  += 1

    print(f"\n{'bucket':<22}{'n':>8}  {'LoRA t1':>8}{'lgb t1':>8}{'Δt1':>9}  {'LoRA t5':>8}{'lgb t5':>8}{'Δt5':>9}")
    print('-'*82)
    for b in order:
        v = buckets[b]
        if v["n"] == 0: continue
        lt1, gt1 = v["lora_t1"]/v["n"], v["lgb_t1"]/v["n"]
        lt5, gt5 = v["lora_t5"]/v["n"], v["lgb_t5"]/v["n"]
        print(f"{b:<22}{v['n']:>8}  {lt1:>8.4f}{gt1:>8.4f}{(lt1-gt1)*100:>+8.2f}pp  "
              f"{lt5:>8.4f}{gt5:>8.4f}{(lt5-gt5)*100:>+8.2f}pp")


if __name__ == "__main__":
    main()
