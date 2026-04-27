#!/usr/bin/env python3
"""18_resample_candidates.py — Resample candidate sets under varying difficulty.

Reads existing inference records (stop, origin, true_poi fixed) and rewrites
only the `candidates` field using a softer popularity weighting. This isolates
the "how hard is the candidate set" variable for the paper ablation:

  easy   freq_power=1.0  (default, current exp02_wcbg)
  medium freq_power=0.5  (square-root softening of popularity bias)
  hard   freq_power=0.0  (popularity signal removed, pure distance decay)

All other record fields (cluster, origin, stop_ts, true_poi_id, work_cbg,
true_polygon_type) are preserved so baselines and LoRA see the same visits.
"""

import argparse
import csv
import json
import math
import os
import random
import re


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def distance_decay(d, primary, fallback, hard_cap):
    if d > hard_cap:
        return 0.02
    if d > fallback:
        return 0.10
    if d > primary:
        return 0.35
    if d > primary / 2:
        return 0.70
    return 1.0


def weight(cluster_freq, d, freq_power, primary, fallback, hard_cap):
    pop = math.log1p(max(cluster_freq, 0))
    if freq_power != 1.0:
        pop = pop ** freq_power  # freq_power=0 → pop always 1.0 (uniform)
    return max(pop * distance_decay(d, primary, fallback, hard_cap), 1e-6)


def weighted_sample_without_replacement(items, weights, n, rng):
    chosen = []
    pool = list(zip(items, weights))
    for _ in range(min(n, len(pool))):
        total = sum(w for _, w in pool)
        if total <= 0:
            break
        r = rng.random() * total
        acc = 0.0
        for i, (item, w) in enumerate(pool):
            acc += w
            if acc >= r:
                chosen.append(item)
                pool.pop(i)
                break
    return chosen


def load_pool(cbg_poi_dir):
    pool_re = re.compile(r"^cluster_(\d+)_pois\.csv$")
    pools = {}
    for fname in sorted(os.listdir(cbg_poi_dir)):
        m = pool_re.match(fname)
        if not m:
            continue
        cid = m.group(1)
        rows = []
        with open(f"{cbg_poi_dir}/{fname}") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "poi_id": row["poi_id"],
                    "poi_name": row["poi_name"],
                    "poi_lat": float(row["poi_lat"]),
                    "poi_lon": float(row["poi_lon"]),
                    "polygon_type": row.get("polygon_type", ""),
                    "sub_category": row.get("sub_category", ""),
                    "cluster_freq": int(row.get("cluster_freq", 0)),
                    "synthetic_unseen": int(float(row.get("synthetic_unseen", 0) or 0)),
                })
        pools[cid] = rows
    return pools


def resample(pool, true_id, origin_lat, origin_lon, k, freq_power,
             primary, fallback, hard_cap, rng):
    annotated = []
    true_rec = None
    for c in pool:
        d = haversine_m(origin_lat, origin_lon, c["poi_lat"], c["poi_lon"])
        r = dict(c)
        r["dist_m"] = d
        annotated.append(r)
        if c["poi_id"] == true_id:
            true_rec = r

    if len(annotated) <= k:
        out = list(annotated)
        if true_rec and true_rec not in out:
            out.insert(0, true_rec)
        return out[:k]

    local = [r for r in annotated if r["dist_m"] <= primary]
    near  = [r for r in annotated if r["dist_m"] <= fallback]
    cap   = [r for r in annotated if r["dist_m"] <= hard_cap]
    min_viable = max(3, min(k, 10))
    if   len(local) >= min_viable: primary_pool = local
    elif len(near)  >= min_viable: primary_pool = near
    elif len(cap)   >= min_viable: primary_pool = cap
    else:                           primary_pool = annotated
    backup_pool = cap if len(cap) >= min_viable else (near if len(near) >= min_viable else annotated)

    chosen = []
    chosen_ids = set()
    if true_rec is not None:
        chosen.append(true_rec)
        chosen_ids.add(true_rec["poi_id"])

    def extend_from(src):
        remaining = k - len(chosen)
        if remaining <= 0:
            return
        avail = [r for r in src if r["poi_id"] not in chosen_ids]
        if not avail:
            return
        ws = [weight(r["cluster_freq"], r["dist_m"], freq_power, primary, fallback, hard_cap)
              for r in avail]
        for rec in weighted_sample_without_replacement(avail, ws, remaining, rng):
            chosen.append(rec)
            chosen_ids.add(rec["poi_id"])

    extend_from(primary_pool)
    extend_from(backup_pool)
    extend_from(annotated)
    return chosen[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_in", required=True, help="records_{insample,oos}*.jsonl")
    p.add_argument("--records_out", required=True)
    p.add_argument("--cbg_poi_dir", required=True)
    p.add_argument("--freq_power", type=float, required=True,
                   help="1.0=current, 0.5=soft, 0.0=uniform popularity")
    p.add_argument("--primary_radius_m", type=float, default=1500.0)
    p.add_argument("--fallback_radius_m", type=float, default=3000.0)
    p.add_argument("--hard_cap_radius_m", type=float, default=5000.0)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== Resample candidates ===")
    print(f"  in:  {args.records_in}")
    print(f"  out: {args.records_out}")
    print(f"  freq_power={args.freq_power}  radii={args.primary_radius_m:.0f}/{args.fallback_radius_m:.0f}/{args.hard_cap_radius_m:.0f}m  k={args.k}")

    pools = load_pool(args.cbg_poi_dir)
    print(f"  loaded {len(pools)} cluster pools")

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.records_out) or ".", exist_ok=True)
    n_in, n_out, n_truth_missing = 0, 0, 0
    with open(args.records_in) as fin, open(args.records_out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n_in += 1
            cid = str(r["cluster"])
            pool = pools.get(cid)
            if not pool:
                continue
            new_cands = resample(
                pool, r["true_poi_id"],
                r["origin_lat"], r["origin_lon"],
                args.k, args.freq_power,
                args.primary_radius_m, args.fallback_radius_m, args.hard_cap_radius_m,
                rng,
            )
            if not any(c["poi_id"] == r["true_poi_id"] for c in new_cands):
                n_truth_missing += 1
                continue
            r["candidates"] = new_cands
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
            if n_in % 10000 == 0:
                print(f"    {n_in:,} in → {n_out:,} out")
    print(f"  done: {n_in:,} in → {n_out:,} out (truth_missing={n_truth_missing})")


if __name__ == "__main__":
    main()
