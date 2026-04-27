#!/usr/bin/env python3
"""
10_sft_data_prep.py — SFT data preparation pipeline for NYC Metro.

Major rewrite (v2, 2026-04-08): aligned with Tokyo POI prediction approach.

Key changes from v1:
  1. Cluster pool entries include `poi_id` (compatible with sample_candidates).
  2. Uses sample_candidates() with distance-aware weighting (1.5km/3km/5km radii).
  3. Uses build_cluster_frequency_top5() for realistic top_5 distribution
     (NOT degenerate prob=1.0).
  4. Personas include borough, work area, top cluster restaurants, JSON signals.
  5. Temporal val split: last 60 days of 2021 as val (not random 5%).
  6. Streaming write (no full data load).
  7. Experiment-scoped train-prior snapshots: cluster pools, fallback POI universe,
     personas, and optional work_cbg-based origin anchors are all built strictly
     from rows before the temporal validation cutoff.

Pipeline stages:
  Step A: Generate per-cluster candidate pools (with poi_id)
  Step B: Generate enriched personas (borough + top restaurants + JSON)
  Step C: Build train.jsonl / val.jsonl with proper supervision targets
"""

import argparse
import gc
import heapq
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tom.utils import (
    FALLBACK_RADIUS_M,
    HARD_CAP_RADIUS_M,
    POLYGON_POLICIES,
    PRIMARY_RADIUS_M,
    apply_poi_policy_to_frame,
    build_cluster_frequency_top5,
    haversine_m,
    load_placekey_point_member_lookup,
    load_poi_id_set,
    load_work_cbg_origin_map,
    resolve_origin_coords,
    sample_candidates,
)

BASE = "/scratch/sx2490/econai/nyc_metro"
ACS_PATH = f"{BASE}/data/acs_cbg_demographics.csv"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"

K_CANDIDATES = 30
SYSTEM_PROMPT = (
    "You are predicting lunch-time restaurant choices for a worker in New York City. "
    "The worker chooses exactly one restaurant per lunch trip. "
    "Predict which restaurant the worker will visit based on their persona profile "
    "and the list of candidate restaurants."
)

BOROUGH_NAMES = {
    "061": "Manhattan",
    "047": "Brooklyn",
    "081": "Queens",
}


def format_poi_id(name, lat, lon):
    return f"{name}|{float(lat):.6f}|{float(lon):.6f}"


# ── Step A: Generate per-cluster candidate pools ─────────────────────────────

def step_a_generate_cbg_poi(
    a_work_path,
    out_dir,
    cluster_col="demo_cluster",
    train_prior_end_date=None,
    fallback_path=None,
    polygon_policy="all",
    point_member_lookup=None,
    exclude_poi_ids=None,
):
    """Build per-cluster POI candidate pools from a_work.

    Each pool entry has: poi_id, poi_name, poi_lat, poi_lon, sub_category, cluster_freq.
    """
    print("\n=== Step A: Generate per-cluster candidate pools ===")
    os.makedirs(out_dir, exist_ok=True)
    if fallback_path:
        fallback_dir = os.path.dirname(fallback_path)
        if fallback_dir:
            os.makedirs(fallback_dir, exist_ok=True)

    poi_by_cluster = defaultdict(list)
    for chunk in pd.read_csv(
        a_work_path,
        chunksize=500_000,
        dtype={"cuebiq_id": str},
        usecols=[cluster_col, "poi_name", "poi_lat", "poi_lon", "PLACEKEY",
                 "polygon_type", "temporal_status", "sub_category",
                 "dwell_time_minutes", "stop_ts"],
    ):
        chunk = chunk.dropna(subset=[cluster_col, "poi_name", "poi_lat", "poi_lon", "stop_ts"])
        if train_prior_end_date:
            chunk["stop_ts"] = chunk["stop_ts"].astype(str)
            chunk = chunk[chunk["stop_ts"].str[:10] < train_prior_end_date]
        chunk = apply_poi_policy_to_frame(
            chunk,
            polygon_policy=polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_poi_ids,
        )
        if len(chunk) == 0:
            continue
        for cluster_id, grp in chunk.groupby(cluster_col):
            poi_by_cluster[cluster_id].append(
                grp[["poi_name", "poi_lat", "poi_lon",
                     "polygon_type", "sub_category", "dwell_time_minutes"]].copy()
            )

    # Aggregate per cluster
    all_pois_for_global = []
    for cluster_id in sorted(poi_by_cluster.keys()):
        df = pd.concat(poi_by_cluster[cluster_id], ignore_index=True)
        # Use rounded coords for grouping (4 decimal ~ 11m)
        df["_key"] = (df["poi_name"].astype(str) + "|" +
                      df["poi_lat"].round(4).astype(str) + "|" +
                      df["poi_lon"].round(4).astype(str))

        agg = df.groupby("_key").agg(
            poi_name=("poi_name", "first"),
            poi_lat=("poi_lat", "mean"),
            poi_lon=("poi_lon", "mean"),
            sub_category=("sub_category", "first"),
            polygon_type=("polygon_type", "first"),
            poi_median_dwell=("dwell_time_minutes", "median"),
            cluster_freq=("poi_name", "count"),
        ).reset_index(drop=True)

        # Build canonical poi_id
        agg["poi_id"] = agg.apply(
            lambda r: format_poi_id(r["poi_name"], r["poi_lat"], r["poi_lon"]),
            axis=1,
        )
        agg = agg[["poi_id", "poi_name", "poi_lat", "poi_lon", "polygon_type",
                   "sub_category", "poi_median_dwell", "cluster_freq"]]
        agg = agg.sort_values("cluster_freq", ascending=False).reset_index(drop=True)

        out_path = f"{out_dir}/cluster_{cluster_id}_pois.csv"
        agg.to_csv(out_path, index=False)
        print(f"  Cluster {cluster_id}: {len(agg):,} POIs ({df['_key'].count():,} visits) → {out_path}")
        all_pois_for_global.append(agg.copy())
        del df, agg

    # Global fallback
    if not all_pois_for_global:
        raise RuntimeError(
            f"No train-prior POIs were found before {train_prior_end_date or 'end of file'} "
            f"when building cluster pools from {a_work_path}."
        )
    df_all = pd.concat(all_pois_for_global, ignore_index=True)
    fallback = df_all.groupby("poi_id").agg(
        poi_name=("poi_name", "first"),
        poi_lat=("poi_lat", "first"),
        poi_lon=("poi_lon", "first"),
        polygon_type=("polygon_type", "first"),
        sub_category=("sub_category", "first"),
        cluster_freq=("cluster_freq", "sum"),
    ).reset_index()
    fallback = fallback.sort_values("cluster_freq", ascending=False).reset_index(drop=True)
    fallback_path = fallback_path or f"{BASE}/data/poi_unique_food_with_freq.csv"
    fallback.to_csv(fallback_path, index=False)
    print(f"  Global fallback: {len(fallback):,} POIs → {fallback_path}")

    del poi_by_cluster, all_pois_for_global
    gc.collect()
    return out_dir


def step_a_generate_work_cbg_origins(a_work_path, out_path, train_prior_end_date=None):
    """Build a work_cbg → empirical lunch-origin anchor table from train-prior rows."""
    print("\n=== Step A2: Generate work_cbg origin anchors ===")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stats = defaultdict(lambda: {"sum_lat": 0.0, "sum_lon": 0.0, "count": 0})

    for chunk in pd.read_csv(
        a_work_path,
        chunksize=500_000,
        dtype={"cuebiq_id": str, "work_cbg": str},
        usecols=["work_cbg", "lat", "lng", "stop_ts"],
    ):
        chunk = chunk.dropna(subset=["work_cbg", "lat", "lng", "stop_ts"])
        if train_prior_end_date:
            chunk["stop_ts"] = chunk["stop_ts"].astype(str)
            chunk = chunk[chunk["stop_ts"].str[:10] < train_prior_end_date]
        if len(chunk) == 0:
            continue

        grouped = chunk.groupby("work_cbg").agg(
            sum_lat=("lat", "sum"),
            sum_lon=("lng", "sum"),
            count=("lat", "count"),
        )
        for work_cbg, row in grouped.iterrows():
            rec = stats[str(work_cbg)]
            rec["sum_lat"] += float(row["sum_lat"])
            rec["sum_lon"] += float(row["sum_lon"])
            rec["count"] += int(row["count"])

    rows = []
    for work_cbg, rec in sorted(stats.items()):
        if rec["count"] <= 0:
            continue
        rows.append({
            "work_cbg": work_cbg,
            "origin_lat": rec["sum_lat"] / rec["count"],
            "origin_lon": rec["sum_lon"] / rec["count"],
            "n_visits": rec["count"],
        })

    pd.DataFrame(
        rows,
        columns=["work_cbg", "origin_lat", "origin_lon", "n_visits"],
    ).to_csv(out_path, index=False)
    print(f"  Work CBG anchors: {len(rows):,} → {out_path}")
    return out_path


# ── Step B: Generate enriched personas ───────────────────────────────────────

def step_b_generate_personas(
    a_work_path,
    cbg_poi_dir,
    out_path,
    cluster_col="demo_cluster",
    n_personas_per_cluster=500,
    train_prior_end_date=None,
    polygon_policy="all",
    point_member_lookup=None,
    exclude_poi_ids=None,
):
    """Generate enriched persona descriptions.

    Each persona includes:
      - borough composition of the cluster's home + work areas
      - demographic summary (income, race, education, employment)
      - top 5 most frequented restaurants in the cluster
      - PREDICTION SIGNALS JSON block
    """
    print("\n=== Step B: Generate enriched personas ===")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pass 1: cluster demographic + borough stats
    print("  Aggregating cluster stats from a_work...")
    cluster_acs = defaultdict(lambda: {
        "income": [], "white": [], "black": [], "asian": [], "hispanic": [],
        "rent": [], "employ": [],
    })
    cluster_work_borough = defaultdict(lambda: defaultdict(int))

    for chunk in pd.read_csv(
        a_work_path,
        chunksize=500_000,
        dtype={"cuebiq_id": str, "work_cbg": str, "home_cbg": str},
        usecols=[cluster_col, "work_cbg", "stop_ts", "poi_name", "poi_lat", "poi_lon",
                 "PLACEKEY", "polygon_type", "sub_category", "temporal_status",
                 "median_household_income", "white_share", "black_share",
                 "asian_share", "hispanic_share", "rent_share", "employment_rate"],
    ):
        chunk = chunk.dropna(subset=[cluster_col, "stop_ts"])
        if train_prior_end_date:
            chunk["stop_ts"] = chunk["stop_ts"].astype(str)
            chunk = chunk[chunk["stop_ts"].str[:10] < train_prior_end_date]
        chunk = apply_poi_policy_to_frame(
            chunk,
            polygon_policy=polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_poi_ids,
        )
        if len(chunk) == 0:
            continue
        chunk[cluster_col] = chunk[cluster_col].astype(int)
        for cid, grp in chunk.groupby(cluster_col):
            stats = cluster_acs[cid]
            stats["income"].append(grp["median_household_income"].median())
            stats["white"].append(grp["white_share"].median())
            stats["black"].append(grp["black_share"].median())
            stats["asian"].append(grp["asian_share"].median())
            stats["hispanic"].append(grp["hispanic_share"].median())
            stats["rent"].append(grp["rent_share"].median())
            stats["employ"].append(grp["employment_rate"].median())

            # Work borough from work_cbg (US.NY.061.xxx -> 061)
            for wcbg in grp["work_cbg"].dropna():
                parts = str(wcbg).split(".")
                if len(parts) >= 3:
                    cluster_work_borough[cid][parts[2]] += 1

    # Pass 2: top 5 restaurants per cluster (from cluster pool CSVs)
    # Strict match: cluster_<int>_pois.csv (skips legacy cluster_X_cleaned_pois.csv)
    import re as _re
    pool_re = _re.compile(r"^cluster_(\d+)_pois\.csv$")
    cluster_top_pois = {}
    for fname in sorted(os.listdir(cbg_poi_dir)):
        m = pool_re.match(fname)
        if not m:
            continue
        cid = int(m.group(1))
        df = pd.read_csv(f"{cbg_poi_dir}/{fname}")
        # Already sorted by cluster_freq desc
        top = df.head(5)[["poi_name", "cluster_freq"]].to_dict("records")
        cluster_top_pois[cid] = top

    # Build personas
    records = []
    rng = random.Random(42)
    for cid in sorted(cluster_acs.keys()):
        s = cluster_acs[cid]

        def med(values):
            vs = [v for v in values if pd.notna(v)]
            if not vs:
                return 0
            vs.sort()
            return vs[len(vs) // 2]

        income = med(s["income"])
        white_pct = med(s["white"]) * 100
        black_pct = med(s["black"]) * 100
        asian_pct = med(s["asian"]) * 100
        hispanic_pct = med(s["hispanic"]) * 100
        rent_pct = med(s["rent"]) * 100
        employ_pct = med(s["employ"]) * 100

        income_desc = "low-income" if income < 40000 else "middle-income" if income < 80000 else "high-income"

        race_shares = {
            "White": white_pct, "Black": black_pct,
            "Asian": asian_pct, "Hispanic": hispanic_pct,
        }
        dom_race = max(race_shares, key=race_shares.get)
        dom_pct = race_shares[dom_race]

        # Work borough
        boro_counts = cluster_work_borough[cid]
        boro_total = sum(boro_counts.values()) or 1
        boro_desc_parts = []
        for code, name in BOROUGH_NAMES.items():
            pct = boro_counts.get(code, 0) / boro_total * 100
            if pct >= 5:
                boro_desc_parts.append(f"{name} ({pct:.0f}%)")
        work_boro_desc = ", ".join(boro_desc_parts) if boro_desc_parts else "the metro area"

        # Top restaurants
        top_pois = cluster_top_pois.get(cid, [])
        if top_pois:
            top_names = ", ".join(
                f"{p['poi_name']} ({int(p['cluster_freq'])}x)" for p in top_pois
            )
            top_desc = f"Their cluster's most-visited restaurants: {top_names}."
        else:
            top_desc = ""

        for i in range(n_personas_per_cluster):
            persona = (
                f"### PERSONA\n"
                f"This worker belongs to NYC demographic cluster {cid}. "
                f"They commute to work in {work_boro_desc} and eat lunch nearby on weekdays. "
                f"Their home neighborhood is predominantly {dom_race} ({dom_pct:.0f}%), "
                f"{income_desc} (median household income ~${income:,.0f}), "
                f"{rent_pct:.0f}% renters, employment rate {employ_pct:.0f}%. "
                f"{top_desc}\n\n"
                f"### PREDICTION SIGNALS\n"
                f'{{"cluster_id": {cid}, '
                f'"income": {int(income)}, '
                f'"income_class": "{income_desc}", '
                f'"dominant_race": "{dom_race}", '
                f'"race_pct": {dom_pct:.1f}, '
                f'"rent_pct": {rent_pct:.1f}, '
                f'"employment_pct": {employ_pct:.1f}, '
                f'"work_borough_mix": {{'
                + ", ".join(
                    f'"{name}": {boro_counts.get(code, 0) / boro_total * 100:.1f}'
                    for code, name in BOROUGH_NAMES.items()
                )
                + "}}}"
            )
            records.append({
                "cluster": str(cid),
                "persona_index": i,
                "persona": persona,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Generated {len(records):,} personas for {len(cluster_acs)} clusters → {out_path}")
    return out_path


def write_snapshot_manifest(
    manifest_path,
    a_work_path,
    cbg_poi_dir,
    personas_path,
    fallback_poi_csv,
    work_cbg_origin_csv,
    train_prior_end_date,
    origin_mode,
    polygon_policy="all",
    exclude_poi_csv="",
    n_excluded_pois=0,
    point_member_authority_csv="",
):
    """Write a manifest describing the experiment-scoped train-prior inputs."""
    if not manifest_path:
        return

    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "a_work_path": os.path.abspath(a_work_path),
        "train_prior_end_date": train_prior_end_date,
        "cbg_poi_dir": os.path.abspath(cbg_poi_dir),
        "personas_path": os.path.abspath(personas_path),
        "fallback_poi_csv": os.path.abspath(fallback_poi_csv),
        "work_cbg_origin_csv": os.path.abspath(work_cbg_origin_csv) if work_cbg_origin_csv else "",
        "default_origin_mode": origin_mode,
        "origin_lookup_type": "empirical_mean_train_prior_stop_coords",
        "polygon_policy": polygon_policy,
        "exclude_poi_csv": os.path.abspath(exclude_poi_csv) if exclude_poi_csv else "",
        "n_excluded_pois": n_excluded_pois,
        "point_member_authority_csv": os.path.abspath(point_member_authority_csv) if point_member_authority_csv else "",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n=== Snapshot Manifest ===\n  {manifest_path}")


# ── Step C: Build training data ──────────────────────────────────────────────

def load_personas_grouped(path):
    by_cluster = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_cluster[str(r["cluster"])].append(r["persona"])
    return by_cluster


def load_cluster_pools(cbg_poi_dir):
    import re as _re
    pool_re = _re.compile(r"^cluster_(\d+)_pois\.csv$")
    pools = {}
    for fname in sorted(os.listdir(cbg_poi_dir)):
        m = pool_re.match(fname)
        if not m:
            continue
        cid = m.group(1)
        rows = []
        with open(f"{cbg_poi_dir}/{fname}", "r", encoding="utf-8") as f:
            import csv as _csv
            for row in _csv.DictReader(f):
                rows.append({
                    "poi_id": row["poi_id"],
                    "poi_name": row["poi_name"],
                    "poi_lat": float(row["poi_lat"]),
                    "poi_lon": float(row["poi_lon"]),
                    "polygon_type": row.get("polygon_type", ""),
                    "sub_category": row.get("sub_category", ""),
                    "cluster_freq": int(row.get("cluster_freq", 0)),
                })
        pools[cid] = rows
    return pools


def build_candidate_table(candidates, orig_lat, orig_lon):
    """Build candidate restaurant table sorted by distance from origin."""
    lines = []
    for c in candidates:
        dist = haversine_m(orig_lat, orig_lon, c["poi_lat"], c["poi_lon"])
        cat = str(c.get("sub_category", "") or "")
        parts = [f"- {c['poi_name']}", f"id={c['poi_id']}"]
        if cat:
            parts.append(f"category={cat}")
        parts.append(f"dist={int(dist)}m")
        lines.append((dist, " | ".join(parts)))
    lines.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in lines)


def _reservoir_add(heap, limit, priority, item):
    """Keep the `limit` smallest-priority items in a max-heap."""
    if limit <= 0:
        return
    entry = (-priority, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _sample_uniform_rows_by_split(
    a_work_path,
    personas_by_cluster,
    pools,
    val_date_from,
    cluster_col,
    max_train_rows,
    max_val_rows,
    rng,
    polygon_policy="all",
    point_member_lookup=None,
    exclude_poi_ids=None,
):
    """Uniformly sample rows within train/val time windows using reservoir sampling."""
    if max_train_rows <= 0 or max_val_rows <= 0:
        raise ValueError(
            "Uniform split sampling requires both max_train_rows and max_val_rows > 0."
        )

    print("  Sampling uniformly within train/val windows...")
    train_heap = []
    val_heap = []
    np_rng = np.random.default_rng(rng.randrange(1 << 63))
    eligible_train = 0
    eligible_val = 0
    eligible_total = 0
    n_skipped = 0
    valid_clusters = {
        cid for cid, pool in pools.items()
        if len(pool) >= 5 and personas_by_cluster.get(cid)
    }

    for chunk in pd.read_csv(
        a_work_path,
        chunksize=200_000,
        dtype={"cuebiq_id": str, "work_cbg": str},
        usecols=[cluster_col, "lat", "lng", "work_cbg", "stop_ts", "poi_name", "poi_lat", "poi_lon",
                 "PLACEKEY", "polygon_type", "sub_category", "temporal_status"],
    ):
        chunk = chunk.dropna(subset=[cluster_col, "poi_name", "poi_lat", "poi_lon",
                                     "lat", "lng", "stop_ts"])
        if len(chunk) == 0:
            continue
        chunk = apply_poi_policy_to_frame(
            chunk,
            polygon_policy=polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_poi_ids,
        )
        if len(chunk) == 0:
            continue
        chunk[cluster_col] = chunk[cluster_col].astype(int).astype(str)
        chunk["stop_ts"] = chunk["stop_ts"].astype(str)
        before_filter = len(chunk)
        chunk = chunk[chunk[cluster_col].isin(valid_clusters)].copy()
        n_skipped += before_filter - len(chunk)
        if len(chunk) == 0:
            continue

        chunk["_is_val"] = chunk["stop_ts"].str[:10] >= val_date_from
        eligible_total += len(chunk)
        eligible_val += int(chunk["_is_val"].sum())
        eligible_train += int((~chunk["_is_val"]).sum())

        for is_val, heap, limit in [
            (False, train_heap, max_train_rows),
            (True, val_heap, max_val_rows),
        ]:
            sub = chunk[chunk["_is_val"] == is_val]
            if len(sub) == 0 or limit <= 0:
                continue

            priorities = np_rng.random(len(sub))
            if len(heap) >= limit:
                threshold = -heap[0][0]
                mask = priorities < threshold
                if not mask.any():
                    continue
                sub = sub.loc[mask].copy()
                priorities = priorities[mask]
            else:
                sub = sub.copy()

            for priority, row in zip(priorities, sub.itertuples(index=False)):
                item = (
                    getattr(row, cluster_col),
                    float(row.lat),
                    float(row.lng),
                    str(getattr(row, "work_cbg", "") or ""),
                    row.stop_ts,
                    row.poi_name,
                    float(row.poi_lat),
                    float(row.poi_lon),
                    "" if pd.isna(row.polygon_type) else str(row.polygon_type),
                )
                _reservoir_add(heap, limit, float(priority), item)

        if eligible_total and eligible_total % 1_000_000 < len(chunk):
            print(f"    eligible={eligible_total:,} "
                  f"(train={eligible_train:,}, val={eligible_val:,}, skipped={n_skipped:,})")

        del chunk
        gc.collect()

    train_rows = [item for _, item in train_heap]
    val_rows = [item for _, item in val_heap]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    print(f"  Eligible rows: {eligible_total:,} "
          f"(train={eligible_train:,}, val={eligible_val:,}, skipped={n_skipped:,})")
    print(f"  Sampled rows:  train={len(train_rows):,}, val={len(val_rows):,}")
    return train_rows, val_rows, {
        "eligible_total": eligible_total,
        "eligible_train": eligible_train,
        "eligible_val": eligible_val,
        "n_skipped": n_skipped,
    }


# ---- Parallel write support (fork-based, COW-shared globals) ----
_PAR_PERSONAS = None
_PAR_POOLS = None
_PAR_WORK_CBG = None
_PAR_K = None
_PAR_PRIMARY = None
_PAR_FALLBACK = None
_PAR_HARD_CAP = None
_PAR_ORIGIN_MODE = None
_PAR_FREQ_POWER = None


def _process_chunk_to_tmp(args):
    """Worker: materialize a row chunk to its own tmp jsonl file."""
    chunk_idx, rows, tmp_path, seed = args
    rng = random.Random(seed)
    written = 0
    n_skipped = 0
    n_origin_fallback = 0
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for cid, stop_lat, stop_lon, work_cbg, stop_ts, poi_name, poi_lat, poi_lon, polygon_type in rows:
            pool = _PAR_POOLS.get(cid)
            cluster_personas = _PAR_PERSONAS.get(cid)
            if not pool or len(pool) < 5 or not cluster_personas:
                n_skipped += 1
                continue

            orig_lat, orig_lon, origin_source = resolve_origin_coords(
                stop_lat, stop_lon, work_cbg,
                origin_mode=_PAR_ORIGIN_MODE,
                work_cbg_origin_map=_PAR_WORK_CBG,
            )
            if origin_source != _PAR_ORIGIN_MODE:
                n_origin_fallback += 1

            true_id = format_poi_id(poi_name, poi_lat, poi_lon)
            candidates = sample_candidates(
                pool, true_id, _PAR_K, rng,
                orig_lat=orig_lat, orig_lon=orig_lon,
                primary_radius_m=_PAR_PRIMARY,
                fallback_radius_m=_PAR_FALLBACK,
                hard_cap_radius_m=_PAR_HARD_CAP,
                freq_power=_PAR_FREQ_POWER,
            )
            if len(candidates) < 5:
                n_skipped += 1
                continue

            cand_table = build_candidate_table(candidates, orig_lat, orig_lon)
            cluster_top_choice, top5 = build_cluster_frequency_top5(candidates, true_id)
            persona = rng.choice(cluster_personas)

            user_msg = (
                f"--- AGENT PERSONA ---\n"
                f"{persona}\n\n"
                f"--- CANDIDATE RESTAURANTS ---\n"
                f"{cand_table}\n\n"
                f"Each restaurant id is listed after 'id='. Use the exact id string in your output.\n\n"
                f"--- TASK ---\n"
                f"Predict the SINGLE most likely restaurant the worker will visit next. "
                f"Provide a ranked Top-5 list with probabilities that sum to 1.0. "
                f"Output JSON only, no explanation.\n"
                f'{{"next_choice": "<poi_id>", "top_5": [{{"poi": "<poi_id>", "prob": <float>}}, ...]}}'
            )
            assistant_msg = json.dumps({
                "next_choice": true_id,
                "top_5": top5,
            }, ensure_ascii=False)

            record = {
                "system": SYSTEM_PROMPT,
                "user": user_msg,
                "assistant": assistant_msg,
                "meta": {
                    "cluster_id": cid,
                    "true_poi_id": true_id,
                    "cluster_top_choice": cluster_top_choice,
                    "stop_ts": stop_ts,
                    "work_cbg": work_cbg,
                    "origin_mode": _PAR_ORIGIN_MODE,
                    "origin_source": origin_source,
                    "origin_lat": orig_lat,
                    "origin_lon": orig_lon,
                    "true_polygon_type": polygon_type,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return chunk_idx, written, n_skipped, n_origin_fallback


def _write_sampled_records_parallel(
    rows, out_path, split_name, personas_by_cluster, pools, k, base_seed,
    primary_radius_m, fallback_radius_m, hard_cap_radius_m,
    origin_mode, work_cbg_origin_map, freq_power, n_workers,
):
    """Parallel materializer. Splits rows into n_workers chunks, each writes to a
    tmp jsonl, then we cat them together. Globals are inherited via fork (COW)."""
    import multiprocessing as mp
    import tempfile
    import shutil
    import time

    global _PAR_PERSONAS, _PAR_POOLS, _PAR_WORK_CBG, _PAR_K
    global _PAR_PRIMARY, _PAR_FALLBACK, _PAR_HARD_CAP, _PAR_ORIGIN_MODE, _PAR_FREQ_POWER
    _PAR_PERSONAS = personas_by_cluster
    _PAR_POOLS = pools
    _PAR_WORK_CBG = work_cbg_origin_map
    _PAR_K = k
    _PAR_PRIMARY = primary_radius_m
    _PAR_FALLBACK = fallback_radius_m
    _PAR_HARD_CAP = hard_cap_radius_m
    _PAR_ORIGIN_MODE = origin_mode
    _PAR_FREQ_POWER = freq_power

    n = len(rows)
    if n == 0:
        with open(out_path, "w") as f:
            pass
        print(f"  {split_name}: 0 written to {out_path}")
        return 0, 0, 0

    chunk_size = (n + n_workers - 1) // n_workers
    chunks = [rows[i:i + chunk_size] for i in range(0, n, chunk_size)]
    tmp_dir = tempfile.mkdtemp(prefix=f"sft_{split_name}_", dir=os.path.dirname(out_path) or ".")
    tasks = [
        (i, chunks[i], os.path.join(tmp_dir, f"part_{i:03d}.jsonl"), base_seed + i * 1009 + 1)
        for i in range(len(chunks))
    ]
    print(f"  {split_name}: {n:,} rows split into {len(chunks)} chunks "
          f"(~{chunk_size:,} each), running on {n_workers} workers")

    t0 = time.time()
    ctx = mp.get_context("fork")  # COW inheritance of large globals
    with ctx.Pool(processes=n_workers) as pool_proc:
        results = pool_proc.map(_process_chunk_to_tmp, tasks)
    results.sort(key=lambda r: r[0])

    total_written = sum(r[1] for r in results)
    total_skipped = sum(r[2] for r in results)
    total_origin_fb = sum(r[3] for r in results)

    elapsed = time.time() - t0
    rate = total_written / max(elapsed, 1e-6)
    print(f"  {split_name}: workers done in {elapsed:.0f}s "
          f"({rate:.0f} rec/s; {total_written:,} written, "
          f"{total_skipped:,} skipped, {total_origin_fb:,} origin fallback)")

    print(f"  {split_name}: concatenating {len(chunks)} parts → {out_path}")
    with open(out_path, "wb") as fout:
        for i in range(len(chunks)):
            part = os.path.join(tmp_dir, f"part_{i:03d}.jsonl")
            with open(part, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return total_written, total_skipped, total_origin_fb


def _write_sampled_records(
    rows,
    out_path,
    split_name,
    personas_by_cluster,
    pools,
    k,
    rng,
    primary_radius_m,
    fallback_radius_m,
    hard_cap_radius_m,
    origin_mode,
    work_cbg_origin_map,
    freq_power=1.0,
    n_workers=1,
):
    """Materialize sampled rows into SFT JSONL records."""
    if n_workers and n_workers > 1:
        # Use the rng's state as a deterministic seed source
        base_seed = rng.randint(0, 2**31 - 1)
        return _write_sampled_records_parallel(
            rows, out_path, split_name, personas_by_cluster, pools, k, base_seed,
            primary_radius_m, fallback_radius_m, hard_cap_radius_m,
            origin_mode, work_cbg_origin_map, freq_power, n_workers,
        )

    written = 0
    n_skipped = 0
    n_origin_fallback = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for cid, stop_lat, stop_lon, work_cbg, stop_ts, poi_name, poi_lat, poi_lon, polygon_type in rows:
            pool = pools.get(cid)
            cluster_personas = personas_by_cluster.get(cid)
            if not pool or len(pool) < 5 or not cluster_personas:
                n_skipped += 1
                continue

            orig_lat, orig_lon, origin_source = resolve_origin_coords(
                stop_lat, stop_lon, work_cbg,
                origin_mode=origin_mode,
                work_cbg_origin_map=work_cbg_origin_map,
            )
            if origin_source != origin_mode:
                n_origin_fallback += 1

            true_id = format_poi_id(poi_name, poi_lat, poi_lon)
            candidates = sample_candidates(
                pool, true_id, k, rng,
                orig_lat=orig_lat, orig_lon=orig_lon,
                primary_radius_m=primary_radius_m,
                fallback_radius_m=fallback_radius_m,
                hard_cap_radius_m=hard_cap_radius_m,
                freq_power=freq_power,
            )
            if len(candidates) < 5:
                n_skipped += 1
                continue

            cand_table = build_candidate_table(candidates, orig_lat, orig_lon)
            cluster_top_choice, top5 = build_cluster_frequency_top5(candidates, true_id)
            persona = rng.choice(cluster_personas)

            user_msg = (
                f"--- AGENT PERSONA ---\n"
                f"{persona}\n\n"
                f"--- CANDIDATE RESTAURANTS ---\n"
                f"{cand_table}\n\n"
                f"Each restaurant id is listed after 'id='. Use the exact id string in your output.\n\n"
                f"--- TASK ---\n"
                f"Predict the SINGLE most likely restaurant the worker will visit next. "
                f"Provide a ranked Top-5 list with probabilities that sum to 1.0. "
                f"Output JSON only, no explanation.\n"
                f'{{"next_choice": "<poi_id>", "top_5": [{{"poi": "<poi_id>", "prob": <float>}}, ...]}}'
            )
            assistant_msg = json.dumps({
                "next_choice": true_id,
                "top_5": top5,
            }, ensure_ascii=False)

            record = {
                "system": SYSTEM_PROMPT,
                "user": user_msg,
                "assistant": assistant_msg,
                "meta": {
                    "cluster_id": cid,
                    "true_poi_id": true_id,
                    "cluster_top_choice": cluster_top_choice,
                    "stop_ts": stop_ts,
                    "work_cbg": work_cbg,
                    "origin_mode": origin_mode,
                    "origin_source": origin_source,
                    "origin_lat": orig_lat,
                    "origin_lon": orig_lon,
                    "true_polygon_type": polygon_type,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 50000 == 0:
                print(f"    {split_name}: {written:,} written")

    print(f"  {split_name}: {written:,} written to {out_path} "
          f"(skipped {n_skipped:,}, origin fallback {n_origin_fallback:,})")
    return written, n_skipped, n_origin_fallback


def step_c_build_training_data(
    a_work_path,
    cbg_poi_dir,
    personas_path,
    out_train,
    out_val,
    fallback_poi_csv="",
    work_cbg_origin_csv="",
    k=K_CANDIDATES,
    val_date_from="2021-11-01",
    cluster_col="demo_cluster",
    max_rows=0,
    max_train_rows=0,
    max_val_rows=0,
    seed=42,
    origin_mode="stop",
    primary_radius_m=PRIMARY_RADIUS_M,
    fallback_radius_m=FALLBACK_RADIUS_M,
    hard_cap_radius_m=HARD_CAP_RADIUS_M,
    polygon_policy="all",
    point_member_lookup=None,
    exclude_poi_ids=None,
    freq_power=1.0,
    n_workers=1,
):
    """Build SFT training data with proper supervision.

    Temporal val split: visits with stop_ts >= val_date_from go to val.
    Default 2021-11-01 → ~16% val (Nov-Dec).
    """
    print("\n=== Step C: Build training data ===")
    print(f"  Temporal val split from: {val_date_from}")
    print(f"  Polygon policy: {polygon_policy}")
    if exclude_poi_ids:
        print(f"  Hidden/excluded POIs: {len(exclude_poi_ids):,}")
    if max_train_rows > 0 or max_val_rows > 0:
        print(f"  Target rows: train={max_train_rows or 'all'}, val={max_val_rows or 'all'}")
    elif max_rows > 0:
        print(f"  Legacy max_rows cap: {max_rows}")
    rng = random.Random(seed)

    personas_by_cluster = load_personas_grouped(personas_path)
    print(f"  Personas: {sum(len(v) for v in personas_by_cluster.values()):,} "
          f"across {len(personas_by_cluster)} clusters")

    pools = load_cluster_pools(cbg_poi_dir)
    print(f"  Candidate pools: {len(pools)} clusters")
    if fallback_poi_csv:
        print(f"  Snapshot fallback POIs: {fallback_poi_csv}")

    work_cbg_origin_map = {}
    if origin_mode != "stop":
        work_cbg_origin_map = load_work_cbg_origin_map(work_cbg_origin_csv)
        print(f"  Work CBG anchors: {len(work_cbg_origin_map):,} "
              f"from {work_cbg_origin_csv}")

    if max_train_rows > 0 or max_val_rows > 0:
        train_rows, val_rows, sample_stats = _sample_uniform_rows_by_split(
            a_work_path,
            personas_by_cluster,
            pools,
            val_date_from,
            cluster_col,
            max_train_rows,
            max_val_rows,
            rng,
            polygon_policy=polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_poi_ids,
        )

        train_written, train_skipped, train_origin_fallback = _write_sampled_records(
            train_rows, out_train, "train", personas_by_cluster, pools, k, rng,
            primary_radius_m, fallback_radius_m, hard_cap_radius_m,
            origin_mode, work_cbg_origin_map, freq_power=freq_power,
            n_workers=n_workers,
        )
        val_written, val_skipped, val_origin_fallback = _write_sampled_records(
            val_rows, out_val, "val", personas_by_cluster, pools, k, rng,
            primary_radius_m, fallback_radius_m, hard_cap_radius_m,
            origin_mode, work_cbg_origin_map, freq_power=freq_power,
            n_workers=n_workers,
        )

        total_written = train_written + val_written
        total_skipped = sample_stats["n_skipped"] + train_skipped + val_skipped
        total_origin_fallback = train_origin_fallback + val_origin_fallback
        print(f"  Processed: {sample_stats['eligible_total']:,} eligible rows")
        print(f"  Total: {total_written:,} samples "
              f"(skipped {total_skipped:,}, origin fallback {total_origin_fallback:,})")
        print(f"  Train: {train_written:,} → {out_train}")
        print(f"  Val:   {val_written:,} → {out_val}")
        return

    # Streaming pass
    train_written = 0
    val_written = 0
    n_skipped = 0
    n_total = 0
    n_seen = 0
    n_origin_fallback = 0

    f_train = open(out_train, "w", encoding="utf-8")
    f_val = open(out_val, "w", encoding="utf-8")

    try:
        for chunk in pd.read_csv(
            a_work_path,
            chunksize=200_000,
            dtype={"cuebiq_id": str, "work_cbg": str},
            usecols=[cluster_col, "lat", "lng", "work_cbg", "stop_ts",
                     "poi_name", "poi_lat", "poi_lon", "sub_category",
                     "PLACEKEY", "polygon_type", "temporal_status"],
        ):
            chunk = chunk.dropna(subset=[cluster_col, "poi_name", "poi_lat", "poi_lon",
                                          "lat", "lng", "stop_ts"])
            if len(chunk) == 0:
                continue
            chunk = apply_poi_policy_to_frame(
                chunk,
                polygon_policy=polygon_policy,
                point_member_lookup=point_member_lookup,
                exclude_poi_ids=exclude_poi_ids,
            )
            if len(chunk) == 0:
                continue
            chunk[cluster_col] = chunk[cluster_col].astype(int).astype(str)
            chunk["stop_ts"] = chunk["stop_ts"].astype(str)
            chunk_dates = chunk["stop_ts"].str[:10]

            # Once one split cap is full, drop rows from that side before any
            # per-row sampling work. This keeps temporal capped runs fast even
            # when the source file is much larger than the target dataset.
            if max_train_rows > 0 and train_written >= max_train_rows:
                chunk = chunk[chunk_dates >= val_date_from]
                chunk_dates = chunk["stop_ts"].str[:10]
            if max_val_rows > 0 and val_written >= max_val_rows:
                chunk = chunk[chunk_dates < val_date_from]

            if len(chunk) == 0:
                if n_seen and n_seen % 1_000_000 == 0:
                    print(f"    processed {n_seen:,} eligible rows, written {n_total:,} "
                          f"(train={train_written:,}, val={val_written:,}, skipped={n_skipped:,})")
                continue

            for _, row in chunk.iterrows():
                n_seen += 1
                if max_rows > 0 and n_total >= max_rows:
                    break

                # Temporal split first, so split-specific caps can skip cheaply
                # without building prompts for rows we no longer need.
                is_val = row["stop_ts"][:10] >= val_date_from
                if is_val and max_val_rows > 0 and val_written >= max_val_rows:
                    if n_seen % 1_000_000 == 0:
                        print(f"    processed {n_seen:,} eligible rows, written {n_total:,} "
                              f"(train={train_written:,}, val={val_written:,}, skipped={n_skipped:,})")
                    continue
                if (not is_val) and max_train_rows > 0 and train_written >= max_train_rows:
                    if n_seen % 1_000_000 == 0:
                        print(f"    processed {n_seen:,} eligible rows, written {n_total:,} "
                              f"(train={train_written:,}, val={val_written:,}, skipped={n_skipped:,})")
                    continue

                cid = row[cluster_col]
                pool = pools.get(cid)
                if not pool or len(pool) < 5:
                    n_skipped += 1
                    continue
                cluster_personas = personas_by_cluster.get(cid)
                if not cluster_personas:
                    n_skipped += 1
                    continue

                orig_lat, orig_lon, origin_source = resolve_origin_coords(
                    float(row["lat"]),
                    float(row["lng"]),
                    row.get("work_cbg"),
                    origin_mode=origin_mode,
                    work_cbg_origin_map=work_cbg_origin_map,
                )
                if origin_source != origin_mode:
                    n_origin_fallback += 1
                true_id = format_poi_id(row["poi_name"],
                                        float(row["poi_lat"]), float(row["poi_lon"]))

                candidates = sample_candidates(
                    pool, true_id, k, rng,
                    orig_lat=orig_lat, orig_lon=orig_lon,
                    primary_radius_m=primary_radius_m,
                    fallback_radius_m=fallback_radius_m,
                    hard_cap_radius_m=hard_cap_radius_m,
                    freq_power=freq_power,
                )
                if len(candidates) < 5:
                    n_skipped += 1
                    continue

                cand_table = build_candidate_table(candidates, orig_lat, orig_lon)
                cluster_top_choice, top5 = build_cluster_frequency_top5(candidates, true_id)
                persona = rng.choice(cluster_personas)

                user_msg = (
                    f"--- AGENT PERSONA ---\n"
                    f"{persona}\n\n"
                    f"--- CANDIDATE RESTAURANTS ---\n"
                    f"{cand_table}\n\n"
                    f"Each restaurant id is listed after 'id='. Use the exact id string in your output.\n\n"
                    f"--- TASK ---\n"
                    f"Predict the SINGLE most likely restaurant the worker will visit next. "
                    f"Provide a ranked Top-5 list with probabilities that sum to 1.0. "
                    f"Output JSON only, no explanation.\n"
                    f'{{"next_choice": "<poi_id>", "top_5": [{{"poi": "<poi_id>", "prob": <float>}}, ...]}}'
                )
                assistant_msg = json.dumps({
                    "next_choice": true_id,
                    "top_5": top5,
                }, ensure_ascii=False)

                record = {
                    "system": SYSTEM_PROMPT,
                    "user": user_msg,
                    "assistant": assistant_msg,
                    "meta": {
                        "cluster_id": cid,
                        "true_poi_id": true_id,
                        "cluster_top_choice": cluster_top_choice,
                        "stop_ts": row["stop_ts"],
                        "work_cbg": row.get("work_cbg"),
                        "origin_mode": origin_mode,
                        "origin_source": origin_source,
                        "origin_lat": orig_lat,
                        "origin_lon": orig_lon,
                        "true_polygon_type": row.get("polygon_type", ""),
                    },
                }

                fout = f_val if is_val else f_train
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if is_val:
                    val_written += 1
                else:
                    train_written += 1
                n_total += 1

                if n_total % 50000 == 0:
                    print(f"    {n_total:,} written (train={train_written:,}, val={val_written:,})")
                elif n_seen % 1_000_000 == 0:
                    print(f"    processed {n_seen:,} eligible rows, written {n_total:,} "
                          f"(train={train_written:,}, val={val_written:,}, skipped={n_skipped:,})")

            split_caps_reached = (
                max_train_rows > 0 and train_written >= max_train_rows and
                max_val_rows > 0 and val_written >= max_val_rows
            )
            if (max_rows > 0 and n_total >= max_rows) or split_caps_reached:
                break
            del chunk
            gc.collect()
    finally:
        f_train.close()
        f_val.close()

    print(f"  Processed: {n_seen:,} eligible rows")
    print(f"  Total: {n_total:,} samples "
          f"(skipped {n_skipped:,}, origin fallback {n_origin_fallback:,})")
    print(f"  Train: {train_written:,} → {out_train}")
    print(f"  Val:   {val_written:,} → {out_val}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_work", default=f"{BASE}/data/a_work_2021.csv")
    parser.add_argument("--cbg_poi_dir", default=f"{BASE}/cbg_poi")
    parser.add_argument("--personas", default=f"{BASE}/data/personas_nyc.jsonl")
    parser.add_argument("--fallback_poi_csv", default=f"{BASE}/data/poi_unique_food_with_freq.csv")
    parser.add_argument("--work_cbg_origin_csv", default="")
    parser.add_argument("--snapshot_manifest", default="")
    parser.add_argument("--out_train", default=f"{BASE}/train.jsonl")
    parser.add_argument("--out_val", default=f"{BASE}/val.jsonl")
    parser.add_argument("--k", type=int, default=K_CANDIDATES)
    parser.add_argument("--val_date_from", type=str, default="2021-11-01",
                        help="Visits with stop_ts >= this date go to val (temporal split).")
    parser.add_argument("--train_prior_end_date", type=str, default="",
                        help="Rows with stop_ts < this date are used to build experiment-scoped priors. "
                             "Default: same as --val_date_from.")
    parser.add_argument("--max_rows", type=int, default=500_000,
                        help="Max samples (0 = all). Default 500k.")
    parser.add_argument("--max_train_rows", type=int, default=0,
                        help="Cap training rows only (0 = all). Prefer this with temporal split.")
    parser.add_argument("--max_val_rows", type=int, default=0,
                        help="Cap validation rows only (0 = all). Prefer this with temporal split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_personas", type=int, default=500)
    parser.add_argument("--origin_mode", choices=["stop", "work_cbg_centroid"], default="stop")
    parser.add_argument("--polygon_policy", choices=sorted(POLYGON_POLICIES), default="all",
                        help="POI polygon sensitivity policy for all prep stages.")
    parser.add_argument("--exclude_poi_csv", default="",
                        help="Optional CSV of POIs to hide from train priors and train/val rows.")
    parser.add_argument("--point_member_authority_csv", default=AUTH_PATH,
                        help="poi_name_authority.csv used by --polygon_policy point_members.")
    parser.add_argument("--primary_radius_m", type=float, default=PRIMARY_RADIUS_M)
    parser.add_argument("--fallback_radius_m", type=float, default=FALLBACK_RADIUS_M)
    parser.add_argument("--hard_cap_radius_m", type=float, default=HARD_CAP_RADIUS_M)
    parser.add_argument("--freq_power", type=float, default=1.0,
                        help="Popularity exponent on log1p(cluster_freq). "
                             "1.0=default, 0.5=softer, 0.0=popularity-agnostic.")
    parser.add_argument("--skip_step_a", action="store_true")
    parser.add_argument("--skip_step_b", action="store_true")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel workers for Step C row materialization "
                             "(uses fork; pools/personas inherited via COW). "
                             "1=serial (default).")
    args = parser.parse_args()

    if not args.train_prior_end_date:
        args.train_prior_end_date = args.val_date_from
    if not args.work_cbg_origin_csv:
        default_dir = os.path.dirname(args.personas) or BASE
        args.work_cbg_origin_csv = os.path.join(default_dir, "work_cbg_mean_origins.csv")

    print("=== SFT Pipeline: NYC Metro (v2) ===")
    print(f"  a_work: {args.a_work}")
    print(f"  max_rows: {args.max_rows}")
    print(f"  max_train_rows: {args.max_train_rows}")
    print(f"  max_val_rows: {args.max_val_rows}")
    print(f"  val_date_from: {args.val_date_from}")
    print(f"  train_prior_end_date: {args.train_prior_end_date}")
    print(f"  origin_mode: {args.origin_mode}")
    print(f"  polygon_policy: {args.polygon_policy}")
    print(f"  radii: primary={args.primary_radius_m}m, "
          f"fallback={args.fallback_radius_m}m, hard_cap={args.hard_cap_radius_m}m")

    exclude_poi_ids = set()
    if args.exclude_poi_csv:
        if not os.path.exists(args.exclude_poi_csv):
            print(f"ERROR: --exclude_poi_csv does not exist: {args.exclude_poi_csv}")
            sys.exit(1)
        exclude_poi_ids = load_poi_id_set(args.exclude_poi_csv)
    print(f"  excluded POIs: {len(exclude_poi_ids):,}")

    point_member_lookup = {}
    if args.polygon_policy == "point_members":
        if not os.path.exists(args.point_member_authority_csv):
            print(f"ERROR: --point_member_authority_csv does not exist: {args.point_member_authority_csv}")
            sys.exit(1)
        point_member_lookup = load_placekey_point_member_lookup(args.point_member_authority_csv)
        print(f"  point-member lookup: {len(point_member_lookup):,} PLACEKEYs")

    if not args.skip_step_a:
        step_a_generate_cbg_poi(
            args.a_work,
            args.cbg_poi_dir,
            train_prior_end_date=args.train_prior_end_date,
            fallback_path=args.fallback_poi_csv,
            polygon_policy=args.polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_poi_ids,
        )
        step_a_generate_work_cbg_origins(
            args.a_work,
            args.work_cbg_origin_csv,
            train_prior_end_date=args.train_prior_end_date,
        )

    if not args.skip_step_b:
        step_b_generate_personas(
            args.a_work, args.cbg_poi_dir, args.personas,
            n_personas_per_cluster=args.n_personas,
            train_prior_end_date=args.train_prior_end_date,
            polygon_policy=args.polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_poi_ids,
        )

    write_snapshot_manifest(
        args.snapshot_manifest,
        args.a_work,
        args.cbg_poi_dir,
        args.personas,
        args.fallback_poi_csv,
        args.work_cbg_origin_csv,
        args.train_prior_end_date,
        args.origin_mode,
        polygon_policy=args.polygon_policy,
        exclude_poi_csv=args.exclude_poi_csv,
        n_excluded_pois=len(exclude_poi_ids),
        point_member_authority_csv=args.point_member_authority_csv,
    )

    step_c_build_training_data(
        args.a_work, args.cbg_poi_dir, args.personas,
        args.out_train, args.out_val,
        fallback_poi_csv=args.fallback_poi_csv,
        work_cbg_origin_csv=args.work_cbg_origin_csv,
        k=args.k, val_date_from=args.val_date_from,
        max_rows=args.max_rows,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        seed=args.seed,
        origin_mode=args.origin_mode,
        primary_radius_m=args.primary_radius_m,
        fallback_radius_m=args.fallback_radius_m,
        hard_cap_radius_m=args.hard_cap_radius_m,
        polygon_policy=args.polygon_policy,
        point_member_lookup=point_member_lookup,
        exclude_poi_ids=exclude_poi_ids,
        freq_power=args.freq_power,
        n_workers=args.n_workers,
    )

    print("\n=== SFT Pipeline Done ===")


if __name__ == "__main__":
    main()
