#!/usr/bin/env python3
"""
16_prepare_synthetic_unseen.py — Select synthetic unseen POIs for Exp03.

The selected POIs are hidden from train-prior pool construction and SFT rows,
then optionally re-injected into inference-time candidate pools. This creates a
controlled POI-OOS split while keeping the 2022 temporal test protocol intact.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

from tom.utils import (
    POLYGON_POLICIES,
    apply_poi_policy_to_frame,
    load_placekey_point_member_lookup,
)


BASE = "/scratch/sx2490/econai/nyc_metro"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"


def format_poi_id(name, lat, lon):
    return f"{name}|{float(lat):.6f}|{float(lon):.6f}"


def collect_train_prior_pois(
    a_work_path,
    train_prior_end_date,
    polygon_policy,
    point_member_lookup,
):
    chunks = []
    for chunk in pd.read_csv(
        a_work_path,
        chunksize=500_000,
        dtype={"cuebiq_id": str},
        usecols=[
            "poi_name", "poi_lat", "poi_lon", "PLACEKEY", "polygon_type",
            "sub_category", "temporal_status", "stop_ts",
        ],
    ):
        chunk = chunk.dropna(subset=["poi_name", "poi_lat", "poi_lon", "stop_ts"])
        chunk["stop_ts"] = chunk["stop_ts"].astype(str)
        chunk = chunk[chunk["stop_ts"].str[:10] < train_prior_end_date]
        if len(chunk) == 0:
            continue
        chunk = apply_poi_policy_to_frame(
            chunk,
            polygon_policy=polygon_policy,
            point_member_lookup=point_member_lookup,
        )
        if len(chunk) == 0:
            continue

        chunk["_key"] = (
            chunk["poi_name"].astype(str)
            + "|"
            + chunk["poi_lat"].round(4).astype(str)
            + "|"
            + chunk["poi_lon"].round(4).astype(str)
        )
        grouped = chunk.groupby("_key").agg(
            poi_name=("poi_name", "first"),
            poi_lat=("poi_lat", "mean"),
            poi_lon=("poi_lon", "mean"),
            sub_category=("sub_category", "first"),
            polygon_type=("polygon_type", "first"),
            temporal_status=("temporal_status", "first"),
            visits_train=("poi_name", "count"),
        ).reset_index(drop=True)
        chunks.append(grouped)

    if not chunks:
        raise RuntimeError("No eligible train-prior POIs found.")

    df = pd.concat(chunks, ignore_index=True)
    df["_key"] = (
        df["poi_name"].astype(str)
        + "|"
        + df["poi_lat"].round(4).astype(str)
        + "|"
        + df["poi_lon"].round(4).astype(str)
    )
    df = df.groupby("_key").agg(
        poi_name=("poi_name", "first"),
        poi_lat=("poi_lat", "mean"),
        poi_lon=("poi_lon", "mean"),
        sub_category=("sub_category", "first"),
        polygon_type=("polygon_type", "first"),
        temporal_status=("temporal_status", "first"),
        visits_train=("visits_train", "sum"),
    ).reset_index(drop=True)
    df["poi_id"] = df.apply(lambda r: format_poi_id(r["poi_name"], r["poi_lat"], r["poi_lon"]), axis=1)
    df["cluster_freq"] = df["visits_train"].astype(int)
    return df


def select_hidden_pois(df, n_pois, min_visits, min_visit_quantile, max_visit_quantile, n_bins, seed):
    if len(df) == 0:
        return df

    df = df.copy()
    df["visit_rank_pct"] = df["visits_train"].rank(method="average", pct=True)
    candidates = df[
        (df["visits_train"] >= min_visits)
        & (df["visit_rank_pct"] >= min_visit_quantile)
        & (df["visit_rank_pct"] <= max_visit_quantile)
    ].copy()

    if len(candidates) < n_pois:
        relaxed = df[
            (df["visits_train"] >= min_visits)
            & (df["visit_rank_pct"] <= max_visit_quantile)
        ].copy()
        if len(relaxed) > len(candidates):
            print(f"  Relaxed lower quantile: {len(candidates):,} -> {len(relaxed):,} candidates")
            candidates = relaxed

    if len(candidates) <= n_pois:
        selected = candidates.sort_values("visits_train", ascending=False).copy()
    else:
        rng = random.Random(seed)
        candidates["_log_visits"] = np.log1p(candidates["visits_train"].astype(float))
        try:
            candidates["_bin"] = pd.qcut(
                candidates["_log_visits"],
                q=min(n_bins, candidates["_log_visits"].nunique()),
                duplicates="drop",
            )
            bins = list(candidates["_bin"].dropna().unique())
        except ValueError:
            candidates["_bin"] = "all"
            bins = ["all"]

        per_bin = max(1, int(np.ceil(n_pois / max(len(bins), 1))))
        selected_parts = []
        for _, sub in candidates.groupby("_bin", observed=False):
            rows = sub.to_dict("records")
            rng.shuffle(rows)
            selected_parts.extend(rows[:per_bin])

        selected = pd.DataFrame(selected_parts)
        if len(selected) < n_pois:
            remaining = candidates[~candidates["poi_id"].isin(set(selected["poi_id"]))].copy()
            remaining = remaining.sample(
                n=min(n_pois - len(selected), len(remaining)),
                random_state=seed + 1,
            )
            selected = pd.concat([selected, remaining], ignore_index=True)
        selected = selected.head(n_pois).copy()

    selected = selected.sort_values("visits_train", ascending=False).reset_index(drop=True)
    selected["synthetic_unseen"] = 1
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    # Inference-time injection should not leak train-prior popularity for these
    # hidden POIs. Keep `visits_train` for diagnostics, but expose zero prior mass.
    selected["cluster_freq"] = 0
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_work", default=f"{BASE}/data/a_work_2021.csv")
    parser.add_argument("--out", default=f"{BASE}/experiments/exp03/hidden_pois.csv")
    parser.add_argument("--train_prior_end_date", default="2021-11-01")
    parser.add_argument("--polygon_policy", choices=sorted(POLYGON_POLICIES), default="exclude_shared_building")
    parser.add_argument("--point_member_authority_csv", default=AUTH_PATH)
    parser.add_argument("--n_pois", type=int, default=500)
    parser.add_argument("--min_visits", type=int, default=100)
    parser.add_argument("--min_visit_quantile", type=float, default=0.50)
    parser.add_argument("--max_visit_quantile", type=float, default=0.995)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Synthetic Unseen POI Selection ===")
    print(f"  a_work: {args.a_work}")
    print(f"  train_prior_end_date: {args.train_prior_end_date}")
    print(f"  polygon_policy: {args.polygon_policy}")
    print(f"  target hidden POIs: {args.n_pois:,}")

    point_member_lookup = {}
    if args.polygon_policy == "point_members":
        point_member_lookup = load_placekey_point_member_lookup(args.point_member_authority_csv)
        print(f"  point-member lookup: {len(point_member_lookup):,} PLACEKEYs")

    df = collect_train_prior_pois(
        args.a_work,
        args.train_prior_end_date,
        args.polygon_policy,
        point_member_lookup,
    )
    print(f"  Eligible train-prior POIs: {len(df):,}")
    print(f"  Visits range: min={int(df['visits_train'].min())}, median={int(df['visits_train'].median())}, max={int(df['visits_train'].max())}")

    selected = select_hidden_pois(
        df,
        args.n_pois,
        args.min_visits,
        args.min_visit_quantile,
        args.max_visit_quantile,
        args.n_bins,
        args.seed,
    )
    if len(selected) == 0:
        raise RuntimeError("No POIs selected; lower --min_visits or quantile thresholds.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cols = [
        "poi_id", "poi_name", "poi_lat", "poi_lon", "sub_category",
        "polygon_type", "temporal_status", "cluster_freq", "visits_train",
        "visit_rank_pct", "synthetic_unseen", "selection_rank",
    ]
    selected[cols].to_csv(args.out, index=False)
    print(f"  Selected: {len(selected):,} POIs")
    print(f"  Hidden visits in train-prior window: {int(selected['visits_train'].sum()):,}")
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
