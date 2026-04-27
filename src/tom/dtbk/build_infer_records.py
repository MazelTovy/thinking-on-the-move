"""Produce nyc_metro-schema infer records from dtbk trip + cbg_poi data.

One record per trip (a_work row). Candidates = k POIs sampled from
cluster_{N}_cleaned_pois.csv weighted by cluster_freq (same logic as
sft_04_vllm.py sample_pois_by_cluster), forced to include the true POI so
baselines can score it.

Output schema matches nyc_metro experiments/<exp>/infer_records/records_*.jsonl,
consumable directly by nyc_metro/17_classical_baselines.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def format_poi_id(name, lat, lon, ndigits=6):
    if pd.isna(name) or pd.isna(lat) or pd.isna(lon):
        return None
    return f"{str(name)}|{float(lat):.{ndigits}f}|{float(lon):.{ndigits}f}"


def load_cluster_pool(cbg_poi_dir, cluster, valid_poi_ids, fallback_df):
    path = Path(cbg_poi_dir) / f"cluster_{cluster}_cleaned_pois.csv"
    df = pd.read_csv(path) if path.exists() else fallback_df.copy()
    df = df.dropna(subset=["poi_LOCATION_NAME", "poi_LATITUDE", "poi_LONGITUDE"]).copy()
    df["poi_id"] = df.apply(
        lambda r: format_poi_id(r["poi_LOCATION_NAME"], r["poi_LATITUDE"], r["poi_LONGITUDE"]),
        axis=1,
    )
    df = df.dropna(subset=["poi_id"]).drop_duplicates(subset=["poi_id"]).copy()
    df = df[df["poi_id"].isin(valid_poi_ids)].copy()

    # Join freq from fallback when cluster CSV lacks it (matches sft_04_vllm).
    if "freq" not in df.columns and "freq" in fallback_df.columns:
        fb = fallback_df.dropna(subset=["poi_LOCATION_NAME", "poi_LATITUDE", "poi_LONGITUDE"]).copy()
        fb["_pid"] = fb.apply(
            lambda r: format_poi_id(r["poi_LOCATION_NAME"], r["poi_LATITUDE"], r["poi_LONGITUDE"]),
            axis=1,
        )
        fmap = fb.dropna(subset=["_pid"]).drop_duplicates("_pid").set_index("_pid")["freq"]
        df["freq"] = df["poi_id"].map(fmap)

    return df


def sample_candidates(pool_df, k, rng, force_pid=None):
    """Weighted sample of k POIs. If force_pid is given, ensure it's included."""
    if "cluster_freq" in pool_df.columns and pool_df["cluster_freq"].fillna(0).sum() > 0:
        w = pool_df["cluster_freq"].fillna(0).astype(float).clip(lower=0)
    elif "freq" in pool_df.columns and pool_df["freq"].fillna(0).sum() > 0:
        w = pool_df["freq"].fillna(0).astype(float).clip(lower=0)
    elif "poi_RAW_VISIT_COUNTS" in pool_df.columns:
        w = pool_df["poi_RAW_VISIT_COUNTS"].fillna(0).astype(float).clip(lower=0)
    else:
        w = pd.Series(np.ones(len(pool_df)), index=pool_df.index)
    w = np.log1p(w)
    total = w.sum()
    probs = (w / total).values if total > 0 else None

    n = len(pool_df)
    if n <= k:
        return pool_df.copy()

    idx = rng.choice(n, size=k, replace=False, p=probs)
    sampled = pool_df.iloc[idx].copy()

    if force_pid is not None and force_pid not in set(sampled["poi_id"]):
        true_row = pool_df[pool_df["poi_id"] == force_pid]
        if len(true_row) > 0:
            # Drop the lowest-weighted sampled candidate to make room
            sampled_w = w.iloc[idx].values
            drop_local = int(np.argmin(sampled_w))
            sampled = sampled.drop(sampled.index[drop_local])
            sampled = pd.concat([sampled, true_row.iloc[[0]]], ignore_index=False)

    return sampled


def build_records(a_work_csv, cbg_poi_dir, fallback_csv, out_path,
                  k=30, max_records=0, seed=42):
    print(f"Loading a_work: {a_work_csv}")
    df = pd.read_csv(a_work_csv, low_memory=False)
    df = df.dropna(subset=["poi_LOCATION_NAME", "poi_LATITUDE", "poi_LONGITUDE",
                           "lat", "lng", "demo_cluster_pure"]).copy()
    df["cluster"] = df["demo_cluster_pure"].astype(int).astype(str)
    df["true_poi_id"] = df.apply(
        lambda r: format_poi_id(r["poi_LOCATION_NAME"], r["poi_LATITUDE"], r["poi_LONGITUDE"]),
        axis=1,
    )
    df = df.dropna(subset=["true_poi_id"]).copy()

    print(f"Loading fallback: {fallback_csv}")
    fallback_df = pd.read_csv(fallback_csv)
    valid_poi_ids = set(
        fallback_df.dropna(subset=["poi_LOCATION_NAME", "poi_LATITUDE", "poi_LONGITUDE"])
        .apply(lambda r: format_poi_id(r["poi_LOCATION_NAME"], r["poi_LATITUDE"],
                                       r["poi_LONGITUDE"]), axis=1)
        .dropna()
    )
    print(f"Valid POI universe: {len(valid_poi_ids)}")

    rng = np.random.default_rng(seed)
    pools = {}
    n_written = 0
    n_skipped_no_true = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            if max_records and n_written >= max_records:
                break

            cluster = row["cluster"]
            if cluster not in pools:
                pools[cluster] = load_cluster_pool(cbg_poi_dir, cluster, valid_poi_ids, fallback_df)
            pool = pools[cluster]
            if len(pool) == 0:
                continue

            true_pid = row["true_poi_id"]
            if true_pid not in valid_poi_ids:
                n_skipped_no_true += 1
                continue

            cand_df = sample_candidates(pool, k, rng, force_pid=true_pid)
            origin_lat = float(row["lat"]); origin_lon = float(row["lng"])
            cand_df = cand_df.copy()
            cand_df["dist_m"] = haversine(
                origin_lat, origin_lon,
                cand_df["poi_LATITUDE"].astype(float),
                cand_df["poi_LONGITUDE"].astype(float),
            )

            candidates = []
            for _, c in cand_df.iterrows():
                candidates.append({
                    "poi_id": c["poi_id"],
                    "poi_name": str(c["poi_LOCATION_NAME"]),
                    "poi_lat": float(c["poi_LATITUDE"]),
                    "poi_lon": float(c["poi_LONGITUDE"]),
                    "sub_category": str(c.get("poi_SUB_CATEGORY", "") or ""),
                    "cluster_freq": int(c.get("cluster_freq", 0) or 0),
                    "polygon_type": "",
                    "synthetic_unseen": 0,
                    "dist_m": float(c["dist_m"]),
                })

            rec = {
                "cuebiq_id": str(row.get("cuebiq_id", "")),
                "cluster": cluster,
                "admin2_id": str(row.get("admin2_id", "US.NY.047")),
                "stop_ts": str(row.get("stop_ts", "")),
                "work_cbg": str(row.get("block_group_id_x", "")),
                "true_poi_id": true_pid,
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "origin_mode": "stop",
                "origin_source": "stop",
                "true_polygon_type": "",
                "candidates": candidates,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {out_path}")
    print(f"Skipped {n_skipped_no_true} rows where true POI not in valid universe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_work", required=True)
    parser.add_argument("--cbg_poi_dir", required=True)
    parser.add_argument("--poi_fallback", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_records(args.a_work, args.cbg_poi_dir, args.poi_fallback, args.out,
                  k=args.k, max_records=args.max_records, seed=args.seed)


if __name__ == "__main__":
    main()
