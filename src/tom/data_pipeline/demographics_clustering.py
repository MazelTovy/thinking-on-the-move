#!/usr/bin/env python3
"""
build_demographics_clustering.py — Demographics join + Ward clustering

1. Join matched stops with ACS demographics via home_cbg
2. Ward hierarchical clustering on CBG-level demographics
3. Auto-K selection via silhouette score (target: 30-80 clusters)
4. Output: a_work_2021.csv, a_work_2022.csv with demo_cluster column

Based on DTBK workDTBKhalfyear.ipynb Ward clustering approach.
"""

import gc
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

BASE = "/scratch/sx2490/econai/nyc_metro"
ACS_PATH = f"{BASE}/data/acs_cbg_demographics.csv"
CHUNK_SIZE = 500_000

# Clustering features (same as DTBK)
CLUSTER_FEATURES = [
    "median_household_income",
    "white_share", "black_share", "asian_share", "hispanic_share",
    "bachelor_degree", "master_degree", "professional_degree", "phd_degree",
    "employment_rate", "rent_share",
]


def load_acs():
    """Load ACS demographics, keyed by Cuebiq-format CBG ID."""
    acs = pd.read_csv(ACS_PATH)
    print(f"ACS demographics: {len(acs):,} CBGs")
    return acs


def get_worker_home_cbgs(year):
    """Extract unique (cuebiq_id, home_cbg) from matched stops."""
    workers = {}
    for chunk in pd.read_csv(f"{BASE}/data/matched_stops_{year}.csv",
                             usecols=["cuebiq_id", "home_cbg"],
                             dtype={"cuebiq_id": str}, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            if row["cuebiq_id"] not in workers:
                workers[row["cuebiq_id"]] = row["home_cbg"]
    df = pd.DataFrame([{"cuebiq_id": k, "home_cbg": v} for k, v in workers.items()])
    print(f"  {year}: {len(df):,} unique workers")
    return df


def build_cbg_features(worker_df, acs_df):
    """Join workers with ACS and aggregate to CBG level for clustering."""
    # Join
    merged = worker_df.merge(acs_df, left_on="home_cbg", right_on="cbg_cuebiq", how="inner")
    print(f"  Workers with ACS match: {len(merged):,}")

    # Aggregate to CBG level (unique CBGs with at least one worker)
    cbg_agg = merged.groupby("home_cbg").agg(
        n_workers=("cuebiq_id", "nunique"),
        **{feat: (feat, "first") for feat in CLUSTER_FEATURES}
    ).reset_index()
    print(f"  Unique home CBGs: {len(cbg_agg):,}")

    return merged, cbg_agg


def ward_clustering(cbg_df, k_range=(30, 80)):
    """Ward hierarchical clustering with auto-K selection."""
    # Prepare features
    features = cbg_df[CLUSTER_FEATURES].copy()

    # Fill NaN with median
    for col in CLUSTER_FEATURES:
        features[col] = features[col].fillna(features[col].median())

    # Normalize: separate scaling for different feature groups
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    print(f"  Feature matrix: {X.shape}")

    # Ward linkage
    print("  Computing Ward linkage...")
    Z = linkage(X, method="ward")

    # Auto-K: test range and pick best silhouette
    print(f"  Testing K={k_range[0]} to {k_range[1]}...")
    best_k = k_range[0]
    best_sil = -1
    results = []

    for k in range(k_range[0], k_range[1] + 1, 5):  # step by 5 for speed
        labels = fcluster(Z, t=k, criterion="maxclust")
        # Need at least 2 samples per cluster for silhouette
        if len(set(labels)) < 2:
            continue
        try:
            sil = silhouette_score(X, labels, sample_size=min(10000, len(X)))
            ch = calinski_harabasz_score(X, labels)
            results.append({"k": k, "silhouette": sil, "calinski_harabasz": ch})
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            continue

    results_df = pd.DataFrame(results)
    print(f"\n  K selection results:")
    print(results_df.to_string(index=False))
    print(f"\n  Best K={best_k} (silhouette={best_sil:.4f})")

    # Final clustering with best K
    labels = fcluster(Z, t=best_k, criterion="maxclust")
    cbg_df = cbg_df.copy()
    cbg_df["demo_cluster"] = labels

    # Cluster stats
    print(f"\n  Cluster distribution:")
    cluster_sizes = cbg_df.groupby("demo_cluster")["n_workers"].sum().sort_values(ascending=False)
    for c, n in cluster_sizes.items():
        print(f"    Cluster {c:3d}: {n:,} workers")

    return cbg_df, best_k


def build_a_work(year, cbg_cluster_map, acs_df):
    """Build final a_work CSV by joining matched stops with demographics + cluster."""
    in_path = f"{BASE}/data/matched_stops_{year}.csv"
    out_path = f"{BASE}/data/a_work_{year}.csv"

    print(f"\n  Building a_work_{year}.csv...")

    # Build lookup dicts for fast joining
    cbg_to_cluster = dict(zip(cbg_cluster_map["home_cbg"], cbg_cluster_map["demo_cluster"]))

    # ACS columns to include
    acs_cols = ["cbg_cuebiq"] + CLUSTER_FEATURES + [
        "median_age", "per_capita_income", "total_population",
        "unemployment_rate", "median_home_value", "median_gross_rent"
    ]
    acs_cols = [c for c in acs_cols if c in acs_df.columns]
    acs_lookup = acs_df[acs_cols].set_index("cbg_cuebiq")

    header_written = False
    total_out = 0

    for chunk in pd.read_csv(in_path, dtype={"cuebiq_id": str}, chunksize=CHUNK_SIZE):
        # Add cluster
        chunk["demo_cluster"] = chunk["home_cbg"].map(cbg_to_cluster)

        # Add ACS demographics
        acs_join = chunk["home_cbg"].map(lambda x: acs_lookup.loc[x] if x in acs_lookup.index else pd.Series(dtype=float))
        acs_joined = pd.DataFrame(acs_join.tolist(), index=chunk.index)
        for col in acs_joined.columns:
            chunk[col] = acs_joined[col]

        # Drop rows without cluster (home CBG not in clustering)
        chunk = chunk.dropna(subset=["demo_cluster"])
        chunk["demo_cluster"] = chunk["demo_cluster"].astype(int)

        total_out += len(chunk)
        chunk.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        del chunk; gc.collect()

    print(f"  Saved {total_out:,} rows to {out_path}")
    return total_out


def main():
    print("=== build_demographics_clustering.py ===\n")

    acs = load_acs()

    # Get workers from 2021 (in-sample) for clustering
    print("\nExtracting worker home CBGs...")
    workers_2021 = get_worker_home_cbgs(2021)
    workers_2022 = get_worker_home_cbgs(2022)

    # Combine for clustering (use both years' workers)
    all_workers = pd.concat([workers_2021, workers_2022]).drop_duplicates("cuebiq_id")
    print(f"Total unique workers (2021∪2022): {len(all_workers):,}")

    # Build CBG features
    print("\nBuilding CBG-level features...")
    merged, cbg_features = build_cbg_features(all_workers, acs)

    # Ward clustering
    print("\nWard clustering...")
    cbg_clustered, best_k = ward_clustering(cbg_features, k_range=(30, 80))

    # Save cluster mapping
    mapping_path = f"{BASE}/data/cbg_cluster_mapping.csv"
    cbg_clustered[["home_cbg", "demo_cluster", "n_workers"]].to_csv(mapping_path, index=False)
    print(f"\nSaved cluster mapping: {len(cbg_clustered):,} CBGs → {mapping_path}")

    # Build a_work files
    print("\nBuilding a_work files...")
    n_2021 = build_a_work(2021, cbg_clustered, acs)
    n_2022 = build_a_work(2022, cbg_clustered, acs)

    print(f"\n=== DONE ===")
    print(f"  Clusters: {best_k}")
    print(f"  a_work_2021: {n_2021:,} rows")
    print(f"  a_work_2022: {n_2022:,} rows")


if __name__ == "__main__":
    main()
