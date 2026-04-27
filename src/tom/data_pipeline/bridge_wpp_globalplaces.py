#!/usr/bin/env python3
"""
Bridge Weekly Patterns Plus (WPP) with SafeGraph Global Places to:
1. Give WPP stores real names (from Global Places)
2. Give WPP stores polygon geometries (from Global Places)
3. Classify temporal status: stable / new / closed (from WPP year presence)

Matching strategy: coordinate nearest-neighbor between WPP and Global Places.
Both use Advan/SafeGraph standardized coordinates (not GPS), so drift is minimal.

Input:
  - data/advan/wpp_raw/wpp_*.parquet (104 weekly files, 2021-2022)
  - data/global_places_nyc_restaurants.csv (38k POIs with names + polygons)

Output:
  - data/poi_universe.csv (master POI list with names, polygons, temporal status)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Use full (unfiltered) WPP data, filter locally to NAICS 722%
WPP_DIR = "/scratch/sx2490/econai/nyc_metro/data/advan/wpp_full_raw"
GP_PATH = "/scratch/sx2490/econai/nyc_metro/data/global_places_nyc_restaurants.csv"
OUT_PATH = "/scratch/sx2490/econai/nyc_metro/data/poi_universe.csv"

# Borough bounding box (Manhattan + Brooklyn + Queens)
LAT_MIN, LAT_MAX = 40.49, 40.88
LON_MIN, LON_MAX = -74.06, -73.70

# Max distance (meters) for coordinate matching
MAX_MATCH_DIST_M = 50  # 50m tolerance for standardized coords


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_wpp_stores():
    """Load all WPP files, extract unique stores per year."""
    files = sorted([f for f in os.listdir(WPP_DIR) if f.endswith('.parquet')])
    if not files:
        print(f"ERROR: No parquet files in {WPP_DIR}")
        sys.exit(1)

    print(f"Loading {len(files)} WPP files...")

    stores_2021 = {}  # persistent_id_store -> {lat, lon, name, visits, ...}
    stores_2022 = {}

    for i, fname in enumerate(files):
        path = os.path.join(WPP_DIR, fname)
        df = pd.read_parquet(path, engine="fastparquet",
                             columns=["PERSISTENT_ID_STORE", "LOCATION_NAME",
                                      "LATITUDE", "LONGITUDE", "STREET_ADDRESS",
                                      "NAICS_CODE", "SUB_CATEGORY", "BRAND",
                                      "VISIT_COUNTS", "VISITOR_COUNTS",
                                      "DATE_RANGE_START", "MEDIAN_DWELL",
                                      "POI_CBG", "POSTAL_CODE", "CITY"])

        # Filter to NAICS 722% (all food services) + 3 boroughs by bounding box
        df["NAICS_CODE"] = df["NAICS_CODE"].astype(str)
        df = df[df["NAICS_CODE"].str.startswith("722")]
        df = df[
            (df["LATITUDE"] >= LAT_MIN) & (df["LATITUDE"] <= LAT_MAX) &
            (df["LONGITUDE"] >= LON_MIN) & (df["LONGITUDE"] <= LON_MAX)
        ]

        # Determine year from DATE_RANGE_START
        df["year"] = pd.to_datetime(df["DATE_RANGE_START"]).dt.year

        for year_val, year_dict in [(2021, stores_2021), (2022, stores_2022)]:
            subset = df[df["year"] == year_val]
            for _, row in subset.iterrows():
                pid = str(row["PERSISTENT_ID_STORE"])
                vc = float(row["VISIT_COUNTS"]) if pd.notna(row["VISIT_COUNTS"]) else 0
                if pid in year_dict:
                    year_dict[pid]["total_visits"] += vc
                    year_dict[pid]["n_weeks"] += 1
                else:
                    year_dict[pid] = {
                        "persistent_id_store": pid,
                        "wpp_name": row["LOCATION_NAME"],
                        "wpp_lat": row["LATITUDE"],
                        "wpp_lon": row["LONGITUDE"],
                        "street_address": row["STREET_ADDRESS"],
                        "naics_code": str(row["NAICS_CODE"]),
                        "sub_category": row["SUB_CATEGORY"],
                        "brand": row["BRAND"],
                        "postal_code": row["POSTAL_CODE"],
                        "city": row["CITY"],
                        "poi_cbg": row["POI_CBG"],
                        "total_visits": vc,
                        "n_weeks": 1,
                    }

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] 2021: {len(stores_2021):,} stores, "
                  f"2022: {len(stores_2022):,} stores")

    df_2021 = pd.DataFrame(stores_2021.values())
    df_2022 = pd.DataFrame(stores_2022.values())
    return df_2021, df_2022


def classify_temporal(df_2021, df_2022):
    """Classify stores as stable/new/closed based on year presence."""
    ids_2021 = set(df_2021["persistent_id_store"])
    ids_2022 = set(df_2022["persistent_id_store"])

    stable = ids_2021 & ids_2022
    new = ids_2022 - ids_2021
    closed = ids_2021 - ids_2022

    print(f"\n=== Temporal Classification ===")
    print(f"  Stable (both years): {len(stable):,}")
    print(f"  New (2022 only):     {len(new):,}")
    print(f"  Closed (2021 only):  {len(closed):,}")

    # Build unified store list
    # For stable/closed: use 2021 info (original). For new: use 2022 info.
    rows = []
    for pid in stable:
        r = df_2021[df_2021["persistent_id_store"] == pid].iloc[0].to_dict()
        r2022 = df_2022[df_2022["persistent_id_store"] == pid].iloc[0]
        r["temporal_status"] = "stable"
        r["visits_2021"] = r["total_visits"]
        r["visits_2022"] = r2022["total_visits"]
        rows.append(r)
    for pid in new:
        r = df_2022[df_2022["persistent_id_store"] == pid].iloc[0].to_dict()
        r["temporal_status"] = "new"
        r["visits_2021"] = 0
        r["visits_2022"] = r["total_visits"]
        rows.append(r)
    for pid in closed:
        r = df_2021[df_2021["persistent_id_store"] == pid].iloc[0].to_dict()
        r["temporal_status"] = "closed"
        r["visits_2021"] = r["total_visits"]
        r["visits_2022"] = 0
        rows.append(r)

    return pd.DataFrame(rows)


def match_to_global_places(df_wpp, df_gp):
    """Match WPP stores to Global Places using coordinate nearest-neighbor."""
    print(f"\n=== Matching WPP → Global Places ===")
    print(f"  WPP stores: {len(df_wpp):,}")
    print(f"  Global Places: {len(df_gp):,}")

    # Build KD-tree from Global Places coordinates
    # Convert to approximate meters for distance threshold
    gp_coords = np.deg2rad(df_gp[["LATITUDE", "LONGITUDE"]].values)
    gp_tree = cKDTree(gp_coords)

    wpp_coords = np.deg2rad(df_wpp[["wpp_lat", "wpp_lon"]].values)

    # Query nearest neighbor
    # At NYC latitude, 1 degree ≈ 111km, so 50m ≈ 0.00045 degrees ≈ 7.85e-6 radians
    max_dist_rad = MAX_MATCH_DIST_M / 6371000.0
    distances, indices = gp_tree.query(wpp_coords, k=1)

    # Compute actual haversine distances for matched pairs
    matched_mask = distances <= max_dist_rad
    actual_dist_m = haversine_m(
        df_wpp["wpp_lat"].values, df_wpp["wpp_lon"].values,
        df_gp["LATITUDE"].values[indices], df_gp["LONGITUDE"].values[indices]
    )
    matched_mask = actual_dist_m <= MAX_MATCH_DIST_M

    print(f"  Matched within {MAX_MATCH_DIST_M}m: {matched_mask.sum():,} ({matched_mask.sum()/len(df_wpp)*100:.1f}%)")
    print(f"  Unmatched: {(~matched_mask).sum():,}")

    # Assign Global Places info to matched WPP stores
    df_wpp = df_wpp.copy()
    df_wpp["gp_matched"] = matched_mask
    df_wpp["match_dist_m"] = actual_dist_m
    df_wpp["PLACEKEY"] = None
    df_wpp["gp_name"] = None
    df_wpp["POLYGON_WKT"] = None
    df_wpp["POLYGON_CLASS"] = None
    df_wpp["PARENT_PLACEKEY"] = None
    df_wpp["gp_lat"] = None
    df_wpp["gp_lon"] = None

    gp_cols = ["PLACEKEY", "LOCATION_NAME", "POLYGON_WKT", "POLYGON_CLASS",
               "PARENT_PLACEKEY", "LATITUDE", "LONGITUDE"]
    for col, out_col in zip(gp_cols, ["PLACEKEY", "gp_name", "POLYGON_WKT",
                                       "POLYGON_CLASS", "PARENT_PLACEKEY",
                                       "gp_lat", "gp_lon"]):
        df_wpp.loc[matched_mask, out_col] = df_gp[col].values[indices[matched_mask]]

    # Use Global Places name when available, otherwise keep WPP name
    df_wpp["poi_name"] = df_wpp["gp_name"].fillna(df_wpp["wpp_name"])
    df_wpp["poi_lat"] = df_wpp["gp_lat"].fillna(df_wpp["wpp_lat"]).astype(float)
    df_wpp["poi_lon"] = df_wpp["gp_lon"].fillna(df_wpp["wpp_lon"]).astype(float)

    # Summary of name quality
    generic_names = ["Full-Service Restaurants", "Limited-Service Restaurants",
                     "Snack and Nonalcoholic Beverage Bars"]
    has_real_name = ~df_wpp["poi_name"].isin(generic_names)
    print(f"\n  Name quality after matching:")
    print(f"    Real names: {has_real_name.sum():,} ({has_real_name.sum()/len(df_wpp)*100:.1f}%)")
    print(f"    Generic names: {(~has_real_name).sum():,} ({(~has_real_name).sum()/len(df_wpp)*100:.1f}%)")

    return df_wpp


def main():
    # Step 1: Load WPP data
    df_2021, df_2022 = load_wpp_stores()

    # Step 2: Classify temporal status
    df_all = classify_temporal(df_2021, df_2022)

    # Step 3: Load Global Places
    df_gp = pd.read_csv(GP_PATH)
    print(f"\nGlobal Places loaded: {len(df_gp):,} restaurants")

    # Step 4: Match
    df_result = match_to_global_places(df_all, df_gp)

    # Step 5: Build final output
    out_cols = [
        "persistent_id_store", "PLACEKEY", "poi_name", "poi_lat", "poi_lon",
        "street_address", "city", "postal_code", "poi_cbg",
        "naics_code", "sub_category", "brand",
        "temporal_status", "visits_2021", "visits_2022",
        "POLYGON_WKT", "POLYGON_CLASS", "PARENT_PLACEKEY",
        "gp_matched", "match_dist_m", "wpp_name", "wpp_lat", "wpp_lon"
    ]
    df_out = df_result[[c for c in out_cols if c in df_result.columns]]

    # Summary
    print(f"\n=== FINAL POI UNIVERSE ===")
    print(f"Total stores: {len(df_out):,}")
    for status in ["stable", "new", "closed"]:
        n = (df_out["temporal_status"] == status).sum()
        matched = df_out.loc[df_out["temporal_status"] == status, "gp_matched"].sum()
        print(f"  {status:8s}: {n:6,} ({matched:,} matched to Global Places)")

    has_polygon = df_out["POLYGON_WKT"].notna().sum()
    print(f"\nHas polygon: {has_polygon:,} ({has_polygon/len(df_out)*100:.1f}%)")

    generic = df_out["poi_name"].isin(["Full-Service Restaurants",
                                        "Limited-Service Restaurants",
                                        "Snack and Nonalcoholic Beverage Bars"])
    print(f"Has real name: {(~generic).sum():,} ({(~generic).sum()/len(df_out)*100:.1f}%)")
    print(f"Generic name (no match): {generic.sum():,} ({generic.sum()/len(df_out)*100:.1f}%)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
