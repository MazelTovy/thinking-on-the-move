#!/usr/bin/env python3
"""
resolve_poi_names.py — Script 1 of POI Polygon Pipeline

Build a definitive name + temporal status for every POI by merging:
  1. safegraph_2021_merged.csv  (42k NYC restaurants, 100% real names, PLACEKEY)
  2. global_places_nyc_restaurants.csv  (50k, 100% real names, PLACEKEY, polygons)
  3. poi_universe.csv  (27k WPP stores, temporal tracking, PERSISTENT_ID_STORE)

Name priority:  safegraph_2021 > Global Places > WPP > DROP if all generic
Temporal:       WPP year-presence cross-validated against safegraph_2021

Output: data/poi_name_authority.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = "/scratch/sx2490/econai/nyc_metro"
SG21_PATH = "/scratch/sx2490/econai/dtbk-sft-half-year/safegraph_2021_merged.csv"
GP_PATH   = f"{BASE}/data/global_places_nyc_restaurants.csv"
WPP_PATH  = f"{BASE}/data/poi_universe.csv"
OUT_PATH  = f"{BASE}/data/poi_name_authority.csv"

# ── NYC 3-borough bounding box ───────────────────────────────────────────────
LAT_MIN, LAT_MAX = 40.49, 40.88
LON_MIN, LON_MAX = -74.06, -73.70

GENERIC_NAMES = {
    "Full-Service Restaurants",
    "Limited-Service Restaurants",
    "Snack and Nonalcoholic Beverage Bars",
    "Drinking Places (Alcoholic Beverages)",
    "Restaurants and Other Eating Places",
    "Caterers",
    "Mobile Food Services",
    "Food Service Contractors",
    "Special Food Services",
}


def is_generic(name):
    if pd.isna(name) or str(name).strip() == "":
        return True
    return str(name).strip() in GENERIC_NAMES


def load_safegraph_2021():
    """Load safegraph_2021 NYC food POIs with PLACEKEY."""
    df = pd.read_csv(SG21_PATH, usecols=[
        "PLACEKEY", "LOCATION_NAME", "LATITUDE", "LONGITUDE",
        "SUB_CATEGORY", "RAW_VISIT_COUNTS", "RAW_VISITOR_COUNTS", "MEDIAN_DWELL"
    ])
    # NYC bounding box
    df = df[(df["LATITUDE"] >= LAT_MIN) & (df["LATITUDE"] <= LAT_MAX) &
            (df["LONGITUDE"] >= LON_MIN) & (df["LONGITUDE"] <= LON_MAX)]
    # All food categories (SUB_CATEGORY contains restaurant-related terms)
    food_cats = [s.strip() for s in GENERIC_NAMES]
    df = df[df["SUB_CATEGORY"].str.strip().isin(food_cats)]
    df = df.drop_duplicates("PLACEKEY")
    print(f"safegraph_2021 NYC food: {len(df):,} POIs")
    return df


def load_global_places():
    """Load Global Places with polygons."""
    df = pd.read_csv(GP_PATH, low_memory=False)
    df = df.drop_duplicates("PLACEKEY")
    print(f"Global Places NYC food: {len(df):,} POIs")
    return df


def load_wpp_universe():
    """Load WPP-based POI universe (has temporal status)."""
    df = pd.read_csv(WPP_PATH, low_memory=False)
    print(f"WPP poi_universe: {len(df):,} stores")
    return df


def build_name_authority(df_sg, df_gp, df_wpp):
    """
    Merge all sources into a single name authority table.

    Strategy:
    - Start from Global Places (most complete polygon coverage)
    - Enrich with safegraph_2021 names and visit data via PLACEKEY join
    - Add WPP temporal data via PLACEKEY join
    - Add safegraph_2021-only POIs (no GP match) as fallback entries
    """
    # ── Base: Global Places (50k, has polygons + real names) ─────────────────
    auth = df_gp[["PLACEKEY", "LOCATION_NAME", "LATITUDE", "LONGITUDE",
                   "NAICS_CODE", "SUB_CATEGORY", "STREET_ADDRESS", "CITY",
                   "POSTAL_CODE", "POLYGON_WKT", "POLYGON_CLASS",
                   "PARENT_PLACEKEY", "OPENED_ON", "CLOSED_ON",
                   "CATEGORY_TAGS", "BRANDS"]].copy()
    auth = auth.rename(columns={"LOCATION_NAME": "gp_name",
                                "LATITUDE": "gp_lat", "LONGITUDE": "gp_lon"})

    # ── Join safegraph_2021 by PLACEKEY ──────────────────────────────────────
    sg_cols = df_sg[["PLACEKEY", "LOCATION_NAME", "LATITUDE", "LONGITUDE",
                     "RAW_VISIT_COUNTS", "RAW_VISITOR_COUNTS", "MEDIAN_DWELL"]].copy()
    sg_cols = sg_cols.rename(columns={
        "LOCATION_NAME": "sg21_name", "LATITUDE": "sg21_lat", "LONGITUDE": "sg21_lon",
        "RAW_VISIT_COUNTS": "sg21_visits", "RAW_VISITOR_COUNTS": "sg21_visitors",
        "MEDIAN_DWELL": "sg21_dwell"
    })
    auth = auth.merge(sg_cols, on="PLACEKEY", how="left")

    # ── Join WPP temporal data by PLACEKEY ───────────────────────────────────
    wpp_with_pk = df_wpp[df_wpp["PLACEKEY"].notna()].copy()
    # Deduplicate WPP by PLACEKEY (keep highest visits)
    wpp_with_pk["total_visits"] = wpp_with_pk["visits_2021"].fillna(0) + wpp_with_pk["visits_2022"].fillna(0)
    wpp_with_pk = wpp_with_pk.sort_values("total_visits", ascending=False).drop_duplicates("PLACEKEY")
    wpp_cols = wpp_with_pk[["PLACEKEY", "persistent_id_store", "wpp_name",
                             "temporal_status", "visits_2021", "visits_2022"]].copy()
    wpp_cols = wpp_cols.rename(columns={
        "temporal_status": "wpp_temporal", "visits_2021": "wpp_visits_2021",
        "visits_2022": "wpp_visits_2022"
    })
    auth = auth.merge(wpp_cols, on="PLACEKEY", how="left")

    # ── Name resolution cascade ──────────────────────────────────────────────
    def resolve_name(row):
        # Priority 1: safegraph_2021
        if not is_generic(row.get("sg21_name")):
            return row["sg21_name"], "safegraph_2021"
        # Priority 2: Global Places
        if not is_generic(row.get("gp_name")):
            return row["gp_name"], "global_places"
        # Priority 3: WPP
        if not is_generic(row.get("wpp_name")):
            return row["wpp_name"], "wpp"
        return None, "generic_dropped"

    results = auth.apply(resolve_name, axis=1, result_type="expand")
    auth["final_name"] = results[0]
    auth["name_source"] = results[1]

    # ── Temporal classification (conservative — only mark closed with hard evidence) ─
    #
    # Closed signals (high confidence):
    #   1. GP CLOSED_ON date <= 2022-12-31 (SafeGraph confirmed closure)
    #   2. WPP tracked it in 2021 but NOT in 2022 (panel dropped it)
    #
    # NOT a closed signal:
    #   - "In safegraph_2021 but not in Global Places" — could be NAICS reclassification,
    #     PLACEKEY change, or GP coverage gap. Default to stable.

    # Parse CLOSED_ON
    auth["_closed_date"] = pd.to_datetime(auth["CLOSED_ON"], errors="coerce")
    gp_closed_by_2021 = auth["_closed_date"] <= "2021-12-31"
    gp_closed_in_2022 = (auth["_closed_date"] > "2021-12-31") & (auth["_closed_date"] <= "2022-12-31")
    gp_closed_after_2022 = auth["_closed_date"] > "2022-12-31"

    def resolve_temporal(row):
        wpp_t = row.get("wpp_temporal")
        in_sg21 = pd.notna(row.get("sg21_name"))
        in_gp = pd.notna(row.get("gp_name"))
        pk_idx = row.name  # dataframe index for boolean masks

        # Hard evidence: GP says closed before our study
        if gp_closed_by_2021.get(pk_idx, False):
            return "closed_before_study"
        # Hard evidence: GP says closed during 2022
        if gp_closed_in_2022.get(pk_idx, False):
            return "closed_2022"

        # WPP-based classification
        if wpp_t == "new":
            if in_sg21:
                return "stable"  # WPP newly tracked, but SG21 already knew it
            return "new"
        elif wpp_t == "closed":
            return "closed_2022"  # WPP saw it in 2021, lost it in 2022
        elif wpp_t == "stable":
            return "stable"
        elif wpp_t == "gp_only":
            if in_sg21:
                return "stable"
            return "stable"  # conservative: in GP now → assume exists

        # No WPP data — use conservative defaults
        if in_sg21:
            return "stable"  # existed in 2021, no evidence of closure → stable
        elif in_gp:
            return "stable"  # in current GP → open
        return "stable"  # absolute fallback: no evidence of closure

    auth["temporal_status"] = auth.apply(resolve_temporal, axis=1)
    auth = auth.drop(columns=["_closed_date"])

    # ── Coordinates: prefer GP (polygon-consistent) ──────────────────────────
    auth["poi_lat"] = auth["gp_lat"]
    auth["poi_lon"] = auth["gp_lon"]

    # ── Add safegraph_2021-only POIs not in Global Places ────────────────────
    gp_pks = set(auth["PLACEKEY"])
    sg_only = df_sg[~df_sg["PLACEKEY"].isin(gp_pks)].copy()
    print(f"safegraph_2021 POIs not in Global Places: {len(sg_only):,}")

    if len(sg_only) > 0:
        sg_extra = pd.DataFrame({
            "PLACEKEY": sg_only["PLACEKEY"].values,
            "gp_name": None,
            "gp_lat": None,
            "gp_lon": None,
            "sg21_name": sg_only["LOCATION_NAME"].values,
            "sg21_lat": sg_only["LATITUDE"].values,
            "sg21_lon": sg_only["LONGITUDE"].values,
            "sg21_visits": sg_only["RAW_VISIT_COUNTS"].values,
            "sg21_visitors": sg_only["RAW_VISITOR_COUNTS"].values,
            "sg21_dwell": sg_only["MEDIAN_DWELL"].values,
            "SUB_CATEGORY": sg_only["SUB_CATEGORY"].values,
            "poi_lat": sg_only["LATITUDE"].values,
            "poi_lon": sg_only["LONGITUDE"].values,
            "POLYGON_WKT": None,
            "POLYGON_CLASS": None,
            "PARENT_PLACEKEY": None,
        })
        # Name + temporal for sg-only
        sg_extra["final_name"] = sg_extra["sg21_name"].where(
            ~sg_extra["sg21_name"].apply(is_generic), None)
        sg_extra["name_source"] = sg_extra["final_name"].apply(
            lambda x: "safegraph_2021" if pd.notna(x) else "generic_dropped")
        sg_extra["temporal_status"] = "stable"  # conservative: no evidence of closure

        auth = pd.concat([auth, sg_extra], ignore_index=True)

    return auth


def add_wpp_only_pois(auth, df_wpp):
    """Add WPP stores that have no PLACEKEY match to Global Places or safegraph."""
    # WPP stores without PLACEKEY (i.e., not matched to GP during bridge step)
    existing_pids = set(auth.loc[auth["persistent_id_store"].notna(), "persistent_id_store"])
    existing_pks = set(auth["PLACEKEY"])

    wpp_new = df_wpp[
        (~df_wpp["persistent_id_store"].isin(existing_pids)) &
        (df_wpp["PLACEKEY"].isna() | ~df_wpp["PLACEKEY"].isin(existing_pks))
    ].copy()

    if len(wpp_new) == 0:
        return auth

    print(f"WPP-only stores (no GP/SG match): {len(wpp_new):,}")

    wpp_extra = pd.DataFrame({
        "PLACEKEY": None,
        "persistent_id_store": wpp_new["persistent_id_store"].values,
        "gp_name": None,
        "wpp_name": wpp_new["wpp_name"].values if "wpp_name" in wpp_new else wpp_new["poi_name"].values,
        "poi_lat": wpp_new["poi_lat"].values,
        "poi_lon": wpp_new["poi_lon"].values,
        "SUB_CATEGORY": wpp_new["sub_category"].values,
        "NAICS_CODE": wpp_new["naics_code"].values,
        "STREET_ADDRESS": wpp_new["street_address"].values,
        "CITY": wpp_new["city"].values,
        "POSTAL_CODE": wpp_new["postal_code"].values,
        "wpp_temporal": wpp_new["temporal_status"].values,
        "wpp_visits_2021": wpp_new["visits_2021"].values,
        "wpp_visits_2022": wpp_new["visits_2022"].values,
        "POLYGON_WKT": wpp_new.get("POLYGON_WKT", pd.Series([None]*len(wpp_new))).values,
    })

    # Resolve names
    wpp_extra["final_name"] = wpp_extra.apply(
        lambda r: r["wpp_name"] if not is_generic(r["wpp_name"]) else None, axis=1)
    wpp_extra["name_source"] = wpp_extra["final_name"].apply(
        lambda x: "wpp" if pd.notna(x) else "generic_dropped")
    wpp_extra["temporal_status"] = wpp_extra["wpp_temporal"]

    auth = pd.concat([auth, wpp_extra], ignore_index=True)
    return auth


def main():
    print("=== resolve_poi_names.py ===\n")

    df_sg = load_safegraph_2021()
    df_gp = load_global_places()
    df_wpp = load_wpp_universe()

    print("\nBuilding name authority...")
    auth = build_name_authority(df_sg, df_gp, df_wpp)

    print("Adding WPP-only stores...")
    auth = add_wpp_only_pois(auth, df_wpp)

    # ── Drop generic-only POIs ───────────────────────────────────────────────
    n_before = len(auth)
    dropped = auth[auth["name_source"] == "generic_dropped"]
    auth = auth[auth["name_source"] != "generic_dropped"]
    print(f"\nDropped {len(dropped):,} POIs with all-generic names ({len(dropped)/n_before*100:.1f}%)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n=== POI NAME AUTHORITY ===")
    print(f"Total POIs: {len(auth):,}")
    print(f"\nBy name source:")
    for src, cnt in auth["name_source"].value_counts().items():
        print(f"  {src:20s}: {cnt:,}")
    print(f"\nBy temporal status:")
    for st, cnt in auth["temporal_status"].value_counts().items():
        print(f"  {st:10s}: {cnt:,}")
    print(f"\nHas polygon: {auth['POLYGON_WKT'].notna().sum():,} "
          f"({auth['POLYGON_WKT'].notna().mean()*100:.1f}%)")

    # ── Verification: Jasper Kane ────────────────────────────────────────────
    jk = auth[auth["final_name"].str.contains("Jasper", case=False, na=False)]
    print(f"\nJasper matches: {len(jk)}")
    if len(jk) > 0:
        for _, r in jk.head(5).iterrows():
            print(f"  {r['final_name']:40s} src={r['name_source']:15s} "
                  f"temporal={r['temporal_status']:8s} poly={'Y' if pd.notna(r.get('POLYGON_WKT')) else 'N'}")

    # ── Verification: NYU Starbucks ──────────────────────────────────────────
    starbucks_nyu = auth[(auth["final_name"].str.contains("Starbucks", case=False, na=False)) &
                         (auth["poi_lat"].between(40.690, 40.698)) &
                         (auth["poi_lon"].between(-73.990, -73.983))]
    print(f"\nStarbucks near NYU: {len(starbucks_nyu)}")
    if len(starbucks_nyu) > 0:
        for _, r in starbucks_nyu.iterrows():
            print(f"  PK={r.get('PLACEKEY','?'):25s} ({r['poi_lat']:.6f}, {r['poi_lon']:.6f}) "
                  f"src={r['name_source']} temporal={r['temporal_status']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    auth.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
