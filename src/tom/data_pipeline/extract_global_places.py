#!/usr/bin/env python3
"""
Extract NYC restaurant POIs from SafeGraph Global Places parquet files.
Filters for Manhattan (061), Brooklyn (047), Queens (081) restaurants.

Output: data/global_places_nyc_restaurants.csv
"""

import os
import pandas as pd

GP_DIR = "/scratch/sx2490/econai/dtbk-sft-half-year/dewey_poi/"
OUT_PATH = "/scratch/sx2490/econai/nyc_metro/data/global_places_nyc_restaurants.csv"

# NYC 3-borough zip code prefixes (Manhattan: 100xx, Brooklyn: 112xx, Queens: 110xx/111xx/113xx/114xx/116xx)
# More reliable: filter by REGION=NY and lat/lon bounding box
# Manhattan + Brooklyn + Queens bounding box
LAT_MIN, LAT_MAX = 40.49, 40.88
LON_MIN, LON_MAX = -74.06, -73.70

# All food service NAICS codes (722xxx)
# 722511: Full-Service Restaurants
# 722513: Limited-Service Restaurants
# 722514: Cafeteria/Buffet
# 722515: Snack/Beverage Bars
# 72251:  Restaurants (generic 5-digit)
# 7223xx: Special Food Services (dining halls, caterers)
# 722410: Drinking Places (bars)
# We include ALL 722% and filter out non-restaurant later if needed
NAICS_PREFIX = "722"  # match all food services

# Columns to keep
KEEP_COLS = [
    "PLACEKEY", "LOCATION_NAME", "LATITUDE", "LONGITUDE",
    "STREET_ADDRESS", "CITY", "POSTAL_CODE", "REGION",
    "NAICS_CODE", "SUB_CATEGORY", "TOP_CATEGORY",
    "POLYGON_WKT", "POLYGON_CLASS", "PARENT_PLACEKEY",
    "GEOMETRY_TYPE", "IS_SYNTHETIC", "ENCLOSED",
    "WKT_AREA_SQ_METERS", "OPENED_ON", "CLOSED_ON",
    "TRACKING_CLOSED_SINCE", "BRANDS", "CATEGORY_TAGS"
]


def main():
    files = sorted([f for f in os.listdir(GP_DIR)
                    if f.startswith("global-places") and f.endswith(".parquet")])
    print(f"Found {len(files)} Global Places parquet files")

    chunks = []
    total_rows = 0
    for i, fname in enumerate(files):
        path = os.path.join(GP_DIR, fname)
        df = pd.read_parquet(path, engine="fastparquet")
        total_rows += len(df)

        # Filter: NY region
        df = df[df["REGION"] == "NY"]

        # Filter: all food service NAICS codes (722%)
        df = df[df["NAICS_CODE"].astype(str).str.startswith(NAICS_PREFIX)]

        # Filter: bounding box (Manhattan + Brooklyn + Queens)
        df = df[
            (df["LATITUDE"] >= LAT_MIN) & (df["LATITUDE"] <= LAT_MAX) &
            (df["LONGITUDE"] >= LON_MIN) & (df["LONGITUDE"] <= LON_MAX)
        ]

        if len(df) > 0:
            # Keep only needed columns (handle missing cols gracefully)
            cols = [c for c in KEEP_COLS if c in df.columns]
            chunks.append(df[cols].copy())

        print(f"  [{i+1}/{len(files)}] {fname}: {len(df)} restaurants "
              f"(cumulative: {sum(len(c) for c in chunks)})")

    # Combine
    result = pd.concat(chunks, ignore_index=True)

    # Deduplicate by PLACEKEY (should be unique but just in case)
    before = len(result)
    result = result.drop_duplicates(subset=["PLACEKEY"], keep="first")
    if before != len(result):
        print(f"\nDeduplicated: {before} -> {len(result)} ({before - len(result)} duplicates)")

    # Summary
    print(f"\n=== RESULT ===")
    print(f"Total Global Places rows scanned: {total_rows:,}")
    print(f"NYC 3-borough restaurants: {len(result):,}")
    print(f"  Has POLYGON_WKT: {result['POLYGON_WKT'].notna().sum()} ({result['POLYGON_WKT'].notna().mean()*100:.1f}%)")
    print(f"  Has PARENT_PLACEKEY: {result['PARENT_PLACEKEY'].notna().sum()} ({result['PARENT_PLACEKEY'].notna().mean()*100:.1f}%)")
    print(f"  Unique LOCATION_NAME: {result['LOCATION_NAME'].nunique()}")

    print(f"\nBy NAICS:")
    for code, grp in result.groupby("NAICS_CODE"):
        sub = grp["SUB_CATEGORY"].iloc[0] if "SUB_CATEGORY" in grp.columns else ""
        print(f"  {code:8s} {len(grp):6,}  {sub}")

    print(f"\nBy SUB_CATEGORY:")
    print(result["SUB_CATEGORY"].value_counts().to_string())

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
