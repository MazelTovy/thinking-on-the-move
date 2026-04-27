#!/usr/bin/env python3
"""
prefilter_and_match.py — Pre-filter lunch stops near restaurants, then polygon match.

Two-stage pipeline:
  Stage 1 (PREFILTER): KDTree to keep only stops within 100m of any known restaurant.
    This reduces 21M/28M stops to ~2-5M, making polygon matching tractable.
    Does NOT affect results — polygon containment is still the final arbiter.

  Stage 2 (MATCH): Polygon containment matching (OWNED/SHARED_DISTINCT/SHARED_BUILDING).

Usage:
  python3 prefilter_and_match.py --year 2021
  python3 prefilter_and_match.py --year 2022 --load_building_map data/building_name_map.csv
"""

import argparse
import math
import os
import pickle
import time
import gc

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

BASE = "/scratch/sx2490/econai/nyc_metro"
PKL_PATH = f"{BASE}/data/polygon_spatial_index.pkl"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"
PREFILTER_RADIUS_M = 100
FALLBACK_MAX_DIST_M = 50
CHUNK_SIZE = 500_000


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def build_prefilter_tree():
    """Build a KDTree from ALL known restaurant coordinates for fast proximity filter."""
    print("Building prefilter KDTree from poi_name_authority...")
    auth = pd.read_csv(AUTH_PATH, usecols=["poi_lat", "poi_lon"], low_memory=False)
    auth = auth.dropna(subset=["poi_lat", "poi_lon"])
    coords = np.deg2rad(auth[["poi_lat", "poi_lon"]].values)
    tree = cKDTree(coords)
    print(f"  {len(coords):,} restaurant coordinates indexed")
    del auth
    return tree


def prefilter_chunk(df, prefilter_tree):
    """Keep only stops within PREFILTER_RADIUS_M of any known restaurant."""
    coords = np.deg2rad(df[["lat", "lng"]].values)
    dist, _ = prefilter_tree.query(coords, k=1)
    dist_m = dist * 6_371_000.0
    mask = dist_m <= PREFILTER_RADIUS_M
    return df[mask].copy()


def build_full_fallback_tree():
    """Build a KDTree from ALL POIs (with and without polygon) for secondary fallback.
    When a stop is near a polygon-having restaurant but GPS drifted outside the polygon,
    this catches it via nearest-centroid matching within 50m."""
    print("Building full fallback tree from poi_name_authority...")
    auth = pd.read_csv(AUTH_PATH, low_memory=False)
    auth = auth.dropna(subset=["poi_lat", "poi_lon", "final_name"])
    pois = []
    for _, r in auth.iterrows():
        pois.append({
            "PLACEKEY": r.get("PLACEKEY"),
            "name": r["final_name"],
            "lat": float(r["poi_lat"]),
            "lon": float(r["poi_lon"]),
            "temporal_status": r.get("temporal_status", "stable"),
            "sub_category": r.get("SUB_CATEGORY", ""),
        })
    coords = np.array([[p["lat"], p["lon"]] for p in pois])
    tree = cKDTree(np.deg2rad(coords))
    print(f"  {len(pois):,} POIs in full fallback tree")
    return tree, pois


def load_spatial_index():
    print(f"Loading polygon spatial index...")
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data['polygons']):,} polygons, {len(data['fallback_pois']):,} fallback POIs")
    return data


def match_stop(lat, lng, idx_data):
    """Match a single GPS stop to a POI via polygon containment."""
    polygons = idx_data["polygons"]
    tree = idx_data["tree"]
    geom_list = idx_data["geom_list"]

    point = Point(lng, lat)

    # Polygon containment
    candidates = tree.query(point)
    for geom_idx in candidates:
        if not point.within(geom_list[geom_idx]):
            continue

        poly = polygons[geom_idx]
        ptype = poly["polygon_type"]

        if ptype == "OWNED":
            m = poly["members"][0]
            return {
                "poi_name": m["name"], "poi_lat": m["lat"], "poi_lon": m["lon"],
                "PLACEKEY": m["PLACEKEY"], "polygon_type": "OWNED",
                "temporal_status": m["temporal_status"],
                "sub_category": m.get("sub_category", ""),
                "building_idx": None,
            }
        elif ptype == "SHARED_DISTINCT":
            best = min(poly["members"],
                       key=lambda m: haversine_m(lat, lng, m["lat"], m["lon"]))
            return {
                "poi_name": best["name"], "poi_lat": best["lat"], "poi_lon": best["lon"],
                "PLACEKEY": best["PLACEKEY"], "polygon_type": "SHARED_DISTINCT",
                "temporal_status": best["temporal_status"],
                "sub_category": best.get("sub_category", ""),
                "building_idx": None,
            }
        elif ptype == "SHARED_BUILDING":
            best_member = min(poly["members"],
                              key=lambda m: haversine_m(lat, lng, m["lat"], m["lon"]))
            return {
                "poi_name": poly["building_name"],
                "poi_lat": poly["centroid_lat"], "poi_lon": poly["centroid_lon"],
                "PLACEKEY": best_member["PLACEKEY"], "polygon_type": "SHARED_BUILDING",
                "temporal_status": best_member["temporal_status"],
                "sub_category": best_member.get("sub_category", ""),
                "building_idx": geom_idx,
                "member_name": best_member["name"],
            }
        break

    # Fallback: coordinate-based nearest using FULL POI tree (includes polygon-having POIs)
    full_fb = idx_data.get("full_fallback_tree")
    full_pois = idx_data.get("full_fallback_pois")
    if full_fb is not None and full_pois:
        query_rad = np.deg2rad([[lat, lng]])
        dist, idx = full_fb.query(query_rad, k=1)
        dist_m = dist[0] * 6_371_000.0
        if dist_m <= FALLBACK_MAX_DIST_M:
            p = full_pois[idx[0]]
            return {
                "poi_name": p["name"], "poi_lat": p["lat"], "poi_lon": p["lon"],
                "PLACEKEY": p["PLACEKEY"], "polygon_type": "FALLBACK",
                "temporal_status": p["temporal_status"],
                "sub_category": p.get("sub_category", ""),
                "building_idx": None,
            }

    return None


def process_year(year, idx_data, prefilter_tree, building_name_map=None):
    stop_path = f"{BASE}/data/cuebiq/lunch_with_home_{year}.csv"
    out_path = f"{BASE}/data/matched_stops_{year}.csv"

    if not os.path.exists(stop_path):
        print(f"ERROR: {stop_path} not found")
        return None

    print(f"\n{'='*60}")
    print(f"Processing {year}")
    print(f"{'='*60}")
    t0_total = time.time()

    # Count total lines (fast via subprocess)
    import subprocess
    print("Counting input rows...", end=" ", flush=True)
    n_total = int(subprocess.run(["wc", "-l", stop_path],
                                 capture_output=True, text=True).stdout.split()[0]) - 1
    print(f"{n_total:,}")

    # Process in chunks
    header_written = False
    stats = {"prefilter_in": 0, "prefilter_out": 0,
             "polygon": 0, "fallback": 0, "unmatched": 0}
    building_member_counts = {}
    polygons = idx_data["polygons"]

    for chunk_i, chunk in enumerate(pd.read_csv(stop_path, chunksize=CHUNK_SIZE)):
        t0 = time.time()
        stats["prefilter_in"] += len(chunk)

        # Stage 1: Prefilter
        chunk = prefilter_chunk(chunk, prefilter_tree)
        stats["prefilter_out"] += len(chunk)

        if len(chunk) == 0:
            elapsed = time.time() - t0
            print(f"  Chunk {chunk_i}: 0 after prefilter ({elapsed:.0f}s)")
            continue

        # Stage 2: Polygon match
        results = []
        for _, row in chunk.iterrows():
            match = match_stop(row["lat"], row["lng"], idx_data)
            if match is None:
                stats["unmatched"] += 1
                continue

            if match["polygon_type"] == "FALLBACK":
                stats["fallback"] += 1
            else:
                stats["polygon"] += 1

            # Track building member counts
            bidx = match.get("building_idx")
            if bidx is not None:
                mname = match.get("member_name", match["poi_name"])
                if bidx not in building_member_counts:
                    building_member_counts[bidx] = {}
                building_member_counts[bidx][mname] = \
                    building_member_counts[bidx].get(mname, 0) + 1

            results.append({
                "cuebiq_id": row["cuebiq_id"],
                "lat": row["lat"], "lng": row["lng"],
                "dwell_time_minutes": row["dwell_time_minutes"],
                "stop_ts": row["stop_ts"],
                "admin2_id": row.get("admin2_id"),
                "home_cbg": row.get("home_cbg"),
                "work_cbg": row.get("work_cbg"),
                "poi_name": match["poi_name"],
                "poi_lat": match["poi_lat"], "poi_lon": match["poi_lon"],
                "PLACEKEY": match["PLACEKEY"],
                "polygon_type": match["polygon_type"],
                "temporal_status": match["temporal_status"],
                "sub_category": match["sub_category"],
            })

        if results:
            df_out = pd.DataFrame(results)
            df_out.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True

        elapsed = time.time() - t0
        matched = stats["polygon"] + stats["fallback"]
        print(f"  Chunk {chunk_i}: {len(chunk):,} prefiltered → {len(results):,} matched "
              f"(total: {matched:,}/{stats['prefilter_in']:,}) [{elapsed:.0f}s]")

        del chunk, results; gc.collect()

    # Pass 2: Finalize building names
    bname_map = dict(building_name_map) if building_name_map else {}
    for bidx, member_counts in building_member_counts.items():
        bkey = f"{polygons[bidx]['centroid_lat']:.6f}_{polygons[bidx]['centroid_lon']:.6f}"
        if bkey not in bname_map:
            bname_map[bkey] = max(member_counts, key=member_counts.get)

    # TODO: Apply finalized building names (read back CSV, update, rewrite)
    # For now, building names are the initial names from build_polygon_index.py

    total_time = time.time() - t0_total
    matched = stats["polygon"] + stats["fallback"]
    print(f"\n=== {year} Summary ===")
    print(f"  Input stops: {stats['prefilter_in']:,}")
    print(f"  After prefilter (<{PREFILTER_RADIUS_M}m): {stats['prefilter_out']:,} "
          f"({stats['prefilter_out']/max(stats['prefilter_in'],1)*100:.1f}%)")
    print(f"  Polygon matched: {stats['polygon']:,}")
    print(f"  Fallback matched: {stats['fallback']:,}")
    print(f"  Unmatched: {stats['unmatched']:,}")
    print(f"  Total matched: {matched:,} "
          f"({matched/max(stats['prefilter_out'],1)*100:.1f}% of prefiltered)")
    print(f"  Saved to: {out_path}")
    print(f"  Time: {total_time/60:.1f} min")

    return bname_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--load_building_map", type=str, default=None)
    parser.add_argument("--save_building_map", type=str,
                        default=f"{BASE}/data/building_name_map.csv")
    args = parser.parse_args()

    prefilter_tree = build_prefilter_tree()
    idx_data = load_spatial_index()

    # Add full fallback tree to idx_data
    full_fb_tree, full_fb_pois = build_full_fallback_tree()
    idx_data["full_fallback_tree"] = full_fb_tree
    idx_data["full_fallback_pois"] = full_fb_pois

    bname_map = None
    if args.load_building_map and os.path.exists(args.load_building_map):
        bm = pd.read_csv(args.load_building_map)
        bname_map = dict(zip(bm["centroid_key"], bm["building_name"]))
        print(f"Loaded building name map: {len(bname_map)} entries")

    result_map = process_year(args.year, idx_data, prefilter_tree, bname_map)

    if result_map and args.save_building_map:
        bm_df = pd.DataFrame([
            {"centroid_key": k, "building_name": v}
            for k, v in result_map.items()
        ])
        bm_df.to_csv(args.save_building_map, index=False)
        print(f"\nSaved building_name_map: {len(bm_df)} entries → {args.save_building_map}")


if __name__ == "__main__":
    main()
