#!/usr/bin/env python3
"""
build_polygon_index.py — Script 2 of POI Polygon Pipeline

Classify polygons into three cases and build spatial index:
  OWNED          — 1 POI → direct assignment
  SHARED_DISTINCT — 2-5 POIs, no parent → nearest centroid within polygon
  SHARED_BUILDING — 6+ POIs or has parent → building-level aggregate

Input:
  data/global_places_nyc_restaurants.csv
  data/poi_name_authority.csv

Output:
  data/polygon_spatial_index.pkl  (STRtree + polygon metadata)
  data/polygon_classification.csv (human-readable summary)
"""

import os
import pickle
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.strtree import STRtree
import math

BASE = "/scratch/sx2490/econai/nyc_metro"
GP_PATH   = f"{BASE}/data/global_places_nyc_restaurants.csv"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"
OUT_PKL   = f"{BASE}/data/polygon_spatial_index.pkl"
OUT_CSV   = f"{BASE}/data/polygon_classification.csv"

# Threshold: shared polygons with <= this many POIs are SHARED_DISTINCT
DISTINCT_MAX_POIS = 5


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def main():
    print("=== build_polygon_index.py ===\n")

    # ── Load name authority ──────────────────────────────────────────────────
    auth = pd.read_csv(AUTH_PATH, low_memory=False)
    print(f"Name authority: {len(auth):,} POIs")

    # Only POIs with polygons can be in the spatial index
    auth_poly = auth[auth["POLYGON_WKT"].notna()].copy()
    print(f"With polygon: {len(auth_poly):,}")

    # ── Parse geometries ─────────────────────────────────────────────────────
    print("Parsing polygon geometries...")
    geoms = []
    valid_mask = []
    for wkt_str in auth_poly["POLYGON_WKT"]:
        try:
            g = shapely_wkt.loads(str(wkt_str))
            if g.is_valid and not g.is_empty:
                geoms.append(g)
                valid_mask.append(True)
            else:
                geoms.append(None)
                valid_mask.append(False)
        except Exception:
            geoms.append(None)
            valid_mask.append(False)

    auth_poly = auth_poly[valid_mask].copy()
    auth_poly["geometry"] = [g for g, v in zip(geoms, valid_mask) if v]
    print(f"Valid geometries: {len(auth_poly):,}")

    # ── Group by POLYGON_WKT to find shared polygons ─────────────────────────
    print("Grouping by POLYGON_WKT...")
    wkt_groups = auth_poly.groupby("POLYGON_WKT")

    polygons = []  # list of polygon records for the spatial index
    stats = {"OWNED": 0, "SHARED_DISTINCT": 0, "SHARED_BUILDING": 0}

    for wkt_str, grp in wkt_groups:
        n_pois = len(grp)
        has_parent = grp["PARENT_PLACEKEY"].notna().any()
        geom = grp.iloc[0]["geometry"]
        centroid = geom.centroid

        # Member POIs info
        members = []
        for _, r in grp.iterrows():
            members.append({
                "PLACEKEY": r.get("PLACEKEY"),
                "name": r["final_name"],
                "lat": float(r["poi_lat"]),
                "lon": float(r["poi_lon"]),
                "temporal_status": r.get("temporal_status", "unknown"),
                "name_source": r.get("name_source", ""),
                "sub_category": r.get("SUB_CATEGORY", ""),
                "sg21_visits": float(r["sg21_visits"]) if pd.notna(r.get("sg21_visits")) else 0,
            })

        # ── Classify ─────────────────────────────────────────────────────────
        # OWNED: single POI per polygon
        # SHARED_DISTINCT: small number of clearly different restaurants
        #   → keep individual identities, match GPS to nearest centroid
        # SHARED_BUILDING: food court / large complex → aggregate to building
        if n_pois == 1:
            ptype = "OWNED"
        elif n_pois <= DISTINCT_MAX_POIS:
            # Even with parent, ≤5 distinct restaurants stay DISTINCT
            # (e.g., NYU dining hall + coffee shop in same building)
            ptype = "SHARED_DISTINCT"
        else:
            ptype = "SHARED_BUILDING"

        stats[ptype] += 1

        # Building name for SHARED_BUILDING (initial; finalized in match pass 2)
        building_name = None
        if ptype == "SHARED_BUILDING":
            # Try parent name from Global Places (look up in authority table)
            parent_pks = grp["PARENT_PLACEKEY"].dropna().unique()
            for ppk in parent_pks:
                parent_row = auth[auth["PLACEKEY"] == ppk]
                if len(parent_row) > 0 and pd.notna(parent_row.iloc[0].get("final_name")):
                    building_name = parent_row.iloc[0]["final_name"]
                    break
            if building_name is None:
                # Fallback: most visited member
                best = max(members, key=lambda m: m["sg21_visits"])
                building_name = best["name"]

        rec = {
            "geometry": geom,
            "centroid_lat": centroid.y,
            "centroid_lon": centroid.x,
            "polygon_type": ptype,
            "n_pois": n_pois,
            "members": members,
            "building_name": building_name,
            "has_parent": has_parent,
        }
        polygons.append(rec)

    print(f"\n=== Polygon Classification ===")
    print(f"Total unique polygons: {len(polygons):,}")
    for ptype, cnt in stats.items():
        print(f"  {ptype:20s}: {cnt:,}")

    # ── Build STRtree spatial index ──────────────────────────────────────────
    print("\nBuilding STRtree spatial index...")
    geom_list = [p["geometry"] for p in polygons]
    tree = STRtree(geom_list)
    print(f"STRtree built with {len(geom_list):,} polygons")

    # ── Also build a coordinate-based fallback index for POIs without polygon
    auth_no_poly = auth[auth["POLYGON_WKT"].isna()].copy()
    print(f"\nPOIs without polygon (for coordinate fallback): {len(auth_no_poly):,}")
    fallback_pois = []
    for _, r in auth_no_poly.iterrows():
        if pd.notna(r.get("poi_lat")) and pd.notna(r.get("poi_lon")) and pd.notna(r.get("final_name")):
            fallback_pois.append({
                "PLACEKEY": r.get("PLACEKEY"),
                "name": r["final_name"],
                "lat": float(r["poi_lat"]),
                "lon": float(r["poi_lon"]),
                "temporal_status": r.get("temporal_status", "unknown"),
                "sub_category": r.get("SUB_CATEGORY", ""),
            })
    print(f"Valid fallback POIs: {len(fallback_pois):,}")

    # Build KDTree for fallback
    if fallback_pois:
        fb_coords = np.array([[p["lat"], p["lon"]] for p in fallback_pois])
        from scipy.spatial import cKDTree
        fb_tree = cKDTree(np.deg2rad(fb_coords))
    else:
        fb_tree = None

    # ── Save pickle ──────────────────────────────────────────────────────────
    index_data = {
        "polygons": polygons,
        "tree": tree,
        "geom_list": geom_list,
        "fallback_pois": fallback_pois,
        "fallback_tree": fb_tree,
        "stats": stats,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(index_data, f)
    pkl_size = os.path.getsize(OUT_PKL) / 1024 / 1024
    print(f"\nSaved {OUT_PKL} ({pkl_size:.1f} MB)")

    # ── Save human-readable CSV ──────────────────────────────────────────────
    rows = []
    for p in polygons:
        member_names = [m["name"] for m in p["members"]]
        rows.append({
            "polygon_type": p["polygon_type"],
            "n_pois": p["n_pois"],
            "centroid_lat": p["centroid_lat"],
            "centroid_lon": p["centroid_lon"],
            "building_name": p["building_name"],
            "has_parent": p["has_parent"],
            "member_names": " | ".join(member_names[:10]),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # ── Verification: Chelsea Market ─────────────────────────────────────────
    big = sorted(polygons, key=lambda p: p["n_pois"], reverse=True)[:5]
    print(f"\n=== Largest shared polygons (food courts) ===")
    for p in big:
        names = [m["name"] for m in p["members"][:5]]
        print(f"  {p['polygon_type']:20s} {p['n_pois']:3d} POIs  "
              f"bldg={p['building_name']!r:30s}  members={names}")

    # ── Verification: NYU area ───────────────────────────────────────────────
    print(f"\n=== NYU area polygons (40.693-40.695, -73.988 to -73.985) ===")
    for p in polygons:
        if 40.693 <= p["centroid_lat"] <= 40.695 and -73.988 <= p["centroid_lon"] <= -73.985:
            names = [m["name"] for m in p["members"]]
            print(f"  {p['polygon_type']:20s} {p['n_pois']} POIs  "
                  f"({p['centroid_lat']:.6f}, {p['centroid_lon']:.6f})  {names[:5]}")


if __name__ == "__main__":
    main()
