"""
Shared utilities for NYC Metro POI prediction.

Adapted from /scratch/sx2490/mobility/tokyo_poi_prediction/scripts/poi_prediction_utils.py.
Keeps train-time and inference-time candidate construction aligned.

Key functions:
  - haversine_m(lat1, lon1, lat2, lon2)
  - sample_candidates(pool, true_poi_id, k, rng, orig_lat, orig_lon, ...)
  - build_cluster_frequency_top5(candidates, true_poi_id)
"""

import csv
import math


# NYC is more compact than Tokyo, smaller lunch radii.
# Manhattan workers typically walk <1km for lunch.
PRIMARY_RADIUS_M = 1500    # walking distance preference
FALLBACK_RADIUS_M = 3000   # short transit/walk
HARD_CAP_RADIUS_M = 5000   # max plausible lunch trip

POLYGON_POLICIES = {"all", "exclude_shared_building", "point_members"}


def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_work_cbg_origin_map(path):
    """Load experiment-scoped workplace anchors keyed by work_cbg."""
    origin_map = {}
    if not path:
        return origin_map

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            work_cbg = str(row.get("work_cbg", "") or "").strip()
            if not work_cbg:
                continue
            try:
                origin_map[work_cbg] = (
                    float(row["origin_lat"]),
                    float(row["origin_lon"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    return origin_map


def format_poi_id(name, lat, lon):
    """Canonical POI id used throughout the NYC pipeline."""
    return f"{name}|{float(lat):.6f}|{float(lon):.6f}"


def load_poi_id_set(path):
    """Load a set of POI ids from a CSV with either `poi_id` or name/lat/lon columns."""
    poi_ids = set()
    if not path:
        return poi_ids
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("poi_id", "") or "").strip()
            if pid:
                poi_ids.add(pid)
                continue
            name = row.get("poi_name") or row.get("final_name") or row.get("name")
            lat = row.get("poi_lat") or row.get("lat")
            lon = row.get("poi_lon") or row.get("lon")
            if name and lat and lon:
                try:
                    poi_ids.add(format_poi_id(name, lat, lon))
                except (TypeError, ValueError):
                    continue
    return poi_ids


def load_poi_records(path, default_cluster_freq=0):
    """Load candidate POI records from a CSV for inference-time injection."""
    records = []
    if not path:
        return records
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                name = row.get("poi_name") or row.get("final_name") or row.get("name")
                lat = float(row.get("poi_lat") or row.get("lat"))
                lon = float(row.get("poi_lon") or row.get("lon"))
            except (TypeError, ValueError):
                continue
            if not name:
                continue
            pid = str(row.get("poi_id", "") or "").strip() or format_poi_id(name, lat, lon)
            try:
                cluster_freq = int(float(row.get("cluster_freq", default_cluster_freq)))
            except (TypeError, ValueError):
                cluster_freq = default_cluster_freq
            records.append({
                "poi_id": pid,
                "poi_name": name,
                "poi_lat": lat,
                "poi_lon": lon,
                "sub_category": row.get("sub_category") or row.get("SUB_CATEGORY") or "",
                "cluster_freq": cluster_freq,
                "polygon_type": row.get("polygon_type", ""),
                "synthetic_unseen": int(float(row.get("synthetic_unseen", 1))),
            })
    return records


def load_placekey_point_member_lookup(authority_path):
    """
    Load PLACEKEY → point-member identity from poi_name_authority.csv.

    Used for the `point_members` polygon sensitivity. For SHARED_BUILDING rows,
    matched stops keep the building aggregate name/centroid but retain the nearest
    member PLACEKEY; this lookup restores that member's point identity.
    """
    lookup = {}
    if not authority_path:
        return lookup
    with open(authority_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            placekey = str(row.get("PLACEKEY", "") or "").strip()
            name = row.get("final_name")
            lat = row.get("poi_lat")
            lon = row.get("poi_lon")
            if not (placekey and name and lat and lon):
                continue
            try:
                lookup[placekey] = {
                    "poi_name": name,
                    "poi_lat": float(lat),
                    "poi_lon": float(lon),
                    "sub_category": row.get("SUB_CATEGORY", ""),
                    "temporal_status": row.get("temporal_status", ""),
                }
            except (TypeError, ValueError):
                continue
    return lookup


def augment_pools_with_pois(pools, poi_records):
    """Append inference-only POIs to every cluster pool, deduplicating by poi_id."""
    if not poi_records:
        return pools
    augmented = {}
    for cid, pool in pools.items():
        rows = list(pool)
        seen = {str(row.get("poi_id")) for row in rows}
        for rec in poi_records:
            pid = str(rec.get("poi_id"))
            if pid in seen:
                continue
            rows.append(dict(rec))
            seen.add(pid)
        augmented[cid] = rows
    return augmented


def apply_poi_policy_to_frame(
    df,
    polygon_policy="all",
    point_member_lookup=None,
    exclude_poi_ids=None,
):
    """
    Apply polygon and hidden-POI policies to an a_work/matched-stop chunk.

    Policies:
      - all: keep the existing matched POI identity.
      - exclude_shared_building: remove rows matched to SHARED_BUILDING.
      - point_members: for SHARED_BUILDING rows, replace the building aggregate
        with the nearest member POI identity recovered from PLACEKEY.

    Hidden POIs are removed after polygon policy normalization so Exp03 can hide
    either building aggregates or restored point-member POIs consistently.
    """
    if polygon_policy not in POLYGON_POLICIES:
        raise ValueError(
            f"polygon_policy must be one of {sorted(POLYGON_POLICIES)}, got {polygon_policy!r}"
        )

    if polygon_policy == "exclude_shared_building" and "polygon_type" in df.columns:
        df = df[df["polygon_type"].astype(str) != "SHARED_BUILDING"].copy()
    elif polygon_policy == "point_members" and {"polygon_type", "PLACEKEY"}.issubset(df.columns):
        point_member_lookup = point_member_lookup or {}
        mask = df["polygon_type"].astype(str).eq("SHARED_BUILDING") & df["PLACEKEY"].astype(str).isin(point_member_lookup)
        if mask.any():
            keys = df.loc[mask, "PLACEKEY"].astype(str)
            df = df.copy()
            df.loc[mask, "poi_name"] = keys.map(lambda key: point_member_lookup[key]["poi_name"]).values
            df.loc[mask, "poi_lat"] = keys.map(lambda key: point_member_lookup[key]["poi_lat"]).values
            df.loc[mask, "poi_lon"] = keys.map(lambda key: point_member_lookup[key]["poi_lon"]).values
            if "sub_category" in df.columns:
                df.loc[mask, "sub_category"] = keys.map(lambda key: point_member_lookup[key]["sub_category"]).values
            if "temporal_status" in df.columns:
                df.loc[mask, "temporal_status"] = keys.map(lambda key: point_member_lookup[key]["temporal_status"]).values
            df.loc[mask, "polygon_type"] = "SHARED_BUILDING_MEMBER"

    exclude_poi_ids = exclude_poi_ids or set()
    if exclude_poi_ids and {"poi_name", "poi_lat", "poi_lon"}.issubset(df.columns):
        poi_ids = (
            df["poi_name"].astype(str)
            + "|"
            + df["poi_lat"].astype(float).map(lambda x: f"{x:.6f}")
            + "|"
            + df["poi_lon"].astype(float).map(lambda x: f"{x:.6f}")
        )
        df = df[~poi_ids.isin(exclude_poi_ids)].copy()

    return df


def resolve_origin_coords(stop_lat, stop_lon, work_cbg, origin_mode="stop", work_cbg_origin_map=None):
    """
    Resolve the origin used for candidate construction.

    Returns:
      (origin_lat, origin_lon, origin_source)
    """
    if origin_mode == "stop":
        return float(stop_lat), float(stop_lon), "stop"

    lookup = work_cbg_origin_map or {}
    key = str(work_cbg or "").strip()
    if key and key in lookup:
        lat, lon = lookup[key]
        return float(lat), float(lon), origin_mode

    return float(stop_lat), float(stop_lon), "stop_fallback"


def _weighted_sample_without_replacement(records, weights, k, rng):
    """Sample up to k records without replacement using weights."""
    pool = list(zip(records, weights))
    chosen = []

    while pool and len(chosen) < k:
        total = sum(max(weight, 0.0) for _, weight in pool)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            draw = rng.random() * total
            acc = 0.0
            idx = len(pool) - 1
            for i, (_, weight) in enumerate(pool):
                acc += max(weight, 0.0)
                if acc >= draw:
                    idx = i
                    break
        rec, _ = pool.pop(idx)
        chosen.append(rec)

    return chosen


def _candidate_weight(cluster_freq, dist_m, primary_radius_m, fallback_radius_m, hard_cap_radius_m,
                      freq_power=1.0):
    """
    Candidate weight for lunch destination choice.

    Cluster popularity is the primary driver, but far destinations are
    strongly down-weighted to keep candidates geographically realistic.
    freq_power exponentiates log1p(cluster_freq): 1.0=default, 0.5=softer,
    0.0=popularity-agnostic (pure distance decay).
    """
    pop = math.log1p(max(cluster_freq, 0))
    if freq_power != 1.0:
        pop = pop ** freq_power

    if dist_m > hard_cap_radius_m:
        pop *= 0.02
    elif dist_m > fallback_radius_m:
        pop *= 0.10
    elif dist_m > primary_radius_m:
        pop *= 0.35
    elif dist_m > primary_radius_m / 2:
        pop *= 0.70

    return max(pop, 1e-6)


def sample_candidates(
    pool,
    true_poi_id,
    k,
    rng,
    orig_lat,
    orig_lon,
    primary_radius_m=PRIMARY_RADIUS_M,
    fallback_radius_m=FALLBACK_RADIUS_M,
    hard_cap_radius_m=HARD_CAP_RADIUS_M,
    freq_power=1.0,
):
    """
    Sample geographically plausible candidate POIs.

    Strategy:
    1. Prefer POIs within a realistic lunch radius from the trip origin.
    2. Back off to a wider radius if the local pool is too small.
    3. Only fall back to the full cluster pool when needed.
    4. ALWAYS include the true POI when present.
    """
    if len(pool) <= k:
        result = list(pool)
        # Even in this case, ensure true POI is present
        if true_poi_id and not any(r.get("poi_id") == true_poi_id for r in result):
            for cand in pool:
                if cand.get("poi_id") == true_poi_id:
                    result.insert(0, cand)
                    break
        return result[:k] if len(result) > k else result

    annotated = []
    true_candidate = None
    for candidate in pool:
        dist_m = haversine_m(orig_lat, orig_lon, candidate["poi_lat"], candidate["poi_lon"])
        rec = dict(candidate)
        rec["_dist_m"] = dist_m
        annotated.append(rec)
        if candidate.get("poi_id") == true_poi_id:
            true_candidate = rec

    local_pool = [r for r in annotated if r["_dist_m"] <= primary_radius_m]
    near_pool = [r for r in annotated if r["_dist_m"] <= fallback_radius_m]
    capped_pool = [r for r in annotated if r["_dist_m"] <= hard_cap_radius_m]

    min_viable = max(3, min(k, 10))
    if len(local_pool) >= min_viable:
        primary_pool = local_pool
    elif len(near_pool) >= min_viable:
        primary_pool = near_pool
    elif len(capped_pool) >= min_viable:
        primary_pool = capped_pool
    else:
        primary_pool = annotated

    if len(capped_pool) >= min_viable:
        backup_pool = capped_pool
    elif len(near_pool) >= min_viable:
        backup_pool = near_pool
    else:
        backup_pool = annotated

    chosen = []
    chosen_ids = set()
    if true_candidate is not None:
        chosen.append(true_candidate)
        chosen_ids.add(true_candidate["poi_id"])

    def extend_from(source_pool):
        remaining = k - len(chosen)
        if remaining <= 0:
            return
        available = [r for r in source_pool if r["poi_id"] not in chosen_ids]
        if not available:
            return
        weights = [
            _candidate_weight(
                r.get("cluster_freq", 0),
                r.get("_dist_m", hard_cap_radius_m + 1),
                primary_radius_m,
                fallback_radius_m,
                hard_cap_radius_m,
                freq_power=freq_power,
            )
            for r in available
        ]
        for rec in _weighted_sample_without_replacement(available, weights, remaining, rng):
            chosen.append(rec)
            chosen_ids.add(rec["poi_id"])

    extend_from(primary_pool)
    extend_from(backup_pool)
    extend_from(annotated)

    return [
        {key: value for key, value in rec.items() if key != "_dist_m"}
        for rec in chosen[:k]
    ]


def build_cluster_frequency_top5(candidates, true_poi_id, p_true=0.6):
    """
    Coherent listwise top-5 target for candidate-choice supervised fine-tuning.

    Target design (replaces the legacy "cluster_freq-sorted with true injected"
    label, which produced incompatible `next_choice` and `top_5[0]` supervision
    — empirically they disagreed 69% of the time on exp02_wcbg OOS 2022):

      Position 0: true POI, probability = p_true (default 0.6)
      Positions 1-4: four non-truth candidates with highest cluster_freq,
                     probability = (1 - p_true) * cluster_freq_share

    Guarantees `top_5[0].poi == true_poi_id == next_choice`, so LoRA learns
    one consistent ranking rather than two contradictory targets.

    Returns:
      cluster_top_choice: most frequent candidate by cluster_freq (informational)
      top5: list of {"poi": id, "prob": float}, sums to 1.0
    """
    truth_cand = next((c for c in candidates if c.get("poi_id") == true_poi_id), None)
    others = [c for c in candidates if c.get("poi_id") != true_poi_id]
    others.sort(key=lambda r: r.get("cluster_freq", 0), reverse=True)
    top_4_others = others[:4]

    top5_ids = [true_poi_id] + [c["poi_id"] for c in top_4_others]
    # Pad if fewer than 5 candidates overall (should not happen with k=30)
    while len(top5_ids) < 5:
        top5_ids.append(true_poi_id)

    cf_values = [float(c.get("cluster_freq", 0) or 0) for c in top_4_others]
    cf_sum = sum(cf_values)
    if cf_sum > 0:
        rest_probs = [(1.0 - p_true) * v / cf_sum for v in cf_values]
    else:
        rest_probs = [(1.0 - p_true) / 4.0] * len(top_4_others)
    probs = [p_true] + rest_probs
    while len(probs) < 5:
        probs.append(0.0)

    total = sum(probs) or 1.0
    probs = [p / total for p in probs]

    top5 = [{"poi": pid, "prob": round(p, 5)} for pid, p in zip(top5_ids, probs)]
    ranked = sorted(candidates, key=lambda r: r.get("cluster_freq", 0), reverse=True)
    cluster_top_choice = ranked[0]["poi_id"] if ranked else true_poi_id
    return cluster_top_choice, top5
