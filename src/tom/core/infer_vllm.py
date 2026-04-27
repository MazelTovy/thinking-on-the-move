#!/usr/bin/env python3
"""
12_infer_vllm.py — vLLM inference for NYC Metro POI prediction.

Current protocol:
  - Sample visits uniformly with reservoir sampling
  - 2021 in-sample evaluation uses the temporal validation window only
  - Formal aggregate metrics are computed from top-5 probability mass
  - A separate Tokyo-style scatter figure can still use top-1 visit mass
  - Candidate construction is aligned with training via the same snapshot-scoped
    cluster pools, fallback POI universe, radii, and origin-mode logic
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import pandas as pd

from tom.utils import (
    FALLBACK_RADIUS_M,
    HARD_CAP_RADIUS_M,
    POLYGON_POLICIES,
    PRIMARY_RADIUS_M,
    apply_poi_policy_to_frame,
    augment_pools_with_pois,
    load_work_cbg_origin_map,
    load_placekey_point_member_lookup,
    load_poi_id_set,
    load_poi_records,
    resolve_origin_coords,
    haversine_m,
    sample_candidates,
)

BASE = "/scratch/sx2490/econai/nyc_metro"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"

K_CANDIDATES = 30
SYSTEM_PROMPT = (
    "You are predicting lunch-time restaurant choices for a worker in New York City. "
    "The worker chooses exactly one restaurant per lunch trip. "
    "Predict which restaurant the worker will visit based on their persona profile "
    "and the list of candidate restaurants."
)


def format_poi_id(name, lat, lon):
    return f"{name}|{float(lat):.6f}|{float(lon):.6f}"


def enforce_hidden_decoys(candidates, hidden_pool, hidden_set, true_id, n,
                          orig_lat, orig_lon, max_radius_m, rng):
    """Force-include `n` POIs from hidden_pool in the candidate set.

    Without this, the only hidden POI in any record is the (force-included) true
    one, making `cluster_freq=0` a deterministic leak label for downstream rankers.
    With m hidden decoys per record, `cluster_freq=0` is no longer label-aligned.

    Decoys are sampled from the closest-by-distance subset (shuffled) so they
    don't all cluster at the trip origin and don't leave 'true is the closest
    hidden' as a residual signal.
    """
    in_hidden = sum(1 for c in candidates if c["poi_id"] in hidden_set)
    needed = n - in_hidden
    if needed <= 0:
        return candidates

    candidate_ids = {c["poi_id"] for c in candidates}
    avail = []
    for r in hidden_pool:
        if r["poi_id"] in candidate_ids or r["poi_id"] == true_id:
            continue
        d = haversine_m(orig_lat, orig_lon, r["poi_lat"], r["poi_lon"])
        if d <= max_radius_m:
            r2 = dict(r); r2["_dist_m"] = d
            avail.append(r2)
    if not avail:
        return candidates

    avail.sort(key=lambda r: r["_dist_m"])
    pool_for_sampling = avail[: max(needed * 4, 8)]
    pick_n = min(needed, len(pool_for_sampling))
    picks = rng.sample(range(len(pool_for_sampling)), pick_n)
    decoys = [pool_for_sampling[i] for i in picks]
    for d in decoys:
        d.pop("_dist_m", None)

    replaceable = [
        i for i, c in enumerate(candidates)
        if c["poi_id"] not in hidden_set and c["poi_id"] != true_id
    ]
    replaceable.sort(key=lambda i: candidates[i].get("cluster_freq", 0))
    n_replace = min(len(decoys), len(replaceable))
    for i in range(n_replace):
        candidates[replaceable[i]] = decoys[i]
    return candidates


def extract_json(text):
    """Extract the first JSON object from model text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def normalize_poi_text(text):
    """Normalize punctuation variants that models often emit differently."""
    return (
        str(text)
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .strip()
        .lower()
    )


def snap_poi_id(pid, valid_set, name_to_valid, tol=5e-4):
    """Map model output -> canonical POI id."""
    if not isinstance(pid, str):
        return None
    pid = pid.strip()
    if pid in valid_set:
        return pid
    pid_norm = normalize_poi_text(pid)
    if pid_norm in valid_set:
        return pid_norm

    parts = pid.split("|")
    if len(parts) < 3:
        return None
    try:
        lat = float(parts[-2])
        lon = float(parts[-1])
    except ValueError:
        return None

    name_key = normalize_poi_text("|".join(parts[:-2]))
    for vpid, vlat, vlon in name_to_valid.get(name_key, []):
        if abs(lat - vlat) <= tol and abs(lon - vlon) <= tol:
            return vpid

    short = name_key.split("|")[0].strip()
    if short != name_key:
        for vpid, vlat, vlon in name_to_valid.get(short, []):
            if abs(lat - vlat) <= tol and abs(lon - vlon) <= tol:
                return vpid
    return None


def load_cluster_pools(cbg_poi_dir):
    """Load per-cluster pools from the experiment snapshot."""
    pool_re = re.compile(r"^cluster_(\d+)_pois\.csv$")
    pools = {}
    for fname in sorted(os.listdir(cbg_poi_dir)):
        m = pool_re.match(fname)
        if not m:
            continue
        cid = m.group(1)
        rows = []
        with open(f"{cbg_poi_dir}/{fname}", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "poi_id": row["poi_id"],
                    "poi_name": row["poi_name"],
                    "poi_lat": float(row["poi_lat"]),
                    "poi_lon": float(row["poi_lon"]),
                    "polygon_type": row.get("polygon_type", ""),
                    "sub_category": row.get("sub_category", ""),
                    "cluster_freq": int(row.get("cluster_freq", 0)),
                    "synthetic_unseen": int(float(row.get("synthetic_unseen", 0) or 0)),
                })
        pools[cid] = rows
    return pools


def build_valid_poi_set(pools, fallback_path):
    """Build canonical POI set from cluster pools + snapshot fallback universe."""
    valid = set()
    name_to_valid = {}

    for rows in pools.values():
        for rec in rows:
            valid.add(rec["poi_id"])
            key = normalize_poi_text(rec["poi_name"])
            name_to_valid.setdefault(key, []).append(
                (rec["poi_id"], rec["poi_lat"], rec["poi_lon"])
            )

    if fallback_path and os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("poi_id")
                if not pid:
                    continue
                valid.add(pid)
                try:
                    lat = float(row["poi_lat"])
                    lon = float(row["poi_lon"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = normalize_poi_text(row.get("poi_name", ""))
                if key:
                    name_to_valid.setdefault(key, []).append((pid, lat, lon))

    print(f"  Valid POI set: {len(valid):,} ids, {len(name_to_valid):,} unique names")
    return valid, name_to_valid


def load_personas_grouped(path):
    by_cluster = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_cluster[str(row["cluster"])].append(row["persona"])
    return by_cluster


def build_candidate_table(candidates, orig_lat, orig_lon):
    """Build candidate table sorted by distance."""
    lines = []
    for candidate in candidates:
        dist = haversine_m(orig_lat, orig_lon, candidate["poi_lat"], candidate["poi_lon"])
        cat = str(candidate.get("sub_category", "") or "")
        parts = [f"- {candidate['poi_name']}", f"id={candidate['poi_id']}"]
        if cat:
            parts.append(f"category={cat}")
        parts.append(f"dist={int(dist)}m")
        lines.append((dist, " | ".join(parts)))
    lines.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in lines)


def build_candidate_snap_index(candidates):
    """Build a per-prompt candidate id index for strict output validation."""
    valid = set()
    by_name = {}
    for candidate in candidates:
        pid = str(candidate["poi_id"])
        valid.add(pid)
        key = normalize_poi_text(candidate.get("poi_name", ""))
        try:
            lat = float(candidate["poi_lat"])
            lon = float(candidate["poi_lon"])
        except (KeyError, TypeError, ValueError):
            continue
        by_name.setdefault(key, []).append((pid, lat, lon))
    return valid, by_name


def write_inference_records(records, path):
    """Persist CPU-built inference records so GPU jobs skip CSV sampling."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            row = dict(record)
            candidate_valid_set = row.get("candidate_valid_set")
            if isinstance(candidate_valid_set, set):
                row["candidate_valid_set"] = sorted(candidate_valid_set)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_inference_records(path):
    """Load prebuilt records and restore candidate-set validation structures."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            candidate_valid_set = row.get("candidate_valid_set")
            if isinstance(candidate_valid_set, list):
                row["candidate_valid_set"] = set(candidate_valid_set)
            records.append(row)
    return records


def sample_visit_rows(
    a_work_path,
    valid_clusters,
    year,
    val_date_from,
    max_samples,
    sampling_unit,
    seed,
    polygon_policy="all",
    point_member_lookup=None,
    exclude_truth_poi_ids=None,
):
    """Uniformly sample eligible visit rows (or workers) from the source CSV."""
    rng = random.Random(seed)
    sampled = []
    seen_units = set()
    n_seen_units = 0
    n_total_rows = 0
    n_dropped_missing = 0
    n_dropped_time = 0
    n_dropped_policy = 0
    n_dropped_cluster = 0

    for chunk in pd.read_csv(
        a_work_path,
        chunksize=200_000,
        dtype={"cuebiq_id": str, "work_cbg": str, "home_cbg": str},
        usecols=[
            "cuebiq_id", "demo_cluster", "lat", "lng", "poi_name", "poi_lat", "poi_lon",
            "stop_ts", "admin2_id", "work_cbg", "home_cbg",
            "PLACEKEY", "polygon_type", "sub_category", "temporal_status",
        ],
    ):
        before_missing = len(chunk)
        n_total_rows += before_missing
        chunk = chunk.dropna(subset=["cuebiq_id", "demo_cluster", "poi_name", "poi_lat", "poi_lon",
                                     "lat", "lng", "stop_ts"])
        n_dropped_missing += before_missing - len(chunk)
        if len(chunk) == 0:
            continue

        chunk["demo_cluster"] = chunk["demo_cluster"].astype(int).astype(str)
        chunk["stop_ts"] = chunk["stop_ts"].astype(str)
        if year == 2021 and val_date_from:
            before = len(chunk)
            chunk = chunk[chunk["stop_ts"].str[:10] >= val_date_from]
            n_dropped_time += before - len(chunk)
            if len(chunk) == 0:
                continue
        if year == 2022:
            # Strict temporal-OOS: only keep 2022 rows. The 2022 a_work file
            # sometimes contains late-2021 rows from the reservoir.
            before = len(chunk)
            chunk = chunk[chunk["stop_ts"].str[:4] == "2022"]
            n_dropped_time += before - len(chunk)
            if len(chunk) == 0:
                continue

        before_policy = len(chunk)
        chunk = apply_poi_policy_to_frame(
            chunk,
            polygon_policy=polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_poi_ids=exclude_truth_poi_ids,
        )
        n_dropped_policy += before_policy - len(chunk)
        if len(chunk) == 0:
            continue

        before = len(chunk)
        chunk = chunk[chunk["demo_cluster"].isin(valid_clusters)]
        n_dropped_cluster += before - len(chunk)
        if len(chunk) == 0:
            continue

        for row in chunk.itertuples(index=False):
            unit_id = str(row.cuebiq_id) if sampling_unit == "worker" else None
            if sampling_unit == "worker":
                if unit_id in seen_units:
                    continue
                seen_units.add(unit_id)

            rec = {
                "cuebiq_id": str(row.cuebiq_id),
                "cluster": str(row.demo_cluster),
                "stop_ts": str(row.stop_ts),
                "admin2_id": "" if pd.isna(row.admin2_id) else str(row.admin2_id),
                "work_cbg": "" if pd.isna(row.work_cbg) else str(row.work_cbg),
                "lat": float(row.lat),
                "lng": float(row.lng),
                "poi_name": row.poi_name,
                "poi_lat": float(row.poi_lat),
                "poi_lon": float(row.poi_lon),
                "polygon_type": "" if pd.isna(row.polygon_type) else str(row.polygon_type),
            }

            n_seen_units += 1
            if max_samples <= 0:
                sampled.append(rec)
                continue

            if len(sampled) < max_samples:
                sampled.append(rec)
                continue

            j = rng.randint(1, n_seen_units)
            if j <= max_samples:
                sampled[j - 1] = rec

    print(f"  Source rows: {n_total_rows:,}")
    print(f"  Eligible {sampling_unit}s seen: {n_seen_units:,}")
    print(f"  Sampled rows kept: {len(sampled):,}")
    print(f"  Dropped: missing={n_dropped_missing:,}, time={n_dropped_time:,}, "
          f"policy={n_dropped_policy:,}, cluster={n_dropped_cluster:,}")
    return sampled


def build_inference_records(
    sampled_rows,
    pools,
    personas_by_cluster,
    k,
    rng,
    origin_mode,
    work_cbg_origin_map,
    primary_radius_m,
    fallback_radius_m,
    hard_cap_radius_m,
    freq_power=1.0,
    hidden_pool=None,
    hidden_id_set=None,
    n_hidden_decoys=0,
):
    """Turn sampled rows into inference prompts."""
    records = []
    n_skipped = 0
    n_origin_fallback = 0

    for row in sampled_rows:
        cid = row["cluster"]
        pool = pools.get(cid)
        if not pool or len(pool) < 5:
            n_skipped += 1
            continue

        cluster_personas = personas_by_cluster.get(cid)
        if not cluster_personas:
            n_skipped += 1
            continue

        orig_lat, orig_lon, origin_source = resolve_origin_coords(
            row["lat"],
            row["lng"],
            row.get("work_cbg"),
            origin_mode=origin_mode,
            work_cbg_origin_map=work_cbg_origin_map,
        )
        if origin_source != origin_mode:
            n_origin_fallback += 1

        true_id = format_poi_id(row["poi_name"], row["poi_lat"], row["poi_lon"])
        candidates = sample_candidates(
            pool,
            true_id,
            k,
            rng,
            orig_lat=orig_lat,
            orig_lon=orig_lon,
            primary_radius_m=primary_radius_m,
            fallback_radius_m=fallback_radius_m,
            hard_cap_radius_m=hard_cap_radius_m,
            freq_power=freq_power,
        )
        if hidden_id_set and n_hidden_decoys > 0:
            candidates = enforce_hidden_decoys(
                candidates, hidden_pool, hidden_id_set, true_id,
                n_hidden_decoys, orig_lat, orig_lon,
                hard_cap_radius_m, rng,
            )
        if len(candidates) < 5:
            n_skipped += 1
            continue

        cand_table = build_candidate_table(candidates, orig_lat, orig_lon)
        candidate_valid_set, candidate_name_to_valid = build_candidate_snap_index(candidates)
        candidate_records = []
        for candidate in candidates:
            candidate_records.append({
                "poi_id": candidate["poi_id"],
                "poi_name": candidate["poi_name"],
                "poi_lat": float(candidate["poi_lat"]),
                "poi_lon": float(candidate["poi_lon"]),
                "sub_category": candidate.get("sub_category", ""),
                "cluster_freq": int(float(candidate.get("cluster_freq", 0) or 0)),
                "polygon_type": candidate.get("polygon_type", ""),
                "synthetic_unseen": int(float(candidate.get("synthetic_unseen", 0) or 0)),
                "dist_m": float(haversine_m(orig_lat, orig_lon, candidate["poi_lat"], candidate["poi_lon"])),
            })
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

        records.append({
            "cuebiq_id": row["cuebiq_id"],
            "cluster": cid,
            "admin2_id": row.get("admin2_id"),
            "stop_ts": row["stop_ts"],
            "work_cbg": row.get("work_cbg", ""),
            "true_poi_id": true_id,
            "origin_lat": orig_lat,
            "origin_lon": orig_lon,
            "origin_mode": origin_mode,
            "origin_source": origin_source,
            "true_polygon_type": row.get("polygon_type", ""),
            "candidates": candidate_records,
            "system": SYSTEM_PROMPT,
            "user": user_msg,
            "candidate_valid_set": candidate_valid_set,
            "candidate_name_to_valid": candidate_name_to_valid,
        })

    print(f"  Inference prompts: {len(records):,} (skipped {n_skipped:,}, "
          f"origin fallback {n_origin_fallback:,})")
    return records


# ---- Parallel inference-prep (fork-based COW; writes directly to tmp jsonl) ----
_PINF_POOLS = None
_PINF_PERSONAS = None
_PINF_WORK_CBG = None
_PINF_K = None
_PINF_ORIGIN_MODE = None
_PINF_PRIMARY = None
_PINF_FALLBACK = None
_PINF_HARD_CAP = None
_PINF_FREQ_POWER = None
_PINF_HIDDEN_POOL = None
_PINF_HIDDEN_IDS = None
_PINF_N_DECOYS = None


def _serialize_inference_record(rec):
    """Mirror of write_inference_records' per-record transform (set→sorted list)."""
    row = dict(rec)
    cvs = row.get("candidate_valid_set")
    if isinstance(cvs, set):
        row["candidate_valid_set"] = sorted(cvs)
    return json.dumps(row, ensure_ascii=False)


def _process_inference_chunk(task):
    """Worker: process a chunk of rows and write directly to a tmp jsonl."""
    chunk_idx, rows, tmp_path, seed = task
    rng = random.Random(seed)
    written = 0
    n_skipped = 0
    n_origin_fallback = 0
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for row in rows:
            cid = row["cluster"]
            pool = _PINF_POOLS.get(cid)
            if not pool or len(pool) < 5:
                n_skipped += 1
                continue
            cluster_personas = _PINF_PERSONAS.get(cid)
            if not cluster_personas:
                n_skipped += 1
                continue

            orig_lat, orig_lon, origin_source = resolve_origin_coords(
                row["lat"], row["lng"], row.get("work_cbg"),
                origin_mode=_PINF_ORIGIN_MODE,
                work_cbg_origin_map=_PINF_WORK_CBG,
            )
            if origin_source != _PINF_ORIGIN_MODE:
                n_origin_fallback += 1

            true_id = format_poi_id(row["poi_name"], row["poi_lat"], row["poi_lon"])
            candidates = sample_candidates(
                pool, true_id, _PINF_K, rng,
                orig_lat=orig_lat, orig_lon=orig_lon,
                primary_radius_m=_PINF_PRIMARY,
                fallback_radius_m=_PINF_FALLBACK,
                hard_cap_radius_m=_PINF_HARD_CAP,
                freq_power=_PINF_FREQ_POWER,
            )
            if _PINF_HIDDEN_IDS and _PINF_N_DECOYS > 0:
                candidates = enforce_hidden_decoys(
                    candidates, _PINF_HIDDEN_POOL, _PINF_HIDDEN_IDS, true_id,
                    _PINF_N_DECOYS, orig_lat, orig_lon,
                    _PINF_HARD_CAP, rng,
                )
            if len(candidates) < 5:
                n_skipped += 1
                continue

            cand_table = build_candidate_table(candidates, orig_lat, orig_lon)
            candidate_valid_set, candidate_name_to_valid = build_candidate_snap_index(candidates)
            candidate_records = []
            for candidate in candidates:
                candidate_records.append({
                    "poi_id": candidate["poi_id"],
                    "poi_name": candidate["poi_name"],
                    "poi_lat": float(candidate["poi_lat"]),
                    "poi_lon": float(candidate["poi_lon"]),
                    "sub_category": candidate.get("sub_category", ""),
                    "cluster_freq": int(float(candidate.get("cluster_freq", 0) or 0)),
                    "polygon_type": candidate.get("polygon_type", ""),
                    "synthetic_unseen": int(float(candidate.get("synthetic_unseen", 0) or 0)),
                    "dist_m": float(haversine_m(orig_lat, orig_lon, candidate["poi_lat"], candidate["poi_lon"])),
                })
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
            rec = {
                "cuebiq_id": row["cuebiq_id"],
                "cluster": cid,
                "admin2_id": row.get("admin2_id"),
                "stop_ts": row["stop_ts"],
                "work_cbg": row.get("work_cbg", ""),
                "true_poi_id": true_id,
                "origin_lat": orig_lat,
                "origin_lon": orig_lon,
                "origin_mode": _PINF_ORIGIN_MODE,
                "origin_source": origin_source,
                "true_polygon_type": row.get("polygon_type", ""),
                "candidates": candidate_records,
                "system": SYSTEM_PROMPT,
                "user": user_msg,
                "candidate_valid_set": candidate_valid_set,
                "candidate_name_to_valid": candidate_name_to_valid,
            }
            fout.write(_serialize_inference_record(rec) + "\n")
            written += 1
    return chunk_idx, written, n_skipped, n_origin_fallback


def build_and_write_inference_records_parallel(
    sampled_rows, out_path, pools, personas_by_cluster, k, base_seed,
    origin_mode, work_cbg_origin_map,
    primary_radius_m, fallback_radius_m, hard_cap_radius_m,
    freq_power=1.0, hidden_pool=None, hidden_id_set=None, n_hidden_decoys=0,
    n_workers=1,
):
    """Parallel inference-prep: each worker writes its chunk to a tmp jsonl,
    main process concatenates to out_path. Globals inherited via fork (COW)."""
    import multiprocessing as mp
    import tempfile
    import shutil
    import time

    global _PINF_POOLS, _PINF_PERSONAS, _PINF_WORK_CBG, _PINF_K
    global _PINF_ORIGIN_MODE, _PINF_PRIMARY, _PINF_FALLBACK, _PINF_HARD_CAP
    global _PINF_FREQ_POWER, _PINF_HIDDEN_POOL, _PINF_HIDDEN_IDS, _PINF_N_DECOYS
    _PINF_POOLS = pools
    _PINF_PERSONAS = personas_by_cluster
    _PINF_WORK_CBG = work_cbg_origin_map
    _PINF_K = k
    _PINF_ORIGIN_MODE = origin_mode
    _PINF_PRIMARY = primary_radius_m
    _PINF_FALLBACK = fallback_radius_m
    _PINF_HARD_CAP = hard_cap_radius_m
    _PINF_FREQ_POWER = freq_power
    _PINF_HIDDEN_POOL = hidden_pool
    _PINF_HIDDEN_IDS = hidden_id_set
    _PINF_N_DECOYS = n_hidden_decoys

    n = len(sampled_rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if n == 0:
        with open(out_path, "w") as f:
            pass
        print("  Inference prompts: 0 written")
        return 0, 0, 0

    chunk_size = (n + n_workers - 1) // n_workers
    chunks = [sampled_rows[i:i + chunk_size] for i in range(0, n, chunk_size)]
    tmp_dir = tempfile.mkdtemp(prefix="infprep_", dir=os.path.dirname(out_path) or ".")
    tasks = [
        (i, chunks[i], os.path.join(tmp_dir, f"part_{i:03d}.jsonl"), base_seed + i * 1009 + 1)
        for i in range(len(chunks))
    ]
    print(f"  Building {n:,} inference prompts in {len(chunks)} chunks "
          f"(~{chunk_size:,} each), {n_workers} workers")

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool_proc:
        results = pool_proc.map(_process_inference_chunk, tasks)
    results.sort(key=lambda r: r[0])

    total_written = sum(r[1] for r in results)
    total_skipped = sum(r[2] for r in results)
    total_origin_fb = sum(r[3] for r in results)
    elapsed = time.time() - t0
    rate = total_written / max(elapsed, 1e-6)
    print(f"  Workers done in {elapsed:.0f}s ({rate:.0f} rec/s; "
          f"{total_written:,} written, {total_skipped:,} skipped, "
          f"{total_origin_fb:,} origin fallback)")

    print(f"  Concatenating {len(chunks)} parts → {out_path}")
    with open(out_path, "wb") as fout:
        for i in range(len(chunks)):
            part = os.path.join(tmp_dir, f"part_{i:03d}.jsonl")
            with open(part, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return total_written, total_skipped, total_origin_fb


def run_inference(
    records,
    model_path,
    lora_path,
    out_path,
    valid_set,
    name_to_valid,
    temperature=0.2,
    top_p=0.9,
    max_tokens=768,
    max_model_len=4096,
    chunk_size=256,
    gpu_mem=0.90,
    min_parse_rate=0.95,
):
    """Run vLLM inference and write predictions to JSONL."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\nLoading vLLM model: {model_path}")
    print(f"  LoRA: {lora_path}")

    llm = LLM(
        model=model_path,
        enable_lora=True,
        max_lora_rank=64,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    lora_req = LoRARequest("nyc_metro", 1, lora_path=lora_path)
    sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    tokenizer = llm.get_tokenizer()
    prompts = []
    kept_records = []
    prompt_lens = []
    n_context_skipped = 0
    max_prompt_tokens = max_model_len - max_tokens
    for record in records:
        messages = [
            {"role": "system", "content": record["system"]},
            {"role": "user", "content": record["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len > max_prompt_tokens:
            n_context_skipped += 1
            continue
        prompts.append(prompt)
        kept_records.append(record)
        prompt_lens.append(prompt_len)

    if not prompts:
        print(
            "ERROR: all inference prompts exceed context window "
            f"(max_model_len={max_model_len}, max_tokens={max_tokens})"
        )
        sys.exit(1)

    prompt_lens_sorted = sorted(prompt_lens)
    p95_idx = min(len(prompt_lens_sorted) - 1, int(0.95 * len(prompt_lens_sorted)))
    print(
        f"  Prompt tokens: min={prompt_lens_sorted[0]}, "
        f"median={prompt_lens_sorted[len(prompt_lens_sorted) // 2]}, "
        f"p95={prompt_lens_sorted[p95_idx]}, max={prompt_lens_sorted[-1]}, "
        f"headroom={max_model_len - prompt_lens_sorted[-1]}"
    )
    if n_context_skipped:
        print(
            f"  Context-skipped prompts: {n_context_skipped:,} / {len(records):,} "
            f"(max_model_len={max_model_len}, max_tokens={max_tokens})"
        )
    records = kept_records

    print(f"  {len(prompts):,} prompts, chunk_size={chunk_size}")

    outputs_all = []
    for i in range(0, len(prompts), chunk_size):
        prompt_chunk = prompts[i:i + chunk_size]
        t0 = time.time()
        outputs = llm.generate(prompt_chunk, sampling, lora_request=lora_req)
        elapsed = max(time.time() - t0, 1e-6)
        outputs_all.extend(outputs)
        print(f"  Chunk {i // chunk_size}: {len(prompt_chunk)} prompts "
              f"({elapsed:.0f}s, {len(prompt_chunk) / elapsed:.0f} prompts/s)")

    print(f"\nParsing {len(outputs_all):,} outputs...")
    n_parsed = 0
    n_failed = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, output in enumerate(outputs_all):
            text = output.outputs[0].text
            parsed = extract_json(text)
            rec = records[idx]
            candidate_valid_set = rec.get("candidate_valid_set") or valid_set
            candidate_name_to_valid = rec.get("candidate_name_to_valid") or name_to_valid

            result = {
                "index": idx,
                "cuebiq_id": rec["cuebiq_id"],
                "cluster": rec["cluster"],
                "admin2_id": rec.get("admin2_id"),
                "stop_ts": rec["stop_ts"],
                "work_cbg": rec.get("work_cbg", ""),
                "true_poi_id": rec["true_poi_id"],
                "true_polygon_type": rec.get("true_polygon_type", ""),
                "candidates": rec.get("candidates", []),
                "origin": {
                    "lat": rec["origin_lat"],
                    "lon": rec["origin_lon"],
                    "mode": rec["origin_mode"],
                    "source": rec["origin_source"],
                },
                "finish_reason": getattr(output.outputs[0], "finish_reason", None),
                "raw_text_len": len(text),
                "raw_text_truncated": len(text) > 2000,
                "raw_text": text[:2000],
            }

            if parsed and isinstance(parsed.get("top_5"), list):
                top5 = []
                for item in parsed["top_5"]:
                    if not isinstance(item, dict):
                        continue
                    raw_pid = str(item.get("poi", ""))
                    snapped = snap_poi_id(
                        raw_pid, candidate_valid_set, candidate_name_to_valid
                    )
                    if not snapped:
                        continue
                    try:
                        prob = max(0.0, float(item.get("prob", 0)))
                    except (TypeError, ValueError):
                        continue
                    top5.append({"poi": snapped, "prob": prob})

                prob_sum = sum(item["prob"] for item in top5)
                if prob_sum > 0:
                    for item in top5:
                        item["prob"] = round(item["prob"] / prob_sum, 4)

                next_choice = None
                next_raw = parsed.get("next_choice")
                if next_raw:
                    next_choice = snap_poi_id(
                        str(next_raw), candidate_valid_set, candidate_name_to_valid
                    )
                if not next_choice and top5:
                    next_choice = top5[0]["poi"]

                result["next_choice"] = next_choice
                result["top_5"] = top5[:5]
                result["parse_ok"] = len(top5) >= 5
                if result["parse_ok"]:
                    n_parsed += 1
                else:
                    n_failed += 1
            else:
                result["next_choice"] = None
                result["top_5"] = []
                result["parse_ok"] = False
                n_failed += 1

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  Parsed: {n_parsed:,}, Failed: {n_failed:,}")
    print(f"  Saved to {out_path}")
    parse_rate = n_parsed / len(outputs_all) if outputs_all else 0.0
    print(f"  Parse rate: {parse_rate:.2%}")
    if parse_rate < min_parse_rate:
        print(
            f"ERROR: parse rate {parse_rate:.2%} is below "
            f"--min_parse_rate {min_parse_rate:.2%}"
        )
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2021, 2022], default=2021)
    parser.add_argument("--a_work", default="")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", default=f"{BASE}/experiments/exp02/lora/final")
    parser.add_argument("--out", default="")
    parser.add_argument("--cbg_poi_dir", default=f"{BASE}/cbg_poi")
    parser.add_argument("--personas", default=f"{BASE}/data/personas_nyc.jsonl")
    parser.add_argument("--fallback_poi_csv", default=f"{BASE}/data/poi_unique_food_with_freq.csv")
    parser.add_argument("--work_cbg_origin_csv", default="")
    parser.add_argument("--records_in", default="")
    parser.add_argument("--records_out", default="")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--origin_mode", choices=["stop", "work_cbg_centroid"], default="stop")
    parser.add_argument("--polygon_policy", choices=sorted(POLYGON_POLICIES), default="all")
    parser.add_argument("--point_member_authority_csv", default=AUTH_PATH)
    parser.add_argument("--exclude_truth_poi_csv", default="",
                        help="Optional CSV of true POIs to exclude from inference/eval sampling.")
    parser.add_argument("--n_hidden_decoys", type=int, default=0,
                        help="Force-include this many POIs from --candidate_inject_poi_csv "
                             "in every candidate set (replacing lowest-cluster_freq non-truth "
                             "candidates). Breaks the single-injection leak where the unique "
                             "cluster_freq=0 candidate trivially identifies the truth.")
    parser.add_argument("--candidate_inject_poi_csv", default="",
                        help="Optional CSV of POIs to append to every candidate pool at inference time.")
    parser.add_argument("--sampling_unit", choices=["visit", "worker"], default="visit")
    parser.add_argument("--val_date_from", default="2021-11-01")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--k", type=int, default=K_CANDIDATES)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--min_parse_rate", type=float, default=0.95)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--gpu_mem", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary_radius_m", type=float, default=PRIMARY_RADIUS_M)
    parser.add_argument("--fallback_radius_m", type=float, default=FALLBACK_RADIUS_M)
    parser.add_argument("--hard_cap_radius_m", type=float, default=HARD_CAP_RADIUS_M)
    parser.add_argument("--freq_power", type=float, default=1.0,
                        help="Popularity exponent on log1p(cluster_freq) in candidate sampler.")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel workers for --prepare_only record building "
                             "(fork-based, COW-shared globals). 1=serial (default).")
    args = parser.parse_args()

    if args.prepare_only and args.records_in:
        print("ERROR: --prepare_only cannot be combined with --records_in")
        sys.exit(1)
    if args.prepare_only and not args.records_out:
        print("ERROR: --prepare_only requires --records_out")
        sys.exit(1)

    if not args.a_work:
        args.a_work = f"{BASE}/data/a_work_{args.year}.csv"
    if not args.out:
        label = "insample" if args.year == 2021 else "oos"
        args.out = f"{BASE}/pred_{label}_{args.year}.jsonl"

    print(f"=== vLLM Inference: {args.year} ===")
    print(f"  Sampling unit: {args.sampling_unit}")
    print(f"  Max samples:   {args.max_samples or 'all'}")
    if args.year == 2021:
        print(f"  In-sample window starts at: {args.val_date_from}")
    print(f"  Origin mode:   {args.origin_mode}")
    print(f"  Polygon policy:{args.polygon_policy}")

    rng = random.Random(args.seed)

    print("Loading cluster pools...")
    pools = load_cluster_pools(args.cbg_poi_dir)
    print(f"  {len(pools)} clusters")
    injected = []
    hidden_id_set = set()
    if args.candidate_inject_poi_csv:
        injected = load_poi_records(args.candidate_inject_poi_csv)
        pools = augment_pools_with_pois(pools, injected)
        hidden_id_set = {r["poi_id"] for r in injected}
        print(f"  Candidate injection: {len(injected):,} POIs from {args.candidate_inject_poi_csv}")
        if args.n_hidden_decoys > 0:
            print(f"  Hidden-decoy enforcement: m={args.n_hidden_decoys} per record")

    print("Loading personas...")
    personas_by_cluster = load_personas_grouped(args.personas)
    print(f"  {sum(len(v) for v in personas_by_cluster.values()):,} personas across "
          f"{len(personas_by_cluster)} clusters")

    valid_clusters = {
        cid for cid, pool in pools.items()
        if len(pool) >= 5 and personas_by_cluster.get(cid)
    }
    print(f"  Valid clusters: {len(valid_clusters):,}")

    valid_set, name_to_valid = build_valid_poi_set(pools, args.fallback_poi_csv)

    exclude_truth_poi_ids = set()
    if args.exclude_truth_poi_csv:
        exclude_truth_poi_ids = load_poi_id_set(args.exclude_truth_poi_csv)
        print(f"  Excluding truth rows for {len(exclude_truth_poi_ids):,} POIs")

    point_member_lookup = {}
    if args.polygon_policy == "point_members":
        point_member_lookup = load_placekey_point_member_lookup(args.point_member_authority_csv)
        print(f"  Point-member lookup: {len(point_member_lookup):,} PLACEKEYs")

    work_cbg_origin_map = {}
    if args.origin_mode != "stop":
        work_cbg_origin_map = load_work_cbg_origin_map(args.work_cbg_origin_csv)
        print(f"  Work CBG anchors: {len(work_cbg_origin_map):,} "
              f"from {args.work_cbg_origin_csv}")

    if args.records_in:
        print(f"Loading prebuilt inference records: {args.records_in}")
        records = load_inference_records(args.records_in)
        print(f"  Loaded records: {len(records):,}")
    else:
        print("Sampling visit rows...")
        sampled_rows = sample_visit_rows(
            args.a_work,
            valid_clusters,
            args.year,
            args.val_date_from,
            args.max_samples,
            args.sampling_unit,
            args.seed,
            polygon_policy=args.polygon_policy,
            point_member_lookup=point_member_lookup,
            exclude_truth_poi_ids=exclude_truth_poi_ids,
        )
        if not sampled_rows:
            print("ERROR: no eligible rows were sampled")
            sys.exit(1)

        print("Building prompts...")
        if args.prepare_only and args.n_workers and args.n_workers > 1:
            base_seed = rng.randint(0, 2**31 - 1)
            n_written, n_skipped, n_origin_fb = build_and_write_inference_records_parallel(
                sampled_rows,
                args.records_out,
                pools, personas_by_cluster, args.k, base_seed,
                args.origin_mode, work_cbg_origin_map,
                args.primary_radius_m, args.fallback_radius_m, args.hard_cap_radius_m,
                freq_power=args.freq_power,
                hidden_pool=injected,
                hidden_id_set=hidden_id_set,
                n_hidden_decoys=args.n_hidden_decoys,
                n_workers=args.n_workers,
            )
            del sampled_rows
            gc.collect()
            if n_written == 0:
                print("ERROR: no inference prompts were built")
                sys.exit(1)
            print(f"  Saved records: {n_written:,}")
            print("Prepare-only mode complete; skipping vLLM generation.")
            return
        records = build_inference_records(
            sampled_rows,
            pools,
            personas_by_cluster,
            args.k,
            rng,
            args.origin_mode,
            work_cbg_origin_map,
            args.primary_radius_m,
            args.fallback_radius_m,
            args.hard_cap_radius_m,
            freq_power=args.freq_power,
            hidden_pool=injected,
            hidden_id_set=hidden_id_set,
            n_hidden_decoys=args.n_hidden_decoys,
        )
        del sampled_rows
        gc.collect()

    if not records:
        print("ERROR: no inference prompts were built")
        sys.exit(1)

    if args.records_out:
        print(f"Writing prebuilt inference records: {args.records_out}")
        write_inference_records(records, args.records_out)
        print(f"  Saved records: {len(records):,}")

    if args.prepare_only:
        print("Prepare-only mode complete; skipping vLLM generation.")
        return

    run_inference(
        records,
        args.model_path,
        args.lora_path,
        args.out,
        valid_set,
        name_to_valid,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        chunk_size=args.chunk_size,
        gpu_mem=args.gpu_mem,
        min_parse_rate=args.min_parse_rate,
    )


if __name__ == "__main__":
    main()
