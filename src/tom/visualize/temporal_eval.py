#!/usr/bin/env python3
"""
14_temporal_eval.py — Visit-level temporal evaluation for NYC Metro.

Aligned protocol:
  - Inference is done on sampled visits
  - Formal aggregate metrics use top-5 probability mass
  - Actual distributions are aligned to the prediction subset itself via true_poi_id
  - A Tokyo-style 4-panel scatter is generated from top-1 visit mass
  - Additional aggregate, calibration, 2D/3D spatial, interactive HTML, and distance diagnostics are generated
  - The POI universe is read from the experiment-scoped train-prior snapshot
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Registers the 3D projection.
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import kendalltau, spearmanr

BASE = "/scratch/sx2490/econai/nyc_metro"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"

BOROUGH_NAMES = {
    "US.NY.061": "Manhattan",
    "US.NY.047": "Brooklyn",
    "US.NY.081": "Queens",
}


def format_poi_id(name, lat, lon):
    if pd.isna(name) or pd.isna(lat) or pd.isna(lon):
        return None
    return f"{name}|{float(lat):.6f}|{float(lon):.6f}"


def normalize_poi_text(text):
    """Normalize punctuation variants consistently with inference parsing."""
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
    """Snap a raw POI id to the canonical eval universe."""
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


def canonical_truth_poi_id(pid, valid_set, name_to_valid):
    """Keep ground-truth POIs even when they are outside the prediction universe."""
    snapped = snap_poi_id(pid, valid_set, name_to_valid)
    if snapped is not None:
        return snapped, True

    if not isinstance(pid, str):
        return None, False
    pid = pid.strip()
    parts = pid.split("|")
    if len(parts) < 3:
        return None, False
    try:
        lat = float(parts[-2])
        lon = float(parts[-1])
    except ValueError:
        return None, False
    name = "|".join(parts[:-2]).strip()
    if not name:
        return None, False
    return format_poi_id(name, lat, lon), False


def normalize_distribution(counts):
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {pid: value / total for pid, value in counts.items()}


def build_valid_poi_set(cbg_poi_dir, fallback_path):
    """Build the canonical POI universe from the experiment snapshot."""
    valid = set()
    name_to_valid = {}
    poi_meta = {}

    pool_re = re.compile(r"^cluster_(\d+)_pois\.csv$")
    for fname in sorted(os.listdir(cbg_poi_dir)):
        if not pool_re.match(fname):
            continue
        df = pd.read_csv(f"{cbg_poi_dir}/{fname}")
        for _, row in df.iterrows():
            pid = row["poi_id"]
            valid.add(pid)
            key = normalize_poi_text(row["poi_name"])
            name_to_valid.setdefault(key, []).append(
                (pid, float(row["poi_lat"]), float(row["poi_lon"]))
            )
            poi_meta[pid] = {
                "name": row["poi_name"],
                "sub_category": row.get("sub_category", ""),
            }

    if fallback_path and os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        for _, row in df.iterrows():
            pid = row.get("poi_id")
            if pd.isna(pid):
                continue
            valid.add(pid)
            key = normalize_poi_text(row.get("poi_name", ""))
            if key:
                name_to_valid.setdefault(key, []).append(
                    (pid, float(row["poi_lat"]), float(row["poi_lon"]))
                )
            poi_meta.setdefault(pid, {
                "name": row.get("poi_name", str(pid).split("|")[0]),
                "sub_category": row.get("sub_category", ""),
            })

    print(f"  Valid POI universe: {len(valid):,}")
    return valid, name_to_valid, poi_meta


def temporal_group(status):
    """Collapse raw authority labels into stable/new/disappeared/unknown."""
    if status == "new":
        return "new"
    if status in {"closed", "closed_2022", "closed_before_study"}:
        return "disappeared"
    if status == "stable":
        return "stable"
    return "unknown"


def load_temporal_status_map(valid_set, name_to_valid):
    """Map canonical poi_id -> temporal_status using poi_name_authority.csv."""
    if not os.path.exists(AUTH_PATH):
        return {}

    status_by_pid = {}
    auth = pd.read_csv(
        AUTH_PATH,
        usecols=["final_name", "poi_lat", "poi_lon", "temporal_status"],
        low_memory=False,
    ).dropna(subset=["final_name", "poi_lat", "poi_lon"])

    for _, row in auth.iterrows():
        raw_pid = format_poi_id(row["final_name"], row["poi_lat"], row["poi_lon"])
        if raw_pid is None:
            continue
        status = row.get("temporal_status", "unknown")
        status_by_pid[raw_pid] = status
        pid = snap_poi_id(raw_pid, valid_set, name_to_valid, tol=1.5e-3)
        if pid is None:
            continue
        status_by_pid[pid] = status

    return status_by_pid


def normalize_top5(entries, valid_set, name_to_valid):
    by_poi = defaultdict(float)
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw_pid = str(entry.get("poi", ""))
        snapped = snap_poi_id(raw_pid, valid_set, name_to_valid)
        if snapped is None:
            continue
        try:
            prob = max(0.0, float(entry.get("prob", 0)))
        except (TypeError, ValueError):
            continue
        by_poi[snapped] += prob

    total = sum(by_poi.values())
    if total <= 0:
        return []

    ranked = sorted(by_poi.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"poi": pid, "prob": value / total} for pid, value in ranked]


def load_visit_level_eval(pred_path, valid_set, name_to_valid):
    """Load visit-level actual/predicted distributions from prediction JSONL."""
    print(f"  Loading {pred_path}...")

    actual_top5_counts = defaultdict(int)
    actual_top1_counts = defaultdict(int)
    pred_top5_mass = defaultdict(float)
    pred_top1_counts = defaultdict(int)
    borough_actual_top5 = defaultdict(lambda: defaultdict(int))
    borough_pred_top5 = defaultdict(lambda: defaultdict(float))
    polygon_actual_top5 = defaultdict(lambda: defaultdict(int))
    polygon_pred_top5 = defaultdict(lambda: defaultdict(float))

    n_total_lines = 0
    n_json_failed = 0
    n_truth_missing = 0
    n_truth_unmapped = 0
    n_truth_valid = 0
    n_truth_in_universe = 0
    n_truth_outside_universe = 0
    n_next_choice_valid = 0
    n_eval_records = 0
    n_top5_missing = 0
    n_top1_hits = 0
    n_top5_hits = 0

    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_total_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_json_failed += 1
                continue

            raw_true = rec.get("true_poi_id")
            if not raw_true:
                n_truth_missing += 1
                continue

            true_pid, truth_in_universe = canonical_truth_poi_id(
                str(raw_true), valid_set, name_to_valid
            )
            if true_pid is None:
                n_truth_unmapped += 1
                continue
            n_truth_valid += 1
            if truth_in_universe:
                n_truth_in_universe += 1
            else:
                n_truth_outside_universe += 1
            actual_top1_counts[true_pid] += 1

            next_pid = None
            next_raw = rec.get("next_choice")
            if next_raw:
                next_pid = snap_poi_id(str(next_raw), valid_set, name_to_valid)
            if next_pid:
                pred_top1_counts[next_pid] += 1
                n_next_choice_valid += 1
            if next_pid == true_pid:
                n_top1_hits += 1

            top5 = normalize_top5(rec.get("top_5", []), valid_set, name_to_valid)
            if not top5:
                n_top5_missing += 1
                continue

            admin2_id = str(rec.get("admin2_id", "") or "")
            polygon_type = str(rec.get("true_polygon_type", "") or "unknown")
            n_eval_records += 1
            actual_top5_counts[true_pid] += 1
            borough_actual_top5[admin2_id][true_pid] += 1
            polygon_actual_top5[polygon_type][true_pid] += 1
            for item in top5:
                pred_top5_mass[item["poi"]] += item["prob"]
                borough_pred_top5[admin2_id][item["poi"]] += item["prob"]
                polygon_pred_top5[polygon_type][item["poi"]] += item["prob"]
            if any(item["poi"] == true_pid for item in top5):
                n_top5_hits += 1

    stats = {
        "records_total": n_total_lines,
        "records_json_failed": n_json_failed,
        "truth_missing": n_truth_missing,
        "truth_unmapped": n_truth_unmapped,
        "truth_valid": n_truth_valid,
        "truth_in_prediction_universe": n_truth_in_universe,
        "truth_outside_prediction_universe": n_truth_outside_universe,
        "records_with_valid_next_choice": n_next_choice_valid,
        "records_with_predicted_top5": n_eval_records,
        "top5_missing_or_unusable": n_top5_missing,
        "truth_in_universe_rate": n_truth_valid / max(n_total_lines - n_json_failed, 1),
        "truth_in_prediction_universe_rate": n_truth_in_universe / max(n_truth_valid, 1),
        "predicted_top5_rate": n_eval_records / max(n_truth_valid, 1),
    }
    accuracy = {
        "n_predictions": n_truth_valid,
        "top1_accuracy": n_top1_hits / max(n_truth_valid, 1),
        "top5_accuracy": n_top5_hits / max(n_truth_valid, 1),
        "n_top1_hits": n_top1_hits,
        "n_top5_hits": n_top5_hits,
    }

    print(f"    total={n_total_lines:,}, truth_valid={n_truth_valid:,}, "
          f"top1_valid={n_next_choice_valid:,}, top5_eval_subset={n_eval_records:,}")
    return {
        "actual_top1": normalize_distribution(actual_top1_counts),
        "predicted_top1": normalize_distribution(pred_top1_counts),
        "actual_top5": normalize_distribution(actual_top5_counts),
        "predicted_top5": normalize_distribution(pred_top5_mass),
        "borough_actual_top5": {
            boro: normalize_distribution(counts)
            for boro, counts in borough_actual_top5.items() if counts
        },
        "borough_predicted_top5": {
            boro: normalize_distribution(counts)
            for boro, counts in borough_pred_top5.items() if counts
        },
        "polygon_actual_top5": {
            polygon_type: normalize_distribution(counts)
            for polygon_type, counts in polygon_actual_top5.items() if counts
        },
        "polygon_predicted_top5": {
            polygon_type: normalize_distribution(counts)
            for polygon_type, counts in polygon_pred_top5.items() if counts
        },
        "stats": stats,
        "accuracy": accuracy,
    }


def compute_metrics(actual, predicted):
    all_pois = sorted(set(actual) | set(predicted))
    if not all_pois:
        return None

    act = np.array([actual.get(pid, 0.0) for pid in all_pois])
    pred = np.array([predicted.get(pid, 0.0) for pid in all_pois])
    mask = act > 0

    if mask.sum() >= 2:
        spear, _ = spearmanr(act[mask], pred[mask])
        kend, _ = kendalltau(act[mask], pred[mask])
    else:
        spear = kend = float("nan")

    m = (act + pred) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        js = (
            0.5 * np.where(act > 0, act * np.log(act / np.where(m > 0, m, 1e-300)), 0).sum()
            + 0.5 * np.where(pred > 0, pred * np.log(pred / np.where(m > 0, m, 1e-300)), 0).sum()
        )

    sorted_actual = sorted(actual, key=actual.get, reverse=True)
    sorted_pred = sorted(predicted, key=predicted.get, reverse=True)
    ndcg = {}
    topk = {}
    for k in [10, 20, 50]:
        top_pred = sorted_pred[:k]
        top_actual = sorted_actual[:k]
        dcg = sum(actual.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(top_pred))
        idcg = sum(actual.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(top_actual))
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
        overlap = len(set(top_actual) & set(top_pred))
        topk[k] = {"overlap": overlap, "recall": overlap / max(len(top_actual), 1)}

    coverage = sum(1 for pid in actual if predicted.get(pid, 0.0) > 0) / max(len(actual), 1)
    return {
        "spearman": float(spear),
        "kendall": float(kend),
        "js_divergence": float(js),
        "ndcg": ndcg,
        "topk": topk,
        "poi_coverage": float(coverage),
        "n_actual_pois": int((act > 0).sum()),
        "n_predicted_pois": int((pred > 0).sum()),
    }


def subset_metrics(actual, predicted, subset):
    actual_sub = {pid: value for pid, value in actual.items() if pid in subset}
    pred_sub = {pid: value for pid, value in predicted.items() if pid in subset}
    actual_sub = normalize_distribution(actual_sub)
    pred_sub = normalize_distribution(pred_sub)
    if not actual_sub or not pred_sub:
        return None
    return compute_metrics(actual_sub, pred_sub)


def load_poi_subset_csv(path):
    subset = set()
    if not path:
        return subset
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("poi_id", "") or "").strip()
            if pid:
                subset.add(pid)
    return subset


def subset_mass_summary(actual, predicted, subset):
    if not subset:
        return None
    actual_mass = sum(value for pid, value in actual.items() if pid in subset)
    pred_mass = sum(value for pid, value in predicted.items() if pid in subset)
    actual_n = sum(1 for pid in actual if pid in subset)
    pred_n = sum(1 for pid in predicted if pid in subset)
    return {
        "n_subset_pois": len(subset),
        "n_actual_subset_pois": actual_n,
        "n_predicted_subset_pois": pred_n,
        "actual_mass": float(actual_mass),
        "predicted_mass": float(pred_mass),
        "mass_ratio_pred_over_actual": float(pred_mass / actual_mass) if actual_mass > 0 else None,
    }


def new_poi_lift(actual, predicted, temporal_status_by_pid):
    actual_new_mass = sum(
        value for pid, value in actual.items()
        if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == "new"
    )
    pred_new_mass = sum(
        value for pid, value in predicted.items()
        if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == "new"
    )
    if actual_new_mass <= 0:
        return None
    return pred_new_mass / actual_new_mass


def pretty_name(pid, poi_meta):
    return str(poi_meta.get(pid, {}).get("name") or pid.split("|")[0])[:18]


def save_figure(fig, out_png, dpi=220):
    """Save a figure as both PNG and PDF for quick preview and paper use."""
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    root, _ = os.path.splitext(out_png)
    fig.savefig(f"{root}.pdf", bbox_inches="tight", pad_inches=0.05)


def parse_poi_lat_lon(pid):
    if not isinstance(pid, str):
        return None
    parts = pid.split("|")
    if len(parts) < 3:
        return None
    try:
        return float(parts[-2]), float(parts[-1])
    except ValueError:
        return None


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.asin(min(1.0, math.sqrt(a)))


def bin_index(value, edges):
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= len(edges) - 1:
        return len(edges) - 2
    return idx


def distance_bin_labels(edges):
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isinf(right):
            labels.append(f">{int(left / 1000)}km")
        elif right < 1000:
            labels.append(f"{int(left)}-{int(right)}m")
        elif left < 1000:
            labels.append(f"{int(left)}m-{right / 1000:g}km")
        else:
            labels.append(f"{left / 1000:g}-{right / 1000:g}km")
    return labels


def load_visit_diagnostics(pred_path, valid_set, name_to_valid):
    """Aggregate visit-level diagnostics for calibration and distance plots."""
    prob_bins = np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.01])
    distance_bins = np.array([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
                              1500.0, 3000.0, 5000.0, 10000.0, 20000.0, np.inf])

    prob_count = np.zeros(len(prob_bins) - 1, dtype=float)
    prob_sum = np.zeros(len(prob_bins) - 1, dtype=float)
    prob_hits = np.zeros(len(prob_bins) - 1, dtype=float)

    actual_distance_mass = np.zeros(len(distance_bins) - 1, dtype=float)
    pred_distance_mass = np.zeros(len(distance_bins) - 1, dtype=float)

    cluster_actual = defaultdict(lambda: np.zeros(len(distance_bins) - 1, dtype=float))
    cluster_pred = defaultdict(lambda: np.zeros(len(distance_bins) - 1, dtype=float))
    cluster_counts = Counter()

    origin_modes = Counter()
    n_eval_records = 0
    n_distance_records = 0
    n_distance_pred_items = 0

    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            true_pid, _ = canonical_truth_poi_id(rec.get("true_poi_id"), valid_set, name_to_valid)
            if true_pid is None:
                continue
            top5 = normalize_top5(rec.get("top_5", []), valid_set, name_to_valid)
            if not top5:
                continue
            n_eval_records += 1

            for item in top5:
                prob = min(max(float(item["prob"]), 0.0), 1.0)
                idx = bin_index(prob, prob_bins)
                prob_count[idx] += 1
                prob_sum[idx] += prob
                if item["poi"] == true_pid:
                    prob_hits[idx] += 1

            origin = rec.get("origin") if isinstance(rec.get("origin"), dict) else {}
            try:
                origin_lat = float(origin.get("lat"))
                origin_lon = float(origin.get("lon"))
            except (TypeError, ValueError):
                continue
            origin_modes[str(origin.get("mode") or origin.get("source") or "unknown")] += 1

            cluster_key = str(rec.get("cluster") or "unknown")

            true_coords = parse_poi_lat_lon(true_pid)
            if true_coords is not None:
                actual_dist = haversine_m(origin_lat, origin_lon, true_coords[0], true_coords[1])
                actual_bin = bin_index(actual_dist, distance_bins)
                actual_distance_mass[actual_bin] += 1
                cluster_actual[cluster_key][actual_bin] += 1
                cluster_counts[cluster_key] += 1
                n_distance_records += 1

            for item in top5:
                coords = parse_poi_lat_lon(item["poi"])
                if coords is None:
                    continue
                pred_dist = haversine_m(origin_lat, origin_lon, coords[0], coords[1])
                pred_bin = bin_index(pred_dist, distance_bins)
                prob_value = float(item["prob"])
                pred_distance_mass[pred_bin] += prob_value
                cluster_pred[cluster_key][pred_bin] += prob_value
                n_distance_pred_items += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        calibration = {
            "bin_left": prob_bins[:-1].tolist(),
            "bin_right": prob_bins[1:].tolist(),
            "n": prob_count.tolist(),
            "avg_prob": np.divide(prob_sum, prob_count, out=np.zeros_like(prob_sum), where=prob_count > 0).tolist(),
            "empirical_hit_rate": np.divide(
                prob_hits, prob_count, out=np.zeros_like(prob_hits), where=prob_count > 0
            ).tolist(),
        }

    actual_total = actual_distance_mass.sum()
    pred_total = pred_distance_mass.sum()
    distance = {
        "bins_m": distance_bins.tolist(),
        "labels": distance_bin_labels(distance_bins),
        "actual_mass": (actual_distance_mass / actual_total if actual_total > 0 else actual_distance_mass).tolist(),
        "predicted_mass": (pred_distance_mass / pred_total if pred_total > 0 else pred_distance_mass).tolist(),
    }

    per_cluster_distance = {}
    for cluster_key, count in cluster_counts.items():
        a = cluster_actual[cluster_key]
        p = cluster_pred[cluster_key]
        a_total = a.sum()
        p_total = p.sum()
        per_cluster_distance[cluster_key] = {
            "n_records": int(count),
            "actual_mass": (a / a_total if a_total > 0 else a).tolist(),
            "predicted_mass": (p / p_total if p_total > 0 else p).tolist(),
        }

    return {
        "calibration": calibration,
        "distance": distance,
        "per_cluster_distance": per_cluster_distance,
        "stats": {
            "records_with_top5": n_eval_records,
            "records_with_origin_distance": n_distance_records,
            "predicted_items_with_origin_distance": n_distance_pred_items,
            "origin_modes": dict(origin_modes),
        },
    }


def build_spatial_records(actual, predicted, temporal_status_by_pid):
    records = []
    for pid in sorted(set(actual) | set(predicted)):
        coords = parse_poi_lat_lon(pid)
        if coords is None:
            continue
        lat, lon = coords
        actual_mass = actual.get(pid, 0.0)
        pred_mass = predicted.get(pid, 0.0)
        if actual_mass <= 0 and pred_mass <= 0:
            continue
        records.append({
            "pid": pid,
            "lat": lat,
            "lon": lon,
            "actual": actual_mass,
            "predicted": pred_mass,
            "residual": pred_mass - actual_mass,
            "status": temporal_group(temporal_status_by_pid.get(pid, "unknown")),
        })
    return records


def spatial_extent(records, q=0.003):
    if not records:
        return None
    lons = np.array([rec["lon"] for rec in records], dtype=float)
    lats = np.array([rec["lat"] for rec in records], dtype=float)
    lon_min, lon_max = np.quantile(lons, [q, 1 - q])
    lat_min, lat_max = np.quantile(lats, [q, 1 - q])
    lon_pad = max((lon_max - lon_min) * 0.05, 0.01)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.01)
    return [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]


def records_to_arrays(records, extent=None):
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        records = [
            rec for rec in records
            if xmin <= rec["lon"] <= xmax and ymin <= rec["lat"] <= ymax
        ]
    return {
        "records": records,
        "lon": np.array([rec["lon"] for rec in records], dtype=float),
        "lat": np.array([rec["lat"] for rec in records], dtype=float),
        "actual": np.array([rec["actual"] for rec in records], dtype=float),
        "predicted": np.array([rec["predicted"] for rec in records], dtype=float),
        "residual": np.array([rec["residual"] for rec in records], dtype=float),
        "status": np.array([rec["status"] for rec in records], dtype=object),
        "pid": np.array([rec["pid"] for rec in records], dtype=object),
    }


def style_map_axis(ax, extent, title):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.18, lw=0.6)


def write_html(path, html):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def aggregate_spatial_grid(records, extent, nx=78, ny=78):
    if extent is None:
        return []
    xmin, xmax, ymin, ymax = extent
    bins = {}
    for rec in records:
        lon = rec["lon"]
        lat = rec["lat"]
        if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
            continue
        ix = min(nx - 1, max(0, int((lon - xmin) / max(xmax - xmin, 1e-12) * nx)))
        iy = min(ny - 1, max(0, int((lat - ymin) / max(ymax - ymin, 1e-12) * ny)))
        key = (ix, iy)
        if key not in bins:
            bins[key] = {
                "ix": ix,
                "iy": iy,
                "actual": 0.0,
                "predicted": 0.0,
                "residual": 0.0,
                "count": 0,
                "status_counts": Counter(),
            }
        item = bins[key]
        item["actual"] += rec["actual"]
        item["predicted"] += rec["predicted"]
        item["residual"] += rec["residual"]
        item["count"] += 1
        item["status_counts"][rec["status"]] += 1

    data = []
    for item in bins.values():
        ix = item["ix"]
        iy = item["iy"]
        status_counts = item.pop("status_counts")
        item["lon"] = xmin + (ix + 0.5) / nx * (xmax - xmin)
        item["lat"] = ymin + (iy + 0.5) / ny * (ymax - ymin)
        item["status"] = status_counts.most_common(1)[0][0] if status_counts else "unknown"
        item["status_counts"] = dict(status_counts)
        data.append(item)
    data.sort(key=lambda d: d["actual"] + d["predicted"])
    return data


def html_spatial_summary(data, records):
    return {
        "n_bins": len(data),
        "n_pois": len(records),
        "max_actual": max([d["actual"] for d in data] or [0.0]),
        "max_predicted": max([d["predicted"] for d in data] or [0.0]),
        "max_abs_residual": max([abs(d["residual"]) for d in data] or [0.0]),
    }


def plotly_surface_payload(records, extent):
    arr = records_to_arrays(records, extent)
    if len(arr["records"]) < 10:
        return None

    x_edges = np.linspace(extent[0], extent[1], 70)
    y_edges = np.linspace(extent[2], extent[3], 70)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    surfaces = []
    for values in [arr["actual"], arr["predicted"], arr["predicted"] - arr["actual"]]:
        hist, _, _ = np.histogram2d(arr["lon"], arr["lat"], bins=[x_edges, y_edges], weights=values)
        surfaces.append(gaussian_filter(hist, sigma=1.15) * 1e4)
    return {
        "x": np.round(x_centers, 6).tolist(),
        "y": np.round(y_centers, 6).tolist(),
        "actual_z": np.round(surfaces[0].T, 8).tolist(),
        "predicted_z": np.round(surfaces[1].T, 8).tolist(),
        "residual_z": np.round(surfaces[2].T, 8).tolist(),
    }


def write_interactive_spatial_grid_html(actual, predicted, temporal_status_by_pid, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    grid = aggregate_spatial_grid(records, extent)
    payload = {
        "extent": [float(x) for x in extent],
        "data": grid,
        "summary": html_spatial_summary(grid, records),
    }
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NYC Exp02 OOS Interactive Spatial Grid</title>
<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101418; color: #f2f4f8; }
#wrap { display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }
#side { padding: 22px; border-right: 1px solid #2b3440; background: #151b22; }
#side h1 { font-size: 20px; margin: 0 0 10px; line-height: 1.25; }
#side p { color: #b8c0cc; line-height: 1.45; font-size: 13px; }
label { display: block; color: #d6dde8; margin-top: 14px; font-size: 13px; }
select, input { width: 100%; margin-top: 6px; padding: 7px; background: #0f141a; color: #f2f4f8; border: 1px solid #394554; border-radius: 6px; }
#mapbox { position: relative; padding: 16px; }
canvas { width: 100%; height: calc(100vh - 32px); background: #f8fafc; border-radius: 14px; box-shadow: 0 20px 70px rgba(0,0,0,0.35); }
#tip { position: absolute; pointer-events: none; background: rgba(15,20,26,0.93); color: white; padding: 9px 11px; border-radius: 8px; font-size: 12px; display: none; max-width: 260px; border: 1px solid rgba(255,255,255,0.16); }
.metric { margin-top: 14px; padding: 10px; background: #0f141a; border-radius: 8px; color: #d6dde8; font-size: 12px; }
</style>
</head>
<body>
<div id="wrap">
<aside id="side">
<h1>Exp02 2022 OOS Spatial Grid</h1>
<p>Self-contained HTML: no Python package and no web tile dependency. Hover bins to inspect actual / predicted / residual demand mass.</p>
<label>Layer
<select id="mode">
<option value="residual">Residual: predicted - actual</option>
<option value="actual">Actual visit mass</option>
<option value="predicted">Predicted top-5 mass</option>
</select>
</label>
<label>Point scale
<input id="scale" type="range" min="0.5" max="2.8" value="1.25" step="0.05"/>
</label>
<div class="metric" id="summary"></div>
<p>Color: red over-predicted, blue under-predicted for residual; sequential colors for mass layers. Size follows magnitude.</p>
</aside>
<main id="mapbox">
<canvas id="canvas" width="1400" height="900"></canvas>
<div id="tip"></div>
</main>
</div>
<script>
const payload = __PAYLOAD__;
const data = payload.data;
const extent = payload.extent;
const summary = payload.summary;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const modeEl = document.getElementById("mode");
const scaleEl = document.getElementById("scale");
const tip = document.getElementById("tip");
document.getElementById("summary").innerHTML =
  `<b>${summary.n_bins.toLocaleString()}</b> spatial bins<br>` +
  `<b>${summary.n_pois.toLocaleString()}</b> POIs represented<br>` +
  `Max actual mass: ${summary.max_actual.toExponential(2)}<br>` +
  `Max predicted mass: ${summary.max_predicted.toExponential(2)}`;

const margin = {left: 72, right: 28, top: 34, bottom: 58};
function sx(lon){ return margin.left + (lon - extent[0]) / (extent[1] - extent[0]) * (canvas.width - margin.left - margin.right); }
function sy(lat){ return canvas.height - margin.bottom - (lat - extent[2]) / (extent[3] - extent[2]) * (canvas.height - margin.top - margin.bottom); }
function lerp(a,b,t){ return Math.round(a + (b-a)*t); }
function mix(c1,c2,t){ return `rgb(${lerp(c1[0],c2[0],t)},${lerp(c1[1],c2[1],t)},${lerp(c1[2],c2[2],t)})`; }
function seq(t){
  const stops = [[24,16,91],[118,42,131],[214,67,103],[253,174,97],[255,247,188]];
  t = Math.max(0, Math.min(1, t));
  const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p));
  return mix(stops[i], stops[i+1], p - i);
}
function div(v){
  const m = Math.max(summary.max_abs_residual, 1e-12);
  const t = Math.max(-1, Math.min(1, v / m));
  return t >= 0 ? mix([247,247,247],[165,0,38], Math.sqrt(t)) : mix([49,54,149],[247,247,247], Math.sqrt(t + 1));
}
function value(d, mode){ return mode === "residual" ? d.residual : d[mode]; }
function color(d, mode){
  if (mode === "residual") return div(d.residual);
  const maxv = Math.max(summary[`max_${mode}`], 1e-12);
  return seq(Math.log10(1 + Math.max(d[mode],0) / maxv * 999) / 3);
}
function radius(d, mode){
  const mag = Math.abs(value(d, mode));
  const denom = mode === "residual" ? Math.max(summary.max_abs_residual,1e-12) : Math.max(summary[`max_${mode}`],1e-12);
  return (2.0 + 16.0 * Math.sqrt(mag / denom)) * parseFloat(scaleEl.value);
}
function axes(){
  ctx.strokeStyle = "#e6edf4"; ctx.lineWidth = 1;
  ctx.fillStyle = "#344054"; ctx.font = "18px sans-serif";
  for (let i=0;i<=6;i++){
    const x = margin.left + i/6*(canvas.width-margin.left-margin.right);
    const y = margin.top + i/6*(canvas.height-margin.top-margin.bottom);
    ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, canvas.height-margin.bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(canvas.width-margin.right, y); ctx.stroke();
  }
  ctx.fillText("Longitude", canvas.width/2 - 46, canvas.height - 16);
  ctx.save(); ctx.translate(24, canvas.height/2 + 40); ctx.rotate(-Math.PI/2); ctx.fillText("Latitude", 0, 0); ctx.restore();
}
function draw(){
  const mode = modeEl.value;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#f8fafc"; ctx.fillRect(0,0,canvas.width,canvas.height);
  axes();
  for (const d of data){
    const r = radius(d, mode);
    if (r <= 2.05 && mode === "residual") continue;
    ctx.beginPath();
    ctx.fillStyle = color(d, mode);
    ctx.globalAlpha = mode === "residual" ? 0.78 : 0.86;
    ctx.arc(sx(d.lon), sy(d.lat), r, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function nearest(mx,my){
  let best=null, bd=1e9;
  const mode = modeEl.value;
  for (const d of data){
    const dx=sx(d.lon)-mx, dy=sy(d.lat)-my, dist=Math.sqrt(dx*dx+dy*dy);
    if (dist < Math.max(10, radius(d, mode)+4) && dist < bd){ best=d; bd=dist; }
  }
  return best;
}
canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) / rect.width * canvas.width;
  const my = (e.clientY - rect.top) / rect.height * canvas.height;
  const d = nearest(mx,my);
  if (!d){ tip.style.display="none"; return; }
  tip.style.display="block"; tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+14)+"px";
  tip.innerHTML = `<b>Bin (${d.count} POIs)</b><br>` +
    `Actual: ${d.actual.toExponential(3)}<br>` +
    `Predicted: ${d.predicted.toExponential(3)}<br>` +
    `Residual: ${d.residual.toExponential(3)}<br>` +
    `Dominant status: ${d.status}`;
});
canvas.addEventListener("mouseleave", () => tip.style.display="none");
modeEl.addEventListener("change", draw);
scaleEl.addEventListener("input", draw);
draw();
</script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    write_html(out_path, html)


def write_temporal_status_poi_html(actual, predicted, temporal_status_by_pid, poi_meta, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    records = [
        rec for rec in records
        if extent[0] <= rec["lon"] <= extent[1] and extent[2] <= rec["lat"] <= extent[3]
    ]
    data = [{
        "name": pretty_name(rec["pid"], poi_meta),
        "lat": rec["lat"],
        "lon": rec["lon"],
        "actual": rec["actual"],
        "predicted": rec["predicted"],
        "residual": rec["residual"],
        "status": rec["status"],
        "mass": rec["actual"] + rec["predicted"],
    } for rec in records]
    payload = {
        "extent": [float(x) for x in extent],
        "data": data,
        "max_mass": max([d["mass"] for d in data] or [0.0]),
    }
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NYC Exp02 POI Temporal Status Map</title>
<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f8fb; color: #1d2430; }
#bar { display: flex; gap: 16px; align-items: center; padding: 12px 18px; background: #111827; color: white; }
#bar h1 { font-size: 18px; margin: 0 18px 0 0; }
#bar label { font-size: 13px; }
#mapbox { position: relative; padding: 14px; }
canvas { width: 100%; height: calc(100vh - 70px); background: white; border-radius: 14px; box-shadow: 0 12px 45px rgba(0,0,0,.14); }
#tip { position: absolute; pointer-events: none; display:none; background: rgba(17,24,39,.94); color: white; padding: 9px 11px; border-radius: 8px; font-size: 12px; max-width: 280px; }
</style>
</head>
<body>
<div id="bar">
<h1>Exp02 2022 OOS POI Status Map</h1>
<label><input type="checkbox" data-status="stable" checked> Stable</label>
<label><input type="checkbox" data-status="disappeared" checked> Disappeared</label>
<label><input type="checkbox" data-status="new" checked> New</label>
<label><input type="checkbox" data-status="unknown" checked> Unknown</label>
<label>Size <input id="scale" type="range" min="0.5" max="3.0" value="1.1" step="0.05"></label>
</div>
<div id="mapbox"><canvas id="canvas" width="1500" height="900"></canvas><div id="tip"></div></div>
<script>
const payload = __PAYLOAD__;
const data = payload.data;
const extent = payload.extent;
const maxMass = Math.max(payload.max_mass, 1e-12);
const canvas = document.getElementById("canvas"), ctx = canvas.getContext("2d"), tip = document.getElementById("tip");
const margin = {left: 72, right: 28, top: 28, bottom: 58};
const colors = {stable:"#9aa0a6", disappeared:"#e68632", new:"#20a65a", unknown:"#756bb1"};
const markers = {stable:"circle", disappeared:"tri", new:"star", unknown:"square"};
function active(){ const s={}; document.querySelectorAll("input[data-status]").forEach(x => s[x.dataset.status]=x.checked); return s; }
function sx(lon){ return margin.left + (lon - extent[0])/(extent[1]-extent[0])*(canvas.width-margin.left-margin.right); }
function sy(lat){ return canvas.height-margin.bottom - (lat - extent[2])/(extent[3]-extent[2])*(canvas.height-margin.top-margin.bottom); }
function r(d){ return (2 + 18*Math.sqrt(d.mass/maxMass))*parseFloat(document.getElementById("scale").value); }
function shape(x,y,rad,type){
  ctx.beginPath();
  if(type==="square"){ ctx.rect(x-rad,y-rad,2*rad,2*rad); return; }
  if(type==="tri"){ ctx.moveTo(x,y-rad); ctx.lineTo(x+rad,y+rad); ctx.lineTo(x-rad,y+rad); ctx.closePath(); return; }
  if(type==="star"){ for(let i=0;i<10;i++){ const a=-Math.PI/2+i*Math.PI/5, rr=i%2?rad*.45:rad; const xx=x+Math.cos(a)*rr, yy=y+Math.sin(a)*rr; if(i===0)ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy);} ctx.closePath(); return; }
  ctx.arc(x,y,rad,0,Math.PI*2);
}
function axes(){
  ctx.strokeStyle="#edf1f5"; ctx.lineWidth=1;
  for(let i=0;i<=6;i++){ const x=margin.left+i/6*(canvas.width-margin.left-margin.right), y=margin.top+i/6*(canvas.height-margin.top-margin.bottom); ctx.beginPath(); ctx.moveTo(x,margin.top); ctx.lineTo(x,canvas.height-margin.bottom); ctx.stroke(); ctx.beginPath(); ctx.moveTo(margin.left,y); ctx.lineTo(canvas.width-margin.right,y); ctx.stroke(); }
  ctx.fillStyle="#344054"; ctx.font="18px sans-serif"; ctx.fillText("Longitude", canvas.width/2-46, canvas.height-16);
  ctx.save(); ctx.translate(24, canvas.height/2+40); ctx.rotate(-Math.PI/2); ctx.fillText("Latitude",0,0); ctx.restore();
}
function draw(){
  const a=active();
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle="white"; ctx.fillRect(0,0,canvas.width,canvas.height); axes();
  const order=["stable","unknown","disappeared","new"];
  for(const status of order){
    for(const d of data){ if(d.status!==status || !a[status]) continue; const rad=r(d); ctx.globalAlpha=status==="stable"?.22:.78; ctx.fillStyle=colors[status]; shape(sx(d.lon),sy(d.lat),rad,markers[status]); ctx.fill(); }
  }
  ctx.globalAlpha=1;
}
function nearest(mx,my){
  const a=active(); let best=null, bd=1e9;
  for(const d of data){ if(!a[d.status]) continue; const dx=sx(d.lon)-mx, dy=sy(d.lat)-my, dist=Math.sqrt(dx*dx+dy*dy); if(dist<Math.max(9,r(d)+4)&&dist<bd){best=d;bd=dist;} }
  return best;
}
canvas.addEventListener("mousemove", e => {
  const rect=canvas.getBoundingClientRect(), mx=(e.clientX-rect.left)/rect.width*canvas.width, my=(e.clientY-rect.top)/rect.height*canvas.height;
  const d=nearest(mx,my); if(!d){tip.style.display="none"; return;}
  tip.style.display="block"; tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+14)+"px";
  tip.innerHTML=`<b>${d.name}</b><br>Status: ${d.status}<br>Actual: ${d.actual.toExponential(3)}<br>Predicted: ${d.predicted.toExponential(3)}<br>Residual: ${d.residual.toExponential(3)}`;
});
canvas.addEventListener("mouseleave",()=>tip.style.display="none");
document.querySelectorAll("input").forEach(x=>x.addEventListener("input",draw));
draw();
</script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    write_html(out_path, html)


def write_plotly_3d_surface_html(actual, predicted, temporal_status_by_pid, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    payload = plotly_surface_payload(records, extent)
    if payload is None:
        return
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NYC Exp02 Interactive 3D Surface</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#111827;color:white}#plot{width:100vw;height:100vh}.note{position:absolute;left:18px;top:14px;background:rgba(17,24,39,.82);padding:10px 12px;border-radius:8px;font-size:13px;z-index:10}</style>
</head>
<body>
<div class="note"><b>Exp02 2022 OOS 3D Surface</b><br>Use the buttons to switch actual / predicted / residual. This file loads Plotly from CDN.</div>
<div id="plot"></div>
<script>
const p = __PAYLOAD__;
const traces = [
  {type:"surface", x:p.x, y:p.y, z:p.actual_z, colorscale:"YlOrBr", name:"Actual", visible:true, showscale:true, colorbar:{title:"Mass density"}},
  {type:"surface", x:p.x, y:p.y, z:p.predicted_z, colorscale:"YlGnBu", name:"Predicted", visible:false, showscale:true, colorbar:{title:"Mass density"}},
  {type:"surface", x:p.x, y:p.y, z:p.residual_z, colorscale:"RdBu", name:"Residual", visible:false, showscale:true, colorbar:{title:"Residual"}}
];
const layout = {
  title:"Interactive 3D Demand Surface: Actual vs Predicted",
  scene:{xaxis:{title:"Longitude"}, yaxis:{title:"Latitude"}, zaxis:{title:"Mass density x1e4"}, camera:{eye:{x:1.55,y:-1.65,z:1.05}}},
  paper_bgcolor:"#111827", plot_bgcolor:"#111827", font:{color:"#f9fafb"},
  margin:{l:0,r:0,t:56,b:0},
  updatemenus:[{type:"buttons", x:.5, y:1.02, xanchor:"center", direction:"right", buttons:[
    {label:"Actual", method:"update", args:[{visible:[true,false,false]}, {title:"Interactive 3D Demand Surface: Actual"}]},
    {label:"Predicted", method:"update", args:[{visible:[false,true,false]}, {title:"Interactive 3D Demand Surface: Predicted"}]},
    {label:"Residual", method:"update", args:[{visible:[false,false,true]}, {title:"Interactive 3D Demand Surface: Residual"}]}
  ]}]
};
Plotly.newPlot("plot", traces, layout, {responsive:true, displaylogo:false});
</script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    write_html(out_path, html)


def write_deckgl_3d_columns_html(actual, predicted, temporal_status_by_pid, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    grid = aggregate_spatial_grid(records, extent, nx=68, ny=68)
    summary = html_spatial_summary(grid, records)
    payload = {
        "data": grid,
        "summary": summary,
        "view": {
            "longitude": float((extent[0] + extent[1]) / 2),
            "latitude": float((extent[2] + extent[3]) / 2),
            "zoom": 9.7,
            "pitch": 54,
            "bearing": -18,
        },
    }
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NYC Exp02 3D Column Map</title>
<script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
<style>
body{margin:0;background:#0b1020;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#deck{position:absolute;inset:0}
#controls{position:absolute;top:16px;left:16px;z-index:3;background:rgba(11,16,32,.86);padding:12px 14px;border-radius:10px;border:1px solid rgba(255,255,255,.16)}
select,input{background:#111827;color:#f8fafc;border:1px solid #475569;border-radius:6px;padding:6px;margin-top:6px}
label{display:block;font-size:13px;margin-top:8px;color:#cbd5e1}
</style>
</head>
<body>
<div id="deck"></div>
<div id="controls">
<b>Exp02 2022 OOS 3D Columns</b><br>
<span style="font-size:12px;color:#cbd5e1">WebGL via deck.gl CDN on a Carto light basemap. Drag to rotate/zoom.</span>
<label>Layer<select id="mode"><option value="actual">Actual mass</option><option value="predicted">Predicted mass</option><option value="residual">Absolute residual height</option></select></label>
<label>Height<input id="height" type="range" min="0.3" max="4.0" value="1.3" step="0.1"></label>
</div>
<script>
const payload = __PAYLOAD__;
const data = payload.data;
const summary = payload.summary;
const view = payload.view;
function lerp(a,b,t){return Math.round(a+(b-a)*t)}
function mix(c1,c2,t){return [lerp(c1[0],c2[0],t),lerp(c1[1],c2[1],t),lerp(c1[2],c2[2],t),210]}
function color(d, mode){
  if(mode==="residual"){ const t=Math.min(1,Math.abs(d.residual)/Math.max(summary.max_abs_residual,1e-12)); return d.residual>=0?mix([255,247,247],[165,0,38],Math.sqrt(t)):mix([49,54,149],[247,247,247],Math.sqrt(t));}
  const maxv=Math.max(summary["max_"+mode],1e-12), t=Math.log10(1+Math.max(d[mode],0)/maxv*999)/3;
  return mix([49,46,129],[255,218,120],Math.max(0,Math.min(1,t)));
}
function elevation(d, mode){
  const h=parseFloat(document.getElementById("height").value);
  if(mode==="residual") return 4200*h*Math.abs(d.residual)/Math.max(summary.max_abs_residual,1e-12);
  return 5200*h*d[mode]/Math.max(summary["max_"+mode],1e-12);
}
function makeLayer(){
  const mode=document.getElementById("mode").value;
  return new deck.ColumnLayer({
    id:"columns", data, diskResolution:6, radius:115, extruded:true, pickable:true,
    getPosition:d=>[d.lon,d.lat], getFillColor:d=>color(d,mode), getElevation:d=>elevation(d,mode),
    material:{ambient:0.35,diffuse:0.6,shininess:32,specularColor:[70,70,70]}
  });
}
function makeBaseLayer(){
  return new deck.TileLayer({
    id:"carto-light",
    data:"https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    minZoom:0,
    maxZoom:19,
    tileSize:256,
    renderSubLayers: props => {
      const {west, south, east, north} = props.tile.bbox;
      return new deck.BitmapLayer(props, {
        id: props.id + "-bitmap",
        image: props.data,
        bounds: [west, south, east, north]
      });
    }
  });
}
const deckgl = new deck.DeckGL({
  container:"deck",
  initialViewState:view,
  controller:true,
  layers:[makeBaseLayer(), makeLayer()],
  getTooltip:({object}) => object && {
    html:`<b>${object.count} POIs</b><br>Actual: ${object.actual.toExponential(3)}<br>Predicted: ${object.predicted.toExponential(3)}<br>Residual: ${object.residual.toExponential(3)}<br>Status: ${object.status}`,
    style:{backgroundColor:"rgba(15,23,42,.93)", color:"white", fontSize:"12px", borderRadius:"8px"}
  }
});
function redraw(){ deckgl.setProps({layers:[makeBaseLayer(), makeLayer()]}); }
document.getElementById("mode").addEventListener("change", redraw);
document.getElementById("height").addEventListener("input", redraw);
</script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    write_html(out_path, html)


def write_leaflet_spatial_grid_map_html(actual, predicted, temporal_status_by_pid, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    grid = aggregate_spatial_grid(records, extent, nx=82, ny=82)
    payload = {
        "extent": [float(x) for x in extent],
        "data": grid,
        "summary": html_spatial_summary(grid, records),
    }
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NYC Exp02 OOS Basemap Demand Grid</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f7f8fb;color:#1d2430}
#map{position:absolute;inset:0;background:#f7f8fb}
.panel{position:absolute;top:18px;left:18px;z-index:600;background:rgba(255,255,255,.94);padding:13px 15px;border-radius:12px;box-shadow:0 12px 40px rgba(15,23,42,.18);max-width:330px}
.panel h1{font-size:18px;margin:0 0 6px}.panel p{font-size:12px;line-height:1.4;color:#526070;margin:6px 0}
.legend{background:rgba(255,255,255,.94);padding:10px 12px;border-radius:10px;box-shadow:0 8px 30px rgba(15,23,42,.14);font-size:12px;color:#344054}
.leaflet-control-layers{border-radius:10px;box-shadow:0 8px 30px rgba(15,23,42,.14)}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
<h1>Exp02 2022 OOS Demand Map</h1>
<p>Elegant basemap version. Toggle actual, predicted, or residual mass; circle size encodes magnitude.</p>
<p><b>Note:</b> this file loads Leaflet and Carto light tiles from CDNs when opened in a browser.</p>
</div>
<script>
const payload = __PAYLOAD__;
const data = payload.data;
const summary = payload.summary;
const map = L.map("map", {preferCanvas:true, zoomControl:false}).setView([40.735, -73.93], 11);
L.control.zoom({position:"bottomright"}).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", {
  maxZoom:20,
  attribution:"&copy; OpenStreetMap &copy; CARTO"
}).addTo(map);
const bounds = [[payload.extent[2], payload.extent[0]], [payload.extent[3], payload.extent[1]]];
map.fitBounds(bounds, {padding:[40,40]});
function lerp(a,b,t){return Math.round(a+(b-a)*t)}
function mix(c1,c2,t){return `rgb(${lerp(c1[0],c2[0],t)},${lerp(c1[1],c2[1],t)},${lerp(c1[2],c2[2],t)})`}
function seq(t){
  const stops=[[43,58,103],[91,124,153],[196,151,95],[230,190,118]];
  t=Math.max(0,Math.min(1,t)); const p=t*(stops.length-1), i=Math.min(stops.length-2,Math.floor(p));
  return mix(stops[i],stops[i+1],p-i);
}
function div(v){
  const m=Math.max(summary.max_abs_residual,1e-12), t=Math.max(-1,Math.min(1,v/m));
  return t>=0 ? mix([250,250,250],[178,24,43],Math.sqrt(t)) : mix([33,102,172],[250,250,250],Math.sqrt(t+1));
}
function val(d, mode){return mode==="residual" ? d.residual : d[mode]}
function radius(d, mode){
  const denom=mode==="residual" ? Math.max(summary.max_abs_residual,1e-12) : Math.max(summary["max_"+mode],1e-12);
  return 2.2 + 15*Math.sqrt(Math.abs(val(d,mode))/denom);
}
function color(d, mode){
  if(mode==="residual") return div(d.residual);
  const denom=Math.max(summary["max_"+mode],1e-12);
  const t=Math.log10(1+Math.max(d[mode],0)/denom*999)/3;
  return seq(t);
}
function popup(d){
  return `<b>${d.count.toLocaleString()} POIs in this bin</b><br>`+
    `Actual mass: ${d.actual.toExponential(3)}<br>`+
    `Predicted mass: ${d.predicted.toExponential(3)}<br>`+
    `Residual: ${d.residual.toExponential(3)}<br>`+
    `Dominant status: ${d.status}`;
}
function makeLayer(mode){
  const group=L.layerGroup();
  data.forEach(d=>{
    if(mode==="residual" && Math.abs(d.residual) < summary.max_abs_residual*0.006) return;
    if(mode!=="residual" && d[mode] <= 0) return;
    const c=color(d,mode);
    L.circleMarker([d.lat,d.lon], {
      radius:radius(d,mode), color:c, fillColor:c, weight:0.8,
      opacity:0.82, fillOpacity:mode==="residual" ? 0.70 : 0.62
    }).bindTooltip(popup(d), {sticky:true, direction:"top", opacity:0.92}).addTo(group);
  });
  return group;
}
const residual=makeLayer("residual"), actual=makeLayer("actual"), predicted=makeLayer("predicted");
residual.addTo(map);
L.control.layers(null, {
  "Residual: predicted - actual": residual,
  "Actual visit mass": actual,
  "Predicted top-5 mass": predicted
}, {collapsed:false, position:"topright"}).addTo(map);
const legend=L.control({position:"bottomleft"});
legend.onAdd=function(){
  const d=L.DomUtil.create("div","legend");
  d.innerHTML="<b>Encoding</b><br>Circle size = mass magnitude<br>Residual: red over, blue under<br>Basemap: Carto Positron";
  return d;
};
legend.addTo(map);
</script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    write_html(out_path, html)


def write_leaflet_temporal_status_map_html(actual, predicted, temporal_status_by_pid, poi_meta, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    data = []
    for rec in records:
        if not (extent[0] <= rec["lon"] <= extent[1] and extent[2] <= rec["lat"] <= extent[3]):
            continue
        data.append({
            "name": pretty_name(rec["pid"], poi_meta),
            "lat": rec["lat"],
            "lon": rec["lon"],
            "actual": rec["actual"],
            "predicted": rec["predicted"],
            "residual": rec["residual"],
            "status": rec["status"],
            "mass": rec["actual"] + rec["predicted"],
        })
    payload = {
        "extent": [float(x) for x in extent],
        "data": data,
        "max_mass": max([d["mass"] for d in data] or [0.0]),
    }
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>NYC Exp02 POI Status Basemap</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f7f8fb;color:#1d2430}
#map{position:absolute;inset:0;background:#f7f8fb}
.panel{position:absolute;top:18px;left:18px;z-index:600;background:rgba(255,255,255,.94);padding:13px 15px;border-radius:12px;box-shadow:0 12px 40px rgba(15,23,42,.18);max-width:340px}
.panel h1{font-size:18px;margin:0 0 6px}.panel p{font-size:12px;line-height:1.4;color:#526070;margin:6px 0}
.legend{background:rgba(255,255,255,.94);padding:10px 12px;border-radius:10px;box-shadow:0 8px 30px rgba(15,23,42,.14);font-size:12px;color:#344054}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:middle}
.leaflet-control-layers{border-radius:10px;box-shadow:0 8px 30px rgba(15,23,42,.14)}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
<h1>Exp02 2022 POI Status Map</h1>
<p>Toggle status layers over a quiet basemap. Stable POIs are off by default to keep the map readable.</p>
<p><b>Note:</b> this file loads Leaflet and Carto light tiles from CDNs when opened in a browser.</p>
</div>
<script>
const payload = __PAYLOAD__;
const data = payload.data;
const maxMass = Math.max(payload.max_mass, 1e-12);
const map = L.map("map", {preferCanvas:true, zoomControl:false}).setView([40.735, -73.93], 11);
L.control.zoom({position:"bottomright"}).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", {
  maxZoom:20,
  attribution:"&copy; OpenStreetMap &copy; CARTO"
}).addTo(map);
const bounds = [[payload.extent[2], payload.extent[0]], [payload.extent[3], payload.extent[1]]];
map.fitBounds(bounds, {padding:[40,40]});
const style={
  stable:{color:"#7d8790", label:"Stable", opacity:0.32},
  disappeared:{color:"#d87c2f", label:"Disappeared", opacity:0.78},
  new:{color:"#159653", label:"New", opacity:0.95},
  unknown:{color:"#756bb1", label:"Unknown", opacity:0.68}
};
function radius(d){ return 2.0 + 12*Math.sqrt(d.mass/maxMass); }
function popup(d){
  return `<b>${d.name}</b><br>Status: ${d.status}<br>`+
    `Actual mass: ${d.actual.toExponential(3)}<br>`+
    `Predicted mass: ${d.predicted.toExponential(3)}<br>`+
    `Residual: ${d.residual.toExponential(3)}`;
}
function makeGroup(status){
  const group=L.layerGroup();
  data.forEach(d=>{
    if(d.status!==status) return;
    const s=style[status] || style.unknown;
    L.circleMarker([d.lat,d.lon], {
      radius:radius(d), color:s.color, fillColor:s.color, weight:0.6,
      opacity:s.opacity, fillOpacity:s.opacity*0.55
    }).bindTooltip(popup(d), {sticky:true, opacity:0.92}).addTo(group);
  });
  return group;
}
const layers={
  stable:makeGroup("stable"),
  disappeared:makeGroup("disappeared"),
  new:makeGroup("new"),
  unknown:makeGroup("unknown")
};
layers.disappeared.addTo(map);
layers.new.addTo(map);
layers.unknown.addTo(map);
L.control.layers(null, {
  "Disappeared": layers.disappeared,
  "New": layers.new,
  "Unknown": layers.unknown,
  "Stable (dense)": layers.stable
}, {collapsed:false, position:"topright"}).addTo(map);
const legend=L.control({position:"bottomleft"});
legend.onAdd=function(){
  const d=L.DomUtil.create("div","legend");
  d.innerHTML="<b>Temporal status</b><br>"+
    '<span class="sw" style="background:#d87c2f"></span>Disappeared<br>'+
    '<span class="sw" style="background:#159653"></span>New<br>'+
    '<span class="sw" style="background:#756bb1"></span>Unknown<br>'+
    '<span class="sw" style="background:#7d8790"></span>Stable<br>'+
    "<br>Circle size = actual + predicted mass";
  return d;
};
legend.addTo(map);
</script>
</body>
</html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    write_html(out_path, html)


def draw_scatter_panel(ax, actual, predicted, temporal_status_by_pid, poi_meta, title, log_mode=False):
    all_pois = set(actual) | set(predicted)
    log_lo = 2e-5
    log_hi = 8e-3

    stable_x, stable_y = [], []
    new_x, new_y = [], []
    dis_x, dis_y = [], []

    for pid in all_pois:
        a = actual.get(pid, 0.0)
        p = predicted.get(pid, 0.0)
        if log_mode:
            if a == 0 and p == 0:
                continue
            a = max(a, log_lo * 0.7)
            p = max(p, log_lo * 0.7)

        group = temporal_group(temporal_status_by_pid.get(pid, "unknown"))
        if group == "new":
            new_x.append(a)
            new_y.append(p)
        elif group == "disappeared":
            dis_x.append(a)
            dis_y.append(p)
        else:
            stable_x.append(a)
            stable_y.append(p)

    ax.scatter(stable_x, stable_y, s=5, alpha=0.3, c="#4C72B0", label="Stable/Other",
               zorder=3, rasterized=True)
    if new_x:
        ax.scatter(new_x, new_y, s=12, alpha=0.55, c="#C44E52", marker="^",
                   label=f"New ({len(new_x):,})", zorder=4)
    if dis_x:
        ax.scatter(dis_x, dis_y, s=12, alpha=0.55, c="#DD8452", marker="v",
                   label=f"Disappeared ({len(dis_x):,})", zorder=4)

    if log_mode:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(log_lo, log_hi)
        ax.set_ylim(log_lo, log_hi)
        ax.plot([log_lo, log_hi], [log_lo, log_hi], "k--", alpha=0.25, lw=1)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x * 1000:.1f}" if x >= 1e-4 else f"{x * 10000:.1f}"
        ))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x * 1000:.1f}" if x >= 1e-4 else f"{x * 10000:.1f}"
        ))
        ax.set_xlabel("Actual visit mass (x1e-3)")
        ax.set_ylabel("Predicted top-1 visit mass (x1e-3)")
    else:
        hi = max(max(actual.values(), default=0.001), max(predicted.values(), default=0.001)) * 1.1
        ax.plot([0, hi], [0, hi], "k--", alpha=0.25, lw=1)
        ax.set_xlim(-hi * 0.02, hi)
        ax.set_ylim(-hi * 0.02, hi)
        ax.set_xlabel("Actual visit mass")
        ax.set_ylabel("Predicted top-1 visit mass")

    top = sorted(actual.items(), key=lambda x: -x[1])[:15 if log_mode else 12]
    for pid, _ in top:
        px = actual.get(pid, log_lo if log_mode else 0.0)
        py = predicted.get(pid, log_lo if log_mode else 0.0)
        if log_mode:
            px = max(px, log_lo * 0.7)
            py = max(py, log_lo * 0.7)
        ax.annotate(
            pretty_name(pid, poi_meta),
            (px, py),
            fontsize=5.5,
            alpha=0.85,
            xytext=(4, 3),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.8, lw=0),
        )

    ax.set_title(title)
    ax.legend(loc="upper left", markerscale=1.5, framealpha=0.9)


def plot_temporal_scatter_4panel(train_actual, train_pred, test_actual, test_pred,
                                 temporal_status_by_pid, poi_meta, out_png, out_pdf):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    draw_scatter_panel(
        axes[0, 0], train_actual, train_pred, temporal_status_by_pid, poi_meta,
        "In-sample 2021 val (top-1 visit mass)", False,
    )
    draw_scatter_panel(
        axes[0, 1], test_actual, test_pred, temporal_status_by_pid, poi_meta,
        "OOS 2022 (top-1 visit mass)", False,
    )
    draw_scatter_panel(
        axes[1, 0], train_actual, train_pred, temporal_status_by_pid, poi_meta,
        "In-sample 2021 val — Log scale", True,
    )
    draw_scatter_panel(
        axes[1, 1], test_actual, test_pred, temporal_status_by_pid, poi_meta,
        "OOS 2022 — Log scale", True,
    )
    fig.suptitle("Actual vs Predicted Visit Mass — Color-coded by POI Status", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_borough_metrics(borough_metrics, out_path):
    if not borough_metrics:
        return
    names = list(borough_metrics.keys())
    spearman_vals = [borough_metrics[name]["spearman"] if borough_metrics[name] else 0 for name in names]
    ndcg_vals = [borough_metrics[name]["ndcg"][10] if borough_metrics[name] else 0 for name in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, spearman_vals, width, label="Spearman", color="#1f77b4")
    ax.bar(x + width / 2, ndcg_vals, width, label="NDCG@10", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Per-Borough OOS Metrics (visit-level)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_top_pois(actual, predicted, poi_meta, out_path, k=20):
    top_actual = sorted(actual.items(), key=lambda x: -x[1])[:k]
    top_pred = sorted(predicted.items(), key=lambda x: -x[1])[:k]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, items, title, color in [
        (axes[0], top_actual, f"Top {k} Actual Visit Mass", "#1f77b4"),
        (axes[1], top_pred, f"Top {k} Predicted Top-5 Mass", "#ff7f0e"),
    ]:
        names = [pretty_name(pid, poi_meta) for pid, _ in items]
        vals = [value for _, value in items]
        ax.barh(range(len(names)), vals, color=color)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Mass")
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_metrics_comparison(metrics_train, metrics_test, out_path):
    if not (metrics_train and metrics_test):
        return
    labels = ["spearman", "kendall", "NDCG@10", "NDCG@20", "NDCG@50"]
    train_vals = [
        metrics_train["spearman"],
        metrics_train["kendall"],
        metrics_train["ndcg"][10],
        metrics_train["ndcg"][20],
        metrics_train["ndcg"][50],
    ]
    test_vals = [
        metrics_test["spearman"],
        metrics_test["kendall"],
        metrics_test["ndcg"][10],
        metrics_test["ndcg"][20],
        metrics_test["ndcg"][50],
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, train_vals, width, label="2021 val", color="#1f77b4")
    ax.bar(x + width / 2, test_vals, width, label="2022 OOS", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Metrics: In-sample vs OOS (visit-level, top-5 mass)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_top5_mass_hexbin(train_actual, train_pred, test_actual, test_pred,
                          metrics_train, metrics_test, out_path):
    panels = [
        ("2021 validation", train_actual, train_pred, metrics_train),
        ("2022 temporal OOS", test_actual, test_pred, metrics_test),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    eps = 1e-7

    for ax, (title, actual, predicted, metrics) in zip(axes, panels):
        all_pois = sorted(set(actual) | set(predicted))
        x = np.log10(np.array([actual.get(pid, 0.0) + eps for pid in all_pois]))
        y = np.log10(np.array([predicted.get(pid, 0.0) + eps for pid in all_pois]))
        hb = ax.hexbin(x, y, gridsize=55, bins="log", mincnt=1, cmap="viridis")
        lo = min(float(x.min()), float(y.min()), -7.0)
        hi = max(float(x.max()), float(y.max()), -2.0)
        ax.plot([lo, hi], [lo, hi], color="white", lw=1.0, ls="--", alpha=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if metrics:
            ax.set_title(f"{title}\nSpearman={metrics['spearman']:.3f}, NDCG@10={metrics['ndcg'][10]:.3f}")
        else:
            ax.set_title(title)
        ax.set_xlabel("log10 actual top-5 visit mass")
        ax.grid(alpha=0.18)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("POI count")
    axes[0].set_ylabel("log10 predicted top-5 mass")
    fig.suptitle("Aggregate POI Demand: Actual vs Predicted Top-5 Probability Mass", y=1.03)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_probability_calibration(train_diag, test_diag, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, diag, color in [
        ("2021 validation", train_diag, "#1f77b4"),
        ("2022 temporal OOS", test_diag, "#ff7f0e"),
    ]:
        cal = diag["calibration"]
        n = np.array(cal["n"])
        x = np.array(cal["avg_prob"])
        y = np.array(cal["empirical_hit_rate"])
        mask = n > 0
        max_n = max(float(n.max()), 1.0) if n.size else 1.0
        sizes = 18 + 80 * np.sqrt(n[mask] / max_n)
        ax.plot(x[mask], y[mask], color=color, lw=1.5, alpha=0.8)
        ax.scatter(x[mask], y[mask], s=sizes, color=color, label=label, alpha=0.8, edgecolor="white")

    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, alpha=0.4, label="Perfect calibration")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Mean predicted probability in bin")
    ax.set_ylabel("Empirical hit rate")
    ax.set_title("Visit-Level Top-5 Probability Calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_cumulative_mass_recovery(train_actual, train_pred, test_actual, test_pred, out_path):
    def curve(actual, ranking):
        if not actual or not ranking:
            return np.array([]), np.array([])
        ranks = np.arange(1, len(ranking) + 1)
        y = np.cumsum([actual.get(pid, 0.0) for pid in ranking])
        total = max(sum(actual.values()), 1e-12)
        return ranks, y / total

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = [
        ("2021 validation", train_actual, train_pred),
        ("2022 temporal OOS", test_actual, test_pred),
    ]
    for ax, (title, actual, predicted) in zip(axes, panels):
        pred_rank = sorted(predicted, key=predicted.get, reverse=True)
        oracle_rank = sorted(actual, key=actual.get, reverse=True)
        x_pred, y_pred = curve(actual, pred_rank)
        x_oracle, y_oracle = curve(actual, oracle_rank)
        ax.plot(x_pred, y_pred, color="#1f77b4", lw=2, label="Model-ranked POIs")
        ax.plot(x_oracle, y_oracle, color="#333333", lw=1.5, ls="--", label="Oracle actual ranking")
        ax.set_xscale("log")
        ax.set_xlim(1, max(len(oracle_rank), len(pred_rank), 2))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Top-K POIs")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=True)
    axes[0].set_ylabel("Cumulative actual visit mass recovered")
    fig.suptitle("How Quickly the Ranked POI List Recovers Actual Demand", y=1.03)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_oos_residual_map(actual, predicted, temporal_status_by_pid, out_path):
    rows = []
    for pid in sorted(set(actual) | set(predicted)):
        coords = parse_poi_lat_lon(pid)
        if coords is None:
            continue
        lat, lon = coords
        a = actual.get(pid, 0.0)
        p = predicted.get(pid, 0.0)
        rows.append((lat, lon, a, p, p - a))
    if not rows:
        return

    data = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], dtype=float)
    lat = data[:, 0]
    lon = data[:, 1]
    actual_mass = data[:, 2]
    pred_mass = data[:, 3]
    residual = data[:, 4]
    mass = actual_mass + pred_mass
    lim = np.quantile(np.abs(residual), 0.985)
    lim = max(lim, 1e-6)
    max_mass = max(float(mass.max()), 1e-12) if mass.size else 1e-12
    sizes = 4 + 120 * np.sqrt(mass / max_mass)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    sc = ax.scatter(
        lon, lat, c=residual, s=sizes, cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
        alpha=0.68, linewidths=0, rasterized=True,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("2022 OOS Spatial Residuals by POI\nred = over-predicted, blue = under-predicted")
    ax.grid(alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted - actual visit mass")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_oos_spatial_hexbin_maps(actual, predicted, temporal_status_by_pid, poi_meta, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    arr = records_to_arrays(records, extent)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharex=True, sharey=True)
    panels = [
        ("Actual 2022 visit mass", arr["actual"], "magma", None),
        ("Predicted 2022 top-5 mass", arr["predicted"], "magma", None),
        ("Residual: predicted - actual", arr["residual"], "RdBu_r", "residual"),
    ]

    for ax, (title, values, cmap, mode) in zip(axes, panels):
        if mode == "residual":
            mask = np.abs(values) > 0
            if not mask.any():
                continue
            lim = max(float(np.quantile(np.abs(values[mask]), 0.985)), 1e-6)
            hb = ax.hexbin(
                arr["lon"][mask],
                arr["lat"][mask],
                C=values[mask],
                reduce_C_function=np.sum,
                gridsize=62,
                extent=extent,
                cmap=cmap,
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                mincnt=1,
            )
            cbar_label = "Summed residual mass"
        else:
            mask = values > 0
            if not mask.any():
                continue
            hb = ax.hexbin(
                arr["lon"][mask],
                arr["lat"][mask],
                C=values[mask],
                reduce_C_function=np.sum,
                gridsize=62,
                extent=extent,
                cmap=cmap,
                bins="log",
                mincnt=1,
            )
            cbar_label = "Summed visit mass (log color)"

        style_map_axis(ax, extent, title)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    top_idx = np.argsort(np.abs(arr["residual"]))[-10:]
    for idx in top_idx:
        axes[2].annotate(
            pretty_name(arr["pid"][idx], poi_meta),
            (arr["lon"][idx], arr["lat"][idx]),
            fontsize=5.5,
            alpha=0.85,
            xytext=(3, 3),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.75, lw=0),
        )

    fig.suptitle("2022 OOS Spatial Demand Maps (hexbin aggregation)", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_oos_3d_demand_surface(actual, predicted, temporal_status_by_pid, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    arr = records_to_arrays(records, extent)
    if len(arr["records"]) < 10:
        return

    x_edges = np.linspace(extent[0], extent[1], 70)
    y_edges = np.linspace(extent[2], extent[3], 70)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

    surfaces = []
    for values in [arr["actual"], arr["predicted"]]:
        hist, _, _ = np.histogram2d(arr["lon"], arr["lat"], bins=[x_edges, y_edges], weights=values)
        surfaces.append(gaussian_filter(hist, sigma=1.15) * 1e4)
    z_max = max(float(surfaces[0].max()), float(surfaces[1].max()), 1e-9)

    fig = plt.figure(figsize=(14, 6.2))
    for i, (title, surface, cmap) in enumerate([
        ("Actual 2022 visit mass surface", surfaces[0], "YlOrBr"),
        ("Predicted 2022 top-5 mass surface", surfaces[1], "PuBuGn"),
    ], start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_surface(
            x_grid,
            y_grid,
            surface,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
            rstride=1,
            cstride=1,
        )
        ax.contour(x_grid, y_grid, surface, zdir="z", offset=0, levels=9, cmap=cmap, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Mass density (x1e4)")
        ax.set_zlim(0, z_max * 1.08)
        ax.view_init(elev=37, azim=-122)
        ax.set_box_aspect((1.0, 1.0, 0.42))
        ax.grid(False)

    fig.suptitle("3D Demand Surface: Actual vs Predicted Spatial Intensity", y=0.98)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=250)
    plt.close(fig)


def plot_temporal_status_spatial_map(actual, predicted, temporal_status_by_pid, poi_meta, out_path):
    records = build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = spatial_extent(records)
    if not records or extent is None:
        return
    arr = records_to_arrays(records, extent)
    total_mass = arr["actual"] + arr["predicted"]
    max_mass = max(float(total_mass.max()), 1e-12) if total_mass.size else 1e-12
    sizes = 6 + 120 * np.sqrt(total_mass / max_mass)

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    status_style = {
        "stable": ("#9AA0A6", "o", 0.22, "Stable"),
        "disappeared": ("#E68632", "v", 0.72, "Disappeared"),
        "new": ("#2CA25F", "*", 0.95, "New"),
        "unknown": ("#756BB1", "s", 0.55, "Unknown"),
    }
    for status, (color, marker, alpha, label) in status_style.items():
        mask = arr["status"] == status
        if not mask.any():
            continue
        ax.scatter(
            arr["lon"][mask],
            arr["lat"][mask],
            s=sizes[mask],
            c=color,
            marker=marker,
            alpha=alpha,
            label=f"{label} ({int(mask.sum()):,})",
            linewidths=0.2,
            edgecolors="white" if status != "stable" else "none",
            rasterized=status == "stable",
        )

    new_idx = np.where(arr["status"] == "new")[0]
    for idx in new_idx[:12]:
        ax.annotate(
            pretty_name(arr["pid"][idx], poi_meta),
            (arr["lon"][idx], arr["lat"][idx]),
            fontsize=6,
            xytext=(4, 3),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.78, lw=0),
        )

    style_map_axis(ax, extent, "2022 OOS POIs by Temporal Status")
    ax.legend(loc="upper right", frameon=True, markerscale=1.2)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))


def plot_distance_decay_by_cluster(diag, out_path, title_tag, min_records=100, ncols=6):
    per_cluster = diag.get("per_cluster_distance") or {}
    labels = diag.get("distance", {}).get("labels") or []
    if not per_cluster or not labels:
        return

    clusters = [
        (c, d) for c, d in per_cluster.items()
        if d.get("n_records", 0) >= min_records
    ]
    clusters.sort(key=lambda kv: -kv[1]["n_records"])
    if not clusters:
        return

    n = len(clusters)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    x = np.arange(len(labels))

    for i, (cluster_key, d) in enumerate(clusters):
        ax = axes[i // ncols, i % ncols]
        a = np.asarray(d["actual_mass"], dtype=float)
        p = np.asarray(d["predicted_mass"], dtype=float)
        js = _js_divergence(a, p)
        shift = int(np.argmax(a)) - int(np.argmax(p)) if a.sum() > 0 and p.sum() > 0 else 0
        ax.plot(x, a, marker="o", ms=3, lw=1.5, color="#1f77b4", label="Actual")
        ax.plot(x, p, marker="s", ms=3, lw=1.5, color="#ff7f0e", label="Pred top-5")
        ax.set_title(f"cluster {cluster_key}  (n={d['n_records']})", fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        ax.text(0.02, 0.95, f"JS={js:.2f}\nshift={shift:+d}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8))

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("mass", fontsize=8)

    handles, h_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, h_labels, loc="upper right", ncol=2, frameon=True, fontsize=9)
    fig.suptitle(f"Per-cluster distance decay — {title_tag}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, out_path)
    plt.close(fig)


def plot_distance_decay(train_diag, test_diag, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (label, diag) in zip(axes, [("2021 validation", train_diag), ("2022 temporal OOS", test_diag)]):
        dist = diag["distance"]
        labels = dist["labels"]
        x = np.arange(len(labels))
        actual = np.array(dist["actual_mass"])
        predicted = np.array(dist["predicted_mass"])
        ax.plot(x, actual, marker="o", lw=2, color="#1f77b4", label="Actual true POI")
        ax.plot(x, predicted, marker="s", lw=2, color="#ff7f0e", label="Predicted top-5 mass")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Distance from inference origin")
        ax.set_title(label)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=True)
    axes[0].set_ylabel("Share of visit/probability mass")
    fig.suptitle("Distance Decay Diagnostic (Exp02 uses stop-origin coordinates)", y=1.03)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_temporal_status_mass(actual, predicted, temporal_status_by_pid, out_path):
    groups = ["stable", "new", "disappeared", "unknown"]
    actual_vals = [
        sum(value for pid, value in actual.items()
            if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == group)
        for group in groups
    ]
    pred_vals = [
        sum(value for pid, value in predicted.items()
            if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == group)
        for group in groups
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(groups))
    width = 0.36
    ax.bar(x - width / 2, actual_vals, width, label="Actual", color="#1f77b4")
    ax.bar(x + width / 2, pred_vals, width, label="Predicted", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(["Stable", "New", "Disappeared", "Unknown"])
    ax.set_ylabel("Share of OOS 2022 visit mass")
    ax.set_title("2022 OOS Mass by POI Temporal Status")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def resolve_snapshot_paths(exp_dir, args):
    manifest_path = args.snapshot_manifest or f"{exp_dir}/inputs/train_prior/manifest.json"
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    cbg_poi_dir = args.cbg_poi_dir or manifest.get("cbg_poi_dir") or f"{BASE}/cbg_poi"
    fallback_poi_csv = (
        args.fallback_poi_csv
        or manifest.get("fallback_poi_csv")
        or f"{BASE}/data/poi_unique_food_with_freq.csv"
    )
    return manifest_path, cbg_poi_dir, fallback_poi_csv


def fmt_metrics(metrics, label):
    if metrics is None:
        return f"  {label}: N/A"
    return (
        f"  {label}:\n"
        f"    Spearman:     {metrics['spearman']:.4f}\n"
        f"    Kendall:      {metrics['kendall']:.4f}\n"
        f"    JS Div:       {metrics['js_divergence']:.4f}\n"
        f"    NDCG@10:      {metrics['ndcg'][10]:.4f}\n"
        f"    NDCG@20:      {metrics['ndcg'][20]:.4f}\n"
        f"    NDCG@50:      {metrics['ndcg'][50]:.4f}\n"
        f"    Top10 recall: {metrics['topk'][10]['recall']:.4f}\n"
        f"    Top50 recall: {metrics['topk'][50]['recall']:.4f}\n"
        f"    POI coverage: {metrics['poi_coverage']:.4f}\n"
        f"    n_actual:     {metrics['n_actual_pois']:,}\n"
        f"    n_predicted:  {metrics['n_predicted_pois']:,}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp01", help="Experiment name under experiments/<exp>/")
    parser.add_argument("--pred_train", default=None)
    parser.add_argument("--pred_test", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--snapshot_manifest", default=None)
    parser.add_argument("--cbg_poi_dir", default=None)
    parser.add_argument("--fallback_poi_csv", default=None)
    parser.add_argument("--synthetic_unseen_poi_csv", default="",
                        help="Optional hidden-POI CSV from 16_prepare_synthetic_unseen.py.")
    args = parser.parse_args()

    exp_dir = f"{BASE}/experiments/{args.exp}"
    pred_train = args.pred_train or f"{exp_dir}/pred_insample_2021.jsonl"
    pred_test = args.pred_test or f"{exp_dir}/pred_oos_2022.jsonl"
    out_dir = args.out_dir or f"{exp_dir}/eval"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Visit-Level Temporal Evaluation: {args.exp} ===\n")
    manifest_path, cbg_poi_dir, fallback_poi_csv = resolve_snapshot_paths(exp_dir, args)
    print(f"Snapshot manifest: {manifest_path}")
    print(f"Cluster pools:     {cbg_poi_dir}")
    print(f"Fallback POIs:     {fallback_poi_csv}")

    print("\nLoading POI universe...")
    valid_set, name_to_valid, poi_meta = build_valid_poi_set(cbg_poi_dir, fallback_poi_csv)
    temporal_status_by_pid = load_temporal_status_map(valid_set, name_to_valid)

    print("\nLoading prediction records...")
    train_eval = load_visit_level_eval(pred_train, valid_set, name_to_valid)
    test_eval = load_visit_level_eval(pred_test, valid_set, name_to_valid)

    print("\nLoading figure diagnostics...")
    train_diag = load_visit_diagnostics(pred_train, valid_set, name_to_valid)
    test_diag = load_visit_diagnostics(pred_test, valid_set, name_to_valid)

    actual_2021 = train_eval["actual_top5"]
    pred_2021 = train_eval["predicted_top5"]
    actual_2022 = test_eval["actual_top5"]
    pred_2022 = test_eval["predicted_top5"]

    print("\nComputing metrics...")
    metrics_train = compute_metrics(actual_2021, pred_2021)
    metrics_test = compute_metrics(actual_2022, pred_2022)

    stable_set = {
        pid for pid in actual_2022
        if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == "stable"
    }
    new_set = {
        pid for pid in actual_2022
        if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == "new"
    }
    disappeared_set = {
        pid for pid in actual_2022
        if temporal_group(temporal_status_by_pid.get(pid, "unknown")) == "disappeared"
    }
    metrics_test_stable = subset_metrics(actual_2022, pred_2022, stable_set)
    lift = new_poi_lift(actual_2022, pred_2022, temporal_status_by_pid)

    synthetic_unseen_set = load_poi_subset_csv(args.synthetic_unseen_poi_csv)
    metrics_synthetic_unseen = subset_metrics(actual_2022, pred_2022, synthetic_unseen_set)
    synthetic_unseen_mass = subset_mass_summary(actual_2022, pred_2022, synthetic_unseen_set)

    borough_metrics = {}
    for boro_code, borough_name in BOROUGH_NAMES.items():
        actual_boro = test_eval["borough_actual_top5"].get(boro_code, {})
        pred_boro = test_eval["borough_predicted_top5"].get(boro_code, {})
        if actual_boro and pred_boro:
            borough_metrics[borough_name] = compute_metrics(actual_boro, pred_boro)

    polygon_metrics = {}
    for polygon_type, actual_poly in test_eval.get("polygon_actual_top5", {}).items():
        pred_poly = test_eval.get("polygon_predicted_top5", {}).get(polygon_type, {})
        if actual_poly and pred_poly:
            polygon_metrics[polygon_type] = compute_metrics(actual_poly, pred_poly)

    temporal_counts_raw = Counter(temporal_status_by_pid.get(pid, "unknown") for pid in valid_set)
    oos_status_counts = Counter(
        temporal_group(temporal_status_by_pid.get(pid, "unknown")) for pid in actual_2022
    )

    report = []
    report.append(f"=== Visit-Level Temporal Evaluation: {args.exp} ===\n")
    report.append("Ground truth source: prediction records (`true_poi_id`) on the same evaluated visit subset.")
    report.append("Formal aggregate metrics use top-5 probability mass. The 4-panel scatter uses top-1 visit mass.")
    report.append("")
    report.append("--- Data Summary ---")
    report.append(f"  Valid POI universe: {len(valid_set):,}")
    report.append(f"  Snapshot manifest: {manifest_path}")
    report.append(f"  2021 prediction rows: {train_eval['stats']['records_total']:,}")
    report.append(f"  2022 prediction rows: {test_eval['stats']['records_total']:,}")
    report.append(f"  2021 truth valid: {train_eval['stats']['truth_valid']:,} "
                  f"({train_eval['stats']['truth_in_universe_rate']:.1%})")
    report.append(f"  2022 truth valid: {test_eval['stats']['truth_valid']:,} "
                  f"({test_eval['stats']['truth_in_universe_rate']:.1%})")
    report.append(f"  2021 truth inside prediction universe: "
                  f"{train_eval['stats']['truth_in_prediction_universe']:,} "
                  f"({train_eval['stats']['truth_in_prediction_universe_rate']:.1%})")
    report.append(f"  2022 truth inside prediction universe: "
                  f"{test_eval['stats']['truth_in_prediction_universe']:,} "
                  f"({test_eval['stats']['truth_in_prediction_universe_rate']:.1%})")
    report.append(f"  2021 eval subset with usable top_5: {train_eval['stats']['records_with_predicted_top5']:,} "
                  f"({train_eval['stats']['predicted_top5_rate']:.1%})")
    report.append(f"  2022 eval subset with usable top_5: {test_eval['stats']['records_with_predicted_top5']:,} "
                  f"({test_eval['stats']['predicted_top5_rate']:.1%})")
    report.append(f"  2021 origin modes: {train_diag['stats']['origin_modes']}")
    report.append(f"  2022 origin modes: {test_diag['stats']['origin_modes']}")
    report.append(f"  2022 observed POI status: stable={oos_status_counts['stable']:,}, "
                  f"new={oos_status_counts['new']:,}, disappeared={oos_status_counts['disappeared']:,}, "
                  f"unknown={oos_status_counts['unknown']:,}")
    report.append(f"  Universe temporal labels: "
                  f"stable={temporal_counts_raw['stable']:,}, "
                  f"new={temporal_counts_raw['new']:,}, "
                  f"closed_2022={temporal_counts_raw['closed_2022']:,}, "
                  f"closed_before_study={temporal_counts_raw['closed_before_study']:,}, "
                  f"unknown={temporal_counts_raw['unknown']:,}")
    report.append("")
    report.append("--- Overall Metrics ---")
    report.append(fmt_metrics(metrics_train, "In-sample 2021 val (top-5 mass)"))
    report.append("")
    report.append(fmt_metrics(metrics_test, "Out-of-sample 2022 (top-5 mass)"))
    report.append("")
    if metrics_test_stable:
        report.append(fmt_metrics(metrics_test_stable, "OOS 2022 stable POIs only"))
        report.append("")

    if synthetic_unseen_set:
        report.append("--- Synthetic Unseen POIs ---")
        report.append(f"  Hidden POI file: {args.synthetic_unseen_poi_csv}")
        if synthetic_unseen_mass:
            report.append(f"  Hidden POIs: {synthetic_unseen_mass['n_subset_pois']:,}; "
                          f"actual observed={synthetic_unseen_mass['n_actual_subset_pois']:,}, "
                          f"predicted={synthetic_unseen_mass['n_predicted_subset_pois']:,}")
            report.append(f"  OOS mass: actual={synthetic_unseen_mass['actual_mass']:.4f}, "
                          f"predicted={synthetic_unseen_mass['predicted_mass']:.4f}, "
                          f"ratio={synthetic_unseen_mass['mass_ratio_pred_over_actual']}")
        report.append(fmt_metrics(metrics_synthetic_unseen, "OOS 2022 synthetic unseen POIs only"))
        report.append("")

    report.append("--- Visit-Level Accuracy ---")
    report.append(f"  In-sample 2021 val: top-1={train_eval['accuracy']['top1_accuracy']:.4f}, "
                  f"top-5={train_eval['accuracy']['top5_accuracy']:.4f}")
    report.append(f"  Out-of-sample 2022: top-1={test_eval['accuracy']['top1_accuracy']:.4f}, "
                  f"top-5={test_eval['accuracy']['top5_accuracy']:.4f}")
    report.append("")

    report.append("--- New POI Lift ---")
    if lift is None:
        report.append("  N/A (no actual mass on POIs labeled `new` within the eval subset)")
    else:
        report.append(f"  Lift = pred_mass(new) / actual_mass(new) = {lift:.4f}")
        report.append("  (1.0 = calibrated; <1 = under-predicting new POIs)")
    report.append("")

    report.append("--- Per-Borough OOS Metrics ---")
    if borough_metrics:
        for borough_name, metrics in borough_metrics.items():
            if metrics:
                report.append(f"  {borough_name}: Spearman={metrics['spearman']:.4f}, "
                              f"NDCG@10={metrics['ndcg'][10]:.4f}")
    else:
        report.append("  N/A (prediction files did not include usable borough labels)")
    report.append("")

    report.append("--- Polygon-Type OOS Metrics ---")
    if polygon_metrics:
        for polygon_type, metrics in sorted(polygon_metrics.items()):
            if metrics:
                report.append(f"  {polygon_type}: Spearman={metrics['spearman']:.4f}, "
                              f"NDCG@10={metrics['ndcg'][10]:.4f}, "
                              f"n_actual={metrics['n_actual_pois']:,}")
    else:
        report.append("  N/A (prediction files did not include true_polygon_type)")
    report.append("")

    if metrics_train and metrics_test:
        report.append("--- Train -> Test Delta ---")
        report.append(f"  Spearman: {metrics_train['spearman']:.4f} -> {metrics_test['spearman']:.4f} "
                      f"(delta={metrics_test['spearman'] - metrics_train['spearman']:+.4f})")
        report.append(f"  NDCG@10:  {metrics_train['ndcg'][10]:.4f} -> {metrics_test['ndcg'][10]:.4f} "
                      f"(delta={metrics_test['ndcg'][10] - metrics_train['ndcg'][10]:+.4f})")

    text = "\n".join(report)
    print("\n" + text)

    with open(f"{out_dir}/temporal_report.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\nGenerating figures...")
    fig_dir = f"{out_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plot_temporal_scatter_4panel(
        train_eval["actual_top1"],
        train_eval["predicted_top1"],
        test_eval["actual_top1"],
        test_eval["predicted_top1"],
        temporal_status_by_pid,
        poi_meta,
        f"{fig_dir}/temporal_scatter_4panel.png",
        f"{fig_dir}/temporal_scatter_4panel.pdf",
    )
    plot_borough_metrics(borough_metrics, f"{fig_dir}/borough_metrics.png")
    plot_top_pois(actual_2022, pred_2022, poi_meta, f"{fig_dir}/top_pois_2022.png", k=20)
    plot_metrics_comparison(metrics_train, metrics_test, f"{fig_dir}/metrics_comparison.png")
    plot_top5_mass_hexbin(
        actual_2021,
        pred_2021,
        actual_2022,
        pred_2022,
        metrics_train,
        metrics_test,
        f"{fig_dir}/top5_mass_hexbin_2panel.png",
    )
    plot_probability_calibration(
        train_diag,
        test_diag,
        f"{fig_dir}/probability_calibration.png",
    )
    plot_cumulative_mass_recovery(
        actual_2021,
        pred_2021,
        actual_2022,
        pred_2022,
        f"{fig_dir}/cumulative_mass_recovery.png",
    )
    plot_oos_residual_map(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/oos_residual_map.png",
    )
    plot_oos_spatial_hexbin_maps(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        poi_meta,
        f"{fig_dir}/oos_spatial_hexbin_maps.png",
    )
    plot_oos_3d_demand_surface(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/oos_3d_demand_surface.png",
    )
    plot_temporal_status_spatial_map(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        poi_meta,
        f"{fig_dir}/temporal_status_spatial_map.png",
    )
    write_interactive_spatial_grid_html(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/oos_spatial_grid_map.html",
    )
    write_temporal_status_poi_html(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        poi_meta,
        f"{fig_dir}/temporal_status_spatial_map.html",
    )
    write_plotly_3d_surface_html(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/oos_3d_demand_surface.html",
    )
    write_deckgl_3d_columns_html(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/oos_deckgl_3d_columns.html",
    )
    write_leaflet_spatial_grid_map_html(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/oos_leaflet_spatial_grid_map.html",
    )
    write_leaflet_temporal_status_map_html(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        poi_meta,
        f"{fig_dir}/temporal_status_leaflet_map.html",
    )
    plot_distance_decay(
        train_diag,
        test_diag,
        f"{fig_dir}/distance_decay_by_year.png",
    )
    plot_distance_decay_by_cluster(
        train_diag,
        f"{fig_dir}/distance_decay_by_cluster_insample_2021.png",
        "2021 validation",
    )
    plot_distance_decay_by_cluster(
        test_diag,
        f"{fig_dir}/distance_decay_by_cluster_oos_2022.png",
        "2022 temporal OOS",
    )
    plot_temporal_status_mass(
        actual_2022,
        pred_2022,
        temporal_status_by_pid,
        f"{fig_dir}/temporal_status_mass_oos.png",
    )

    out_json = {
        "experiment": args.exp,
        "evaluation_mode": "visit_level_sampled",
        "snapshot_manifest": manifest_path,
        "valid_universe": {
            "n_pois": len(valid_set),
            "temporal_status_counts_raw": dict(temporal_counts_raw),
        },
        "data_summary": {
            "train": train_eval["stats"],
            "test": test_eval["stats"],
            "oos_observed_status_counts": dict(oos_status_counts),
        },
        "in_sample_2021_val": metrics_train,
        "oos_2022": metrics_test,
        "oos_2022_stable": metrics_test_stable,
        "oos_2022_synthetic_unseen": metrics_synthetic_unseen,
        "synthetic_unseen_mass": synthetic_unseen_mass,
        "new_poi_lift": lift,
        "borough_metrics": borough_metrics,
        "polygon_metrics": polygon_metrics,
        "visit_accuracy": {
            "in_sample_2021_val": train_eval["accuracy"],
            "oos_2022": test_eval["accuracy"],
        },
        "figure_diagnostics": {
            "in_sample_2021_val": train_diag,
            "oos_2022": test_diag,
        },
    }
    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, default=str)

    print(f"  Saved report to {out_dir}/temporal_report.txt")
    print(f"  Saved metrics to {out_dir}/metrics.json")
    print(f"  Saved figures to {fig_dir}/")


if __name__ == "__main__":
    main()
