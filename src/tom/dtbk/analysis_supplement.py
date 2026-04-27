"""Supplementary analysis: Pearson correlation + per-area distance-bias heatmap.

Designed for pred JSONLs that contain `candidates[].dist_m + dist_m for true POI`
(both nyc_metro and dtbk-via-build_infer_records schemas). Area clusters are
k-means on the origin centroids of work_cbg groups (20 clusters by default).

Outputs:
  - metrics.json  : Spearman + Pearson + Kendall + NDCG@{10,20} + JS + POI coverage
  - area_clusters.csv : work_cbg -> area_id mapping (saved once, reusable)
  - area_distance_bias.csv : (area_id, dist_bin_idx, actual_mass, predicted_mass, residual)
  - area_distance_bias.png : heatmap, rows=area, cols=dist bin, cell=residual

Usage (fit area clusters on first run, reuse for rest):
  python analysis_supplement.py --pred A.jsonl --out A_out/ --fit_areas
  python analysis_supplement.py --pred B.jsonl --out B_out/ --areas_from A_out/area_clusters.csv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau


DIST_BINS = [0, 50, 100, 200, 500, 1000, 1500, 3000, 5000, 10000, 20000, float("inf")]
DIST_LABELS = ["0-50", "50-100", "100-200", "200-500", "500-1k", "1-1.5k",
               "1.5-3k", "3-5k", "5-10k", "10-20k", "20k+"]


def load_pred(path):
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def origin_of(rec):
    o = rec.get("origin")
    if isinstance(o, dict):
        return o.get("lat"), o.get("lon")
    return rec.get("origin_lat"), rec.get("origin_lon")


def fit_area_clusters(records, n=20, seed=0):
    """K-means on origin (mean per work_cbg if available, else per-record).
    Returns (key_to_area dict, centroids array, key_fn callable)."""
    from sklearn.cluster import KMeans
    cbg_coords = defaultdict(list)
    has_cbg = False
    for r in records:
        cbg = r.get("work_cbg") or ""
        if cbg:
            has_cbg = True
        lat, lon = origin_of(r)
        if lat is None or lon is None:
            continue
        # Bucket records without work_cbg by their own (rounded) origin
        key = cbg or f"o_{float(lat):.4f}_{float(lon):.4f}"
        cbg_coords[key].append((float(lat), float(lon)))

    keys = sorted(cbg_coords)
    if not keys:
        raise SystemExit("No records with origin lat/lon — cannot fit areas")
    centroids = np.array([[np.mean([p[0] for p in cbg_coords[k]]),
                           np.mean([p[1] for p in cbg_coords[k]])] for k in keys])
    km = KMeans(n_clusters=min(n, len(keys)), random_state=seed, n_init=10).fit(centroids)
    return dict(zip(keys, km.labels_)), km.cluster_centers_, has_cbg


def record_key(rec, has_cbg):
    cbg = rec.get("work_cbg") or ""
    if has_cbg and cbg:
        return cbg
    lat, lon = origin_of(rec)
    if lat is None or lon is None:
        return None
    return f"o_{float(lat):.4f}_{float(lon):.4f}"


def compute_overall_metrics(records, valid_poi_ids=None):
    """POI-level actual vs predicted mass. Returns spearman, pearson, kendall, ndcg, js, coverage."""
    actual = defaultdict(float)
    predicted = defaultdict(float)
    n_actual = 0
    for r in records:
        tpid = r.get("true_poi_id")
        if tpid and (valid_poi_ids is None or tpid in valid_poi_ids):
            actual[tpid] += 1.0
            n_actual += 1
        s = 0.0
        for e in r.get("top_5") or []:
            s += max(0.0, float(e.get("prob", 0)))
        if s <= 0:
            continue
        for e in r.get("top_5") or []:
            pid = e.get("poi")
            if pid and (valid_poi_ids is None or pid in valid_poi_ids):
                predicted[pid] += max(0.0, float(e.get("prob", 0))) / s

    if n_actual == 0:
        return None
    actual = {p: v / n_actual for p, v in actual.items()}
    total_pred = sum(predicted.values())
    if total_pred > 0:
        predicted = {p: v / total_pred for p, v in predicted.items()}

    all_pois = sorted(set(actual) | set(predicted))
    a = np.array([actual.get(p, 0) for p in all_pois])
    p = np.array([predicted.get(p, 0) for p in all_pois])

    mask = a > 0
    if mask.sum() >= 2:
        sp, _ = spearmanr(a[mask], p[mask])
        pe, _ = pearsonr(a[mask], p[mask])
        ke, _ = kendalltau(a[mask], p[mask])
    else:
        sp = pe = ke = float("nan")

    eps = 1e-12
    m = 0.5 * (a + p)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_ap = np.where(a > 0, a * np.log((a + eps) / (m + eps)), 0).sum()
        kl_pm = np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0).sum()
    js = 0.5 * (kl_ap + kl_pm)

    def ndcg_at(k):
        idx_a = np.argsort(-a)[:k]
        idx_p = np.argsort(-p)[:k]
        dcg = sum(a[idx_p[i]] / np.log2(i + 2) for i in range(len(idx_p)))
        idcg = sum(a[idx_a[i]] / np.log2(i + 2) for i in range(len(idx_a)))
        return dcg / idcg if idcg > 0 else 0.0

    return {
        "n_actual_pois": int((a > 0).sum()),
        "n_predicted_pois": int((p > 0).sum()),
        "spearman": float(sp),
        "pearson": float(pe),
        "kendall": float(ke),
        "ndcg_10": float(ndcg_at(10)),
        "ndcg_20": float(ndcg_at(20)),
        "js_divergence": float(js),
        "poi_coverage": float(sum(1 for pid in actual if predicted.get(pid, 0) > 0) / max(len(actual), 1)),
    }


def per_area_distance_bias(records, key_to_area, n_areas, has_cbg):
    """
    For each (area, dist_bin): accumulate actual mass (1 per trip, bin by true POI dist)
    and predicted mass (prob-weighted, bin by each top_5 POI's distance).

    Predicted POI distance is looked up in candidates[] if available; else computed
    from origin and POI lat/lon parsed from pipe-format poi_id.
    """
    actual_mass = np.zeros((n_areas, len(DIST_LABELS)))
    predicted_mass = np.zeros((n_areas, len(DIST_LABELS)))
    area_trip_count = np.zeros(n_areas)

    for r in records:
        k = record_key(r, has_cbg)
        if k is None or k not in key_to_area:
            continue
        area = key_to_area[k]
        area_trip_count[area] += 1

        cand_dist = {c["poi_id"]: float(c["dist_m"]) for c in (r.get("candidates") or [])
                     if c.get("poi_id") and c.get("dist_m") is not None}

        tpid = r.get("true_poi_id")
        if tpid and tpid in cand_dist:
            bin_idx = np.searchsorted(DIST_BINS, cand_dist[tpid], side="right") - 1
            bin_idx = max(0, min(bin_idx, len(DIST_LABELS) - 1))
            actual_mass[area, bin_idx] += 1.0

        s = sum(max(0.0, float(e.get("prob", 0))) for e in r.get("top_5") or [])
        if s <= 0:
            continue
        for e in r.get("top_5") or []:
            pid = e.get("poi")
            d = cand_dist.get(pid)
            if d is None:
                continue
            bin_idx = np.searchsorted(DIST_BINS, d, side="right") - 1
            bin_idx = max(0, min(bin_idx, len(DIST_LABELS) - 1))
            predicted_mass[area, bin_idx] += max(0.0, float(e.get("prob", 0))) / s

    with np.errstate(divide="ignore", invalid="ignore"):
        actual_frac = np.where(area_trip_count[:, None] > 0,
                               actual_mass / area_trip_count[:, None], 0.0)
        predicted_frac = np.where(area_trip_count[:, None] > 0,
                                  predicted_mass / area_trip_count[:, None], 0.0)
    residual = predicted_frac - actual_frac
    return actual_frac, predicted_frac, residual, area_trip_count


def plot_heatmap(residual, area_trip_count, centroids, out_path, title):
    order = np.argsort(-area_trip_count)
    r_ord = residual[order]
    counts_ord = area_trip_count[order]
    centroids_ord = centroids[order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(order))))
    vmax = float(np.nanmax(np.abs(r_ord))) or 1e-6
    im = ax.imshow(r_ord, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DIST_LABELS)))
    ax.set_xticklabels(DIST_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"A{i:02d} n={int(c):,} ({lat:.3f},{lon:.3f})"
                        for i, c, (lat, lon) in zip(order, counts_ord, centroids_ord)])
    ax.set_xlabel("Distance bin (m)")
    ax.set_ylabel("Area cluster (sorted by trip count)")
    ax.set_title(title + "  (residual = predicted - actual, red = over, blue = under)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="residual mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fit_areas", action="store_true",
                   help="Compute area clusters on this pred and save to out/area_clusters.csv")
    p.add_argument("--areas_from", default="",
                   help="Load pre-computed area_clusters.csv instead of fitting")
    p.add_argument("--n_areas", type=int, default=20)
    p.add_argument("--tag", default="",
                   help="Label for the plot title")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.pred}")
    records = load_pred(args.pred)
    print(f"  {len(records):,} records")

    if args.fit_areas:
        key_to_area, centroids, has_cbg = fit_area_clusters(records, n=args.n_areas)
        pd.DataFrame([{"key": k, "has_cbg": int(has_cbg), "area_id": int(a),
                       "centroid_lat": centroids[a][0], "centroid_lon": centroids[a][1]}
                      for k, a in key_to_area.items()]).to_csv(out / "area_clusters.csv", index=False)
        print(f"  fit {len(set(key_to_area.values()))} area clusters (has_cbg={has_cbg})")
    elif args.areas_from:
        df = pd.read_csv(args.areas_from)
        has_cbg = bool(df.has_cbg.iloc[0])
        key_to_area = dict(zip(df.key.astype(str), df.area_id.astype(int)))
        centroids = np.zeros((df.area_id.max() + 1, 2))
        for aid, sub in df.groupby("area_id"):
            centroids[aid] = [sub.centroid_lat.iloc[0], sub.centroid_lon.iloc[0]]
    else:
        raise SystemExit("Pass --fit_areas or --areas_from")

    print("Computing overall metrics...")
    m = compute_overall_metrics(records)
    print(json.dumps(m, indent=2))

    print("Computing per-area distance bias...")
    n_areas = centroids.shape[0]
    actual_frac, predicted_frac, residual, counts = per_area_distance_bias(records, key_to_area, n_areas, has_cbg)

    rows = []
    for a in range(n_areas):
        for b, lbl in enumerate(DIST_LABELS):
            rows.append({
                "area_id": a,
                "centroid_lat": centroids[a][0],
                "centroid_lon": centroids[a][1],
                "dist_bin_idx": b,
                "dist_bin": lbl,
                "actual_mass": float(actual_frac[a, b]),
                "predicted_mass": float(predicted_frac[a, b]),
                "residual": float(residual[a, b]),
                "area_trips": int(counts[a]),
            })
    pd.DataFrame(rows).to_csv(out / "area_distance_bias.csv", index=False)

    tag = args.tag or Path(args.pred).stem
    plot_heatmap(residual, counts, centroids, out / "area_distance_bias.png", tag)

    with open(out / "metrics.json", "w") as f:
        json.dump({"pred_file": args.pred, "tag": tag, **m}, f, indent=2)
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
