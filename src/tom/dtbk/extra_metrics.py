"""Additional metrics not in sft_09:
 - Trip-level top-1/5 accuracy + MRR@5 (only for pred files with true_poi_id)
 - Distribution-level top-1 match, top-K overlap, top-K mass capture
 - Actual-mass-weighted Pearson/Spearman
"""
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
from sft_09_temporal_eval import (
    load_valid_poi_ids, load_a_work, build_actual_dist, load_predictions,
    build_predicted_dist,
)
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr


def trip_level_topk(pred_records):
    """For pred files with `true_poi_id` field: top-1, top-5, MRR@5."""
    n = 0; t1 = 0; t5 = 0; mrr = 0.0
    for r in pred_records:
        true_pid = r.get("true_poi_id")
        if not true_pid:
            return None  # not trip-level
        top5 = [e["poi"] for e in (r.get("top_5") or [])]
        if not top5: continue
        n += 1
        if top5[0] == true_pid: t1 += 1
        if true_pid in top5:
            t5 += 1
            mrr += 1.0 / (top5.index(true_pid) + 1)
    if n == 0: return None
    return {"trip_top1": t1/n, "trip_top5": t5/n, "trip_mrr5": mrr/n, "n_trips": n}


def distribution_extras(act, pred):
    """Distribution-level extras usable for any pred schema."""
    # Sort actual POIs by mass descending
    act_sorted = sorted(act.items(), key=lambda x: -x[1])
    pred_sorted = sorted(pred.items(), key=lambda x: -x[1])
    a_pois = [p for p, _ in act_sorted]
    p_pois = [p for p, _ in pred_sorted]

    out = {}
    out["top1_match"] = int(a_pois[0] == p_pois[0]) if a_pois and p_pois else 0
    for k in (5, 10, 20):
        a_set = set(a_pois[:k]); p_set = set(p_pois[:k])
        out[f"top{k}_overlap"] = len(a_set & p_set) / k if k else 0.0
        # mass capture: predicted_top_K as fraction of ACTUAL mass
        out[f"top{k}_actual_mass_captured"] = sum(act.get(pid, 0) for pid in p_pois[:k])

    # Weighted correlations: weight each POI by its actual mass
    all_pois = sorted(set(act) | set(pred))
    a = np.array([act.get(p, 0) for p in all_pois])
    p = np.array([pred.get(p, 0) for p in all_pois])
    mask = a > 0
    a_m = a[mask]; p_m = p[mask]
    if mask.sum() >= 2:
        # Weighted Pearson using actual mass as weights
        w = a_m
        wm_a = np.average(a_m, weights=w)
        wm_p = np.average(p_m, weights=w)
        cov = np.average((a_m-wm_a)*(p_m-wm_p), weights=w)
        var_a = np.average((a_m-wm_a)**2, weights=w)
        var_p = np.average((p_m-wm_p)**2, weights=w)
        out["weighted_pearson_by_actual"] = float(cov / np.sqrt(var_a * var_p)) if var_a*var_p > 0 else float("nan")
        # Weighted Spearman (using ranks, weighted by actual)
        ra = np.argsort(np.argsort(-a_m)).astype(float)
        rp = np.argsort(np.argsort(-p_m)).astype(float)
        wm_ra = np.average(ra, weights=w); wm_rp = np.average(rp, weights=w)
        covr = np.average((ra-wm_ra)*(rp-wm_rp), weights=w)
        var_ra = np.average((ra-wm_ra)**2, weights=w)
        var_rp = np.average((rp-wm_rp)**2, weights=w)
        out["weighted_spearman_by_actual"] = float(covr / np.sqrt(var_ra*var_rp)) if var_ra*var_rp > 0 else float("nan")
    return out


def process(label, pred_path, a_work_path, valid_poi_path, display_names_csv=None):
    BASE = Path("/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
    valid, n2v = load_valid_poi_ids(BASE/valid_poi_path)
    df = load_a_work(BASE/a_work_path, valid_poi_ids=valid, name_to_valid=n2v)
    act = build_actual_dist(df)

    display_to_poi = {}
    if display_names_csv:
        dn = pd.read_csv(BASE/display_names_csv)
        display_to_poi = {str(r.display_id).strip().lower(): str(r.poi_id).strip()
                          for r in dn.itertuples() if str(r.display_id) and str(r.poi_id)}
    recs = load_predictions(pred_path)
    pred = build_predicted_dist(recs, valid_poi_ids=valid, name_to_valid=n2v, display_to_poi=display_to_poi)

    trip = trip_level_topk(recs)
    dist = distribution_extras(act, pred)

    row = {"method": label, "n_records": len(recs)}
    if trip:
        row.update(trip)
    row.update(dist)
    return row


A_WORK_TEST = "a_work_q1_polygon.csv"
VALID_POI_TEST = "poi_unique_food_with_freq_extended.csv"
DISPLAY = "poi_display_names_extended_coordnames.csv"
EXP11 = Path("/home/shawn/testing/alex/hpc/baselines_exp11")

rows = []

# Exp 11 LLM (no true_poi_id, only distribution extras)
rows.append(process("Llama_Exp11_historical",
                    "/home/shawn/testing/review/dtbk/dtbk-sft-half-year/pred_q12022_coordnames.jsonl",
                    A_WORK_TEST, VALID_POI_TEST, DISPLAY))

rows.append(process("Llama_v2_repeats3",
                    "/home/shawn/testing/alex/hpc/pred_q12022_llama_v2_coord.jsonl",
                    A_WORK_TEST, VALID_POI_TEST, DISPLAY))

# 8 baselines (have true_poi_id → trip-level works)
for m in ["frequency","gravity","huff","mnl_grid","mnl_rich","deep_gravity","lgb_rank","xgb_rank"]:
    rows.append(process(f"{m}", str(EXP11/m/"pred_oos_2022.jsonl"),
                        A_WORK_TEST, VALID_POI_TEST, DISPLAY))

# Print table
print()
print(f"{'method':<25} {'trip_top1':>10} {'trip_top5':>10} {'trip_mrr5':>10} "
      f"{'top1_m':>7} {'top5_ov':>8} {'top5_mass':>10} {'wPearson':>9} {'wSpear':>9}")
print("-"*115)
for r in rows:
    t1 = r.get("trip_top1","-"); t5 = r.get("trip_top5","-"); mr = r.get("trip_mrr5","-")
    def fmt(v, w=10, d=4): return f"{v:>{w}.{d}f}" if isinstance(v,(int,float)) and not isinstance(v,bool) else f"{str(v):>{w}}"
    print(f"{r['method']:<25} {fmt(t1)} {fmt(t5)} {fmt(mr)} "
          f"{r['top1_match']:>7} {r['top5_overlap']:>8.4f} "
          f"{r['top5_actual_mass_captured']:>10.4f} "
          f"{r.get('weighted_pearson_by_actual',float('nan')):>9.4f} "
          f"{r.get('weighted_spearman_by_actual',float('nan')):>9.4f}")

# Save JSON
import json
out = {"rows": rows}
Path("/home/shawn/testing/alex/hpc/extra_metrics_dtbk_exp11.json").write_text(json.dumps(out, indent=2))
print("\nSaved /home/shawn/testing/alex/hpc/extra_metrics_dtbk_exp11.json")
