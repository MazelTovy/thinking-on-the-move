"""Final unified comparison table across all 10 methods on DTBK Exp 11 eval.
Includes: Spearman / Pearson / Kendall / NDCG@10/20 / JS / Coverage /
          top5_overlap / top5_mass_captured / weighted_pearson / weighted_spearman /
          trip_level top-1/top-5/MRR (baselines only — LLM can't).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
from sft_09_temporal_eval import (
    load_valid_poi_ids, load_a_work, build_actual_dist, load_predictions,
    build_predicted_dist, compute_metrics
)
import numpy as np, pandas as pd
from scipy.stats import pearsonr


def trip_topk(recs):
    n = t1 = t5 = 0; mrr = 0.0
    for r in recs:
        t = r.get("true_poi_id")
        if not t:
            return None
        top5 = [e["poi"] for e in (r.get("top_5") or [])]
        if not top5: continue
        n += 1
        if top5[0] == t: t1 += 1
        if t in top5:
            t5 += 1
            mrr += 1.0 / (top5.index(t) + 1)
    if n == 0: return None
    return {"tt1": t1/n, "tt5": t5/n, "tmrr": mrr/n, "n_trips": n}


def dist_extras(act, pred):
    a_sorted = sorted(act.items(), key=lambda x: -x[1])
    p_sorted = sorted(pred.items(), key=lambda x: -x[1])
    a_pois = [p for p, _ in a_sorted]
    p_pois = [p for p, _ in p_sorted]
    out = {}
    out["top5_overlap"] = len(set(a_pois[:5]) & set(p_pois[:5])) / 5 if a_pois and p_pois else 0
    out["top5_mass_captured"] = sum(act.get(pid, 0) for pid in p_pois[:5])
    all_pois = sorted(set(act) | set(pred))
    a = np.array([act.get(p, 0) for p in all_pois])
    p = np.array([pred.get(p, 0) for p in all_pois])
    mask = a > 0
    if mask.sum() >= 2:
        w = a[mask]
        ma = np.average(a[mask], weights=w); mp = np.average(p[mask], weights=w)
        cov = np.average((a[mask]-ma)*(p[mask]-mp), weights=w)
        va = np.average((a[mask]-ma)**2, weights=w); vp = np.average((p[mask]-mp)**2, weights=w)
        out["wPe"] = float(cov/np.sqrt(va*vp)) if va*vp > 0 else float("nan")
        ra = np.argsort(np.argsort(-a[mask])).astype(float)
        rp = np.argsort(np.argsort(-p[mask])).astype(float)
        mra = np.average(ra, weights=w); mrp = np.average(rp, weights=w)
        c2 = np.average((ra-mra)*(rp-mrp), weights=w)
        vra = np.average((ra-mra)**2, weights=w); vrp = np.average((rp-mrp)**2, weights=w)
        out["wSp"] = float(c2/np.sqrt(vra*vrp)) if vra*vrp > 0 else float("nan")
    return out


def process(label, pred_path):
    BASE = Path("/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
    valid, n2v = load_valid_poi_ids(BASE/"poi_unique_food_with_freq_extended.csv")
    df = load_a_work(BASE/"a_work_q1_polygon.csv", valid_poi_ids=valid, name_to_valid=n2v)
    act = build_actual_dist(df)
    dn = pd.read_csv(BASE/"poi_display_names_extended_coordnames.csv")
    display_to_poi = {str(r.display_id).strip().lower(): str(r.poi_id).strip()
                      for r in dn.itertuples() if str(r.display_id) and str(r.poi_id)}
    recs = load_predictions(pred_path)
    pred = build_predicted_dist(recs, valid_poi_ids=valid, name_to_valid=n2v, display_to_poi=display_to_poi)
    m = compute_metrics(act, pred)
    all_pois = sorted(set(act) | set(pred))
    a = np.array([act.get(p, 0) for p in all_pois])
    p = np.array([pred.get(p, 0) for p in all_pois])
    mask = a > 0
    pe, _ = pearsonr(a[mask], p[mask]) if mask.sum() >= 2 else (float("nan"), None)
    ndcg = m.get("ndcg", {})
    row = {
        "method": label, "n_recs": len(recs), "n_POI": m["n_actual_pois"],
        "Sp": m["spearman"], "Pe": float(pe),
        "Ke": m["kendall"],
        "NDCG10": ndcg.get(10, ndcg.get("10", 0)),
        "NDCG20": ndcg.get(20, ndcg.get("20", 0)),
        "JS": m["js_divergence"],
        "COV": m["poi_coverage"],
    }
    row.update(dist_extras(act, pred))
    tl = trip_topk(recs)
    if tl: row.update(tl)
    return row


EXP11 = Path("/home/shawn/testing/alex/hpc/baselines_exp11")
WORK = Path("/home/shawn/testing/alex/hpc")
SRC = Path("/home/shawn/testing/review/dtbk/dtbk-sft-half-year")

rows = []
# LLM methods (no true_poi_id → no trip-level)
rows.append(process("Llama Exp 11 (hist.)", str(SRC/"pred_q12022_coordnames.jsonl")))
rows.append(process("Llama v2 (rep=3)",    str(WORK/"pred_q12022_llama_v2_coord.jsonl")))
rows.append(process("Qwen v2 (rep=3)",     str(WORK/"pred_q12022_qwen_coord.jsonl")))
# 8 baselines (with trip-level)
for m in ["frequency","gravity","huff","mnl_grid","mnl_rich","deep_gravity","lgb_rank","xgb_rank"]:
    rows.append(process(m, str(EXP11/m/"pred_oos_2022.jsonl")))

# Print sorted by Spearman desc within LLM vs baseline groups
print(f"\n{'method':<22} {'Sp':>6} {'Pe':>6} {'Ke':>6} {'NDCG10':>7} {'NDCG20':>7} {'JS':>6} "
      f"{'COV':>6} {'t5ov':>6} {'t5mass':>7} {'wPe':>6} {'wSp':>6} "
      f"{'tt1':>6} {'tt5':>6} {'tmrr':>6}  n_POI")
print("-"*160)

def fmt(x, w=6, d=4):
    return f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else f"{str(x):>{w}}"

for r in rows:
    print(f"{r['method']:<22} "
          f"{fmt(r['Sp'])} {fmt(r['Pe'])} {fmt(r['Ke'])} "
          f"{fmt(r['NDCG10'],7)} {fmt(r['NDCG20'],7)} {fmt(r['JS'])} "
          f"{fmt(r['COV'])} {fmt(r.get('top5_overlap',float('nan')))} "
          f"{fmt(r.get('top5_mass_captured',float('nan')),7)} "
          f"{fmt(r.get('wPe',float('nan')))} {fmt(r.get('wSp',float('nan')))} "
          f"{fmt(r.get('tt1','-'))} {fmt(r.get('tt5','-'))} {fmt(r.get('tmrr','-'))}  "
          f"{r['n_POI']}")

Path("/home/shawn/testing/alex/hpc/final_unified_metrics.json").write_text(
    json.dumps({"rows": rows, "eval_config": "Exp 11 (a_work_q1_polygon + extended valid_poi + extended coordnames)"}, indent=2)
)
print(f"\nSaved: /home/shawn/testing/alex/hpc/final_unified_metrics.json")
