"""Compute Spearman+Pearson on Llama v2 re-run using same Exp 11 eval config."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
from sft_09_temporal_eval import (
    load_valid_poi_ids, load_a_work, build_actual_dist, load_predictions,
    build_predicted_dist, compute_metrics
)
import numpy as np, pandas as pd
from scipy.stats import pearsonr

BASE = Path("/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
PRED = Path("/home/shawn/testing/alex/hpc/pred_q12022_llama_v2_coord.jsonl")

valid_test, n2v_test = load_valid_poi_ids(BASE / "poi_unique_food_with_freq_extended.csv")
df_test = load_a_work(BASE / "a_work_q1_polygon.csv", valid_poi_ids=valid_test, name_to_valid=n2v_test)
act = build_actual_dist(df_test)

dn = pd.read_csv(BASE / "poi_display_names_extended_coordnames.csv")
display_to_poi = {str(r.display_id).strip().lower(): str(r.poi_id).strip()
                  for r in dn.itertuples() if str(r.display_id) and str(r.poi_id)}

recs = load_predictions(PRED)
print(f"pred: {len(recs):,} recs")
pred = build_predicted_dist(recs, valid_poi_ids=valid_test, name_to_valid=n2v_test, display_to_poi=display_to_poi)

m = compute_metrics(act, pred)
all_pois = sorted(set(act) | set(pred))
a = np.array([act.get(p, 0) for p in all_pois]); p = np.array([pred.get(p, 0) for p in all_pois])
mask = a > 0
pe, _ = pearsonr(a[mask], p[mask])
ndcg = m.get('ndcg', {})
print(f"Llama v2 (repeats=3) Exp 11 setup eval:")
print(f"  Spearman: {m['spearman']:.4f}  Pearson: {pe:.4f}  Kendall: {m['kendall']:.4f}")
print(f"  NDCG@10: {ndcg.get(10, ndcg.get('10', 0)):.4f}  NDCG@20: {ndcg.get(20, ndcg.get('20', 0)):.4f}")
print(f"  JS: {m['js_divergence']:.4f}  POI coverage: {m['poi_coverage']:.4f}  n_actual={m['n_actual_pois']}")
