"""Compute Pearson (+ Spearman/NDCG) on Exp 11 LoRA pred files using a_work as actual source.
Reuses sft_09_temporal_eval.py logic for actual/predicted distribution building."""
import sys, json
from pathlib import Path

sys.path.insert(0, "/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
from sft_09_temporal_eval import (
    load_valid_poi_ids, load_a_work, build_actual_dist, load_predictions,
    build_predicted_dist, compute_metrics
)

import numpy as np
from scipy.stats import pearsonr

BASE = Path("/home/shawn/testing/review/dtbk/dtbk-sft-half-year")

valid_test, n2v_test = load_valid_poi_ids(BASE / "poi_unique_food_with_freq_extended.csv")

# Load a_work_q1_polygon for actual
df_test = load_a_work(BASE / "a_work_q1_polygon.csv", valid_poi_ids=valid_test, name_to_valid=n2v_test)
print(f"OOS records: {len(df_test):,}  unique POIs: {df_test['poi_id'].nunique()}")
act_test = build_actual_dist(df_test)

# Load display_names reverse map
import pandas as pd
dn = pd.read_csv(BASE / "poi_display_names_extended_coordnames.csv")
display_to_poi = {str(r.display_id).strip().lower(): str(r.poi_id).strip()
                  for r in dn.itertuples() if str(r.display_id) and str(r.poi_id)}

# Load pred
recs_test = load_predictions(BASE / "pred_q12022_coordnames.jsonl")
print(f"Test predictions: {len(recs_test):,}")
pred_test = build_predicted_dist(recs_test, valid_poi_ids=valid_test, name_to_valid=n2v_test, display_to_poi=display_to_poi)

# Standard metrics
m = compute_metrics(act_test, pred_test)

# Pearson on non-zero actual POIs (matching my supplement script)
all_pois = sorted(set(act_test) | set(pred_test))
a = np.array([act_test.get(p, 0) for p in all_pois])
p = np.array([pred_test.get(p, 0) for p in all_pois])
mask = a > 0
pe, _ = pearsonr(a[mask], p[mask])

print()
print(f"Exp 11 LoRA (pred_q12022_coordnames.jsonl) OOS metrics:")
print(f"  Spearman     : {m['spearman']:.4f}")
print(f"  Pearson      : {pe:.4f}")
print(f"  Kendall      : {m['kendall']:.4f}")
ndcg = m.get('ndcg', {})
print(f"  NDCG@10      : {ndcg.get(10, ndcg.get('10', float('nan'))):.4f}")
print(f"  NDCG@20      : {ndcg.get(20, ndcg.get('20', float('nan'))):.4f}")
print(f"  JS divergence: {m['js_divergence']:.4f}")
print(f"  POI coverage : {m['poi_coverage']:.4f}")
print(f"  n_actual POIs: {m['n_actual_pois']}")
