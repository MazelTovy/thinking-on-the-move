"""Debug: show actual top-10 POIs and each method's predicted top-5."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
from sft_09_temporal_eval import (
    load_valid_poi_ids, load_a_work, build_actual_dist, load_predictions,
    build_predicted_dist,
)
import pandas as pd

BASE = Path("/home/shawn/testing/review/dtbk/dtbk-sft-half-year")
EXP11 = Path("/home/shawn/testing/alex/hpc/baselines_exp11")

valid, n2v = load_valid_poi_ids(BASE/"poi_unique_food_with_freq_extended.csv")
df = load_a_work(BASE/"a_work_q1_polygon.csv", valid_poi_ids=valid, name_to_valid=n2v)
act = build_actual_dist(df)
act_sorted = sorted(act.items(), key=lambda x: -x[1])
print("ACTUAL TOP-10 POIs in Q1 2022:")
for i, (p, m) in enumerate(act_sorted[:10], 1):
    print(f"  {i}. {p:<60} {m*100:.2f}%")
print()

dn = pd.read_csv(BASE/"poi_display_names_extended_coordnames.csv")
display_to_poi = {str(r.display_id).strip().lower(): str(r.poi_id).strip()
                  for r in dn.itertuples() if str(r.display_id) and str(r.poi_id)}

def show_top5(label, pred_path):
    recs = load_predictions(pred_path)
    pred = build_predicted_dist(recs, valid_poi_ids=valid, name_to_valid=n2v, display_to_poi=display_to_poi)
    pred_sorted = sorted(pred.items(), key=lambda x: -x[1])
    print(f"{label:<25} predicted top-5:")
    for i, (p, m) in enumerate(pred_sorted[:5], 1):
        act_rank = next((k+1 for k, (ap, _) in enumerate(act_sorted) if ap == p), None)
        print(f"  {i}. {p:<60} {m*100:>5.2f}%  (actual rank: {act_rank})")
    print()

show_top5("Llama Exp11", str(BASE/"pred_q12022_coordnames.jsonl"))
show_top5("Llama v2", "/home/shawn/testing/alex/hpc/pred_q12022_llama_v2_coord.jsonl")
for m in ["frequency","huff","lgb_rank","xgb_rank"]:
    show_top5(m, str(EXP11/m/"pred_oos_2022.jsonl"))
