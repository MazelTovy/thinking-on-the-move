#!/bin/bash
# Re-render the poster's key figures from cached predictions. This script does
# NOT re-run training or inference — it only rebuilds plots from JSON inputs
# already shipped under ../results/.
#
# For full end-to-end reproduction (needs Cuebiq/SafeGraph + GPU) see
# ../docs/reproducing_poster.md
set -e

cd "$(dirname "$0")/.."

OUT="figures_regenerated"
mkdir -p $OUT

echo "=== [1/3] Render NYC headline table from results/nyc_metrics.json ==="
python3 - <<'PY'
import json, pandas as pd
m = json.load(open("results/nyc_metrics.json"))
df = pd.DataFrame(m["methods"])
print(df.to_markdown(index=False, floatfmt=".4f"))
PY

echo
echo "=== [2/3] DTBK summary from results/dtbk_metrics.json ==="
python3 - <<'PY'
import json, pandas as pd
m = json.load(open("results/dtbk_metrics.json"))
df = pd.DataFrame(m["methods"])
cols = ["method","spearman","pearson","ndcg_10","ndcg_20","js","coverage"]
print(df[cols].fillna("—").to_markdown(index=False))
PY

echo
echo "=== [3/3] Counterfactual headline from results/counterfactual_brooklyn_pantry.json ==="
python3 - <<'PY'
import json
c = json.load(open("results/counterfactual_brooklyn_pantry.json"))
h = c["headline_numbers"]
print(f"POI: {c['poi']['name']}")
print(f"Original predicted share: {h['original_predicted_share_percent']}%")
print(f"Top 15 substitutes capture: {h['top_15_substitutes_combined_capture_percent']}%")
print(f"Weighted-avg substitute distance: {h['weighted_avg_substitute_distance_m']} m")
PY

echo
echo "Cached static figures already in figures/:"
find figures -name "*.png" -o -name "*.svg" | sort
