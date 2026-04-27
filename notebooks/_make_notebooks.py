"""Generate the four demo notebooks programmatically."""
import json
from pathlib import Path

OUT = Path(__file__).parent

def cell_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def cell_code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

# ---------- 01 pipeline overview ----------
nb01 = nb([
    cell_md("""# 01 — Pipeline overview

This notebook walks through the **Thinking on the Move (ToM)** pipeline using a
fully synthetic toy dataset — **no Cuebiq, no SafeGraph, no GPU required**.
It's the fastest way to grasp what the framework does end-to-end.

We will:
1. Inspect the toy data shape (workers, POIs, visits)
2. Build a single ToM-style prompt
3. Score the same visit with one classical baseline (gravity)
4. Show where the real pipeline differs (LoRA-Llama at scale)
"""),
    cell_md("## 1. Inspect the toy data\n\n"
            "Schema mirrors the production pipeline."),
    cell_code("""import json
import pandas as pd
from pathlib import Path

DATA = Path('../data/synthetic')
pois = pd.read_csv(DATA / 'toy_pois.csv')
visits = pd.read_csv(DATA / 'toy_visits.csv')

print(f'POIs: {len(pois)} across {pois.sub_category.nunique()} categories')
print(f'Visits: {len(visits)} from {visits.cuebiq_id.nunique()} workers'
      f' across {visits.cluster.nunique()} demographic clusters')
pois.head()"""),
    cell_code("""visits.head()"""),
    cell_md("## 2. A single ToM-style prompt"),
    cell_code("""train = [json.loads(l) for l in open(DATA / 'toy_train.jsonl')]
ex = train[0]
print('SYSTEM:'); print(ex['system'])
print()
print('USER:'); print(ex['user'][:1200], '...' if len(ex['user']) > 1200 else '')
print()
print('ASSISTANT (target):'); print(ex['assistant'][:600])"""),
    cell_md("## 3. Classical baseline: gravity score\n\n"
            "Even without an LLM, the candidate set + observed frequencies are enough\n"
            "to fit a plain gravity model. We score the same visit's candidates here."),
    cell_code("""import math, sys
sys.path.insert(0, '../src')
from tom.utils import haversine_m

def gravity_score(cand, alpha=1.0, beta=0.7):
    return (cand['cluster_freq'] ** alpha) * math.exp(-cand['dist_m'] / 800.0) ** beta

cands = ex['meta']['candidates']
ranked = sorted(cands, key=gravity_score, reverse=True)
true_id = ex['meta']['true_poi_id']
print('Truth:', true_id.split('|')[0])
print('Gravity top-5:')
for r in ranked[:5]:
    print(f"  {r['poi_name']:<32}  dist={r['dist_m']:>6.0f}m  freq={r['cluster_freq']}")"""),
    cell_md("""## 4. What the real pipeline adds

In the production setting:
- The candidate set comes from a real SafeGraph snapshot at the visit's week
- The persona text is a sample from a 15 000-persona pool clustered by ACS demographics
- The model is `Llama-3.1-8B + LoRA r=16` — fine-tuned for one epoch on 1 M visits
- Inference runs through vLLM in batches of 256 prompts on an H200

To run real inference on the toy set with the released DTBK adapter:

```bash
huggingface-cli download MazelTovy/thinking-on-the-move-dtbk --local-dir ./dtbk_adapter
python -m tom.core.infer_vllm \\
    --records_in ../data/synthetic/toy_inference_records.jsonl \\
    --model_path meta-llama/Llama-3.1-8B-Instruct \\
    --lora_path ./dtbk_adapter \\
    --out predictions.jsonl \\
    --max_model_len 4096 --temperature 0.2
```

Continue with `02_dtbk_walkthrough.ipynb` for the DTBK story or `03_nyc_scaling.ipynb`
for the NYC headline numbers."""),
])
(OUT / "01_pipeline_overview.ipynb").write_text(json.dumps(nb01, indent=1))


# ---------- 02 DTBK walkthrough ----------
nb02 = nb([
    cell_md("""# 02 — Downtown Brooklyn walkthrough

The DTBK pilot answers: *can a fine-tuned LLaMA recover the per-POI demand
shape that vanilla LLaMA cannot?*

Below we show:
1. The headline scatter (vanilla vs LoRA-Llama)
2. The 9-method comparison numbers
3. The 7-cluster demographic decomposition
"""),
    cell_md("## 1. Vanilla vs fine-tuned LLaMA on DTBK 2022 Q1 OOS"),
    cell_code("""from IPython.display import Image, display, Markdown
display(Image('../figures/dtbk/temporal_scatter_colored.png'))
display(Markdown('Each point is one DTBK POI. **x** = actual 2022-Q1 visits;'
                ' **y** = predicted top-5 mass.'))"""),
    cell_md("## 2. The 9-method comparison"),
    cell_code("""import json, pandas as pd
m = json.load(open('../results/dtbk_metrics.json'))
df = pd.DataFrame(m['methods']).set_index('method')
df[['spearman','pearson','ndcg_10','ndcg_20','js','coverage']].fillna('—')"""),
    cell_md("""**Read this as:** classical baselines retain higher per-POI Pearson because
n ≈ 6 028 trips concentrates signal on the *head* of the demand
distribution, where popularity-distance models fit well. The LoRA-Llama
recovers structure on the *tail* (coverage 0.84 vs gravity 0.64), which is
where the counterfactual analysis becomes useful (see notebook 04)."""),
    cell_md("## 3. 7-cluster demographic decomposition"),
    cell_code("""display(Image('../figures/demographic/radar_per_cluster.png'))"""),
    cell_code("""display(Image('../figures/demographic/heatmap_zscored.png'))"""),
    cell_md("""The 7 clusters are interpreted as:

| Cluster | Name | Hint |
|---|---|---|
| C1 | DTBK Local Affluent | brownstone owners, walks |
| C2 | NYU Tandon Faculty/Staff | campus-anchored |
| C3 | Black Middle-Income | Crown Heights commute |
| C4 | Low-Income Service Workers | tight time, tight budget |
| C5 | White Professional Commuters | suburb-to-Atlantic-Terminal |
| C6 | Asian Tandon Workforce | rotates campus + nearby |
| C7 | Mixed-Race Moderate-Income | exploratory |

Cluster names are descriptive, not normative. They emerge from KMeans on
ACS variables (income, race, education, age) at the worker's home CBG."""),
])
(OUT / "02_dtbk_walkthrough.ipynb").write_text(json.dumps(nb02, indent=1))


# ---------- 03 NYC scaling ----------
nb03 = nb([
    cell_md("""# 03 — NYC scaling

When we scale the pipeline up from DTBK (~4 km², 667 POIs) to NYC Metro
(~785 km², 57 k POIs), the LoRA-Llama becomes the **best Top-1 and Top-5**
predictor of any of the 9 methods we evaluated.

This notebook shows the headline table and the 4-panel + 3D residual
diagnostic figures that ship in the poster.
"""),
    cell_md("## 1. The 9-method comparison (2022 OOS, 500k visits)"),
    cell_code("""import json, pandas as pd
m = json.load(open('../results/nyc_metrics.json'))
df = pd.DataFrame(m['methods']).set_index('name')
def bold_max_min(s, op):
    target = op(s)
    return [f'**{v:.4f}**' if v == target else f'{v:.4f}' for v in s]
out = df.copy()
for col in ['top_1','top_5','spearman','ndcg_10','ndcg_50']:
    out[col] = bold_max_min(df[col], max)
out['js'] = bold_max_min(df['js'], min)
out"""),
    cell_md("""**Read as**: the four popularity/distance-only models collapse to popularity
(β → 0) because the candidate sampler has already pre-filtered by distance.
Among rich-feature models, the LoRA-Llama posts the best Top-1 and Top-5;
the LightGBM ranker dominates aggregate ranking metrics."""),
    cell_md("## 2. The 4-panel actual-vs-predicted scatter"),
    cell_code("""from IPython.display import Image
Image('../figures/nyc/temporal_scatter_4panel.png')"""),
    cell_md("## 3. The 3D residual demand surface"),
    cell_code("""Image('../figures/nyc/oos_3d_demand_surface.png')"""),
    cell_md("""**3D residual** = predicted minus actual visit mass at every NYC location.
Red areas are over-predicted, blue are under-predicted. Notice the model
slightly under-predicts mid-Brooklyn and over-predicts Manhattan's
financial district — both consistent with sparser Cuebiq coverage in
gentrifying neighborhoods."""),
    cell_md("## 4. Spatial hexbin maps"),
    cell_code("""Image('../figures/nyc/poster_temporal_status_spatial_map.png')"""),
])
(OUT / "03_nyc_scaling.ipynb").write_text(json.dumps(nb03, indent=1))


# ---------- 04 counterfactual ----------
nb04 = nb([
    cell_md("""# 04 — Counterfactual: what if Brooklyn Pantry & Coffee closed?

Because the LLM scores all 30 candidates per visit (not just the truth), we
can run *counterfactual closures*: drop a POI from every candidate set,
re-normalise the model's top-5 weights, and measure where the lost mass
redistributes.

This notebook reproduces the closure analysis for **Brooklyn Pantry &
Coffee Shop** that anchors the bottom-right of the capstone poster.
"""),
    cell_md("## 1. The headline numbers"),
    cell_code("""import json
c = json.load(open('../results/counterfactual_brooklyn_pantry.json'))
print(f"POI: {c['poi']['name']}")
print(f"Original predicted share: {c['headline_numbers']['original_predicted_share_percent']}%")
print(f"Appears in {c['headline_numbers']['appeared_in_predictions']:,}/{c['headline_numbers']['of_total_predictions']:,}"
      f" predictions ({c['headline_numbers']['presence_rate_percent']}%)")
print(f"Top 15 substitutes capture: {c['headline_numbers']['top_15_substitutes_combined_capture_percent']}% of lost mass")
print(f"Weighted-avg substitute distance: {c['headline_numbers']['weighted_avg_substitute_distance_m']} m")"""),
    cell_md("## 2. Top substitutes"),
    cell_code("""import pandas as pd
subs = pd.DataFrame(c['top_substitutes'])
subs"""),
    cell_md("## 3. Per-cluster reliance — who feels the loss most?"),
    cell_code("""rel = pd.DataFrame(c['per_cluster_reliance']).sort_values('reliance_pct', ascending=False)
rel"""),
    cell_md("""**Read as**: White Professional Commuters (C5, 10.7% reliance) and Black
Middle-Income workers (C3, 9.1%) lean on Brooklyn Pantry the hardest.
Low-Income Service Workers (C4, 1.6%) barely use it — they substitute to
McDonald's. The LLM's per-cluster persona conditioning makes these
heterogeneous redistributions visible without any explicit demographic
target in the loss."""),
    cell_md("## 4. The closure substitution map"),
    cell_code("""from IPython.display import Image
Image('../figures/counterfactual/brooklyn_pantry/fig_closure_Brooklyn_Pantry_and_Coffee_Shop.png')"""),
    cell_md("""## 5. Reproduce on a different POI

The same machinery works on any POI in the cluster pool. The script
`scripts/counterfactual_closure.py` (TODO in the public repo) takes a POI
name + the prediction JSONL and writes the substitution map.

Pseudocode:

```python
predictions = load_jsonl('preds_oos_2022.jsonl')
target_poi  = 'Some Restaurant Name|lat|lon'
new_preds   = []
for p in predictions:
    if target_poi not in {c['poi'] for c in p['top_5']}:
        new_preds.append(p)  # closure didn't affect this prediction
        continue
    remaining = [c for c in p['top_5'] if c['poi'] != target_poi]
    z = sum(c['prob'] for c in remaining)
    for c in remaining:
        c['prob'] /= z
    new_preds.append({**p, 'top_5': remaining})
substitution_map = aggregate_by_poi(new_preds) - aggregate_by_poi(predictions)
```
"""),
])
(OUT / "04_counterfactual_demo.ipynb").write_text(json.dumps(nb04, indent=1))


print("Wrote 4 notebooks:")
for nb_file in sorted(OUT.glob("*.ipynb")):
    print(f"  {nb_file.name}")
