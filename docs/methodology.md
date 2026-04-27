# Methodology

A reader's guide to *Thinking on the Move*. Pairs with the
[capstone poster (PDF)](poster.pdf) and is the recommended entry point before
diving into the source under [`src/tom/`](../src/tom/).

---

## 1. Why this problem

Cities are shaped by the millions of repeated lunch decisions their workers
make. Where they go, when stores open and close, which neighborhoods absorb
displaced traffic — these questions sit at the intersection of urban
planning, retail strategy, and equity.

Two strands of literature have addressed it:

- **Spatial choice models** (gravity, Huff/MCI, MNL) — interpretable, but
  brittle outside the regime where popularity × distance dominates.
- **Recommender / ranking models** (LambdaRank, XGBRanker, Deep Gravity)
  — accurate on aggregate ranking, but produce no narrative, no persona, and
  no answer to "why this restaurant for this person."

We test whether a **fine-tuned language model** can fill the missing third
position: per-visit decisive, persona-aware, counterfactual-ready.

## 2. Data

Three layers, all anonymized or aggregated before use.

| Source | Granularity | Use |
|---|---|---|
| Cuebiq | de-identified GPS pings | trajectory → home/work CBG → stop visits |
| SafeGraph (Advan) | weekly POI + polygons | candidate universe + temporal snapshot per visit |
| ACS / Spectus demographics | block-group | persona construction + demographic clustering |

The repo never ships any of the raw data. See [`data_sources.md`](data_sources.md)
for procurement; the runnable demos use the synthetic toy dataset under
[`data/synthetic/`](../data/synthetic/) instead.

## 3. The three-layer architecture

### 3.1 Urban Evidence Layer

Two preprocessing steps that the public never sees in the released code,
because they require the licensed data:

1. **POI matching** — a Cuebiq stop is reconciled against the SafeGraph POI
   universe at the visit week. Polygons that contain multiple POIs are
   handled via the `polygon_policy` switch
   (`exclude_shared_building` is the default; `point_members` is an
   ablation).
2. **Demographic clustering** — workers are KMeans-clustered into ~7 groups
   on income / race / education / age. The cluster ID becomes a
   first-class feature carried through every prompt.

### 3.2 Choice Set Engine

For every observed visit we synthesize a candidate set of **k = 30** POIs
that the worker *plausibly could have chosen*:

```
true_POI ∪ k-1 other POIs sampled from the cluster's active universe,
        weighted by  log1p(cluster_freq)^p · φ(distance)
```

The distance kernel φ has three concentric radii: 1.5 km (primary), 3 km
(fallback), 5 km (hard cap). This pre-filtering matters for evaluation —
it removes most of the popularity-vs-distance signal that classical
gravity / Huff models normally learn, which is why those baselines collapse
to near-pure popularity in our table.

Origin coordinates default to the worker's **work-CBG centroid** (not the
raw stop location), to test whether the model can predict
*destination from workplace alone* — a much harder task than predicting
from the stop itself.

### 3.3 ToM Choice Model

The prompt template is intentionally simple:

```
SYSTEM:
You are a behavioral analyst. Predict which restaurant the worker
will visit next based on their persona and the candidate list.

USER:
--- AGENT PERSONA ---
{cluster persona}

--- CANDIDATE RESTAURANTS ---
- id={poi_id} | {sub_category} | {distance_m}m | freq={cluster_freq}
- ...

--- TASK ---
Predict the SINGLE most likely restaurant the worker will visit next.
Provide a ranked Top-5 list with probabilities that sum to 1.0.
Output JSON only, no explanation.
{"next_choice": "<poi_id>", "top_5": [{"poi": "<poi_id>", "prob": <float>}, ...]}
```

The base model is **Llama-3.1-8B-Instruct**. We adapt it with
[LoRA](https://arxiv.org/abs/2106.09685): r = 16, α = 32, dropout = 0.05,
applied to all attention and MLP linear layers. Trainable parameters: ~80 M.

Training: AdamW, lr = 2e-4, cosine schedule with 100 warmup steps,
batch 4 × grad-accum 8 = effective 32, max length 2048, 1 epoch over 1 M
visit-prompts (NYC) or 3 epochs over 18k (DTBK).

Inference: vLLM with `temperature=0.2`, `top_p=0.9`, `max_tokens=768`.
A JSON parser validates each output against the candidate set; predictions
that fail to parse are treated as a uniform distribution over the candidates.

## 4. Evaluation

We score each method on the same **OOS-2022** prediction subset:

| Family | Metric | Reads as |
|---|---|---|
| Per-visit | Top-1, Top-5 | "did the model rank the chosen POI in its top-K?" |
| Aggregate | Spearman, NDCG@10, NDCG@50 | "do POIs that get more predicted visits actually get more real visits?" |
| Distributional | JS divergence | "how close is the predicted visit-mass distribution to the empirical one?" |

For the LLM, predicted visit mass per POI = the sum of `top_5[k].prob`
across all evaluation visits where the POI appears as the predicted *k*-th
choice. For classical baselines, the same aggregation rule applies on top
of their probability outputs.

POIs that exist in the truth but not in the model's prediction universe
(zero-shot / cold-start) remain in the actual distribution with zero
predicted mass — i.e., we do *not* hide our miss rate behind a coverage
filter. This is the "universe penalty" referenced in
[`results/nyc_metrics.json`](../results/nyc_metrics.json).

## 5. Why the LLM wins on Top-1 / Top-5 but loses on NDCG / Spearman

Two structural reasons:

1. **Decisive top probabilities**. The training target gives the true POI
   `prob = 0.6` and spreads `0.4` across four neighbours. The model learns
   to be confident, which helps Top-1 but flattens the long tail of
   per-POI mass — Spearman penalises that flattening.
2. **Aggregate reweighting**. NDCG@K and Spearman score POIs after
   summing the model's per-visit probabilities into a per-POI vector. A
   tree-ranker that always emits a smooth, well-calibrated probability
   vector across all 30 candidates can dominate this metric without ever
   actually choosing the true POI as its top pick.

We discuss this more in the poster's "Discussion" panel, and we believe
the right fix is changing the supervision target rather than the model
class — this is open work.

## 6. Counterfactual analysis

Because the LLM scores all 30 candidates (not just the true one), the
prediction has *full distributional shape* over the local choice set.
That makes "what if POI X closed?" trivially answerable:

```
predicted_share(POI)  →  drop POI from every candidate set  →
re-normalise top_5 weights  →  measure where the lost mass redistributes
```

We instantiate this for **Brooklyn Pantry & Coffee Shop**:

- 6.55% of all DTBK lunch predictions go to it
- Closing it sends 35% of the lost mass to ten neighbours
- Weighted-average substitute distance: 536 m
- The redistribution is asymmetric across demographic clusters (see
  [`results/counterfactual_brooklyn_pantry.json`](../results/counterfactual_brooklyn_pantry.json))

This is the only experiment in the framework that the classical baselines
*cannot* perform, because their score functions are not normalized over
the same choice set.

## 7. Limitations

- **Spatial generalisation**. We trained and evaluated on the same
  metropolitan region (DTBK and NYC separately). Cross-city transfer is
  open future work.
- **Population coverage**. Cuebiq users skew younger and more
  smartphone-active. Demographic re-weighting at the cluster level
  partially corrects this but doesn't eliminate selection bias.
- **POI universe drift**. SafeGraph POIs come and go week to week. Our
  temporal snapshot pinning is correct on the supply side but cannot
  observe POIs that opened and closed *between* weekly snapshots.
- **Counterfactual confidence intervals**. The current closure
  experiment is a single-shot prediction. Bootstrapping the candidate set
  would give confidence intervals on the substitution percentages.
