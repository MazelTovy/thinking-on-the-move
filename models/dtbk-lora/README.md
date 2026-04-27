---
license: mit
language: en
library_name: peft
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- urban-science
- mobility
- spatial-choice
- discrete-choice
- lora
- peft
- llama-3.1
datasets:
- proprietary-cuebiq
- proprietary-safegraph
pipeline_tag: text-generation
---

# Thinking on the Move &mdash; Downtown Brooklyn LoRA adapter

LoRA fine-tune of `meta-llama/Llama-3.1-8B-Instruct` on **lunch-time
restaurant choice** in Downtown Brooklyn (DTBK), built for the NYU CUSP
capstone project *Thinking on the Move: An LLM-Agent Framework for
Modeling Urban Consumption Choices from Mobility Traces*.

> Code: <https://github.com/MazelTovy/thinking-on-the-move>
> Poster: [`docs/poster.pdf`](https://github.com/MazelTovy/thinking-on-the-move/blob/main/docs/poster.pdf)

## What it does

Given a worker's demographic persona + a list of nearby candidate
restaurants, this adapter predicts which restaurant the worker is most
likely to visit next, with a ranked top-5 + probabilities.

Input prompt template:

```
SYSTEM:
You are a behavioral analyst. Predict which restaurant the worker
will visit next based on their persona and the candidate list.

USER:
--- AGENT PERSONA ---
{cluster_persona_text}

--- CANDIDATE RESTAURANTS ---
- id={poi_id} | {sub_category} | {distance_m}m | freq={cluster_freq}
- ...

--- TASK ---
Predict the SINGLE most likely restaurant the worker will visit next.
Provide a ranked Top-5 list with probabilities that sum to 1.0.
Output JSON only, no explanation.
{"next_choice": "<poi_id>", "top_5": [{"poi": "<poi_id>", "prob": <float>}]}
```

Output (JSON):

```json
{
  "next_choice": "Brooklyn Pantry and Coffee Shop|40.693856|-73.988042",
  "top_5": [
    {"poi": "Brooklyn Pantry and Coffee Shop|40.693856|-73.988042", "prob": 0.62},
    {"poi": "Teranga Brooklyn|40.6912|-73.9847",                     "prob": 0.14},
    {"poi": "DGGD Coffee Shop|40.6940|-73.9879",                     "prob": 0.10},
    {"poi": "Starbucks|40.6938|-73.9871",                            "prob": 0.08},
    {"poi": "Cammareri Bakery & Cafe|40.6845|-73.9848",              "prob": 0.06}
  ]
}
```

## Training

| | |
|---|---|
| Base | `meta-llama/Llama-3.1-8B-Instruct` |
| Adapter | LoRA on all attention + MLP linear layers |
| LoRA r / α / dropout | 16 / 32 / 0.05 |
| Trainable params | ~80 M (1.0 % of base) |
| Optimiser | AdamW |
| Learning rate | 2e-4, cosine, 100 warmup steps |
| Batch | 4 × grad-accum 8 = effective 32 |
| Max length | 2048 tokens |
| Epochs | 3 |
| Train visits | ~18 k DTBK visits, Q1-Q4 2021 |
| Validation | last-60-day temporal hold-out |
| Compute | 1 × NVIDIA H100, ~2 h |

## Evaluation (DTBK Q1 2022 OOS, 6 028 trips, 667 active POIs)

| Metric | Value |
|---|---:|
| Spearman (per-POI mass) | 0.629 |
| NDCG@10 | 0.822 |
| JS divergence | 0.185 |
| POI coverage | 0.840 |

For comparison against the 8 classical baselines see
`results/dtbk_metrics.json` in the project repo.

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "MazelTovy/thinking-on-the-move-dtbk"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

For batched inference at scale, prefer **vLLM**:

```bash
huggingface-cli download MazelTovy/thinking-on-the-move-dtbk --local-dir ./adapter
python -m tom.core.infer_vllm \
    --records_in your_inference_records.jsonl \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_path ./adapter \
    --out predictions.jsonl
```

## Intended use & limitations

**Designed for**: research into urban consumption demand, retail
substitution analysis, neighborhood-scale economic resilience studies,
counterfactual closure simulations.

**Not designed for**: identifying or re-identifying individual
users, surveillance, ad targeting, or any decision affecting a specific
named worker.

**Known limitations**:
- Fitted to Downtown Brooklyn (≈ 4 km² in Kings County, NY). Cross-city
  transfer is untested — the NYC-Metro adapter
  (`MazelTovy/thinking-on-the-move-nyc`) is trained on a much larger
  geography.
- Cuebiq users skew younger and more smartphone-active than the general
  population; demographic re-weighting at the cluster level mitigates but
  does not eliminate this.
- POI universe drifts week-to-week; predictions assume the candidate set
  reflects the active universe at the *prediction* week.

## Citation

```bibtex
@unpublished{xu2026tom,
  title  = {Thinking on the Move: An LLM-Agent Framework for Modeling
            Urban Consumption Choices from Mobility Traces},
  author = {Xu, Sizhe (Alex) and Natekar, Divya},
  year   = {2026},
  note   = {NYU CUSP capstone, P38; in collaboration with Downtown
            Brooklyn Partnership.}
}
```

Please also cite the base model (Llama 3 herd):

```bibtex
@misc{dubey2024llama3,
  title={The Llama 3 Herd of Models},
  author={Dubey, Abhimanyu and others},
  year={2024},
  eprint={2407.21783},
  archivePrefix={arXiv}
}
```

## License

MIT for the adapter weights and code in the project repo. The base model
(Llama-3.1-8B) is governed by the [Meta Llama 3.1 Community License
Agreement](https://llama.meta.com/llama3_1/license/) — adhere to its
terms when redistributing or using the merged model.
