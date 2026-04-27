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

# Thinking on the Move &mdash; NYC Metro LoRA adapter

LoRA fine-tune of `meta-llama/Llama-3.1-8B-Instruct` on **lunch-time
restaurant choice** across the NYC Metropolitan Area, scaled up from the
Downtown Brooklyn pilot to ~50× the geography.

> Code: <https://github.com/MazelTovy/thinking-on-the-move>
> Companion DTBK adapter: [`MazelTovy/thinking-on-the-move-dtbk`](https://huggingface.co/MazelTovy/thinking-on-the-move-dtbk)

## Headline result

On 2022 out-of-sample evaluation (500 k visits, k = 30 candidates each),
this adapter posts the **best Top-1 (0.7447)** and **best Top-5 (0.8031)**
of any of the nine methods we tested — beating the strongest classical
ranker (LambdaRank/LightGBM, Top-1 = 0.7300) by **+1.47 pp**.

| Method | Top-1 ↑ | Top-5 ↑ |
|---|---:|---:|
| Frequency prior | 0.2034 | 0.4572 |
| Gravity | 0.2025 | 0.4522 |
| Huff / MCI | 0.2034 | 0.4572 |
| MNL-grid | 0.2034 | 0.4572 |
| MNL-rich (conditional logit) | 0.6495 | 0.7603 |
| Deep Gravity (lite MLP) | 0.7237 | 0.7986 |
| LambdaRank (LightGBM) | 0.7300 | 0.8017 |
| XGBRanker (rank:ndcg) | 0.7282 | 0.8025 |
| **LoRA-Llama-3.1-8B (this adapter)** | **0.7447** | **0.8031** |

Trade-off: aggregate ranking metrics (NDCG, Spearman) are lower because
the supervision target gives the true POI prob = 0.6 (decisive top-1
prior), flattening the per-POI distribution. See the project's
[methodology doc](https://github.com/MazelTovy/thinking-on-the-move/blob/main/docs/methodology.md#5-why-the-llm-wins-on-top-1--top-5-but-loses-on-ndcg--spearman)
for a full discussion.

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
| Epochs | 1 |
| Train visits | 1 M visits sampled from 2021 (pre-Nov-2021 hold-out) |
| Validation | last-60-day temporal hold-out + 500 k 2022 OOS visits |
| Compute | 1 × NVIDIA H200, ~36 h |

## Choice-set construction

Inputs at inference time are pre-built **inference records** rather than
raw trajectories — see
[`tom.core.infer_vllm`](https://github.com/MazelTovy/thinking-on-the-move/blob/main/src/tom/core/infer_vllm.py)
for the canonical builder. Per record:

- 30 candidate POIs sampled from the worker's demographic-cluster pool
- Distance kernel: 1.5 km primary / 3 km fallback / 5 km hard cap
- Origin = worker's **work-CBG centroid** (not raw stop), to test
  destination prediction from workplace alone
- Polygon policy: `exclude_shared_building` (default)

## How to use

For batched inference (recommended), use vLLM:

```bash
huggingface-cli download MazelTovy/thinking-on-the-move-nyc --local-dir ./adapter
python -m tom.core.infer_vllm \
    --records_in your_inference_records.jsonl \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_path ./adapter \
    --out predictions.jsonl \
    --temperature 0.2 --max_model_len 4096
```

For single prompts:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "MazelTovy/thinking-on-the-move-nyc"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Intended use & limitations

**Designed for**: research into urban consumption demand at metro scale,
retail planning, neighborhood-substitution analysis, comparing LLM-agent
behavior against classical spatial-choice models.

**Not designed for**: identifying or re-identifying individual mobility
users, real-time personalised recommendations, surveillance, or any
decision affecting a specific named worker.

**Known limitations**:
- Trained on the 5-borough NYC Metropolitan area. We do not test
  cross-city transfer (e.g., to Chicago or LA).
- Cuebiq users skew younger and more smartphone-active; cluster-level
  demographic re-weighting only partially corrects.
- The choice-set construction pre-filters by distance, which can mask
  true cross-borough substitution patterns.

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

## License

MIT for the adapter weights. Base model is governed by the [Meta Llama
3.1 Community License Agreement](https://llama.meta.com/llama3_1/license/).
