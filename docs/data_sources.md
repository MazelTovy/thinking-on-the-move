# Data sources

The full pipeline depends on three licensed / restricted data products.
**This repository does not redistribute any of them.** The runnable demos
under [`data/synthetic/`](../data/synthetic/) use a fully synthetic toy
dataset that mirrors the schema.

## 1. Cuebiq / Spectus mobility data (2021–2022)

Anonymized GPS pings from opted-in mobile-app users, used to derive home /
work CBG and stop-detected visits.

- **Vendor**: Cuebiq, Inc. (now operating as Spectus / Advan)
- **License**: Commercial; researchers can request access via Spectus's
  Social Impact program (https://spectus.ai/social-impact/) which has
  granted access to several urban-science groups for non-commercial work.
- **Schema we expect** (`a_work.csv` after preprocessing):
  ```
  cuebiq_id, work_cbg, stop_lat, stop_lon, stop_ts,
  poi_name, poi_lat, poi_lon, polygon_type, demo_cluster
  ```

## 2. Advan (SafeGraph) Weekly Patterns + POI Geometry (2021–2022)

POI universe — restaurant names, sub-categories, weekly open status,
building polygons.

- **Vendor**: Advan (acquired SafeGraph's POI products in 2024)
- **License**: Available via the Dewey Data marketplace and through
  university research agreements
- **Schema we expect** (`cbg_poi/cluster_*.csv`):
  ```
  poi_id, poi_name, poi_lat, poi_lon, sub_category,
  cluster_freq, polygon_type, week_active_2021, week_active_2022
  ```

## 3. American Community Survey (ACS) — block-group demographics

- **Vendor**: U.S. Census Bureau (public)
- **Use**: input to the 7-cluster KMeans demographic clustering described
  in the poster's "Demographic Clustering" panel
- **API**: https://www.census.gov/data/developers/data-sets/acs-5year.html
- **Variables**: median household income, unemployment rate, race
  composition, education attainment

## Preparing your own pipeline run

If you have access to Cuebiq + SafeGraph, the full pipeline is:

```bash
# 1. Stop detection + work CBG inference (you write this; we cannot ship it)
python your_stop_detection.py raw_pings.parquet → a_work.csv

# 2. Demographic clustering (open, we can ship)
python -m tom.core.data_prep --skip_step_b ...

# 3. SFT data preparation
python -m tom.core.data_prep \
    --a_work a_work.csv \
    --cbg_poi_dir cbg_poi/ \
    --personas data/personas.jsonl \
    --out_train train.jsonl --out_val val.jsonl \
    --max_train_rows 1000000 --max_val_rows 50000

# 4. LoRA fine-tuning
python -m tom.core.train_lora \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --train_jsonl train.jsonl --val_jsonl val.jsonl \
    --output_dir ./adapter \
    --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 \
    --num_train_epochs 1 --max_length 2048

# 5. Inference + evaluation
python -m tom.core.infer_vllm --records_in val_records.jsonl \
    --lora_path ./adapter/final --out preds.jsonl
python -m tom.visualize.temporal_eval --pred_test preds.jsonl --out_dir eval/
```

Total compute budget for the NYC-Metro run:
- Prep:   ~1 h  on 28-core CPU node
- Train: ~36 h on a single H200 GPU (1 M visits × 1 epoch)
- Infer: ~6 h  on a single H200 GPU (500 k visits, vLLM batched)
- Eval:  ~30 min on 1-core CPU
