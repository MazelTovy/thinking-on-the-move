# Reproducing the poster

The four numbered panels of the poster map directly to artifacts in this
repo:

| Poster panel | Cached output | Regen script |
|---|---|---|
| 1. Architecture diagram | `figures/architecture/architecture.svg` | hand-authored SVG |
| 2. NYC scaling table | `results/nyc_metrics.json` | `scripts/build_nyc_table.py` |
| 3. NYC 4-panel scatter | `figures/nyc/temporal_scatter_4panel.png` | `python -m tom.visualize.scatter_4panel` |
| 4. NYC 3D residual demand surface | `figures/nyc/oos_3d_demand_surface.png` | `python -m tom.visualize.temporal_eval --plot 3d_residual` |
| 5. NYC spatial hexbin maps | `figures/nyc/poster_temporal_status_spatial_map.png` | `python -m tom.visualize.temporal_eval --plot temporal_status_spatial` |
| 6. DTBK vanilla vs fine-tuned | `figures/dtbk/temporal_scatter_colored.png` | `python -m tom.visualize.temporal_eval --plot temporal_scatter_colored` |
| 7. DTBK demographic clustering | `figures/demographic/{radar,heatmap,bars,category_mix}*.png` | `python -m tom.visualize.scatter_4panel --cluster_viz` |
| 8. Brooklyn Pantry counterfactual | `figures/counterfactual/brooklyn_pantry/*.png` | `python scripts/counterfactual_closure.py --poi "Brooklyn Pantry and Coffee Shop"` |

Numerical headlines (also embedded in `results/*.json`):

| Number | Value | Source |
|---|---|---|
| LoRA NYC Top-1 | 0.7447 | `results/nyc_metrics.json` |
| LoRA NYC Top-5 | 0.8031 | `results/nyc_metrics.json` |
| Brooklyn Pantry original share | 6.55 % | `results/counterfactual_brooklyn_pantry.json` |
| Top-15 substitutes capture | ~35 % | `results/counterfactual_brooklyn_pantry.json` |
| Weighted-avg substitute distance | 536 m | `results/counterfactual_brooklyn_pantry.json` |
| DTBK LoRA OOS NDCG@10 | 0.822 | `results/dtbk_metrics.json` |

## End-to-end reproduction

If you have the licensed data and a GPU:

```bash
# Run the headline NYC experiment from scratch (≈ 38 h on H200)
python -m tom.core.data_prep \
    --a_work data/raw/a_work_2021.csv \
    --cbg_poi_dir data/raw/cbg_poi/ \
    --personas data/raw/personas.jsonl \
    --out_train experiments/exp_repro/train.jsonl \
    --out_val   experiments/exp_repro/val.jsonl \
    --max_train_rows 1000000 --max_val_rows 50000 \
    --val_date_from 2021-11-01 \
    --origin_mode work_cbg_centroid \
    --polygon_policy exclude_shared_building \
    --n_workers 28

python -m tom.core.train_lora \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --train_jsonl experiments/exp_repro/train.jsonl \
    --val_jsonl   experiments/exp_repro/val.jsonl \
    --output_dir  experiments/exp_repro/lora \
    --num_train_epochs 1 --learning_rate 2e-4 \
    --lora_r 16 --lora_alpha 32

python -m tom.core.infer_vllm \
    --year 2022 --records_out experiments/exp_repro/records_oos_2022.jsonl \
    --prepare_only \
    --max_samples 500000 --n_workers 28 \
    --cbg_poi_dir data/raw/cbg_poi --personas data/raw/personas.jsonl

python -m tom.core.infer_vllm \
    --records_in experiments/exp_repro/records_oos_2022.jsonl \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_path  experiments/exp_repro/lora/final \
    --out experiments/exp_repro/pred_oos_2022.jsonl

python -m tom.visualize.temporal_eval \
    --exp exp_repro \
    --pred_test experiments/exp_repro/pred_oos_2022.jsonl \
    --out_dir   experiments/exp_repro/eval/

# Run the 8 classical baselines on the same records
for m in frequency gravity huff mnl_grid mnl_rich deep_gravity lgb_rank xgb_rank; do
    python -m tom.baselines.classical \
        --method $m \
        --records_in experiments/exp_repro/records_oos_2022.jsonl \
        --out experiments/exp_repro/baselines/$m/pred_oos_2022.jsonl
done
```

The poster's exact numbers come from the `exp02_wcbg` checkpoint (April
2026 run). With a fresh data drop you may see ±0.5pp drift on Top-1.
