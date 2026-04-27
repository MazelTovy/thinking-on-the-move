# Data

This directory holds **only the synthetic toy demo** that ships with the
repo. The real Cuebiq + SafeGraph + ACS sources are not redistributable —
see [`../docs/data_sources.md`](../docs/data_sources.md) for procurement.

## `synthetic/`

A self-contained micro-dataset that mirrors the production schema:

- 50 workers × 7 demographic clusters
- 80 POIs across 8 sub-categories, scattered in a DTBK-shaped bounding box
- 1000 visits across 6 weeks (2021-09 to 2021-10)
- 800 train prompts + 200 val prompts in the ToM JSONL format
- 200 inference records ready for `tom.infer_vllm`

Regenerate (reproducibly) at any time:

```bash
cd data/synthetic
python build_toy.py            # default seed = 42
python build_toy.py --seed 7   # reproducible variant
```

## `raw/` (gitignored)

If you have access to the real data, place it here:

```
data/raw/
├── a_work_2021.csv         # Cuebiq stops × CBG joins
├── a_work_2022.csv
├── cbg_poi/                # SafeGraph POI snapshots per cluster
│   ├── cluster_00_pois.csv
│   └── ...
├── personas.jsonl          # demographic-cluster personas (1 per visit)
└── poi_unique_food_with_freq.csv  # global food-POI fallback universe
```

Then run:

```bash
python -m tom.data_prep --a_work data/raw/a_work_2021.csv \
    --cbg_poi_dir data/raw/cbg_poi/ --personas data/raw/personas.jsonl \
    --out_train experiments/yours/train.jsonl \
    --out_val   experiments/yours/val.jsonl \
    --n_workers 28
```
