"""
Downtown Brooklyn-side helpers (the DTBK pilot's extra metrics + diagnostics).

These ran on the ss2 server during the DTBK Llama Exp 11 sanity sweep.

- compute_pearson_lora       — per-method Pearson on the Exp 11 LoRA predictions
- compute_pearson_llama_v2   — same for the v2 polygon-aware run
- extra_metrics              — extended metrics (NDCG, JS, coverage, weighted Pe/Sp)
- unified_table              — merge LoRA + classical baselines into one table
- inspect_top1s              — debug helper: dump per-record Top-1 to inspect
- analysis_supplement        — bespoke supplement analyses (counterfactual, etc.)
- build_infer_records        — inference-record builder for DTBK
- plot_cluster_demographics  — generate the 7-cluster radar / heatmap / bar charts
                               (the poster's "Demographic Clustering" panel)
"""
