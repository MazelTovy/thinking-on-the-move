"""
Paper-track ablations. Each script takes inference records and produces a
per-slice metric file or sliced eval figure.

- resample_candidates   — re-sample candidate sets at different (k, β) settings
- tail_slicing          — head vs tail POI slicing (long-tail diagnosis)
- cuisine_slice         — per-sub-category slicing
- rare_poi_slice        — held-out, rare-POI eval
- obfuscate_names       — POI name obfuscation (test the model isn't memorising)
- strip_persona         — strip the persona block (test ablation of persona)
- hybrid_decode         — LLM + ranker hybrid decoding
- hybrid_eval           — eval for hybrid-decode outputs
- reparse_obfuscated    — re-parse outputs from obfuscated runs
- exp14_figures         — ablation-specific figures for exp14
"""
