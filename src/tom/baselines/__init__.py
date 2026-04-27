"""
Eight classical spatial-choice baselines for comparison against LoRA-Llama:

- frequency      — popularity prior only
- gravity        — popularity × distance, β fit by L-BFGS
- huff (MCI)     — multiplicative competitive interaction
- mnl_grid       — multinomial logit, β grid-searched
- mnl_rich       — conditional logit with cluster_freq + distance + polygon +
                   sub_category dummies, L-BFGS-B + L2 ridge
- deep_gravity   — light MLP on (log1p(freq), log(dist), polygon dummies)
- lgb_rank       — LightGBM LambdaRank
- xgb_rank       — XGBRanker (rank:ndcg)

Entry point: ``classical.fit_*`` and ``classical.predict_record_*``.
The ``compare`` module aggregates predictions from multiple baselines into a
single comparison table.
"""
