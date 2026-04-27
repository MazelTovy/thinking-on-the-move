"""
Thinking on the Move (ToM)
==========================

LLM-agent framework for modeling urban consumption choices from mobility traces.

Subpackages
-----------
- ``tom.core``           — the poster pipeline core: data prep, LoRA train,
                           vLLM batch inference, cold-start hold-out
- ``tom.baselines``      — eight classical spatial-choice baselines (frequency,
                           gravity, Huff, MNL grid, MNL rich, Deep Gravity,
                           LightGBM LambdaRank, XGBRanker) + comparison utility
- ``tom.visualize``      — the poster's signature plots: 4-panel scatter,
                           temporal-status spatial maps, 3D residual demand surface
- ``tom.data_pipeline``  — raw-data ingestion (Cuebiq → stops → POI matching →
                           polygons → demographic clustering)
- ``tom.ablations``      — paper-track ablations: cold-start, cuisine slice,
                           rare-POI slice, persona stripping, hybrid decoding, …
- ``tom.dtbk``           — Downtown Brooklyn-side helpers (per-method Pearson,
                           unified table, top-1 inspector, cluster demographics)

Module-level helpers
--------------------
- ``tom.utils`` — POI ID format, haversine distance, candidate sampler

See ``docs/methodology.md`` for a full pipeline walkthrough and
``docs/reproducing_poster.md`` for the per-figure regeneration recipe.
"""

__version__ = "0.1.0"
