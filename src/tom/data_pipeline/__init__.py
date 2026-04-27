"""
Raw-data ingestion pipeline (mostly requires licensed Cuebiq + SafeGraph data
that this repo cannot redistribute — see ``docs/data_sources.md``):

- download_wpp              — World Population Prospects + workplace bridge
- download_acs              — American Community Survey block-group demographics
- extract_global_places     — SafeGraph POI extraction
- bridge_wpp_globalplaces   — POI ↔ geography bridge
- resolve_poi_names         — name canonicalisation (handles chain dedup)
- build_polygon_index       — building-polygon spatial index
- match_stops               — Cuebiq stop ↔ POI matching with polygon policy
- demographics_clustering   — KMeans clustering on ACS variables (the 7-cluster
                              decomposition shown in the poster)

Most scripts accept ``--help`` and print their CLI surface.
"""
