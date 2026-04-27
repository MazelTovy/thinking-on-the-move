"""
Build a small synthetic toy dataset that mirrors the schema used by the NYC
pipeline, so the notebooks and `tom.baselines` runners are end-to-end runnable
without any access to Cuebiq or SafeGraph.

Output (in this directory):
- toy_visits.csv          — 1000 visits across 50 workers, 80 POIs, 6 weeks
- toy_pois.csv            — 80 POIs (subset of categories: cafe, deli, pizza, …)
- toy_personas.jsonl      — 30 personas, one per cluster
- toy_train.jsonl         — 800 ToM-formatted prompts (in-sample)
- toy_val.jsonl           — 200 ToM-formatted prompts (held-out)
- toy_inference_records.jsonl — 200 records ready for `tom.infer_vllm`

Run:
    python build_toy.py            # writes to ./
    python build_toy.py --seed 7   # reproducible variant
"""

from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta

DEFAULT_SEED = 42
N_WORKERS = 50
N_POIS = 80
N_VISITS = 1000
N_WEEKS = 6
N_CLUSTERS = 7
K_CANDIDATES = 12

CATEGORIES = ["cafe", "deli", "pizza", "asian", "burger", "salad", "bbq", "bakery"]
POI_NAMES = [
    "Brooklyn Bagel", "Roebling Tea", "Court St Coffee", "Pine Box Sandwich",
    "Williamsburg Pizza", "Park Slope Pies", "Smith St Wok", "Carroll Garden Sushi",
    "Atlantic Burgers", "Henry St Grill", "Boerum Salad Bar", "Cobble Greens",
    "Red Hook BBQ", "Gowanus Smoke", "Greenpoint Bakery", "Flatbush Pastry",
    "Tandon Cafe", "Jasper H Kane Hall", "Dock St Coffee", "Cesar's Gourmet",
]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon/2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build(seed: int, out_dir: Path):
    rng = random.Random(seed)

    # --- POIs spread around DTBK-ish bbox ---
    BBOX = (40.685, -74.005, 40.703, -73.975)  # (lat_min, lon_min, lat_max, lon_max)
    pois = []
    for i in range(N_POIS):
        name = (POI_NAMES[i % len(POI_NAMES)] +
                ("" if i < len(POI_NAMES) else f" #{i // len(POI_NAMES) + 1}"))
        lat = rng.uniform(BBOX[0], BBOX[2])
        lon = rng.uniform(BBOX[1], BBOX[3])
        cat = CATEGORIES[i % len(CATEGORIES)]
        cluster_freq = max(1, int(rng.lognormvariate(2.5, 1.0)))
        pois.append({
            "poi_id": f"{name}|{lat:.6f}|{lon:.6f}",
            "poi_name": name,
            "poi_lat": round(lat, 6),
            "poi_lon": round(lon, 6),
            "sub_category": cat,
            "cluster_freq": cluster_freq,
            "polygon_type": "POINT",
        })

    # --- Workers + visits ---
    workers = []
    for w in range(N_WORKERS):
        workers.append({
            "cuebiq_id": f"toy_worker_{w:03d}",
            "cluster": w % N_CLUSTERS,
            "work_lat": rng.uniform(BBOX[0], BBOX[2]),
            "work_lon": rng.uniform(BBOX[1], BBOX[3]),
        })

    base_date = datetime(2021, 9, 1)
    visits = []
    for v in range(N_VISITS):
        worker = rng.choice(workers)
        # bias choice toward closer + higher-freq POIs (gravity-ish)
        weights = []
        for poi in pois:
            d = haversine_m(worker["work_lat"], worker["work_lon"],
                            poi["poi_lat"], poi["poi_lon"])
            w = poi["cluster_freq"] * math.exp(-d / 800.0)
            weights.append(w)
        chosen = rng.choices(pois, weights=weights, k=1)[0]
        ts = base_date + timedelta(
            days=rng.randint(0, N_WEEKS * 7 - 1),
            hours=rng.randint(11, 14),
            minutes=rng.randint(0, 59),
        )
        visits.append({
            "cuebiq_id": worker["cuebiq_id"],
            "cluster": worker["cluster"],
            "stop_lat": worker["work_lat"] + rng.gauss(0, 0.0005),
            "stop_lon": worker["work_lon"] + rng.gauss(0, 0.0005),
            "stop_ts": ts.isoformat(),
            "poi_name": chosen["poi_name"],
            "poi_lat": chosen["poi_lat"],
            "poi_lon": chosen["poi_lon"],
            "polygon_type": chosen["polygon_type"],
        })

    # --- Personas (1 per cluster) ---
    persona_templates = [
        "Brooklyn local, prefers small independent cafes within a 10-minute walk; "
        "values quick service and counter seating.",
        "Tandon faculty/staff, eats lunch on campus 3-4 days/week; gravitates to "
        "the campus dining hall and nearby fast-casual spots.",
        "Black middle-income worker commuting from Crown Heights; values affordable "
        "Caribbean and Senegalese options.",
        "Low-income service worker on tight time; chooses the closest place that "
        "consistently offers lunch under $12.",
        "White professional commuter from the suburbs; prefers chain options near "
        "Atlantic Terminal or Fulton Mall.",
        "Asian Tandon researcher; rotates between salad bars, sushi spots, and the "
        "campus dining hall.",
        "Mixed-race moderate-income worker; explores new openings via word of mouth.",
    ]
    personas = []
    for cid in range(N_CLUSTERS):
        for i in range(20):  # 20 paraphrased copies per cluster
            personas.append({
                "cluster": cid,
                "persona": persona_templates[cid] +
                            ("" if i == 0 else f" (paraphrase v{i})"),
            })

    # --- ToM-formatted prompts ---
    SYSTEM_PROMPT = (
        "You are a behavioral analyst. Predict which restaurant the worker will "
        "visit next based on their persona and the candidate list."
    )

    def build_prompt(visit):
        # k=12 candidates: true POI + 11 nearest by gravity-weighted distance
        true_id = f"{visit['poi_name']}|{visit['poi_lat']:.6f}|{visit['poi_lon']:.6f}"
        scored = []
        for poi in pois:
            d = haversine_m(visit["stop_lat"], visit["stop_lon"],
                            poi["poi_lat"], poi["poi_lon"])
            scored.append((d, poi))
        scored.sort(key=lambda x: x[0])
        chosen = [p for _, p in scored[:K_CANDIDATES]]
        if not any(p["poi_id"] == true_id for p in chosen):
            # ensure truth always in candidate set
            true_poi = next((p for p in pois if p["poi_id"] == true_id), None)
            if true_poi:
                chosen[-1] = true_poi

        cand_lines = []
        for p in chosen:
            d = haversine_m(visit["stop_lat"], visit["stop_lon"],
                            p["poi_lat"], p["poi_lon"])
            cand_lines.append(f"- id={p['poi_id']} | {p['sub_category']} | {d:.0f}m | freq={p['cluster_freq']}")
        cand_table = "\n".join(cand_lines)
        persona = rng.choice([p["persona"] for p in personas if p["cluster"] == visit["cluster"]])
        user_msg = (
            f"--- AGENT PERSONA ---\n{persona}\n\n"
            f"--- CANDIDATE RESTAURANTS ---\n{cand_table}\n\n"
            f"--- TASK ---\n"
            f"Predict the SINGLE most likely restaurant the worker will visit next. "
            f"Provide a ranked Top-5 list with probabilities that sum to 1.0. "
            f"Output JSON only, no explanation.\n"
            f'{{"next_choice": "<poi_id>", "top_5": [{{"poi": "<poi_id>", "prob": <float>}}]}}'
        )
        # synthetic supervision: top-5 = true + 4 closest, with prob mass [0.6, 0.15, 0.1, 0.08, 0.07]
        ranked_ids = [p["poi_id"] for p in chosen if p["poi_id"] != true_id][:4]
        top5 = [{"poi": true_id, "prob": 0.6}]
        for j, pid in enumerate(ranked_ids):
            top5.append({"poi": pid, "prob": [0.15, 0.10, 0.08, 0.07][j]})
        assistant = {"next_choice": true_id, "top_5": top5}
        return {
            "system": SYSTEM_PROMPT,
            "user": user_msg,
            "assistant": json.dumps(assistant, ensure_ascii=False),
            "meta": {
                "cuebiq_id": visit["cuebiq_id"],
                "cluster_id": visit["cluster"],
                "true_poi_id": true_id,
                "stop_ts": visit["stop_ts"],
                "origin_lat": visit["stop_lat"],
                "origin_lon": visit["stop_lon"],
                "candidates": [
                    {"poi_id": p["poi_id"], "poi_name": p["poi_name"],
                     "poi_lat": p["poi_lat"], "poi_lon": p["poi_lon"],
                     "sub_category": p["sub_category"],
                     "cluster_freq": p["cluster_freq"],
                     "polygon_type": p["polygon_type"],
                     "dist_m": haversine_m(visit["stop_lat"], visit["stop_lon"],
                                            p["poi_lat"], p["poi_lon"])}
                    for p in chosen
                ],
            },
        }

    rng.shuffle(visits)
    train_visits = visits[:800]
    val_visits = visits[800:]

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "toy_pois.csv").open("w") as f:
        f.write("poi_id,poi_name,poi_lat,poi_lon,sub_category,cluster_freq,polygon_type\n")
        for p in pois:
            f.write(",".join(str(p[k]) for k in ["poi_id", "poi_name", "poi_lat", "poi_lon",
                                                   "sub_category", "cluster_freq", "polygon_type"]) + "\n")
    with (out_dir / "toy_visits.csv").open("w") as f:
        cols = ["cuebiq_id", "cluster", "stop_lat", "stop_lon", "stop_ts",
                "poi_name", "poi_lat", "poi_lon", "polygon_type"]
        f.write(",".join(cols) + "\n")
        for v in visits:
            f.write(",".join(str(v[k]) for k in cols) + "\n")
    with (out_dir / "toy_personas.jsonl").open("w") as f:
        for p in personas:
            f.write(json.dumps(p) + "\n")
    with (out_dir / "toy_train.jsonl").open("w") as f:
        for v in train_visits:
            f.write(json.dumps(build_prompt(v)) + "\n")
    with (out_dir / "toy_val.jsonl").open("w") as f:
        for v in val_visits:
            f.write(json.dumps(build_prompt(v)) + "\n")
    # Re-export val as inference-records format (no assistant target)
    with (out_dir / "toy_inference_records.jsonl").open("w") as f:
        for v in val_visits:
            rec = build_prompt(v)
            del rec["assistant"]
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote synthetic toy dataset to {out_dir}/:")
    print(f"  {N_POIS} POIs, {N_WORKERS} workers, {N_VISITS} visits, {N_CLUSTERS} clusters")
    print(f"  train: 800 prompts, val: 200 prompts (also exported as inference_records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    build(args.seed, args.out_dir)
