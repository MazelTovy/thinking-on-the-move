#!/usr/bin/env python3
"""26_reparse_obfuscated.py — Re-score obfuscated LoRA pred from raw_text.

The obfuscation rerun produced raw_text where LoRA emitted the (obfuscated)
POI IDs correctly, but 12_infer_vllm.py's parser built the valid set from
the unobfuscated pool CSVs, so every emission was rejected → parse_ok=False
→ next_choice=None → top1=0 across the board.

Here we parse raw_text directly and validate only against the record's own
candidate list (since the record has the obfuscated candidates).
"""
import argparse
import json
import re


JSON_PATTERN = re.compile(r'\{.*\}', re.DOTALL)


def extract_next_choice(raw):
    if not raw: return None
    m = JSON_PATTERN.search(raw)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    return obj.get("next_choice"), obj.get("top_5") or []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_in", required=True)
    p.add_argument("--pred_out", required=True)
    args = p.parse_args()

    n = n_parsed = n_top1 = n_top5 = 0
    with open(args.pred_in) as fin, open(args.pred_out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n += 1
            truth = r.get("true_poi_id")
            candidates = r.get("candidates") or []
            valid_ids = {c["poi_id"] for c in candidates}

            raw = r.get("raw_text") or ""
            res = extract_next_choice(raw)
            if res is None or res == (None, None):
                r["parse_ok"] = False
                r["next_choice"] = None
                r["top_5"] = []
            else:
                nc, t5 = res
                # Validate against THIS record's candidate set
                if nc in valid_ids:
                    r["next_choice"] = nc
                else:
                    r["next_choice"] = None
                # Keep only top_5 entries whose poi is in candidate set
                t5_valid = [x for x in (t5 or []) if isinstance(x, dict) and x.get("poi") in valid_ids]
                r["top_5"] = t5_valid
                r["parse_ok"] = bool(r["next_choice"]) and len(t5_valid) >= 5
                if r["parse_ok"]:
                    n_parsed += 1
                    if r["next_choice"] == truth:
                        n_top1 += 1
                    if any(x.get("poi") == truth for x in t5_valid):
                        n_top5 += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"n={n}  parse_ok={n_parsed} ({100*n_parsed/n:.1f}%)")
    if n_parsed:
        print(f"top1 = {n_top1/n:.4f}  top5 = {n_top5/n:.4f}")


if __name__ == "__main__":
    main()
