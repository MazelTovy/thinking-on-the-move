#!/usr/bin/env python3
"""21_obfuscate_names.py — Rewrite records with POI names hashed to "POI_NNNNN".

Keeps all other fields (category, dist, cluster_freq, polygon_type, lat/lon) intact.
This isolates whether LoRA is exploiting POI-name semantics (cuisine, brand, etc.)
vs the structured features a gradient-boosted ranker would also have.
"""
import argparse, hashlib, json, os


def stable_id(poi_id):
    h = hashlib.md5(poi_id.encode('utf-8')).hexdigest()[:8]
    return f"POI_{h}"


def obfuscate_record(r, name_map):
    def get_obf(poi_id):
        if poi_id not in name_map:
            name_map[poi_id] = stable_id(poi_id)
        return name_map[poi_id]

    # Build (original_name, obfuscated_name) pairs as we rewrite candidates;
    # we then use these to patch the prebuilt `user` prompt below.
    name_replacements = []

    # rewrite truth
    old_true = r.get('true_poi_id', '')
    if '|' in old_true:
        orig_name, lat, lon = old_true.split('|', 2)
        obf_name = get_obf(old_true)
        r['true_poi_id'] = f"{obf_name}|{lat}|{lon}"

    # rewrite candidates
    for c in r.get('candidates', []):
        old_id = c.get('poi_id', '')
        if '|' not in old_id: continue
        orig_name, lat, lon = old_id.split('|', 2)
        obf_name = get_obf(old_id)
        c['poi_id'] = f"{obf_name}|{lat}|{lon}"
        c['poi_name'] = obf_name
        name_replacements.append((orig_name, obf_name))

    # Also rewrite the prebuilt prompt so LoRA actually sees obfuscated names.
    # The prompt text lines look like:
    #   - <orig_name> | id=<orig_name>|<lat>|<lon> | category=... | dist=...m
    # Replacing the raw name substring is sufficient because lat/lon are
    # distinct and the `| id=` separator keeps tokens unambiguous.
    user = r.get('user')
    if user:
        # Longest-name-first so "Kao Sushi" does not clobber "Kao Sushi Deluxe".
        for orig, obf in sorted(name_replacements, key=lambda x: -len(x[0])):
            if orig and obf and orig != obf:
                user = user.replace(orig, obf)
        r['user'] = user
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_in", required=True)
    p.add_argument("--records_out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.records_out) or ".", exist_ok=True)
    name_map = {}
    n_in, n_out = 0, 0
    with open(args.records_in) as fin, open(args.records_out, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            r2 = obfuscate_record(r, name_map)
            fout.write(json.dumps(r2, ensure_ascii=False) + '\n')
            n_in += 1
            n_out += 1
            if n_in % 10000 == 0:
                print(f"  {n_in:,} lines, {len(name_map):,} unique POIs")
    print(f"done: {n_in:,} in → {n_out:,} out, {len(name_map):,} unique POI names obfuscated")


if __name__ == "__main__":
    main()
