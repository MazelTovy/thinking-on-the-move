#!/usr/bin/env python3
"""20_cuisine_slice.py — Compare LoRA vs lgb on POIs whose names carry cuisine
semantics, vs generic POI names. Tests the hypothesis that LoRA extracts
world-knowledge signal from POI names.
"""
import json, argparse, re
from collections import defaultdict


CUISINE_KEYWORDS = [
    # Asian
    'sushi', 'ramen', 'japanese', 'japan', 'chinese', 'china', 'dim sum',
    'korean', 'korea', 'bbq', 'thai', 'pho', 'vietnamese', 'noodle', 'wok',
    'teriyaki', 'udon', 'sashimi', 'kimchi',
    # Italian
    'pizza', 'italian', 'pasta', 'trattoria', 'pizzeria', 'osteria',
    # Mexican / Latin
    'taco', 'burrito', 'mexican', 'cantina', 'latin', 'cubano', 'peruvian',
    # American / Casual
    'burger', 'grill', 'diner', 'bbq', 'kitchen', 'deli', 'steakhouse',
    # Middle Eastern / Indian
    'kebab', 'shawarma', 'falafel', 'halal', 'indian', 'curry', 'biryani',
    'mediterranean', 'hummus',
    # Misc
    'bakery', 'pastries', 'pastry', 'crepe', 'bagel', 'donut', 'cafe',
    'coffee', 'tea', 'juice',
]
CUISINE_RE = re.compile('|'.join(r'\b' + re.escape(k) + r'\b' for k in CUISINE_KEYWORDS), re.I)


def has_cuisine_keyword(name):
    return bool(CUISINE_RE.search(name or ''))


def load_preds(path):
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r.get('cuebiq_id'), r.get('stop_ts'), r.get('true_poi_id'))
            recs[key] = {
                'truth': r.get('true_poi_id'),
                'next': r.get('next_choice'),
                'top5': [x.get('poi') for x in (r.get('top_5') or [])],
            }
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--lora_pred", required=True)
    p.add_argument("--lgb_pred", required=True)
    args = p.parse_args()

    lora = load_preds(args.lora_pred)
    lgb  = load_preds(args.lgb_pred)

    # Map truth POI to name (string before first |)
    truth_names = {}
    for key in lora:
        name = (key[2] or '').split('|')[0]
        truth_names[key[2]] = name

    # Aggregate
    buckets = defaultdict(lambda: {'n':0, 'l_t1':0, 'g_t1':0, 'l_t5':0, 'g_t5':0})
    for key, lo in lora.items():
        gb = lgb.get(key);
        if not gb: continue
        truth = lo['truth']
        name = truth_names.get(truth, '')
        b = 'cuisine-named' if has_cuisine_keyword(name) else 'generic-named'
        b2 = buckets[b]
        b2['n'] += 1
        if lo['next']  == truth: b2['l_t1'] += 1
        if gb['next']  == truth: b2['g_t1'] += 1
        if truth in lo['top5']: b2['l_t5'] += 1
        if truth in gb['top5']: b2['g_t5'] += 1

    print("="*80)
    print(" OOS 2022 LoRA vs lgb_rank by whether truth POI name has a cuisine keyword")
    print("="*80)
    print(f"{'bucket':<18}{'n':>8}  {'LoRA t1':>8}{'lgb t1':>8}{'Δt1':>9}  {'LoRA t5':>8}{'lgb t5':>8}{'Δt5':>9}")
    print('-'*80)
    for b in ['cuisine-named', 'generic-named']:
        v = buckets[b]
        if v['n'] == 0: continue
        lt1, gt1 = v['l_t1']/v['n'], v['g_t1']/v['n']
        lt5, gt5 = v['l_t5']/v['n'], v['g_t5']/v['n']
        print(f"{b:<18}{v['n']:>8}  {lt1:>8.4f}{gt1:>8.4f}{(lt1-gt1)*100:>+8.2f}pp  "
              f"{lt5:>8.4f}{gt5:>8.4f}{(lt5-gt5)*100:>+8.2f}pp")

    # Sample non-matched (generic) names
    generic_samples = [n for key, lo in list(lora.items())[:2000]
                       for n in [truth_names.get(key[2], '')]
                       if n and not has_cuisine_keyword(n)][:15]
    print("\nSample 'generic-named' POIs:")
    for s in generic_samples: print(f"  {s}")


if __name__ == "__main__":
    main()
