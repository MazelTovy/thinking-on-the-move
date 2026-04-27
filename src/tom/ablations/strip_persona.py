#!/usr/bin/env python3
"""22_strip_persona.py — Strip the AGENT PERSONA block from LoRA prompts while
keeping the candidate list intact. Tests whether LoRA's top-1 lead over
gradient-boosted rankers depends on the persona text.
"""
import argparse, json, os


MINIMAL_PERSONA = (
    "--- AGENT PERSONA ---\n"
    "### PERSONA\nA worker in New York City, on a weekday lunch break.\n\n"
)


def strip_persona(user_text):
    # Keep everything from '--- CANDIDATE RESTAURANTS ---' onward;
    # replace the persona block with a minimal placeholder.
    idx = user_text.find('--- CANDIDATE RESTAURANTS ---')
    if idx == -1:
        return user_text
    return MINIMAL_PERSONA + user_text[idx:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_in", required=True)
    p.add_argument("--records_out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.records_out) or ".", exist_ok=True)
    n = 0
    with open(args.records_in) as fin, open(args.records_out, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            if 'user' in r:
                r['user'] = strip_persona(r['user'])
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
            if n % 10000 == 0:
                print(f"  {n:,}")
    print(f"done: {n:,} records")


if __name__ == "__main__":
    main()
