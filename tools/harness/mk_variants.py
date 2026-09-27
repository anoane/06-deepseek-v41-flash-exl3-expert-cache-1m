#!/usr/bin/env python3
"""T1 variant list from measured sizings. usage: mk_variants.py <out.json> <a results.jsonl>:<a label>
[<split>=<results.jsonl>:<label> ...] [--c2-from 8]
Split 0 = config a itself (its own pool / RAM slots, SSD backing): the replay's check against the measured counters.
For every family-c split: the measured pool (slots) and RAM = every other cached expert (disk off), under the default
policy and one-factor policy variants (admit, evict, spare, halflife, prefill heat, lazy-exclusive, +160 / +384
slots for a 2 GiB headroom cut / single staging)."""
import json, sys


def sizing(spec):
    path, label = spec.rsplit(":", 1)
    last = None
    for line in open(path):
        j = json.loads(line)
        if j.get("label") == label and j.get("tier_sizing"):
            last = j["tier_sizing"]
    if last is None:
        raise SystemExit(f"no tier sizing for {label} in {path}")
    d = last["devices"][0]
    return {"slots": d["slots"], "spare": d["spare"], "ram_slots": last["tier_slots"], "cached": last["cached"]}


out_path, a_spec = sys.argv[1], sys.argv[2]
splits = {}
c2_from = None
args = sys.argv[3:]
i = 0
while i < len(args):
    if args[i] == "--c2-from":
        c2_from = int(args[i + 1]); i += 2
        continue
    s, spec = args[i].split("=", 1)
    splits[int(s)] = sizing(spec)
    i += 1
a = sizing(a_spec)
V = [{"name": "a-own", "split": 0, "slots": a["slots"], "ram_slots": a["ram_slots"], "disk_off": False,
      "policy": {"spare": a["spare"]}}]
if c2_from is not None and c2_from in splits:
    splits[2] = dict(splits[c2_from], note = f"pool of split {c2_from} (no fit measured)")
POL = [("base", {}), ("admit-heat", {"admit": "heat"}), ("admit-always", {"admit": "always"}), ("evict-lfu", {"evict": "lfu"}),
       ("spare4", {"spare": 4}), ("spare1", {"spare": 1}), ("hl64", {"halflife": 64}), ("hl1024", {"halflife": 1024}),
       ("hp0", {"prefill_inc": 0}), ("hp0.25", {"prefill_inc": 16384}), ("lazy", {"policy": "lazy-exclusive"})]
for s, z in sorted(splits.items()):
    for name, pol in POL:
        p = {"spare": z["spare"], **pol}
        V.append({"name": f"c{s}-{name}", "split": s, "slots": z["slots"], "ram_slots": "all", "disk_off": True,
                  "policy": p})
    for extra, why in ((160, "headroom 2048"), (384, "staging single")):
        V.append({"name": f"c{s}-slots+{extra}", "split": s, "slots": z["slots"] + extra, "ram_slots": "all",
                  "disk_off": True, "policy": {"spare": z["spare"]}, "note": why})
json.dump(V, open(out_path, "w"), indent = 1)
print(f"{len(V)} variants -> {out_path}")
