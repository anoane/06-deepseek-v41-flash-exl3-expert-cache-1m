#!/usr/bin/env python3
"""T1 summary: per replay variant, the decode misses per token and hit rate per scenario group, and the predicted
decode ms/token of a split from the stage-0 decode model (T = s*t_cmp + (40-s)*t_pro + t_cross + misses*t_miss).
usage: replay_summary.py <replay.json> [--tcmp MS --tpro MS --tcross MS --tmiss MS] [--md]"""
import json, statistics as st, sys

a = sys.argv[1:]
o = json.load(open(a[0]))
prm = {"tcmp": None, "tpro": None, "tcross": 0.0, "tmiss": None}
for k in list(prm):
    if "--" + k in a:
        prm[k] = float(a[a.index("--" + k) + 1])
md = "--md" in a
G = {"S0": ["S0"], "S1-32K": ["S1-32K@100000"], "S2-32K": ["S2-32K@100000"], "S3-t1": ["S3_t1"],
     "S3-follow": [f"S3_t{t}" for t in range(2, 7)], "S5-prose": ["S5_t7"], "S5-back": ["S5_t8"], "S5b": ["S5b"]}
print(f"records {o['records']}, truncated last line {o['header'].get('truncated_last_line')}")
hdr = "| variant | split | pool slots | " + " | ".join(f"{g} miss/tok (hit)" for g in G) + " | S1 d2h/tok |" + \
    (" predicted decode tok/s S1 / S2 / S3-follow |" if prm["tmiss"] else "")
print(hdr)
print("|" + "---|" * (hdr.count("|") - 1))
for v in o["variants"]:
    J = {j["label"]: j for j in v["jobs"]}
    cells, miss = [], {}
    for g, labs in G.items():
        m = [J[l]["decode_misses_per_token"] for l in labs if l in J and "decode_misses_per_token" in J[l]]
        h = [J[l]["decode_hit_rate"] for l in labs if l in J and J[l].get("decode_hit_rate") is not None]
        miss[g] = st.mean(m) if m else None
        cells.append(f"{st.mean(m):.1f} ({st.mean(h):.3f})" if m else "-")
    row = f"| {v['name']} | {v['split']} | {v['slots']} | " + " | ".join(cells) + \
        f" | {J['S1-32K@100000']['decode_d2h_per_token']:.1f} |"
    if prm["tmiss"]:
        s = v["split"]
        base = (s * prm["tcmp"] + (40 - s) * prm["tpro"] + (prm["tcross"] if s else 0.0))
        pr = [1000 / (base + miss[g] * prm["tmiss"]) for g in ("S1-32K", "S2-32K", "S3-follow")]
        row += f" {pr[0]:.1f} / {pr[1]:.1f} / {pr[2]:.1f} |"
    print(row)
