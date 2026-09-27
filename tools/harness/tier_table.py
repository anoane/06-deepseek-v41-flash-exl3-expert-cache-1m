#!/usr/bin/env python3
"""tier_table.py results.jsonl [...]: one line per configuration: medians over the repeats of decode and
prefill tok/s and TTFT, the decode hit rate, bytes per decode token (SSD, H2D, D2H), what the prefill
staged and read, the load's time"""
import json
import statistics as st
import sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, fmt):
    return "-" if x is None else format(x, fmt)


rows = []
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        lab = r.get("label")
        if r.get("status") != "ok" or "speed" not in r:
            rows.append((lab, f"{r.get('status')}: {str(r.get('error'))[:80]}"))
            continue
        runs = r["speed"]["runs"]
        dec = [x.get("decode_tps") for x in runs]
        pre = [x.get("prefill_tps") for x in runs]
        ttft = [x.get("ttft_s") for x in runs]
        hit, ssd, h2d, d2h, pssd, pstage = [], [], [], [], [], []
        for x in runs:
            t = x.get("tier") or {}
            d, p = t.get("decode") or {}, t.get("prefill") or {}
            used = d.get("policy_hits", 0) + d.get("policy_admits", 0) + d.get("policy_transients", 0)
            n = max(1, (x.get("new_tokens") or 1) - 1)
            if used:
                hit.append(d.get("policy_hits", 0) / used)
                ssd.append(d.get("ssd_bytes", 0) / n / 1e6)
                h2d.append(d.get("h2d_bytes", 0) / n / 1e6)
                d2h.append(d.get("d2h_bytes", 0) / n / 1e6)
            if p:
                pssd.append(p.get("ssd_bytes", 0) / 1e9)
                pstage.append((p.get("layer_h2d_bytes", 0) + p.get("h2d_bytes", 0)) / 1e9)
        load = (r.get("load") or {}).get("wall_s") or (r.get("load") or {}).get("seconds")
        rows.append((lab, f"decode {f(med(dec), '6.2f')} tok/s [{' '.join(f(x, '.2f') for x in dec)}]  prefill "
                          f"{f(med(pre), '7.1f')} tok/s  TTFT {f(med(ttft), '5.2f')} s  hit {f(med(hit), '.4f')}  "
                          f"SSD {f(med(ssd), '6.1f')} H2D {f(med(h2d), '6.1f')} D2H {f(med(d2h), '6.1f')} MB/tok  "
                          f"prefill SSD {f(med(pssd), '5.2f')} GB staged {f(med(pstage), '6.1f')} GB  load {f(load, '.0f')} s"))
w = max(len(str(l)) for l, _ in rows) if rows else 10
for l, s in rows:
    print(f"{str(l):{w}}  {s}")
