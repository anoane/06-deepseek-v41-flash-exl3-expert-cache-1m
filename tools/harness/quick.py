#!/usr/bin/env python3
"""quick.py <results.jsonl> [label ...]: per line, per scenario group: decode / prefill / hit, and against the
reference of its layout (C12-R*, C11-R*, stage-1 A1M), plus greedy-token equality with that reference."""
import json, re, statistics as st, sys

sys.path.insert(0, "/root/rnd/release/recipe/bench3")
import h2h_tables as H

res = sys.argv[1]
want = sys.argv[2:]
recs = [json.loads(l) for l in open(res)]
s1 = [json.loads(l) for l in open("/root/rnd/release/recipe/stage1/w1/results.jsonl")]


def groups(r):
    out = {}
    for sec in ("h2h", "longctx"):
        for m in (r.get(sec) or {}).get("jobs", []):
            g = H.group_of(m)
            ts = m.get("tier_summary") or {}
            pf = m.get("delta_prefill_tps") if g.startswith(("S3", "S5")) and m["scenario"] != "S5b" else m.get("prefill_tps")
            o = out.setdefault(g, {"d": [], "p": [], "h": [], "t": [], "dig": {}})
            if m.get("decode_tps") is not None:
                o["d"].append(m["decode_tps"])
            if pf is not None:
                o["p"].append(pf)
            if ts.get("decode_hit_rate") is not None:
                o["h"].append(ts["decode_hit_rate"])
            if m.get("ttft_s") is not None:
                o["t"].append(m["ttft_s"])
            o["dig"][(m["scenario"], m.get("turn"))] = m.get("tokens_sha256_12")
    return out


def ref_recs(label):
    if label.startswith("C12"):
        return [r for r in recs if re.match(r"^C12-R\d+$", r["label"]) and r.get("status") == "ok"] + \
            [r for r in s1 if re.match(r"^C12-\d+$", r["label"])]
    if label.startswith("C11"):
        rr = [r for r in recs if re.match(r"^C11-R\d+$", r["label"]) and r.get("status") == "ok"]
        return rr or [r for r in s1 if re.match(r"^C11-\d+$", r["label"])]
    if label.startswith("A1M"):
        return [r for r in s1 if re.match(r"^A1M-\d+$", r["label"])]
    return []


def m(v):
    return st.mean(v) if v else None


for r in recs:
    if want and r["label"] not in want:
        continue
    ts = r.get("tier_sizing") or {}
    dv = (ts.get("devices") or [{}])[0] if ts else {}
    procs = H.g(r, "peaks", "host_end", "procs") or []
    rss = max([p.get("hwm_gib") or 0 for p in procs] or [0])
    print(f"=== {r['label']} {r.get('status')} {str(r.get('error'))[:300] if r.get('status') != 'ok' else ''}")
    print(f"    pool {dv.get('slots')} staging {dv.get('staging_slots')} ram {ts.get('ram_gib')} ngram {ts.get('ngram_gib')} "
          f"unpinned {ts.get('left_unpinned_gib')} ssd_after_load {r.get('tier_ssd_reads_after_load')} rss {rss:.2f} "
          f"minfree {H.g(r, 'peaks', 'gpu_mem_min_free_mib')} oomkill {H.g(r, 'peaks', 'cgroup_end', 'events_oom_kill')}")
    if r.get("status") != "ok":
        continue
    G = groups(r)
    refs = [groups(x) for x in ref_recs(r["label"]) if x is not r]
    for g, o in G.items():
        rd = [m(x[g]["d"]) for x in refs if g in x and x[g]["d"]]
        rp = [m(x[g]["p"]) for x in refs if g in x and x[g]["p"]]
        d, p = m(o["d"]), m(o["p"])
        dd = f"{100 * (d / m(rd) - 1):+5.1f}%" if d and rd else "   -  "
        pd = f"{100 * (p / m(rp) - 1):+5.1f}%" if p and rp else "   -  "
        spr = f"ref d {min(rd):.2f}-{max(rd):.2f}" if len(rd) > 1 else ""
        print(f"    {g:16s} dec {d if d is None else round(d, 2)!s:>7} {dd}  pf {p if p is None else round(p):>6} {pd}  "
              f"hit {m(o['h']) and round(m(o['h']), 3)}  ttft {m(o['t']) and round(m(o['t']), 2)}  {spr}")
    if refs:
        rdig = {}
        for x in ref_recs(r["label"]):
            if x is r:
                continue
            for sec in ("h2h", "longctx"):
                for mm in (x.get(sec) or {}).get("jobs", []):
                    rdig.setdefault((mm["scenario"], mm.get("turn")), set()).add(mm.get("tokens_sha256_12"))
        mine = {}
        for g, o in G.items():
            mine.update(o["dig"])
        cmp = [(k, v in rdig[k]) for k, v in mine.items() if k in rdig]
        diff = [f"{k[0]}{'_t' + str(k[1]) if k[1] else ''}" for k, eq in cmp if not eq]
        print(f"    tokens equal to the reference: {sum(eq for _, eq in cmp)} / {len(cmp)}  differ: {', '.join(diff) or '-'}")
