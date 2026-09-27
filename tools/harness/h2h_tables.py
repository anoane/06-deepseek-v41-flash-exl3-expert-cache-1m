#!/usr/bin/env python3
"""M1 for the head-to-head modes: markdown tables of h2h / longctx result lines (harness version 4).

usage: h2h_tables.py <run dir or results.jsonl> [...] [--out FILE] [--ref ARM] [--exclude ARM,...] [--jsonl FILE]

Tables: processes (status, load, sizing, memory, VRAM), per scenario and arm (mean [min-max] over every job of the
arm in that scenario: decode tok/s, prefill tok/s, TTFT, hit rate, H2D / D2H MB per decode token), S3 per turn, S5
switch / recovery, S4 long context, H7 device time, and one verdict line per scenario: the reference arm (--ref,
default A1M = config a) against the best other arm in decode and in prefill. Arm = label without a trailing -<n>
or -rerun. Provisional jobs (foreign load) are listed but left out of the means (--keep-provisional keeps them).
--jsonl writes one flat row per job (arm, process, scenario, metrics) for later stages.
"""
import json, os, re, statistics as st, sys


def load(paths):
    recs = []
    for p in paths:
        f = os.path.join(p, "results.jsonl") if os.path.isdir(p) else p
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return recs


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return None
    return d


def f(v, nd = 1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def arm_of(label):
    return re.sub(r"-\d+$", "", re.sub(r"-rerun$", "", label or ""))


def group_of(m):
    """scenario group used for the head-to-head: S1-32K / S2-32K pool both offsets; S3 turn 1 vs follow-ups"""
    s = m["scenario"]
    if s.startswith(("S1-", "S2-")):
        return s.split("@")[0]
    if s == "S3":
        return "S3-t1" if m.get("turn") == 1 else "S3-follow"
    if s == "S5":
        return "S5-prose" if m.get("source") == "prose" else "S5-back"
    if s.startswith("S4"):
        return s + ("-prefill" if m.get("turn") == 0 else "-decode")
    return s


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return (st.mean(v), min(v), max(v), len(v))


def fs(s, nd = 1):
    if s is None:
        return "-"
    m, lo, hi, n = s
    return f"{m:.{nd}f} [{lo:.{nd}f}-{hi:.{nd}f}] n{n}" if n > 1 else f"{m:.{nd}f} n1"


def rows_of(recs, keep_prov):
    out = []
    for r in recs:
        if r.get("status") != "ok":
            continue
        for sec in ("h2h", "longctx"):
            for m in (r.get(sec) or {}).get("jobs", []):
                ts = m.get("tier_summary") or {}
                pc = m.get("pcie") or {}
                prov = bool(g(m, "phase", "provisional"))
                row = {"arm": arm_of(r["label"]), "label": r["label"], "run_id": r.get("run_id"),
                       "scenario": m["scenario"], "turn": m.get("turn"), "group": group_of(m),
                       "source": m.get("source"), "P": m.get("prompt_tokens"), "cached": m.get("cached_tokens"),
                       "prefilled": m.get("prefilled_tokens"), "G": m.get("new_tokens"),
                       "ttft_s": m.get("ttft_s"), "prefill_tps": m.get("prefill_tps"),
                       "delta_prefill_tps": m.get("delta_prefill_tps"), "decode_tps": m.get("decode_tps"),
                       "tpot_p50": m.get("tpot_ms_p50"), "tpot_p99": m.get("tpot_ms_p99"),
                       "stalls": m.get("stall_count"), "hit": ts.get("decode_hit_rate"),
                       "miss_tok": ts.get("decode_misses_per_token"), "h2d_mb_tok": ts.get("decode_h2d_mb_per_token"),
                       "d2h_mb_tok": ts.get("decode_d2h_mb_per_token"), "ssd_mb_tok": ts.get("decode_ssd_mb_per_token"),
                       "staged_gb": ts.get("prefill_staged_gb"), "pf_ssd_gb": ts.get("prefill_ssd_gb"),
                       "rows_hit": g(m, "phase", "ngram_row_cache", "hit_rate"),
                       "pcie": pc, "provisional": prov, "timeline": m.get("timeline"),
                       "recovery": m.get("recovery_vs_last_S3_turn"),
                       "rc_entries": g(m, "recurrent_cache", "entries"), "rc_gib": g(m, "recurrent_cache", "bytes_gib")}
                if prov and not keep_prov:
                    row["excluded"] = True
                out.append(row)
    return out


def proc_table(recs):
    L = ["| label | status | load s | placement | tier slots / RAM slots / RAM GiB / disk-only | SSD reads after load | "
         "max RSS GiB | cgroup peak GiB | min MemAvail GiB | VRAM used max MiB per nvsmi GPU | min free MiB | engram drop GiB | "
         "provisional |", "|" + "---|" * 13]
    for r in recs:
        ts = r.get("tier_sizing") or {}
        dv = (ts.get("devices") or [{}])[0] if ts else {}
        rss = max([p.get("rss_gib", 0) for p in g(r, "load_monitor", "max_rss_gib") or []] +
                  [p.get("hwm_gib", 0) for p in g(r, "peaks", "host_end", "procs") or []] or [None]) \
            if (g(r, "load_monitor", "max_rss_gib") or g(r, "peaks", "host_end", "procs")) else None
        mav = [x for x in (g(r, "batch_monitor", "min_mem_available_gib"), g(r, "load_monitor", "min_mem_available_gib"))
               if x is not None]
        sz = f"{dv.get('slots')} / {ts.get('tier_slots')} / {ts.get('ram_gib')} / {ts.get('disk_only')}" if ts else "-"
        L.append(f"| {r.get('label')} | {r.get('status')}{(': ' + str(r.get('error'))[:120]) if r.get('status') != 'ok' else ''} | "
                 f"{f(g(r, 'load', 'seconds'))} | `{g(r, 'config', 'placement') or g(r, 'env', 'EXL3_DSV41_PLACEMENT') or '-'}` | {sz} | "
                 f"{f(r.get('tier_ssd_reads_after_load'))} | {f(rss, 2)} | {f(g(r, 'peaks', 'cgroup_end', 'peak'), 2)} | "
                 f"{f(min(mav) if mav else None, 1)} | {g(r, 'peaks', 'gpu_mem_max_mib')} | {g(r, 'peaks', 'gpu_mem_min_free_mib')} | "
                 f"{f((g(r, 'engram_drop', 'bytes') or 0) / 2**30, 1) if r.get('engram_drop') else '-'} | {f(r.get('provisional'))} |")
    return L


ORDER = ["S0", "S1-32K", "S2-32K", "S1-128K", "S3-t1", "S3-follow", "S5-prose", "S5-back", "S5b",
         "S4-512K-prefill", "S4-512K-decode", "S4-1016K-prefill", "S4-1016K-decode"]


def scen_table(rows):
    arms = sorted({r["arm"] for r in rows})
    groups = [x for x in ORDER if any(r["group"] == x for r in rows)] + \
        sorted({r["group"] for r in rows} - set(ORDER))
    L = ["| scenario | arm | decode tok/s | prefill tok/s (S3/S5: new rows / TTFT) | TTFT s | hit | miss / token | "
         "H2D MB/token | D2H MB/token | staged GB (prefill) | n-gram rows hit |", "|" + "---|" * 11]
    cells = {}
    for gname in groups:
        for arm in arms:
            rs = [r for r in rows if r["group"] == gname and r["arm"] == arm and not r.get("excluded")]
            if not rs:
                continue
            pf = [r["delta_prefill_tps"] if gname.startswith(("S3", "S5")) and r["scenario"] != "S5b" else r["prefill_tps"]
                  for r in rs]
            c = {"decode": stat([r["decode_tps"] for r in rs]), "prefill": stat(pf), "ttft": stat([r["ttft_s"] for r in rs]),
                 "hit": stat([r["hit"] for r in rs]), "miss": stat([r["miss_tok"] for r in rs]),
                 "h2d": stat([r["h2d_mb_tok"] for r in rs]), "d2h": stat([r["d2h_mb_tok"] for r in rs]),
                 "staged": stat([r["staged_gb"] for r in rs]), "rows": stat([r["rows_hit"] for r in rs])}
            cells[(gname, arm)] = c
            L.append(f"| {gname} | {arm} | {fs(c['decode'], 2)} | {fs(c['prefill'], 0)} | {fs(c['ttft'], 2)} | "
                     f"{fs(c['hit'], 3)} | {fs(c['miss'], 1)} | {fs(c['h2d'], 0)} | {fs(c['d2h'], 0)} | {fs(c['staged'], 1)} | "
                     f"{fs(c['rows'], 3)} |")
    return L, cells, groups, arms


def turn_table(rows):
    L = ["| arm | process | turn | source | P | cached | prefilled | TTFT s | new rows / s | decode tok/s | hit | "
         "recurrent stashes / GiB |", "|" + "---|" * 12]
    for r in rows:
        if r["scenario"] in ("S3", "S5", "S5b"):
            L.append(f"| {r['arm']} | {r['run_id'][-24:] if r['run_id'] else '-'} | {r['scenario']} {r['turn'] or ''} | "
                     f"{r['source']} | {r['P']} | {r['cached']} | {r['prefilled']} | {f(r['ttft_s'], 2)} | "
                     f"{f(r['delta_prefill_tps'], 0)} | {f(r['decode_tps'], 2)} | {f(r['hit'], 3)} | "
                     f"{f(r['rc_entries'])} / {f(r['rc_gib'], 3)} |")
    return L


def recovery_table(rows):
    L = ["| arm | process | job | ref decode (S3 t6) | worst window tok/s | collapse % | decode recovered after tokens | "
         "worst hit | hit recovered after tokens |", "|" + "---|" * 9]
    for r in rows:
        rv = r.get("recovery")
        if rv:
            L.append(f"| {r['arm']} | {r['run_id'][-24:] if r['run_id'] else '-'} | {r['scenario']} {r['turn'] or ''} | "
                     f"{f(rv.get('ref_tps'), 2)} | {f(rv.get('worst_tps'), 2)} | {f(rv.get('collapse_pct'), 1)} | "
                     f"{f(rv.get('recovered_tps_after_tokens'))} | {f(rv.get('worst_hit'), 3)} | "
                     f"{f(rv.get('recovered_hit_after_tokens'))} |")
    return L


def devtime_table(recs):
    L = ["| label | wall ms/token | per device: module ms/token, blocks, layer ms mean / p50 / p90 | residual ms/token |",
         "|---|---|---|---|"]
    for r in recs:
        d = g(r, "h2h", "devtime")
        if not d:
            continue
        s = d.get("summary")
        if not s:
            L.append(f"| {r['label']} | - | error: {d.get('error')} | - |")
            continue
        parts = []
        for dev, x in s["devices"].items():
            parts.append(f"cuda:{dev}: {x['module_ms_per_token']:.2f} ms, {x.get('blocks', '-')} blocks, "
                         f"{f(x.get('layer_ms_mean'), 3)} / {f(x.get('layer_ms_p50'), 3)} / {f(x.get('layer_ms_p90'), 3)}")
        L.append(f"| {r['label']} | {s['wall_ms_per_token']:.2f} | {'; '.join(parts)} | {s['residual_ms_per_token']:.2f} |")
    return L


def verdicts(cells, groups, arms, ref):
    L = []
    for gname in groups:
        a = cells.get((gname, ref))
        if a is None:
            continue
        for what, key in (("decode", "decode"), ("prefill", "prefill")):
            if gname.startswith("S4") and ((what == "prefill") != gname.endswith("prefill")):
                continue
            others = [(arm, cells[(gname, arm)][key]) for arm in arms if arm != ref and (gname, arm) in cells
                      and cells[(gname, arm)][key] is not None]
            if a[key] is None or not others:
                continue
            best_arm, best = max(others, key = lambda x: x[1][0])
            av, bv = a[key][0], best[0]
            win = ref if av >= bv else best_arm
            pct = 100 * (max(av, bv) / min(av, bv) - 1) if min(av, bv) > 0 else float("inf")
            L.append(f"- {gname} {what}: {ref} {av:.{2 if what == 'decode' else 0}f} vs {best_arm} "
                     f"{bv:.{2 if what == 'decode' else 0}f} tok/s: **{win} wins by {pct:.1f}%** "
                     f"(n {a[key][3]} vs {best[3]})")
    return L


def main():
    args = sys.argv[1:]
    out, ref, jl, keep, excl = None, "A1M", None, False, set()
    paths = []
    i = 0
    while i < len(args):
        if args[i] == "--out":
            out = args[i + 1]; i += 2
        elif args[i] == "--ref":
            ref = args[i + 1]; i += 2
        elif args[i] == "--jsonl":
            jl = args[i + 1]; i += 2
        elif args[i] == "--exclude":
            excl = set(args[i + 1].split(",")); i += 2
        elif args[i] == "--keep-provisional":
            keep = True; i += 1
        else:
            paths.append(args[i]); i += 1
    recs = [r for r in load(paths) if any(m in (g(r, "args", "modes") or []) for m in ("h2h", "longctx"))
            or r.get("h2h") or r.get("longctx")]
    rows = [r for r in rows_of(recs, keep) if r["arm"] not in excl]
    L = ["### Processes (h2h / longctx lines)", ""] + proc_table(recs) + [""]
    t, cells, groups, arms = scen_table(rows)
    L += ["### Per scenario (mean [min-max] over jobs; provisional jobs excluded)", ""] + t + [""]
    L += ["### Verdicts (reference " + ref + " vs the best other arm)", ""] + verdicts(cells, groups, arms, ref) + [""]
    L += ["### S3 / S5 per turn", ""] + turn_table(rows) + [""]
    L += ["### S5 collapse and recovery (vs the last S3 turn)", ""] + recovery_table(rows) + [""]
    L += ["### H7 device time (32 decode steps, warm 32K text)", ""] + devtime_table(recs) + [""]
    txt = "\n".join(L)
    if out:
        open(out, "w").write(txt + "\n")
    print(txt)
    if jl:
        with open(jl, "w") as fh:
            for r in rows:
                r2 = {k: v for k, v in r.items() if k not in ("timeline",)}
                fh.write(json.dumps(r2) + "\n")


if __name__ == "__main__":
    main()
