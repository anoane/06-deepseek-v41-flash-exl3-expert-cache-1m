#!/usr/bin/env python3
"""M1: markdown tables of recipe runs (dsv41_bench.py result lines, harness version 2 or 3).

usage: stage_tables.py <run dir or results.jsonl> [...] [--out FILE] [--drop-provisional]

Tables: processes (status, load, memory, VRAM, temperature, tier sizing, SSD reads after the load), generator runs
(one row per timed job), raw loops, legacy mode, quality (per anchor), and per-arm means with min / max over
processes and repeats (arm = label without a trailing -<n> or -rerun, split by --vary cell and prompt length).
A run is excluded from the arm means when it is provisional (foreign load on cores 0-15 / a CPU-MoE core) or its
phase had cores 16-23 more than 25% busy, with --drop-provisional.
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


def vary_str(v):
    if not v:
        return ""
    out = []
    for k, x in v.items():
        if isinstance(x, bool):
            x = int(x)
        out.append(f"{k}={x}")
    return " ".join(out)


def tier_rates(run):
    t = run.get("tier") or {}
    d, p = t.get("decode") or {}, t.get("prefill") or {}
    used = d.get("policy_hits", 0) + d.get("policy_admits", 0) + d.get("policy_transients", 0)
    n = max(1, (run.get("new_tokens") or 1) - 1)
    if not used:
        return None, None, None, None
    return (d.get("policy_hits", 0) / used, d.get("h2d_bytes", 0) / n / 1e6,
            (p.get("layer_h2d_bytes", 0) + p.get("h2d_bytes", 0)) / 1e9, d.get("ssd_reads", 0) + p.get("ssd_reads", 0))


def excluded(ph, drop):
    return drop and bool(ph and (ph.get("provisional") or ph.get("cores_16_23_contended")))


def main():
    args = sys.argv[1:]
    out = None
    drop = False
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    if "--drop-provisional" in args:
        args.remove("--drop-provisional")
        drop = True
    recs = load(args)
    L = []

    L.append("### Processes\n")
    L.append("| label | status | load s | max RSS GiB | cgroup peak GiB | min MemAvail GiB (load / batch) | nvidia-smi min free MiB (0 PRO / 1 CMP) | VRAM peak alloc GiB (cuda:0 / 1) | temp max C (0 / 1) | tier pool slots / RAM slots / RAM GiB / disk-only | SSD reads after load | provisional / 16-23 | note |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        st_ = r.get("status")
        procs = g(r, "peaks", "host_end", "procs") or []
        hwm = max([p.get("hwm_gib") or 0 for p in procs], default = None)
        lm = r.get("load_monitor") or {}
        rss_l = max([x.get("rss_gib") or 0 for x in lm.get("max_rss_gib", [])], default = None)
        rss = max([x for x in (hwm, rss_l) if x is not None], default = None)
        cgp = g(r, "peaks", "cgroup_end", "peak")
        mf = g(r, "peaks", "gpu_mem_min_free_mib") or {}
        vp = g(r, "peaks", "vram_peak_alloc_gib") or []
        tmx = g(r, "peaks", "gpu_temp_max_c") or {}
        ts = r.get("tier_sizing") or {}
        dv = (ts.get("devices") or [{}])[0] if ts else {}
        tier = f"{dv.get('slots')} / {ts.get('tier_slots')} / {f(ts.get('ram_gib'))} / {ts.get('disk_only')}" if ts else "-"
        note = ""
        if st_ != "ok":
            note = (r.get("error") or "")[:160].replace("|", "/")
        L.append(f"| {r.get('label')} | {st_} | {f(g(r, 'load', 'seconds'), 0)} | {f(rss)} | {f(cgp)} | "
                 f"{f(lm.get('min_mem_available_gib'))} / {f(g(r, 'batch_monitor', 'min_mem_available_gib'))} | "
                 f"{mf.get('0', '-')} / {mf.get('1', '-')} | {' / '.join(f(x) for x in vp) if vp else '-'} | "
                 f"{tmx.get('0', '-')} / {tmx.get('1', '-')} | {tier} | {f(r.get('tier_ssd_reads_after_load'))} | "
                 f"{f(r.get('provisional'))} / {f(r.get('cores_16_23_contended'))} | {note} |")

    rows = []
    for r in recs:
        for x in g(r, "speed", "runs") or []:
            rows.append((r, x))
    if rows:
        L.append("\n### Generator runs (one timed job each)\n")
        L.append("| label | rep | cell | P | TTFT s | prefill tok/s | decode tok/s | TPOT p50 / p99 ms | stalls | cached | rc entries | tier hit | H2D MB/tok | prefill staged GB | tier SSD reads | read MiB | 16-23 busy % | prov. | tokens |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r, x in rows:
            ph = x.get("phase") or {}
            hit, h2d, stg, ssd = tier_rates(x)
            L.append(f"| {r.get('label')} | {x.get('repeat')} | {vary_str(x.get('vary'))} | {x.get('prompt_tokens')} | "
                     f"{f(x.get('ttft_s'), 2)} | {f(x.get('prefill_tps'), 0)} | {f(x.get('decode_tps'), 2)} | "
                     f"{f(x.get('tpot_ms_p50'))} / {f(x.get('tpot_ms_p99'))} | {x.get('stall_count')} | "
                     f"{x.get('job_cached_tokens')} | {g(x, 'recurrent_cache', 'entries') if x.get('recurrent_cache') else '-'} | "
                     f"{f(hit, 3)} | {f(h2d)} | {f(stg)} | {f(ph.get('tier_ssd_reads'))} | {f(ph.get('proc_read_mib'))} | "
                     f"{f(ph.get('cores_16_23_busy_mean_pct'))} | {'P' if ph.get('provisional') else ''}"
                     f"{'C' if ph.get('cores_16_23_contended') else ''} | {x.get('tokens_sha256_12')} |")

    raws = [(r, x) for r in recs for x in (g(r, "speed", "raw") or [])]
    if raws:
        L.append("\n### Raw forward loops\n")
        L.append("| label | P | prefill tok/s | TTFT s | decode tok/s | TPOT p50 ms | stalls | greedy prefix = generator | prov. |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r, x in raws:
            ph = x.get("phase") or {}
            L.append(f"| {r.get('label')} | {x.get('prompt_tokens')} | {f(x.get('prefill_tps'), 0)} | {f(x.get('ttft_s'), 2)} | "
                     f"{f(x.get('decode_tps'), 2)} | {f(x.get('tpot_ms_p50'))} | {x.get('stall_count')} | "
                     f"{x.get('greedy_prefix_equal_to_generator', '-')} | {'P' if ph.get('provisional') else ''}"
                     f"{'C' if ph.get('cores_16_23_contended') else ''} |")

    leg = [r for r in recs if r.get("legacy")]
    if leg:
        L.append("\n### Legacy mode (the 23.4 record's shape)\n")
        L.append("| label | prefill 8192 tok/s | prefill 33600 tok/s | decode ctx 24 tok/s (steady) | decode ctx 20000 tok/s (steady) | prov. |")
        L.append("|---|---|---|---|---|---|")
        for r in leg:
            lg = r["legacy"]
            pf = {x.get("prompt_tokens"): x for x in lg.get("prefill", [])}
            dc = {x.get("ctx"): x for x in lg.get("decode", [])}
            def pv(n):
                x = next((v for k, v in pf.items() if k == n), None)
                return f(x.get("prefill_tps"), 0) if x else "-"
            def dv_(n):
                x = dc.get(n)
                return f"{f(x.get('decode_tps'), 2)} ({f(x.get('steady_tps'), 2)})" if x else "-"
            L.append(f"| {r.get('label')} | {pv(8192)} | {pv(33600)} | {dv_(24)} | {dv_(20000)} | {f(r.get('provisional'))} |")

    qs = [r for r in recs if r.get("quality")]
    if qs:
        L.append("\n### Quality (64K teacher-forced)\n")
        L.append("| label | chunk | NLL | anchor | bitwise equal / compared | first diff | top-1 agree | dNLL |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in qs:
            q = r["quality"]
            for i, c in enumerate(q.get("anchors") or [{}]):
                name = os.path.basename(c.get("path", "-"))
                if "error" in c:
                    L.append(f"| {r.get('label') if i == 0 else ''} | {q.get('chunk') if i == 0 else ''} | "
                             f"{f(q.get('nll'), 6) if i == 0 else ''} | {name} | error: {c['error'][:80]} | | | |")
                    continue
                L.append(f"| {r.get('label') if i == 0 else ''} | {q.get('chunk') if i == 0 else ''} | "
                         f"{f(q.get('nll'), 6) if i == 0 else ''} | {name} | {c.get('bitwise_equal')} / {c.get('compared')} | "
                         f"{f(c.get('first_diff_pos'))} | {f(c.get('top1_agree_frac'), 4)} | "
                         f"{'-' if c.get('dnll') is None else format(c['dnll'], '+.6f')} |")

    arms = {}
    for r, x in rows:
        if r.get("status") != "ok":
            continue
        ph = x.get("phase") or {}
        if excluded(ph, drop):
            continue
        key = (arm_of(r.get("label")), vary_str(x.get("vary")), x.get("prompt_tokens"))
        a = arms.setdefault(key, {"dec": [], "pre": [], "ttft": [], "procs": set()})
        if x.get("decode_tps"):
            a["dec"].append(x["decode_tps"])
        if x.get("prefill_tps"):
            a["pre"].append(x["prefill_tps"])
        if x.get("ttft_s"):
            a["ttft"].append(x["ttft_s"])
        a["procs"].add(r.get("run_id"))
    if arms:
        L.append("\n### Arms: mean [min, max] over processes and repeats\n")
        L.append("| arm | cell | P | n (procs) | decode tok/s | prefill tok/s | TTFT s |")
        L.append("|---|---|---|---|---|---|---|")
        def mm(v, nd):
            if not v:
                return "-"
            return f"{st.mean(v):.{nd}f} [{min(v):.{nd}f}, {max(v):.{nd}f}]"
        for (arm, cell, P), a in arms.items():
            L.append(f"| {arm} | {cell} | {P} | {len(a['dec'])} ({len(a['procs'])}) | {mm(a['dec'], 2)} | {mm(a['pre'], 0)} | "
                     f"{mm(a['ttft'], 2)} |")
    txt = "\n".join(L) + "\n"
    if out:
        open(out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
