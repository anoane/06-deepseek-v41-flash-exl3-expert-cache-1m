#!/usr/bin/env python3
"""
Markdown tables from a batch's result lines (dsv41_bench.py JSON lines; any harness version).

usage: make_tables.py <run dir or results.jsonl> [more ...] [--drop-provisional] [--drop-stalls] [--out FILE]

  --drop-provisional  leave provisional phases out of every mean (foreign load on cores 0-15 > 0.5, or on a
                      CPU-MoE-pinned core > 0.15; older lines: the phase's own provisional flag). The per-run
                      tables still list them, marked *
  --drop-stalls       also leave out decode measurements with a stall (a step > 5 x the run's median step)

Arms: a configuration's label without a trailing -<number> and without -rerun (A-old-1, A-old-2, A-old-1-rerun
-> A-old), split further by the run's --vary values. Quality lines of one arm are also compared pairwise (a
run-to-run noise floor: bitwise-equal log-probs, top-1 agreement, NLL spread).
"""
import json, os, re, statistics as st, sys

import numpy as np


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


def f(v, nd = 1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list) and isinstance(k, int) and k < len(d):
            d = d[k]
        else:
            return None
    return d


def arm_of(label):
    b = re.sub(r"-rerun$", "", label or "")
    return re.sub(r"-\d+$", "", b)


def vary_key(x):
    v = x.get("vary") or {}
    return ", ".join(f"{k}={v[k]}" for k in sorted(v)) if v else ""


def bad(ph, drop_prov):
    return drop_prov and bool((ph or {}).get("provisional") or (ph or {}).get("moe_cores_contended"))


def ms(xs, nd = 2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{st.mean(xs):.{nd}f} +- {st.stdev(xs):.{nd}f}"


def status_table(rs, out):
    out.append("| config | status | harness | load s | wall s | provisional | MoE cores contended | stalls | cores 16-23 contended | "
               "GPU max C | min MemAvailable during load GiB | GPU mem max during load MiB | error |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        pk = r.get("peaks") or {}
        t = pk.get("gpu_temp_max_c") or {}
        lm = r.get("load_monitor") or {}
        out.append(f"| {r.get('label')} | {r.get('status')} | {r.get('harness_version', 1)} | {f(g(r, 'load', 'seconds'))} | "
                   f"{f(r.get('wall_s'))} | {f(r.get('provisional'))} | {f(r.get('moe_cores_contended'))} | "
                   f"{f(r.get('stall_flag'))} | {f(r.get('cores_16_23_contended'))} | "
                   f"{'/'.join(f(v, 0) for v in t.values()) or '-'} | {f(lm.get('min_mem_available_gib'))} | "
                   f"{'/'.join(f(v, 0) for v in (lm.get('gpu_mem_max_mib') or {}).values()) or '-'} | "
                   f"{(r.get('error') or '')[:140]} |")


def phase_cols(ph):
    return (f"{f(ph.get('tree_read_mib', ph.get('proc_read_mib')), 0)} / {f(ph.get('tree_majflt'), 0)} | "
            f"{f(ph.get('foreign_cores_0_15'), 2)} | {f(ph.get('moe_core_foreign_max'), 2)} / {f(ph.get('moe_core_residual_excess'), 2)} | "
            f"{f(ph.get('cores_16_23_busy_mean_pct'), 0)} | {f(ph.get('tick_coverage_0_15'), 3)} |")


def speed_table(rs, out):
    out.append("| config | run | vary | TTFT s | prefill tok/s | decode tok/s | excl. stalls | TPOT p50 / p90 / p99 ms | stalls | "
               "read MiB / majflt (tree) | foreign 0-15 | MoE core foreign direct / residual excess | 16-23 busy % | tick cov |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        s = r.get("speed")
        if not s:
            continue
        for x in s.get("runs", []):
            ph = x.get("phase", {})
            mark = "*" if ph.get("provisional") else ""
            out.append(f"| {r['label']}{mark} | gen P={x['prompt_tokens']} r{x.get('repeat')} | {vary_key(x)} | {f(x.get('ttft_s'), 2)} | "
                       f"{f(x.get('prefill_tps'), 0)} | {f(x.get('decode_tps'), 2)} | {f(x.get('decode_tps_excl_stalls'), 2)} | "
                       f"{f(x.get('tpot_ms_p50'))} / {f(x.get('tpot_ms_p90'))} / {f(x.get('tpot_ms_p99'))} | {f(x.get('stall_count'))} | "
                       + phase_cols(ph))
        for x in s.get("raw", []):
            ph = x.get("phase", {})
            mark = "*" if ph.get("provisional") else ""
            out.append(f"| {r['label']}{mark} | raw P={x['prompt_tokens']} | | {f(x.get('ttft_s'), 2)} | {f(x.get('prefill_tps'), 0)} | "
                       f"{f(x.get('decode_tps'), 2)} | {f(x.get('decode_tps_excl_stalls'), 2)} | {f(x.get('tpot_ms_p50'))} / "
                       f"{f(x.get('tpot_ms_p90'))} / - | {f(x.get('stall_count'))} | " + phase_cols(ph))


def legacy_table(rs, out):
    out.append("| config | job | TTFT s | prefill tok/s | decode tok/s | steady tok/s | TPOT p50 ms | stalls | "
               "read MiB / majflt (tree) | foreign 0-15 | MoE core foreign direct / residual excess | 16-23 busy % | tick cov |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        L = r.get("legacy")
        if not L:
            continue
        for x in L.get("prefill", []) + L.get("decode", []):
            ph = x.get("phase", {})
            mark = "*" if ph.get("provisional") else ""
            out.append(f"| {r['label']}{mark} | {x.get('label')} | {f(x.get('ttft_s'), 2)} | {f(x.get('prefill_tps'), 0)} | "
                       f"{f(x.get('decode_tps'), 2)} | {f(x.get('steady_tps'), 2)} | {f(x.get('tpot_ms_p50'))} | {f(x.get('stall_count'))} | "
                       + phase_cols(ph))


def arm_summary(rs, out, drop_prov, drop_stalls):
    arms = {}
    for r in rs:
        if r.get("status") != "ok":
            continue
        a = arm_of(r.get("label"))
        s = r.get("speed") or {}
        for x in s.get("runs", []):
            arms.setdefault((a, f"gen P={x['prompt_tokens']}", vary_key(x)), []).append(x)
        for x in s.get("raw", []):
            arms.setdefault((a, f"raw P={x['prompt_tokens']}", ""), []).append(x)
        L = r.get("legacy") or {}
        for x in L.get("prefill", []):
            arms.setdefault((a, f"legacy prefill {x.get('prompt_tokens')}", ""), []).append(x)
        for x in L.get("decode", []):
            arms.setdefault((a, f"legacy decode ctx {x.get('ctx')}", ""), []).append(x)
    out.append(f"| arm | measurement | vary | n (used) | TTFT s | prefill tok/s | decode tok/s | decode excl. stalls | TPOT p50 ms | steady tok/s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for (a, kind, vk), xs in sorted(arms.items()):
        used = [x for x in xs if not bad(x.get("phase"), drop_prov)]
        dec = [x for x in used if not (drop_stalls and x.get("stall_flag"))]
        out.append(f"| {a} | {kind} | {vk} | {len(xs)} ({len(used)}) | {ms([x.get('ttft_s') for x in used])} | "
                   f"{ms([x.get('prefill_tps') for x in used], 0)} | {ms([x.get('decode_tps') for x in dec])} | "
                   f"{ms([x.get('decode_tps_excl_stalls') for x in dec])} | {ms([x.get('tpot_ms_p50') for x in dec])} | "
                   f"{ms([x.get('steady_tps') for x in dec])} |")


def quality_table(rs, out):
    out.append("| config | numerics | N | NLL (all) | score tok/s | anchor | bitwise equal / compared | first diff | top-1 agree | dNLL vs anchor |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        q = r.get("quality")
        if not q:
            continue
        anchors = q.get("anchors") or [{}]
        for i, a in enumerate(anchors):
            head = (f"| {r['label']}{'*' if g(q, 'phase', 'provisional') else ''} | {q.get('numerics')} | {q['tokens']} | "
                    f"{q['nll']:.12f} | {f(q.get('score_tps'), 0)} |" if i == 0 else "| | | | | |")
            if not a:
                out.append(head + " - | - | - | - | - |")
            elif "error" in a:
                out.append(head + f" {os.path.basename(a.get('path', ''))} | error {str(a['error'])[:60]} | | | |")
            else:
                out.append(head + f" {os.path.basename(a['path'])} | {a['bitwise_equal']} / {a['compared']} | {a['first_diff_pos']} | "
                           f"{a['top1_agree_frac']:.6f} | {a['dnll']:+.6f} |")


def quality_pairs(rs, out):
    arms = {}
    for r in rs:
        q = r.get("quality")
        if r.get("status") == "ok" and q and q.get("npz") and os.path.exists(q["npz"]):
            arms.setdefault(arm_of(r["label"]), []).append((r["label"], q))
    rows = []
    for a, qs in sorted(arms.items()):
        if len(qs) < 2:
            continue
        nlls = [q["nll"] for _, q in qs]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                (la, qa), (lb, qb) = qs[i], qs[j]
                za, zb = np.load(qa["npz"]), np.load(qb["npz"])
                x, y = za["actual_lp"], zb["actual_lp"]
                m = min(len(x), len(y))
                eq = x[:m].astype(np.float32).view(np.int32) == y[:m].astype(np.float32).view(np.int32)
                t1 = za["top1"][:m] == zb["top1"][:m]
                d = np.abs(x[:m].astype(np.float64) - y[:m].astype(np.float64))
                nd = np.nonzero(~(eq & t1))[0]
                rows.append(f"| {a} | {la} vs {lb} | {int((eq & t1).sum())} / {m} | {int(nd[0]) + 1 if len(nd) else '-'} | "
                            f"{float(t1.mean()):.6f} | {qa['nll'] - qb['nll']:+.6f} | {float(np.quantile(d, 0.99)):.4f} / {float(d.max()):.3f} | "
                            f"{max(nlls) - min(nlls):.6f} |")
    if rows:
        out.append("| arm | pair | bitwise equal / compared | first diff pos | top-1 agree | dNLL (a - b) | abs dlp p99 / max | arm NLL spread (max - min) |")
        out.append("|---|---|---|---|---|---|---|---|")
        out += rows


def main():
    args = [x for x in sys.argv[1:]]
    drop_prov = "--drop-provisional" in args
    drop_stalls = "--drop-stalls" in args
    outf = None
    if "--out" in args:
        outf = args[args.index("--out") + 1]
        args.remove("--out"); args.remove(outf)
    paths = [x for x in args if not x.startswith("--")]
    if not paths:
        print(__doc__); sys.exit(2)
    rs = load(paths)
    out = [f"# tables for {', '.join(paths)}", "",
           f"Means {'without' if drop_prov else 'including'} provisional phases"
           f"{', without stalled decode measurements' if drop_stalls else ''}; * = provisional phase.", "",
           "## status", ""]
    status_table(rs, out)
    if any(r.get("speed") or r.get("legacy") for r in rs):
        out += ["", "## arms (means +- sd)", ""]
        arm_summary(rs, out, drop_prov, drop_stalls)
    if any(r.get("speed") for r in rs):
        out += ["", "## speed runs", ""]
        speed_table(rs, out)
    if any(r.get("legacy") for r in rs):
        out += ["", "## legacy-shape runs", ""]
        legacy_table(rs, out)
    if any(r.get("quality") for r in rs):
        out += ["", "## quality", ""]
        quality_table(rs, out)
        out += ["", "## quality, same arm pairwise (run-to-run floor)", ""]
        quality_pairs(rs, out)
    text = "\n".join(out) + "\n"
    if outf:
        open(outf, "w").write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
