#!/usr/bin/env python3
"""Stage-1 tables: config a vs the family-c splits in S0-S5, pooled over window 0c and the stage-1 window(s).

usage: stage1_tables.py --out FILE --jsonl FILE --runs FILE <results.jsonl or run dir> [...]

Uses h2h_tables.py (M1) for the row extraction and the per-scenario cells, with these arm rules:
  C12-base-<n> (window 0c)   -> C12   (the same plain C12 line as stage 1's C12-<n>)
  <arm>-S4                   -> <arm> (the S4-512K lines: A1M-S4 from 0c, C12-S4 / C11-S4 / ... from stage 1)
Adds: per-process means (decode, prefill) per scenario group, so the spread between processes is visible; one verdict
line per scenario and c arm against a, and one against the best c arm; a compact summary table (decode / prefill a vs
every c arm); the quality lines (Q-*) against their anchors.
"""
import json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2h_tables as H


def arm_of(label):
    a = re.sub(r"-\d+$", "", re.sub(r"-rerun$", "", label or ""))
    a = re.sub(r"-S4$", "", a)
    if a == "C12-base":
        a = "C12"
    return a


H.arm_of = arm_of

GROUP_ORDER = H.ORDER


def per_process(rows):
    """{(group, arm): {run_id: (decode mean, prefill mean)}}"""
    out = {}
    for r in rows:
        if r.get("excluded"):
            continue
        k = (r["group"], r["arm"])
        pf = r["delta_prefill_tps"] if r["group"].startswith(("S3", "S5")) and r["scenario"] != "S5b" else r["prefill_tps"]
        out.setdefault(k, {}).setdefault(r["run_id"], ([], []))
        if r["decode_tps"] is not None:
            out[k][r["run_id"]][0].append(r["decode_tps"])
        if pf is not None:
            out[k][r["run_id"]][1].append(pf)
    res = {}
    for k, d in out.items():
        res[k] = {rid: (st.mean(a) if a else None, st.mean(b) if b else None) for rid, (a, b) in d.items()}
    return res


def fmt_pp(vals, nd):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    return " / ".join(f"{x:.{nd}f}" for x in v)


def main():
    args = sys.argv[1:]
    out = jl = runs = None
    paths = []
    i = 0
    while i < len(args):
        if args[i] in ("--out", "--jsonl", "--runs"):
            if args[i] == "--out":
                out = args[i + 1]
            elif args[i] == "--jsonl":
                jl = args[i + 1]
            else:
                runs = args[i + 1]
            i += 2
        else:
            paths.append(args[i]); i += 1
    allrecs = H.load(paths)
    recs = [r for r in allrecs if r.get("h2h") or r.get("longctx")]
    recs = [r for r in recs if "trace" not in (r.get("label") or "")]   # trace captures: not timed
    rows = H.rows_of(recs, False)
    for r in rows:   # S4 decode repeats: the 64-token suffix's prefill rate is not a prefill measurement
        if r["group"].startswith("S4") and r["group"].endswith("decode"):
            r["prefill_tps"] = None
    L = ["### Processes (h2h / longctx lines)", ""] + H.proc_table(recs) + [""]
    t, cells, groups, arms = H.scen_table(rows)
    L += ["### Per scenario (mean [min-max] over jobs; provisional jobs excluded)", ""] + t + [""]

    # per-process table
    pp = per_process(rows)
    cs = [a for a in arms if a.startswith("C")]
    order = ["A1M"] + sorted(cs, key = lambda a: (not re.match(r"^C\d+$", a), -int(re.sub(r"\D.*", "", a[1:]) or 0), a)) + \
        [a for a in arms if a != "A1M" and not a.startswith("C")]
    L += ["### Per process (decode tok/s | prefill tok/s, one value per process, in run order)", "",
          "| scenario | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for gname in groups:
        cellsx = []
        for a in order:
            d = pp.get((gname, a))
            if not d:
                cellsx.append("-")
                continue
            ids = sorted(d)
            cellsx.append(fmt_pp([d[x][0] for x in ids], 1) + " \\| " + fmt_pp([d[x][1] for x in ids], 0))
        L.append(f"| {gname} | " + " | ".join(cellsx) + " |")
    L.append("")

    # summary a vs each c
    L += ["### Summary: decode and prefill tok/s, mean over jobs (a = A1M; + = c faster than a)", "",
          "| scenario | a decode | " + " | ".join(f"{c} decode" for c in order[1:] if c.startswith("C")) +
          " | a prefill | " + " | ".join(f"{c} prefill" for c in order[1:] if c.startswith("C")) + " |",
          "|---|" + "---|" * (2 * len([c for c in order if c.startswith("C")]) + 2)]
    cord = [c for c in order if c.startswith("C")]

    def cell(gname, arm, key, ref, nd):
        c = cells.get((gname, arm))
        if not c or c[key] is None:
            return "-"
        v = c[key][0]
        r = cells.get((gname, ref))
        if arm != ref and r and r[key] is not None and r[key][0] > 0:
            return f"{v:.{nd}f} ({100 * (v / r[key][0] - 1):+.0f}%)"
        return f"{v:.{nd}f}"

    for gname in groups:
        L.append(f"| {gname} | {cell(gname, 'A1M', 'decode', 'A1M', 2)} | " +
                 " | ".join(cell(gname, c, "decode", "A1M", 2) for c in cord) +
                 f" | {cell(gname, 'A1M', 'prefill', 'A1M', 0)} | " +
                 " | ".join(cell(gname, c, "prefill", "A1M", 0) for c in cord) + " |")
    L.append("")

    # verdicts: a vs best whole-layer c arm (C<digits>) and vs each
    main_c = [c for c in cord if re.match(r"^C\d+$", c)]
    L += ["### Verdicts per scenario: a vs the best family-c split (C<split> arms only)", ""]
    for gname in groups:
        a = cells.get((gname, "A1M"))
        if a is None:
            continue
        for what in ("decode", "prefill"):
            if gname.startswith("S4") and ((what == "prefill") != gname.endswith("prefill")):
                continue
            others = [(c, cells[(gname, c)][what]) for c in main_c if (gname, c) in cells and cells[(gname, c)][what]]
            if a[what] is None or not others:
                continue
            best, bv = max(others, key = lambda x: x[1][0])
            av = a[what][0]
            win = "a" if av >= bv[0] else best
            pct = 100 * (max(av, bv[0]) / min(av, bv[0]) - 1)
            nd = 2 if what == "decode" else 0
            # process-level overlap: min/max of per-process means
            pa = [v[0 if what == "decode" else 1] for v in pp.get((gname, "A1M"), {}).values()]
            pb = [v[0 if what == "decode" else 1] for v in pp.get((gname, best), {}).values()]
            pa = [x for x in pa if x is not None]; pb = [x for x in pb if x is not None]
            overlap = ""
            if pa and pb:
                if max(pa) >= min(pb) and max(pb) >= min(pa):
                    overlap = " (process means overlap: within noise)"
            L.append(f"- {gname} {what}: a {av:.{nd}f} vs {best} {bv[0]:.{nd}f}: **{win} wins by {pct:.1f}%**"
                     f" (a per process {fmt_pp(pa, nd)}; {best} per process {fmt_pp(pb, nd)}){overlap}")
    L.append("")
    L += ["### S3 / S5 per turn", ""] + H.turn_table(rows) + [""]
    L += ["### S5 collapse and recovery (vs the last S3 turn)", ""] + H.recovery_table(rows) + [""]
    L += ["### H7 device time (32 decode steps, warm 32K text)", ""] + H.devtime_table(recs) + [""]

    # quality lines
    q = [r for r in allrecs if (r.get("label") or "").startswith("Q-") or "quality" in (H.g(r, "args", "modes") or [])]
    if q:
        L += ["### Quality lines (64K teacher-forced)", "", "| label | status | NLL | vs anchor: top-1 agreement / dNLL / bitwise |",
              "|---|---|---|---|"]
        for r in q:
            qq = r.get("quality") or {}
            parts = []
            for an in qq.get("anchors", []) or []:
                bw = f"{an.get('bitwise_equal')}/{an.get('compared')}"
                parts.append(f"{os.path.basename(str(an.get('path') or an.get('label')))[:44]}: "
                             f"{H.f(an.get('top1_agree_frac'), 4)} / {an.get('dnll', 0):+.5f} / {bw}")
            L.append(f"| {r.get('label')} | {r.get('status')} | {H.f(qq.get('nll'), 6)} | {'; '.join(parts) or json.dumps(qq)[:300]} |")
        L.append("")
    txt = "\n".join(L)
    if out:
        open(out, "w").write(txt + "\n")
    print(txt)
    if jl:
        with open(jl, "w") as fh:
            for r in rows:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "timeline"}) + "\n")
    if runs:
        with open(runs, "w") as fh:
            for r in allrecs:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
