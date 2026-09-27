#!/usr/bin/env python3
"""Rebuild RESULTS.md from every results.jsonl record of the DeepSeek-V4.1 recipe search and the earlier campaigns.

Every run of the harness appends one JSON record (env, args, host, all metrics) and writes a log with one "==" line
per measurement. This script reads all of them on every pass and renders one Markdown file, so nothing is copied by
hand and nothing is lost: summary tables first, then every run with its configuration, every measurement line, the
tier / memory / timing lines, and the complete flattened JSON record. Run it repeatedly; it is idempotent.
"""
import datetime
import glob
import json
import os
import re
import statistics
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import master_table

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/rnd/release/recipe/RESULTS.md"
CAMPAIGNS = [
    ("Recipe search (family c vs a, 1M context)", ["/root/rnd/release/recipe/stage*/**/results.jsonl",
                                                  "/root/rnd/release/recipe/stage*/results.jsonl"]),
    ("Validation of dsv41-all (2026-09-26)", ["/root/rnd/release/all/val/**/results.jsonl"]),
    ("Expert-tier development runs", ["/root/rnd/release/ws-tier/**/results.jsonl"]),
    ("Earlier benchmark windows (baseline, fixes, rebases)", ["/root/rnd/release/bench/runs/**/results.jsonl",
                                                             "/root/rnd/release/all/run-*/**/results.jsonl"]),
]

NUM = r"([0-9.]+|None)"
RE_H2H = re.compile(r"^== (?P<scn>\S+)(?: (?P<vary>\{.*?\}))? P=(?P<P>\d+) cached=(?P<cached>\d+) G=(?P<G>\d+): TTFT "
                    + NUM + r" s, prefill " + NUM + r" \(delta " + NUM + r"\) tok/s, decode " + NUM + r" tok/s")
RE_GEN = re.compile(r"^== (?P<scn>gen_P\d+_r\d+|raw_P\d+)(?: (?P<vary>\{.*?\}))?: TTFT " + NUM + r" s, prefill " + NUM
                    + r" tok/s, decode " + NUM + r" tok/s")
RE_RAW = re.compile(r"^== (?P<scn>raw_P\d+): prefill " + NUM + r" tok/s, TTFT " + NUM + r" s, decode " + NUM + r" tok/s")
RE_Q = re.compile(r"^== quality N=(\d+): NLL ([0-9.]+)")
RE_HIT = re.compile(r"decode hit ([0-9.]+)")
KEEP_LINE = re.compile(r"^(== |-- (loaded|warmup|prewarm|cgroup|RAM budget|DSV41 placement|CPU MoE|tier|scenario|turn)|"
                       r"\s+anchor |.*(Error|Traceback|OutOfMemory|does not fit|refused|FAILED|KILLED))")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_records():
    seen = {}
    for title, pats in CAMPAIGNS:
        for pat in pats:
            for path in sorted(set(glob.glob(pat, recursive = True))):
                if "/dry/" in path or "/dry-" in path:
                    continue   # dry runs of planned batches: no measurements
                try:
                    lines = open(path, errors = "replace").read().splitlines()
                except OSError:
                    continue
                for ln in lines:
                    ln = ln.strip()
                    if not ln.startswith("{"):
                        continue
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    rid = r.get("run_id") or r.get("label") or ln[:80]
                    if rid in seen:
                        continue
                    r["_campaign"] = title
                    r["_source"] = path
                    seen[rid] = r
    return list(seen.values())


def log_path(r):
    a = r.get("args") or {}
    outdir = a.get("outdir") or os.path.dirname(r.get("_source", ""))
    rid = r.get("run_id") or ""
    for cand in (os.path.join(outdir, rid + ".log"), *glob.glob(os.path.join(outdir, "**", rid + ".log"), recursive = True)):
        if os.path.isfile(cand):
            return cand
    return None


def parse_log(path):
    out = {"lines": [], "scen": [], "quality": [], "hits": []}
    if not path:
        return out
    try:
        text = open(path, errors = "replace").read().splitlines()
    except OSError:
        return out
    for ln in text:
        if "Overriding" in ln or "Replaced" in ln:
            continue
        if KEEP_LINE.match(ln):
            out["lines"].append(ln.rstrip()[:400])
        m = RE_H2H.match(ln)
        if m:
            g = m.groups()
            out["scen"].append({"scn": m.group("scn"), "vary": m.group("vary") or "", "P": int(m.group("P")),
                                "cached": int(m.group("cached")), "G": int(m.group("G")), "ttft": f(g[5]),
                                "prefill": f(g[6]), "delta": f(g[7]), "decode": f(g[8])})
            continue
        m = RE_RAW.match(ln)
        if m:
            out["scen"].append({"scn": m.group("scn"), "vary": "", "P": int(m.group("scn").split("_P")[1]), "cached": 0,
                                "G": None, "ttft": f(m.group(3)), "prefill": f(m.group(2)), "delta": None,
                                "decode": f(m.group(4))})
            continue
        m = RE_GEN.match(ln)
        if m:
            g = m.groups()
            out["scen"].append({"scn": m.group("scn"), "vary": m.group("vary") or "", "P": int(re.search(r"P(\d+)", m.group("scn")).group(1)),
                                "cached": 0, "G": None, "ttft": f(g[2]), "prefill": f(g[3]), "delta": None, "decode": f(g[4])})
            continue
        m = RE_Q.match(ln)
        if m:
            out["quality"].append((int(m.group(1)), float(m.group(2))))
        m = RE_HIT.search(ln)
        if m and "tier" in ln:
            out["hits"].append(float(m.group(1)))
    return out


def flatten(obj, prefix = "", out = None):
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if len(obj) > 16 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            xs = sorted(obj)
            out.append((prefix, f"list[{len(obj)}] min {xs[0]:.4g} p50 {statistics.median(xs):.4g} "
                                f"p90 {xs[int(0.9 * (len(xs) - 1))]:.4g} max {xs[-1]:.4g}"))
        elif len(obj) > 40:
            out.append((prefix, f"list[{len(obj)}] (first: {json.dumps(obj[0])[:120]})"))
        else:
            for i, v in enumerate(obj):
                flatten(v, f"{prefix}[{i}]", out)
    else:
        s = json.dumps(obj) if not isinstance(obj, str) else obj
        out.append((prefix, s[:300]))
    return out


def short_place(env):
    p = (env or {}).get("EXL3_PLACEMENT") or (env or {}).get("EXL3_DSV41_PLACEMENT") or "(none)"
    return p.replace("|", "/")


def cfg_key(env):
    keys = sorted(k for k in (env or {}) if k.startswith(("EXL3_", "CUDA_VISIBLE")) and k != "EXL3_PLACEMENT")
    return ", ".join(f"{k}={env[k]}" for k in keys)


def fmt(x, nd = 1):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    recs = load_records()
    order = {t: i for i, (t, _) in enumerate(CAMPAIGNS)}
    recs.sort(key = lambda r: (order.get(r["_campaign"], 99), r.get("t_start") or "", r.get("run_id") or ""))
    parsed = {id(r): parse_log(log_path(r)) for r in recs}
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    L = []
    L.append("# DeepSeek-V4.1-Flash: every measured run (auto-generated)")
    L.append("")
    L.append(f"Regenerated {now} from every `results.jsonl` record and run log on the AI VM ({len(recs)} runs). Do not edit by hand:"
             " the next pass overwrites it. Sections: best-so-far, head-to-head by scenario, all runs in one table, then every"
             " run in full (configuration, every measurement line, and the complete JSON record).")
    L.append("")
    L.append("Rules in force for the recipe: FP16 KV cache only (no KV quantization), 1M (1,048,576-token) cache mandatory,"
             " host RAM <= 128 GB for the serving process, engrams on the SSD, zero expert SSD reads for family c.")
    L.append("")

    recipe = [r for r in recs if r["_campaign"].startswith("Recipe")]
    # best so far (recipe, h2h scenario lines)
    rows = []
    for r in recipe:
        for s in parsed[id(r)]["scen"]:
            rows.append((r, s))
    L.append("## Best so far (recipe search)")
    L.append("")
    if rows:
        def best(metric, flt):
            c = [(s[metric], r, s) for r, s in rows if s.get(metric) is not None and flt(s)]
            return max(c, key = lambda t: t[0]) if c else None
        items = [
            ("decode, cold 32K (S1-32K*)", "decode", lambda s: s["scn"].startswith("S1-32K")),
            ("decode, warm same prompt (S2*)", "decode", lambda s: s["scn"].startswith("S2")),
            ("decode at 128K context (S1-128K*)", "decode", lambda s: s["scn"].startswith("S1-128K")),
            ("decode, multi-turn (S3_*)", "decode", lambda s: s["scn"].startswith("S3")),
            ("decode, topic switch (S5*)", "decode", lambda s: s["scn"].startswith("S5")),
            ("prefill, cold 32K (S1-32K*)", "prefill", lambda s: s["scn"].startswith("S1-32K")),
            ("prefill, 128K (S1-128K*)", "prefill", lambda s: s["scn"].startswith("S1-128K")),
            ("prefill, any gen_P32768", "prefill", lambda s: s["scn"].startswith("gen_P32768")),
            ("decode, any gen_P32768", "decode", lambda s: s["scn"].startswith("gen_P32768")),
        ]
        L.append("| Metric | Best | Run | Placement | Other settings |")
        L.append("|---|---|---|---|---|")
        for name, metric, flt in items:
            b = best(metric, flt)
            if b:
                v, r, s = b
                L.append(f"| {name} | **{v:.1f}** tok/s ({s['scn']}{' ' + s['vary'] if s['vary'] else ''}) | `{r.get('label')}` |"
                         f" `{short_place(r.get('env'))}` | {cfg_key(r.get('env'))[:160]} |")
    else:
        L.append("(no recipe scenario lines yet)")
    L.append("")

    # master table: one row per measurement, every knob and every metric as a column
    cols, mrows = master_table.master_rows(recs, log_path)
    csv_path = os.path.splitext(OUT)[0] + ".csv"
    L.extend(master_table.leaderboard(cols, mrows))
    L.append(f"## Master table: {len(mrows)} measurements x {len(cols)} columns")
    L.append("")
    L.append(f"One row per measurement (run x scenario); every configuration knob and every metric is a column. The same table is in `{os.path.basename(csv_path)}` next to this file. Empty cell = not set / not measured.")
    L.append("")
    L.extend(master_table.render(cols, mrows, csv_path))
    L.append("")

    # head-to-head table: scenario x run
    L.append("## Head-to-head by scenario (recipe search, every run)")
    L.append("")
    scen_names = []
    for r, s in rows:
        if s["scn"] not in scen_names:
            scen_names.append(s["scn"])
    if rows:
        L.append("Decode tok/s / prefill tok/s (delta prefill for turns with a cached prefix) / TTFT s. One column per run.")
        L.append("")
        runs_with = [r for r in recipe if parsed[id(r)]["scen"]]
        L.append("| Scenario | " + " | ".join(f"`{r.get('label')}`" for r in runs_with) + " |")
        L.append("|---|" + "---|" * len(runs_with))
        for sc in scen_names:
            cells = []
            for r in runs_with:
                ss = [s for s in parsed[id(r)]["scen"] if s["scn"] == sc]
                if not ss:
                    cells.append("")
                    continue
                s = ss[0]
                pre = s["delta"] if (s["cached"] and s["delta"] is not None) else s["prefill"]
                cells.append(f"{fmt(s['decode'])} / {fmt(pre, 0)} / {fmt(s['ttft'])}" + (f" {s['vary']}" if s["vary"] else ""))
            L.append(f"| {sc} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Placement per run: " + "; ".join(f"`{r.get('label')}` = `{short_place(r.get('env'))}`" for r in runs_with))
        L.append("")

    # all runs table
    L.append("## All runs")
    L.append("")
    L.append("| # | Campaign | Run | Status | Placement | Load s | Max decode | Max prefill | NLL | Hit rates | Settings |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(recs, 1):
        p = parsed[id(r)]
        decs = [s["decode"] for s in p["scen"] if s["decode"] is not None]
        pres = [s["prefill"] for s in p["scen"] if s["prefill"] is not None and not s["cached"]]
        load = None
        for ln in p["lines"]:
            m = re.search(r"loaded in ([0-9.]+) s", ln)
            if m:
                load = float(m.group(1))
                break
        nll = ", ".join(f"{n // 1024}K {v:.6f}" for n, v in p["quality"])
        hits = ", ".join(f"{h:.2f}" for h in p["hits"][:6])
        L.append(f"| {i} | {r['_campaign'].split(' (')[0]} | `{r.get('run_id')}` | {r.get('status')}"
                 f"{' ' + str(r.get('error'))[:60] if r.get('error') else ''} | `{short_place(r.get('env'))[:90]}` | {fmt(load, 0)} |"
                 f" {fmt(max(decs) if decs else None)} | {fmt(max(pres) if pres else None, 0)} | {nll} | {hits} |"
                 f" {cfg_key(r.get('env'))[:140]} |")
    L.append("")

    # every run in full
    L.append("## Every run in full")
    L.append("")
    cur = None
    for r in recs:
        if r["_campaign"] != cur:
            cur = r["_campaign"]
            L.append(f"### {cur}")
            L.append("")
        p = parsed[id(r)]
        L.append(f"#### `{r.get('run_id')}` ({r.get('label')}): {r.get('status')}")
        L.append("")
        L.append(f"- Started: {r.get('t_start')}; record: `{r['_source']}`; log: `{log_path(r)}`")
        L.append(f"- Placement: `{short_place(r.get('env'))}`")
        env = r.get("env") or {}
        L.append("- Environment: " + (", ".join(f"`{k}={v}`" for k, v in sorted(env.items())
                                               if k not in ("PYTHONPATH", "PATH", "CUDA_HOME")) or "(none)"))
        a = r.get("args") or {}
        L.append("- Arguments: " + ", ".join(f"`{k}={json.dumps(v)[:80]}`" for k, v in sorted(a.items())
                                            if k not in ("results", "outdir", "run_id", "label", "model_dir", "ids")))
        if r.get("error"):
            L.append(f"- **Error:** {str(r.get('error'))[:600]}")
        if p["lines"]:
            L.append("")
            L.append("```")
            L.extend(p["lines"][:200])
            L.append("```")
        fl = flatten({k: v for k, v in r.items() if k not in ("env", "args")})
        L.append("")
        L.append(f"<details><summary>Complete JSON record ({len(fl)} fields)</summary>")
        L.append("")
        L.append("| Field | Value |")
        L.append("|---|---|")
        for k, v in fl:
            L.append(f"| `{k}` | {str(v).replace('|', '/').replace(chr(10), ' ')} |")
        L.append("")
        L.append("</details>")
        L.append("")

    # stage reports of the recipe search
    stages = sorted(glob.glob("/root/rnd/release/recipe/STAGE_*.md")) + sorted(glob.glob("/root/rnd/release/recipe/*/STAGE_*.md"))
    if stages:
        L.append("## Stage reports (the stage reports of the recipe search)")
        L.append("")
        for s in stages:
            L.append(f"### `{s}`")
            L.append("")
            try:
                L.append(open(s, errors = "replace").read())
            except OSError:
                pass
            L.append("")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(recs)} runs, {sum(len(parsed[id(r)]['scen']) for r in recs)} measurement lines")


if __name__ == "__main__":
    main()
