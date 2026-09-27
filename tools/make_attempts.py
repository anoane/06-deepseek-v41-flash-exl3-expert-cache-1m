#!/usr/bin/env python3
"""Build docs/ATTEMPTS.md and results/LEADERBOARD.md of this repository from results/RESULTS.csv,
results/RESULTS.md.gz and the batch files in configs/.

    python3 tools/make_attempts.py            # from the repository root

ATTEMPTS.md: one row per run of the recipe search (every attempt, including the ones that failed or were not run),
with the exact configuration taken from the batch line that ran it and the key measurements taken from RESULTS.csv.
LEADERBOARD.md: the leaderboard recomputed with the allocator as part of the configuration key (the auto-generated
RESULTS.md groups runs that differ only in the allocator), then the best-so-far, leaderboard and head-to-head sections
of RESULTS.md exactly as generated.
"""
import csv
import gzip
import os
import re
import shlex
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import master_table as mt  # noqa: E402

csv.field_size_limit(1 << 30)

# batch files -> stage / window, in the order the windows ran
STAGES = [
    ("0a", ["stage0a.batch"]),
    ("0b", ["stage0b.batch", "stage0b-requeue.batch"]),
    ("0c", ["stage0c.batch"]),
    ("1", ["stage1.batch", "stage1-requeue.batch"]),
    ("2", ["stage2.batch", "stage2-requeue.batch"]),
]
BOILER_ENV = {"BENCH_PYTHON", "MEMWATCH_PY", "PYTHONDONTWRITEBYTECODE", "MEMMAX", "PYTORCH_ALLOC_CONF",
              "PYTORCH_CUDA_ALLOC_CONF", "EXL3_PLACEMENT", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "WS_SRC", "WS_NATIVE",
              "EXL3_DSV41_PLACEMENT", "BENCH_CORES"}
BOILER_ARGS = ["--cache-tokens 1048576", "--max-output-size 32", "--engram-drop", "--timeline 32",
               "--expect-native-sha 823174af46fb", "--expect-native-sha 4fa4d0749921"]
SHORT_ENV = {
    "EXL3_DSV41_PIPELINE": "pipe", "EXL3_DSV41_PIPELINE_CHUNK": "sub", "EXL3_DSV41_XDEV_BF16": "bf16x",
    "EXL3_DSV41_NUMERICS": "numerics", "EXL3_MOE_TIER_STAGING": "staging", "EXL3_MOE_TIER_HEADROOM_MB": "headroom",
    "EXL3_MOE_TIER_PREFILL_ROWS": "prefill_rows", "EXL3_MOE_HEAT_FILE": "heat_file", "EXL3_STABLE_ARITHMETIC": "stable",
    "EXL3_HGEMM_FIXED_ROWS": "fixed_rows", "EXL3_MOE_FUSED_PREFILL": "fused_prefill", "EXL3_MOE_FUSED_DET": "fused_det",
    "EXL3_DSV41_ENGRAM_PREFETCH": "engram_prefetch", "EXL3_MOE_PINNED_ARENA": "pinned_arena",
    "EXL3_MOE_CPU_WSLOTS": "cpu_wslots", "EXL3_MOE_CPU_OFFLOAD": "cpu_offload", "EXL3_MOE_TIER_HUGEPAGE": "hugepage",
    "EXL3_MOE_TIER_SPIN_US": "spin_us", "EXL3_MOE_COOP_WIDE": "coop_wide", "EXL3_MOE_COOP_KSPLIT": "coop_ksplit",
    "EXL3_HGEMM_FP32_REDUCTION": "fp32_reduction", "EXL3_DSV41_FUSED_COMPRESS": "fused_compress",
    "EXL3_MOE_TIER_TRACE": "tier_trace", "OMP_NUM_THREADS": "omp",
}


# known facts about single runs, from the stage reports (docs/stages/)
NOTES = {
    "A-old-3": "contaminated: a foreign find over the root file system ran during its decode (kcompactd); excluded",
    "B-det": "EXL3_MOE_FUSED_DET=0: every 32K prefill ran out of memory on the CMP at 1M",
    "DIAG-L0-det0": "diagnostic of B-det: CUDA OOM on the CMP (the atomic path's scratch)",
    "D-rc": "recurrent_cache_mb=0: upstream RecurrentCache.put() KeyError; fixed on dsv41-all 5e55c062, not re-run",
    "T-fit": "pipelined 32K jobs produced no tokens (CMP out of memory at 1M); the numbers are the plain jobs",
    "T-pipe-128k": "the pipelined 128K job produced no tokens (CMP out of memory); the number is the plain job",
    "DIAG-C12-pipe": "C12 pipelined 4096 under cudaMallocAsync: CUDA OOM on the CMP in stage 1's engram forward",
    "DIAG-C12-pipe2048": "2048-row sub-chunks fit at split 12 but are no faster than plain",
    "C14-fit": "13 whole layers do not fit on the CMP at 1M: refused at load, before any RAM was pinned",
    "C12-K8": "load chunk 8192 at split 12: layer 11 does not fit on the CMP (refused at load)",
    "C12-P3": "1 of 15 jobs (S2-32K@140000) hit a CMP out-of-memory under cudaMallocAsync",
    "C12-E2": "ram ngram=auto sized the row cache at 96 GiB on 5e55c062 (sizing bug); see C12-E2F",
    "C12-E2F": "the same as C12-E2 on a Python-only sizing fix (not on the published tip yet): 30.8 GiB of row cache",
    "A1M-trace": "EXL3_MOE_TIER_TRACE capture for the offline replay; excluded from speed comparisons",
    "C10-P8X": "the configuration serving now (temporary)",
}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def fmt(v, nd=1):
    if v is None:
        return ""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def parse_batches():
    """label -> (stage, env dict, args string, comment block before it)"""
    out = {}
    for stage, files in STAGES:
        for fn in files:
            path = os.path.join(ROOT, "configs", fn)
            comment = []
            first = True
            for line in open(path):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if line.startswith("#@"):
                    continue
                if line.startswith("#"):
                    text = line.lstrip("# ").strip()
                    if text.startswith("----"):
                        comment = []          # a section rule, not a note about the next line
                    else:
                        comment.append(text)
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                label = parts[0].strip()
                env = {}
                for w in shlex.split(parts[1]):
                    k, _, v = w.partition("=")
                    env[k] = v
                args = "|".join(parts[2:]).strip()
                if label not in out:
                    # the comments before a file's first line are the file header, not a note about that line
                    out[label] = (stage, env, args, "" if first else " ".join(comment))
                comment = []
                first = False
    return out


def layout(env):
    p = env.get("EXL3_PLACEMENT", "")
    if env.get("EXL3_DSV41_PLACEMENT"):
        return "OLD"
    if not p:
        return "OLD" if "wt-dsv41" in env.get("PYTHONPATH", "") else "?"
    if p.startswith("*=cuda:0 experts=cache"):
        return "a"
    if p.startswith("0-11=cuda:0; 12-22=cuda:1 experts=cpu"):
        return "L0"
    if "experts=stream hot=" in p:
        return "C12-S"
    if "experts=cpu hot=" in p:
        return "C12-C"
    m = re.match(r"0-(\d+)=cuda:0; (\d+)-39=cuda:1 experts=cache", p)
    if m:
        return f"C{int(m.group(2))}"
    return p


def knobs(env):
    k = []
    p = env.get("EXL3_PLACEMENT", "")
    for word in ("admit=always", "evict=lfu", "ngram=auto", "ngram=40GiB", "io=direct"):
        if word in p:
            k.append(word)
    for key, val in env.items():
        if key in BOILER_ENV:
            continue
        name = SHORT_ENV.get(key, key)
        if key == "EXL3_MOE_HEAT_FILE":
            val = os.path.basename(val)
        k.append(f"{name}={val}")
    if "PYTORCH_CUDA_ALLOC_CONF" not in env and "BENCH_PYTHON" in env:
        k.append("alloc=expandable")
    if "src-e451b5f97412" in env.get("PYTHONPATH", ""):
        k.append("tree=fixup e451b5f9")
    return " ".join(k)


def short_args(args):
    a = args
    for b in BOILER_ARGS:
        a = a.replace(b, "")
    a = re.sub(r"--anchor \S+", "", a)
    a = a.replace("--modes ", "").replace("--scenarios ", "")
    return re.sub(r"\s+", " ", a).strip()


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "RESULTS.csv"))))
    rec = [r for r in rows if r["campaign"].startswith("Recipe")]
    batches = parse_batches()
    runs = {}
    order = []
    for r in rec:
        key = r["run_id"] or ("skipped:" + r["label"])
        if key not in runs:
            runs[key] = []
            order.append(key)
        runs[key].append(r)

    def first(rs, col):
        for r in rs:
            if r.get(col) not in (None, ""):
                return r[col]
        return ""

    def scen(rs, prefixes, metric, exact=False):
        v = []
        for r in rs:
            s = r.get("scenario", "")
            if (s in prefixes) if exact else any(s.startswith(p) for p in prefixes):
                v.append(f(r.get(metric)))
        return mean(v)

    stage_rank = {s: i for i, (s, _) in enumerate(STAGES)}
    items = []
    for key in order:
        rs = runs[key]
        label = rs[0]["label"]
        stage, env, args, comment = batches.get(label, ("?", {}, "", ""))
        items.append((stage_rank.get(stage, 9), first(rs, "t_start") or "9999", label, key, rs, stage, env, args, comment))
    items.sort(key=lambda t: (t[0], t[1]))

    L = ["# Every attempt of the recipe search (generated)", "",
         "Generated by `tools/make_attempts.py` from `results/RESULTS.csv` (measurements) and `configs/*.batch` (the exact",
         "configuration of each run). One row per run, in the order the runs were started; runs that were queued but not",
         "started when the search was paused are listed as `skipped`. Do not edit by hand.", "",
         "Layout codes (all at a 1,048,576-token FP16 cache unless the run says otherwise; CUDA_VISIBLE_DEVICES=1,0 so",
         "cuda:0 = CMP 170HX and cuda:1 = RTX PRO 6000, except config a, which runs on the PRO alone):", "",
         "| Code | Placement |", "|---|---|",
         "| C12 | `0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |",
         "| C11 | `0-10=cuda:0; 11-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |",
         "| C10 | `0-9=cuda:0; 10-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |",
         "| C8 | `0-7=cuda:0; 8-39=cuda:1 experts=cache; ram experts=all; disk experts=off` |",
         "| C14 | `0-13=cuda:0; 14-39=cuda:1 experts=cache; ram experts=all; disk experts=off` (refused at load) |",
         "| C12-S | `0-11=cuda:0; 12-39=cuda:1 experts=stream hot=208` |",
         "| C12-C | `0-11=cuda:0; 12-39=cuda:1 experts=cpu hot=208` |",
         "| a | `*=cuda:0 experts=cache; ram experts=102GiB` (PRO alone, `CUDA_VISIBLE_DEVICES=0`; ~1,486 experts only on the SSD) |",
         "| L0 | `0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1` (the old production layout on the new tree) |",
         "| OLD | the old production tree and settings (`EXL3_DSV41_PLACEMENT='split=12;cpu=12-22'`, `EXL3_MOE_CPU_OFFLOAD=11`) |",
         "", "Knob words in `knobs` are the placement attributes and environment variables that differ from the layout's",
         "plain form: `pipe=1 sub=N` = `EXL3_DSV41_PIPELINE=1 EXL3_DSV41_PIPELINE_CHUNK=N`, `bf16x=1` = `EXL3_DSV41_XDEV_BF16=1`,",
         "`alloc=expandable` = no `PYTORCH_CUDA_ALLOC_CONF`, so the library default (expandable segments) instead of `cudaMallocAsync`",
         "(every other run sets cudaMallocAsync, as tabby's `cuda_malloc_async: true` does); `other` shows decode / prefill,",
         "`staging`, `headroom`, `prefill_rows` = `EXL3_MOE_TIER_*`, `heat_file` = `EXL3_MOE_HEAT_FILE`; see docs/KNOBS.md.",
         "`args` is the harness command line without the options every run shares (`--cache-tokens 1048576`, `--engram-drop`, ...).", "",
         "Columns: decode tok/s in S1 (cold 32K, mean of two texts), S2 (the same 32K prompt again), S1-128K, S3 turns 2-6,",
         "S5 turns 7-8 (topic switch); prefill tok/s in S1-32K, S2-32K and S1-128K; TTFT of S1-32K; `other` = measurements",
         "of runs that are not head-to-head runs (32K speed repeats, 64K quality NLL, 512K long context). RSS is the",
         "serving process's peak; free MiB = nvidia-smi minimum free memory over the run (PRO / CMP). See docs/METHOD.md.",
         "`note`: known facts about the run from the stage reports, or, for stage 2, the batch comment written before the run",
         "(its plan item and intent; the plan's classes N = bit-exact, A = last-bit numerics, L = lossy).", "",
         "| # | stage | label | status | layout | knobs | args | dec S1 | dec S2 | dec 128K | dec S3 | dec S5 | pre S1 | pre S2 | pre 128K | TTFT S1 s | other | load s | RSS GiB | free MiB PRO / CMP | note |",
         "|" + "---|" * 21]
    n = 0
    counts = {}
    for _, _, label, key, rs, stage, env, args, comment in items:
        n += 1
        status = rs[0]["status"]
        counts[(stage, status)] = counts.get((stage, status), 0) + 1
        err = first(rs, "error") or first(rs, "job.error")
        dec = [scen(rs, ["S1-32K"], "decode_tok_s"), scen(rs, ["S2"], "decode_tok_s"), scen(rs, ["S1-128K"], "decode_tok_s"),
               scen(rs, ["S3_t2", "S3_t3", "S3_t4", "S3_t5", "S3_t6"], "decode_tok_s", True),
               scen(rs, ["S5_t7", "S5_t8"], "decode_tok_s", True)]
        pre = [scen(rs, ["S1-32K"], "prefill_tok_s"), scen(rs, ["S2"], "prefill_tok_s"), scen(rs, ["S1-128K"], "prefill_tok_s")]
        ttft = scen(rs, ["S1-32K"], "ttft_s")
        other = []
        gens = {}
        for r in rs:
            s = r.get("scenario", "")
            m = re.match(r"(gen_P\d+|raw_P\d+|legacy-[a-z]+-\d+)", s)
            if m:
                d = f(r.get("decode_tok_s"))
                if d is None:
                    d = f(r.get("decode_steady_tok_s"))
                gens.setdefault(m.group(1), []).append((d, f(r.get("prefill_tok_s"))))
        for g, vals in gens.items():
            d = mean([v[0] for v in vals]); p = mean([v[1] for v in vals])
            if d is None and p is None:
                other.append(f"{g}: no tokens")
            elif d is None:
                other.append(f"{g}: prefill {fmt(p, 0)}")
            elif p is None:
                other.append(f"{g}: decode {fmt(d)}")
            else:
                other.append(f"{g}: {fmt(d)} / {fmt(p, 0)}")
        s4p = scen(rs, ["S4-512K_t0"], "prefill_tok_s", True)
        s4d = scen(rs, ["S4-512K_t1", "S4-512K_t2", "S4-512K_t3"], "decode_tok_s", True)
        if s4p is not None:
            other.append(f"512K prefill {fmt(s4p, 0)}, decode at 512K {fmt(s4d)}")
        nll = first(rs, "nll")
        if nll:
            other.append(f"NLL {nll}")
        rss = None
        for r in rs:
            for i in range(4):
                for src in ("loadmon.max_rss_gib", "peaks.host_end.procs", "load.host.procs"):
                    if r.get(f"{src}[{i}].name") == "python":
                        v = f(r.get(f"{src}[{i}].rss_gib"))
                        if v is not None and (rss is None or v > rss):
                            rss = v
        if rss is None:
            rss = f(first(rs, "peaks.host_end.procs[0].rss_gib"))
        fp = f(first(rs, "peaks.gpu_mem_min_free_mib.0")); fc = f(first(rs, "peaks.gpu_mem_min_free_mib.1"))
        free = "" if fp is None else (f"{fp:,.0f} / {fc:,.0f}" if fc is not None and layout(env) != "a" else f"{fp:,.0f} / -")
        note = NOTES.get(label, "")
        if note:
            pass
        elif status == "skipped":
            note = "not started: the search was paused after C12-HR"
        elif err:
            note = re.sub(r"\s+", " ", err)[:160].replace("|", "/")
        elif comment and stage == "2":
            note = comment[:160].replace("|", "/")
        L.append("| " + " | ".join([
            str(n), stage, f"`{label}`", status, layout(env), knobs(env).replace("|", "/"), short_args(args).replace("|", "/"),
            *[fmt(x) for x in dec], *[fmt(x, 0) for x in pre], fmt(ttft), "; ".join(other).replace("|", "/"),
            fmt(f(first(rs, "load_s")), 0), fmt(rss), free, note]) + " |")
    L += ["", "## Counts", "", "| stage | ok | error | skipped |", "|---|---|---|---|"]
    for stage, _ in STAGES:
        L.append(f"| {stage} | {counts.get((stage, 'ok'), 0)} | {counts.get((stage, 'error'), 0)} | {counts.get((stage, 'skipped'), 0)} |")
    L += ["", "## Stage reports and tables", "",
          "| Stage | Report | Tables |", "|---|---|---|",
          "| 0 (0a, 0b, 0c) | [stages/STAGE_0.md](stages/STAGE_0.md) | [0a](stages/tables/tables_0a.md), [0b](stages/tables/tables_0b.md), "
          "[0c](stages/tables/tables_0c.md), [a vs C12](stages/tables/tables_0c_a_vs_c12.md), offline replay "
          "[a](stages/tables/t1_pred_a.md) / [c](stages/tables/t1_pred_c.md) |",
          "| 1 | [stages/STAGE_1.md](stages/STAGE_1.md) | [tables_1.md](stages/tables/tables_1.md), "
          "[decode model](stages/tables/decode_model2.txt), [quality intervals](stages/tables/quality_ci.txt) |",
          "| 2 (partial) | [stages/STAGE_2_PARTIAL.md](stages/STAGE_2_PARTIAL.md) | "
          "[tables_2_partial.md](stages/tables/tables_2_partial.md) |", ""]
    open(os.path.join(ROOT, "docs", "ATTEMPTS.md"), "w").write("\n".join(L).rstrip("\n") + "\n")

    # ---- leaderboard, allocator-aware, plus the RESULTS.md sections as generated
    mt._SIG_SKIP = {"memmax"}
    lb = mt.leaderboard(list(rows[0].keys()), rows)
    lb[0] = lb[0].replace("## Leaderboard", "## Leaderboard with the allocator in the configuration key")
    md = gzip.open(os.path.join(ROOT, "results", "RESULTS.md.gz"), "rt", encoding="utf-8").read().split("\n")

    def section(title_prefix):
        s = next(i for i, l in enumerate(md) if l.startswith(title_prefix))
        e = next(i for i in range(s + 1, len(md)) if md[i].startswith("## "))
        return md[s:e]

    stamp = md[2] if len(md) > 2 else ""
    out = ["# Leaderboard of the recipe search (partial: stages 0 and 1, 37 of the stage-2 runs)", "",
           "Generated by `tools/make_attempts.py`. The first table is the leaderboard of `tools/master_table.py`",
           "recomputed with `PYTORCH_CUDA_ALLOC_CONF` as part of the configuration key: the auto-generated RESULTS.md",
           "groups runs that differ only in the allocator, so there `DIAG-C12-pipe` (C12 pipelined under cudaMallocAsync,",
           "which ran out of CMP memory and produced no numbers) carries the numbers of `C12-ES-P4` (the same layout under",
           "expandable segments, which fits), and `C12-base` mixes eight cudaMallocAsync runs with two expandable-segment",
           "runs. The sections after it are copied from RESULTS.md as generated. Scores are defined in docs/METHOD.md.", "",
           "Scores come from 1 to 8 processes per configuration in the first table (the `runs` column, which counts every",
           "process of a configuration, 512K and fit runs included; up to 10 in the generated leaderboard, which merges the",
           "allocators); in stage 2 most arms have one. `C12-base` includes `C12-HR` (the heat-file restart, identical tokens).",
           "`family_c_ok` = the CMP holds layers, zero expert reads from the SSD, 1M cache. `1M only` = a 1M cache but",
           "not family c (config a reads experts from the SSD; OLD / L0 compute experts on the CPU).", "",
           "RESULTS.md: " + stamp, ""]
    out += lb + [""]
    for t in ("## Best so far", "## Leaderboard:", "## Head-to-head by scenario"):
        sec = section(t)
        sec[0] = sec[0] + " (as generated in RESULTS.md)"
        out += sec + [""]
    open(os.path.join(ROOT, "results", "LEADERBOARD.md"), "w").write("\n".join(out).rstrip("\n") + "\n")
    print("wrote docs/ATTEMPTS.md (%d runs) and results/LEADERBOARD.md" % n)


if __name__ == "__main__":
    main()
