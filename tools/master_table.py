# Master table helpers, imported by collect_results.py: one row per measurement (run x scenario), dozens of columns.
import csv, glob, json, os, re

def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

RX = {
    "ttft_s": r"TTFT ([0-9.]+|None) s",
    "prefill_tok_s": r"prefill ([0-9.]+|None)(?: \(delta (?:[0-9.]+|None)\))? tok/s",
    "delta_prefill_tok_s": r"\(delta ([0-9.]+|None)\)",
    "decode_tok_s": r"decode ([0-9.]+|None) tok/s",
    "p50_ms": r"p50 ([0-9.]+|None) ms",
    "p90_ms": r"p90 ([0-9.]+|None) ms",
    "stalls": r"stalls (\d+|None)",
    "read_mib": r"read ([0-9.]+) MiB",
    "read_tree_mib": r"\(tree ([0-9.]+)",
    "major_faults": r"majflt (\d+)",
    "foreign_cores": r"(?<!MoE-core )foreign ([0-9.]+|None)",
    "moe_core_foreign": r"MoE-core foreign ([0-9.]+|None)",
    "cores16_23_busy_pct": r"16-23 busy ([0-9.]+)%",
    "decode_steady_tok_s": r"steady ([0-9.]+)",
    "hit_rate": r"\bhit ([0-9.]+)",
    "miss_per_tok": r"miss/token ([0-9.]+)",
    "h2d_mb_per_tok": r"H2D ([0-9.]+) MB/token",
    "d2h_mb_per_tok": r"D2H ([0-9.]+) MB/token",
    "staged_gb": r"\bstaged ([0-9.]+) GB",
    "ssd_gb_inline": r"SSD ([0-9.]+)/[0-9.]+",
    "cgroup_now_gib": r"cgroup ([0-9.]+)",
}
RT = {
    "hit_rate": r"decode hit ([0-9.]+)",
    "hits": r"decode hit [0-9.]+ \((\d+)/",
    "lookups": r"decode hit [0-9.]+ \(\d+/(\d+)\)",
    "ssd_mb_per_tok": r"SSD ([0-9.]+) MB/token",
    "h2d_mb_per_tok": r"H2D ([0-9.]+) MB/token",
    "d2h_mb_per_tok": r"D2H ([0-9.]+) MB/token",
    "predicted_reads": r"predicted reads (\d+)",
    "prefill_staged_gb": r"prefill staged ([0-9.]+) GB",
    "prefill_ssd_gb": r"prefill staged [0-9.]+ GB, SSD ([0-9.]+) GB",
    "layer_plans": r"layer plans (\d+)",
}

def parse_measurements(path):
    """-> (run_level dict, list of per-scenario dicts)"""
    run, rows, tier = {}, [], {}
    if not path or not os.path.isfile(path):
        return run, rows
    for ln in open(path, errors="replace").read().splitlines():
        m = re.search(r"loaded in ([0-9.]+) s; VRAM alloc \[([^\]]*)\] GiB", ln)
        if m:
            run["load_s"] = _f(m.group(1))
            v = [x.strip() for x in m.group(2).split(",")]
            run["vram_alloc_gpu_a_gib"] = _f(v[0]) if v else None
            run["vram_alloc_gpu_b_gib"] = _f(v[1]) if len(v) > 1 else None
        m = re.search(r"cgroup at the end: .*'peak': ([0-9.]+).*'max': ([0-9.]+|None)", ln)
        if m:
            run["ram_cgroup_peak_gib"] = _f(m.group(1)); run["ram_cgroup_max_gib"] = _f(m.group(2))
        m = re.search(r"cache (\d+) slots \(([0-9.]+) GiB", ln)
        if m: run["gpu_cache_slots"] = _f(m.group(1)); run["gpu_cache_gib"] = _f(m.group(2))
        m = re.search(r"ram experts ([0-9.]+) GiB pinned \(static [0-9.]+ \w+ \+ tier (\d+) slots\)", ln)
        if m: run["ram_tier_gib"] = _f(m.group(1)); run["ram_tier_slots"] = _f(m.group(2))
        m = re.search(r"RAM budget.*routed experts ([0-9.]+) GiB", ln)
        if m: run["ram_experts_gib"] = _f(m.group(1))
        m = re.match(r"^== quality N=(\d+): NLL ([0-9.]+) \(([0-9.]+) tok/s scored\)", ln)
        if m:
            run.setdefault("quality", []).append({"N": int(m.group(1)), "nll": float(m.group(2)), "scored_tok_s": _f(m.group(3))})
        m = re.match(r"^\s+anchor (\S+): (\d+)/(\d+) bitwise equal, first diff (\S+), top1 ([0-9.]+), dNLL ([-+0-9.]+)", ln)
        if m:
            run.setdefault("anchors", []).append(f"{m.group(1)} {m.group(2)}/{m.group(3)} top1 {m.group(5)} dNLL {m.group(6)}")
        m = re.match(r"^== (\S+) tier: (.*)$", ln)
        if m:
            tier[m.group(1)] = {k: _f(re.search(rx, m.group(2)).group(1)) if re.search(rx, m.group(2)) else None for k, rx in RT.items()}
            continue
        m = re.match(r"^== (?!quality)(\S+)(?: (\{.*?\}))?(?: P=(\d+) cached=(\d+) G=(\d+))?: (.*)$", ln)
        if m and ("tok/s" in ln):
            d = {"scenario": m.group(1), "vary": m.group(2) or "", "P": _f(m.group(3)), "cached": _f(m.group(4)), "G": _f(m.group(5))}
            if d["P"] is None:
                mp = re.search(r"P(\d+)", m.group(1)); d["P"] = _f(mp.group(1)) if mp else None
            for k, rx in RX.items():
                mm = re.search(rx, m.group(6)); d[k] = _f(mm.group(1)) if mm else None
            rows.append(d)
        m = re.match(r"^== legacy (PREFILL|DECODE) (?:n|ctx)=(\d+): (.*)$", ln)
        if m:
            d = {"scenario": f"legacy-{m.group(1).lower()}-{m.group(2)}", "vary": "", "P": _f(m.group(2)), "cached": 0, "G": None}
            for k, rx in RX.items():
                mm = re.search(rx, m.group(3)); d[k] = _f(mm.group(1)) if mm else None
            if m.group(1) == "PREFILL":
                mm = re.search(r"([0-9.]+) tok/s", m.group(3)); d["prefill_tok_s"] = _f(mm.group(1)) if mm else None
                d["decode_tok_s"] = None
            rows.append(d)
    for d in rows:
        d.update({k: v for k, v in tier.get(d["scenario"], {}).items() if v is not None})
    # nvidia-smi samples of the run
    for csvp in glob.glob(path[:-4] + "_nvsmi.csv"):
        try:
            rd = list(csv.reader(open(csvp)))
            hdr = [h.strip().lower() for h in rd[0]]
            ti = [i for i, h in enumerate(hdr) if "temp" in h]; pi = [i for i, h in enumerate(hdr) if "power.draw" in h or h.startswith("power")]
            ii = [i for i, h in enumerate(hdr) if h in ("index", "gpu")]
            by = {}
            for row in rd[1:]:
                g = row[ii[0]].strip() if ii else "?"
                t = _f(row[ti[0]].split()[0]) if ti else None; p = _f(row[pi[0]].split()[0]) if pi else None
                e = by.setdefault(g, [None, None])
                if t is not None: e[0] = t if e[0] is None else max(e[0], t)
                if p is not None: e[1] = p if e[1] is None else max(e[1], p)
            for g, (t, p) in sorted(by.items()):
                run[f"gpu{g}_temp_max_c"] = t; run[f"gpu{g}_power_max_w"] = p
        except Exception:
            pass
    return run, rows

def placement_fields(env):
    env = env or {}
    p = env.get("EXL3_PLACEMENT") or ""
    legacy = env.get("EXL3_DSV41_PLACEMENT") or ""
    vis = env.get("CUDA_VISIBLE_DEVICES", "")
    cmp_dev = "cuda:0" if vis.replace(" ", "") in ("1,0",) else ("cuda:1" if vis.replace(" ", "") in ("0,1",) else None)
    out = {"placement": p or (f"legacy {legacy}" if legacy else ""), "gpus_visible": vis}
    layers_on = {}
    for rule in [r.strip() for r in p.split(";") if r.strip()]:
        m = re.match(r"^([0-9*][0-9\-*]*)=(cuda:\d+)(.*)$", rule)
        if m:
            rng, dev, rest = m.groups()
            n = 40 if rng == "*" else (int(rng.split("-")[1]) - int(rng.split("-")[0]) + 1 if "-" in rng else 1)
            layers_on[dev] = layers_on.get(dev, 0) + n
            me = re.search(r"experts=(\w+)", rest)
            out.setdefault("expert_modes", set()).add(f"{rng}:{me.group(1) if me else 'vram'}")
            mh = re.search(r"hot=(\S+)", rest)
            if mh: out["hot"] = mh.group(1)
        mr = re.match(r"^ram (.*)$", rule)
        if mr:
            for k in ("experts", "policy", "demote", "evict", "ngram", "pagecache"):
                mm = re.search(rf"\b{k}=(\S+)", mr.group(1))
                if mm: out[f"ram_{k}"] = mm.group(1)
        md = re.match(r"^disk (.*)$", rule)
        if md:
            for k in ("experts", "ngram", "io"):
                mm = re.search(rf"\b{k}=(\S+)", md.group(1))
                if mm: out[f"disk_{k}"] = mm.group(1)
        mc = re.match(r"^(cuda:\d+) (.*)$", rule)
        if mc:
            for k in ("cache", "spare", "evict", "admit"):
                mm = re.search(rf"\b{k}=(\S+)", mc.group(2))
                if mm: out[f"gpu_cache_{k}"] = mm.group(1)
    out["expert_modes"] = " ".join(sorted(out.get("expert_modes", [])))
    out["cmp_layers"] = layers_on.get(cmp_dev) if cmp_dev else (0 if vis.strip() == "0" else None)
    out["pro_layers"] = (40 - out["cmp_layers"]) if isinstance(out.get("cmp_layers"), int) else None
    return out

ENV_COLS = [
    ("numerics", "EXL3_DSV41_NUMERICS"), ("xdev_bf16", "EXL3_DSV41_XDEV_BF16"), ("pipeline", "EXL3_DSV41_PIPELINE"),
    ("pipeline_chunk", "EXL3_DSV41_PIPELINE_CHUNK"), ("fused_prefill", "EXL3_MOE_FUSED_PREFILL"),
    ("stable", "EXL3_STABLE_ARITHMETIC"), ("fp32_reduction", "EXL3_HGEMM_FP32_REDUCTION"),
    ("engram_prefetch", "EXL3_DSV41_ENGRAM_PREFETCH"), ("expert_ram_env", "EXL3_EXPERT_RAM"), ("ngram_ram_env", "EXL3_NGRAM_RAM"),
    ("tier_demote_env", "EXL3_MOE_TIER_DEMOTE"), ("tier_staging", "EXL3_MOE_TIER_STAGING"),
    ("tier_prefill_rows", "EXL3_MOE_TIER_PREFILL_ROWS"), ("tier_decode_rows", "EXL3_MOE_TIER_DECODE_ROWS"),
    ("heat_halflife", "EXL3_MOE_HEAT_HALFLIFE"), ("tier_hugepage", "EXL3_MOE_TIER_HUGEPAGE"), ("tier_headroom_mb", "EXL3_MOE_TIER_HEADROOM_MB"),
    ("disk_backend", "EXL3_DISK_BACKEND"), ("disk_keepalive_ms", "EXL3_DISK_KEEPALIVE_MS"), ("disk_qd", "EXL3_DISK_QD"),
    ("disk_sqpoll", "EXL3_DISK_SQPOLL"), ("disk_register", "EXL3_DISK_REGISTER"), ("disk_window_expert", "EXL3_DISK_WINDOW_EXPERT"),
    ("moe_cpu_threads", "EXL3_MOE_CPU_THREADS"), ("moe_cpu_wslots", "EXL3_MOE_CPU_WSLOTS"), ("moe_pinned_arena", "EXL3_MOE_PINNED_ARENA"),
    ("moe_cpu_offload", "EXL3_MOE_CPU_OFFLOAD"), ("stream_min_rows", "EXL3_MOE_STREAM_MIN_ROWS"),
    ("alloc_conf", "PYTORCH_CUDA_ALLOC_CONF"), ("memmax", "MEMMAX"),
]
ARG_COLS = ["chunk", "gen_chunk", "cache_tokens", "cache_mode", "max_output_size", "gen_tokens", "repeats", "recurrent_cache_mb", "prewarm"]
MEAS_COLS = ["scenario", "vary", "P", "cached", "G", "ttft_s", "prefill_tok_s", "delta_prefill_tok_s", "decode_tok_s",
             "decode_steady_tok_s", "p50_ms", "p90_ms", "stalls", "hit_rate", "hits", "lookups", "ssd_mb_per_tok", "h2d_mb_per_tok",
             "d2h_mb_per_tok", "miss_per_tok", "staged_gb", "ssd_gb_inline", "cgroup_now_gib", "predicted_reads", "prefill_staged_gb", "prefill_ssd_gb", "layer_plans", "read_mib",
             "read_tree_mib", "major_faults", "foreign_cores", "moe_core_foreign", "cores16_23_busy_pct"]
RUN_COLS = ["load_s", "vram_alloc_gpu_a_gib", "vram_alloc_gpu_b_gib", "gpu_cache_slots", "gpu_cache_gib", "ram_tier_gib", "ram_tier_slots", "ram_experts_gib", "ram_cgroup_peak_gib",
            "ram_cgroup_max_gib", "nll", "anchors", "gpu0_temp_max_c", "gpu0_power_max_w", "gpu1_temp_max_c", "gpu1_power_max_w"]
PLACE_COLS = ["gpus_visible", "cmp_layers", "pro_layers", "expert_modes", "hot", "ram_experts", "ram_policy", "ram_demote",
              "ram_evict", "ram_ngram", "gpu_cache_cache", "gpu_cache_spare", "gpu_cache_evict", "gpu_cache_admit",
              "disk_experts", "disk_ngram", "disk_io"]

def master_rows(recs, logpath_fn):
    out = []
    for i, r in enumerate(recs, 1):
        run, rows = parse_measurements(logpath_fn(r))
        env = r.get("env") or {}; a = r.get("args") or {}
        pf = placement_fields(env)
        base = {"row_run": i, "campaign": r["_campaign"].split(" (")[0], "run_id": r.get("run_id"), "label": r.get("label"),
                "status": r.get("status"), "error": (str(r.get("error"))[:120] if r.get("error") else ""),
                "t_start": r.get("t_start"), "tree": (env.get("PYTHONPATH", "").split(":")[0].split("/")[-1]),
                "native_sha": a.get("expect_native_sha") or ""}
        base.update({k: pf.get(k, "") for k in ["placement"] + PLACE_COLS})
        for col, key in ENV_COLS:
            base[col] = env.get(key, "")
        for k in ARG_COLS:
            v = a.get(k, ""); base[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        q = run.get("quality") or []
        run2 = dict(run); run2["nll"] = "; ".join(f"{x['N']//1024}K {x['nll']:.6f}" for x in q); run2["anchors"] = "; ".join(run.get("anchors", []))
        for k in RUN_COLS:
            base[k] = run2.get(k, "")
        if not rows:
            rows = [{"scenario": "(run)"}]
        for d in rows:
            row = dict(base)
            vary = {}
            try: vary = json.loads(d.get("vary") or "{}")
            except ValueError: pass
            for k, v in vary.items():
                if k in ("numerics", "xdev_bf16", "pipeline", "recurrent_cache_mb"):
                    row[k] = v
            for k in MEAS_COLS:
                row[k] = d.get(k, "")
            out.append(row)
    cols = (["row_run", "campaign", "run_id", "label", "status", "t_start", "tree", "native_sha", "placement"] + PLACE_COLS
            + [c for c, _ in ENV_COLS] + ARG_COLS + MEAS_COLS + RUN_COLS + ["error"])
    return cols, out

def render(cols, rows, csv_path):
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in cols})
    def cell(v):
        if v is None: return ""
        if isinstance(v, float): return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
        return str(v).replace("|", "/").replace("\n", " ")
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |")
    return L


# ---- extension (2026-09-27): every scalar the JSON record holds for a measurement, as more columns ----
_SKIP_SUFFIX = ("_list", "first_tokens", "pythonpath", "native_sha256", "native_build_txt", "path", "npz", "text")

def _flat(obj, prefix, out, depth=0):
    if depth > 5:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj) and len(obj) <= 4:
            for i, v in enumerate(obj):
                _flat(v, f"{prefix}[{i}]", out, depth + 1)
        elif obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj) and len(obj) <= 4:
            for i, v in enumerate(obj):
                out[f"{prefix}[{i}]"] = v
        return
    else:
        if any(prefix.endswith(s) for s in _SKIP_SUFFIX):
            return
        out[prefix] = obj

def _pick(r, *keys):
    o = r
    for k in keys:
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o

def json_run_fields(r):
    out = {}
    groups = [
        ("tier", ("tier_end",)), ("sizing", ("tier_sizing",)), ("loadmon", ("load_monitor",)),
        ("load", ("load",)), ("disk", ("config", "disk_engine")), ("pipe", ("config", "dsv41_pipeline")),
        ("devtime", ("h2h", "devtime", "summary")), ("peaks", ("peaks",)), ("q", ("quality",)),
        ("engramdrop", ("engram_drop",)), ("settings", ("config", "settings")),
    ]
    for pre, path in groups:
        v = _pick(r, *path)
        if v is not None:
            tmp = {}
            _flat(v, "", tmp)
            for k, val in tmp.items():
                if pre == "load" and (k.startswith("host.procs[1]") or ".swap." in k):
                    continue
                out[f"{pre}.{k}"] = val
    for k in ("stable_arithmetic", "hgemm_fixed_rows", "disk_ngram_route", "max_output_size_source"):
        v = _pick(r, "config", k)
        if v is not None:
            out[f"cfg.{k}"] = v
    for k in ("bench_sha256_12", "native_sha256_12", "source_tree"):
        v = _pick(r, "build", k)
        if v is not None:
            out[f"build.{k}"] = v
    for k in ("provisional", "moe_cores_contended", "stall_flag", "cores_16_23_contended", "wall_s",
              "tier_ssd_reads_after_load", "tier_ssd_bytes_after_load", "harness_version"):
        if k in r and not isinstance(r[k], (dict, list)):
            out[f"flag.{k}"] = r[k]
    for i, d in enumerate(r.get("devices") or []):
        if isinstance(d, dict):
            for k in ("name", "cc", "pci_bus_id", "total_gib", "power_limit_w", "nvsmi_index"):
                if k in d:
                    out[f"dev{i}.{k}"] = d[k]
    return out

def json_scen_fields(r, scn):
    out = {}
    jobs = _pick(r, "h2h", "jobs") or []
    for j in jobs:
        if isinstance(j, dict) and j.get("scenario") == scn:
            tmp = {}; _flat(j, "", tmp)
            out.update({f"job.{k}": v for k, v in tmp.items() if k != "scenario"})
            break
    s = _pick(r, "h2h", "summary", scn)
    if isinstance(s, dict):
        tmp = {}; _flat(s, "", tmp); out.update({f"sum.{k}": v for k, v in tmp.items()})
    runs = _pick(r, "speed", "runs") or []
    for j in runs:
        if isinstance(j, dict) and scn in (j.get("name"), j.get("label"), j.get("tag"), j.get("phase"), j.get("id")):
            tmp = {}; _flat(j, "", tmp); out.update({f"job.{k}": v for k, v in tmp.items()})
            break
    g = _pick(r, "gpu", scn)
    if isinstance(g, dict):
        for dev, gd in g.items():
            if isinstance(gd, dict):
                for k, v in gd.items():
                    if not isinstance(v, (dict, list)):
                        out[f"gpu{dev}.{k}"] = v
    return out

_GROUP_ORDER = ["sum.", "job.", "gpu0.", "gpu1.", "devtime.", "tier.", "sizing.", "settings.", "pipe.", "disk.", "cfg.",
                "q.", "loadmon.", "load.", "peaks.", "engramdrop.", "dev0.", "dev1.", "build.", "flag."]

_base_master_rows = master_rows

def master_rows(recs, logpath_fn):
    cols, rows = _base_master_rows(recs, logpath_fn)
    by_id = {r.get("run_id"): r for r in recs}
    extra = set()
    for row in rows:
        r = by_id.get(row.get("run_id")) or {}
        add = json_run_fields(r)
        if row.get("scenario") and row["scenario"] != "(run)":
            add.update(json_scen_fields(r, row["scenario"]))
        for k, v in add.items():
            if isinstance(v, bool):
                v = int(v)
            row[k] = v
            extra.add(k)
    def okey(k):
        for i, g in enumerate(_GROUP_ORDER):
            if k.startswith(g):
                return (i, k)
        return (len(_GROUP_ORDER), k)
    ecols = sorted(extra, key=okey)
    err = [c for c in cols if c == "error"]
    cols = [c for c in cols if c != "error"] + ecols + err
    return cols, rows


# ---- leaderboard (2026-09-27): one row per configuration, aggregated over every run and scenario ----
_SIG_SKIP = {"memmax", "alloc_conf"}

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None

def leaderboard(cols, rows):
    rec = [r for r in rows if str(r.get("campaign", "")).startswith("Recipe") and r.get("scenario") not in (None, "", "(run)")]
    sig_cols = ["placement", "gpus_visible"] + [c for c, _ in ENV_COLS if c not in _SIG_SKIP] + ["chunk", "gen_chunk", "cache_tokens", "recurrent_cache_mb"]
    groups = {}
    for r in rec:
        sig = tuple(str(r.get(c, "")) for c in sig_cols)
        groups.setdefault(sig, []).append(r)
    out = []
    for sig, rs in groups.items():
        labels = []
        for r in rs:
            if r["label"] not in labels:
                labels.append(r["label"])
        def pick(prefixes, metric, rows_=rs):
            return [_num(r.get(metric)) for r in rows_ if any(str(r.get("scenario", "")).startswith(p) for p in prefixes)]
        def pick_exact(names, metric):
            return [_num(r.get(metric)) for r in rs if r.get("scenario") in names]
        d = {"config": re.sub(r"-\d+$", "", labels[0]), "runs": len(labels), "labels": " ".join(labels),
             "placement": rs[0].get("placement", ""), "cmp_layers": rs[0].get("cmp_layers", ""),
             "pipeline": rs[0].get("pipeline", ""), "pipe_chunk": rs[0].get("pipeline_chunk", ""),
             "numerics": rs[0].get("numerics", ""), "xdev_bf16": rs[0].get("xdev_bf16", ""),
             "chunk": rs[0].get("chunk", ""), "gen_chunk": rs[0].get("gen_chunk", "")}
        other = [f"{c}={rs[0].get(c)}" for c, _ in ENV_COLS if c not in _SIG_SKIP and c not in ("numerics", "xdev_bf16", "pipeline", "pipeline_chunk") and str(rs[0].get(c, "")) != ""]
        d["other_knobs"] = " ".join(other)
        d["dec_S0_2K"] = _mean(pick(["S0"], "decode_tok_s"))
        d["dec_S1_32K_cold"] = _mean(pick(["S1-32K"], "decode_tok_s"))
        v = pick(["S1-32K"], "decode_tok_s"); v = [x for x in v if x is not None]
        d["dec_S1_min"] = min(v) if v else None; d["dec_S1_max"] = max(v) if v else None
        d["dec_S2_warm"] = _mean(pick(["S2"], "decode_tok_s"))
        d["dec_S1_128K"] = _mean(pick(["S1-128K"], "decode_tok_s"))
        d["dec_S3_turn1"] = _mean(pick_exact(["S3_t1"], "decode_tok_s"))
        d["dec_S3_turns2_6"] = _mean(pick_exact(["S3_t2", "S3_t3", "S3_t4", "S3_t5", "S3_t6"], "decode_tok_s"))
        d["dec_S5_switch"] = _mean(pick_exact(["S5_t7", "S5_t8"], "decode_tok_s"))
        d["dec_S5b_fresh"] = _mean(pick_exact(["S5b"], "decode_tok_s"))
        d["dec_S4_512K"] = _mean(pick_exact(["S4-512K_t1", "S4-512K_t2", "S4-512K_t3"], "decode_tok_s"))
        d["pre_S1_32K_cold"] = _mean(pick(["S1-32K"], "prefill_tok_s"))
        d["pre_S2_warm"] = _mean(pick(["S2"], "prefill_tok_s"))
        d["pre_S1_128K"] = _mean(pick(["S1-128K"], "prefill_tok_s"))
        d["pre_S4_512K"] = _mean(pick_exact(["S4-512K_t0"], "prefill_tok_s"))
        d["ttft_S1_32K_s"] = _mean(pick(["S1-32K"], "ttft_s"))
        d["ttft_S3_turns_s"] = _mean(pick_exact(["S3_t2", "S3_t3", "S3_t4", "S3_t5", "S3_t6"], "ttft_s"))
        d["delta_pre_S3_turns"] = _mean(pick_exact(["S3_t2", "S3_t3", "S3_t4", "S3_t5", "S3_t6"], "delta_prefill_tok_s"))
        d["hit_S1"] = _mean(pick(["S1-32K"], "hit_rate")); d["hit_S2"] = _mean(pick(["S2"], "hit_rate"))
        d["miss_per_tok_S1"] = _mean(pick(["S1-32K"], "miss_per_tok"))
        d["h2d_MB_tok_S1"] = _mean(pick(["S1-32K"], "h2d_mb_per_tok"))
        d["p90_ms_S1"] = _mean(pick(["S1-32K"], "p90_ms"))
        d["gpu_cache_GiB"] = _num(rs[0].get("gpu_cache_gib")); d["ram_tier_GiB"] = _num(rs[0].get("ram_tier_gib"))
        d["vram_a_GiB"] = _num(rs[0].get("vram_alloc_gpu_a_gib")); d["vram_b_GiB"] = _num(rs[0].get("vram_alloc_gpu_b_gib"))
        d["load_s"] = _mean([_num(r.get("load_s")) for r in rs])
        d["disk_experts"] = rs[0].get("disk_experts", "")
        dec_parts = [d["dec_S1_32K_cold"], d["dec_S2_warm"], d["dec_S1_128K"], d["dec_S3_turns2_6"], d["dec_S5_switch"]]
        pre_parts = [d["pre_S1_32K_cold"], d["pre_S2_warm"], d["pre_S1_128K"]]
        d["DECODE_score"] = _mean(dec_parts) if all(x is not None for x in dec_parts) else None
        d["PREFILL_score"] = _mean(pre_parts) if all(x is not None for x in pre_parts) else None
        cmp_ = str(d["cmp_layers"]).strip()
        fam_c = cmp_ not in ("", "0", "None") and (d["disk_experts"] in ("off",) or "disk experts=off" in d["placement"])
        onem = str(rs[0].get("cache_tokens", "")) == "1048576"
        d["family_c_ok"] = "yes" if (fam_c and onem) else ("1M only" if onem else "no")
        out.append(d)
    ranked = list(out)   # configurations with a partial scenario set are kept, ranked after the complete ones
    for key in ("DECODE_score", "PREFILL_score"):
        srt = sorted([d for d in ranked if d[key] is not None], key=lambda d: -d[key])
        for i, d in enumerate(srt, 1):
            d[key.split("_")[0].lower() + "_rank"] = i
    ranked.sort(key=lambda d: (d["DECODE_score"] is None, -(d["DECODE_score"] or 0), -(d["PREFILL_score"] or 0)))
    lcols = ["decode_rank", "prefill_rank", "config", "family_c_ok", "DECODE_score", "PREFILL_score", "runs", "cmp_layers",
             "pipeline", "pipe_chunk", "numerics", "xdev_bf16", "chunk", "gen_chunk",
             "dec_S0_2K", "dec_S1_32K_cold", "dec_S1_min", "dec_S1_max", "dec_S2_warm", "dec_S1_128K", "dec_S3_turn1",
             "dec_S3_turns2_6", "dec_S5_switch", "dec_S5b_fresh", "dec_S4_512K",
             "pre_S1_32K_cold", "pre_S2_warm", "pre_S1_128K", "pre_S4_512K", "ttft_S1_32K_s", "ttft_S3_turns_s",
             "delta_pre_S3_turns", "hit_S1", "hit_S2", "miss_per_tok_S1", "h2d_MB_tok_S1", "p90_ms_S1",
             "gpu_cache_GiB", "ram_tier_GiB", "vram_a_GiB", "vram_b_GiB", "load_s", "disk_experts", "placement", "other_knobs", "labels"]
    def cell(v):
        if v is None: return ""
        if isinstance(v, float): return f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v).replace("|", "/")
    L = [f"## Leaderboard: {len(ranked)} configurations x {len(lcols)} columns (recipe search)", "",
         "One row per configuration (identical placement and knobs), averaged over every run and scenario of it. DECODE_score ="
         " mean decode of cold 32K, warm 32K, cold 128K, multi-turn turns 2-6 and topic switch; PREFILL_score = mean prefill of cold"
         " 32K, warm 32K and cold 128K. family_c_ok = CMP holds layers, zero expert SSD reads and a 1M cache (the recipe rules)."
         " Sorted by DECODE_score.", "",
         "| " + " | ".join(lcols) + " |", "|" + "---|" * len(lcols)]
    for d in ranked:
        L.append("| " + " | ".join(cell(d.get(c)) for c in lcols) + " |")
    L.append("")
    return L
