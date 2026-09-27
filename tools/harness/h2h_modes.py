"""Head-to-head scenarios for the recipe harness (PLAN.md v2, section 3): modes `h2h` and `longctx`.

Imported by dsv41_bench.py (same directory). Everything here drives the Generator exactly as the speed mode does
(greedy CustomSampler([SS_Argmax()]), no stop conditions, max_rq_tokens = generator chunk); what is new:

  H1  prompts at a fixed offset of a token source (canonical code ids or the prose ids), the same for every arm
  H2  sessions: ONE Generator for several turns, each turn = the previous turn's input ids + its generated ids + a
      delta, appended as exact token ids, so the page hashes (and the recurrent-state stashes) match as they do for a
      chat client that re-sends the raw history; per turn the cached (reused) tokens are recorded
  H3  timeline: a timestamp and the expert tier's host counters every N new tokens (windowed decode and hit rate)
  H5  long context: one prompt, then short suffixes on the same Generator (prefix reuse), then a longer prompt
  H7  per-device decode time: CUDA events around every forwarded module for N decode tokens (per-layer device time;
      no profiler, nothing synchronizes inside the measured tokens; the events are read after the job)

Scenario list (--scenarios, comma separated; the sets `full` and `short` expand to the plan's lists):
  S0                      first request after the load: canonical @700000, P 2048, G 512, fresh generator
  S1:<off>:<P>            novel text: canonical @off, P tokens, G = --gen-tokens, fresh generator, no prewarm
  S2:<off>:<P>            the same request again on a fresh generator (no prefix reuse: a full re-prefill)
  S3                      session: T1 = --session-first, T2.. = --session-deltas appended (one Generator)
  S5                      continues S3's Generator: --session-switch turns, then --session-fresh on a new Generator
  DT:<off>:<P>            H7 device timing on a fresh generator (warm text: use an S1 offset seen before)
"""
import gc, hashlib, json, os, time

import numpy as np

GIB = 1024 ** 3
SETS = {
    "full": "S0,S1:100000:32768,S2:100000:32768,S1:140000:32768,S2:140000:32768,S1:200000:131072,S3,S5,DT:100000:32768",
    "short": "S0,S1:100000:32768,S2:100000:32768,S3,S5,DT:100000:32768",
}
DEFAULT_PROSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prose_ids.npy")


# ------------------------------------------------------------------------------------------------------------------
# argument helpers (pure; tested on the CPU)
# ------------------------------------------------------------------------------------------------------------------

def parse_span(s, default_src = "canonical"):
    """'[src:]N@OFF' -> (src, N, OFF); e.g. '16384@400000', 'prose:4096@0'"""
    s = s.strip()
    src = default_src
    if ":" in s.split("@")[0]:
        src, s = s.split(":", 1)
    n, _, off = s.partition("@")
    if src not in ("canonical", "prose"):
        raise ValueError(f"unknown token source {src!r} (canonical | prose)")
    return src, int(n), int(off or 0)


def parse_turns(s, default_src = "canonical"):
    """'1024,4096,512@420000' (consecutive deltas from one offset) or 'prose:4096@0:512,canonical:2048@460000'
    (each its own source / offset, optional :G) -> [(src, n, off, G or None)]"""
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if all("@" not in p for p in parts[:-1]) and "@" in parts[-1] and ":" not in parts[-1].split("@")[0]:
        # consecutive form: sizes, the offset on the last one
        last_n, _, off = parts[-1].partition("@")
        sizes = [int(p) for p in parts[:-1]] + [int(last_n)]
        out, o = [], int(off)
        for n in sizes:
            out.append((default_src, n, o, None))
            o += n
        return out
    out = []
    for p in parts:
        g = None
        head, _, tail = p.partition("@")
        if ":" in tail:
            tail, _, gs = tail.partition(":")
            g = int(gs)
        src, n, off = parse_span(head + "@" + tail, default_src)
        out.append((src, n, off, g))
    return out


def parse_scenarios(s):
    s = SETS.get(s.strip(), s)
    out = []
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        p = tok.split(":")
        kind = p[0]
        if kind in ("S0", "S3", "S5") and len(p) == 1:
            out.append((kind, None, None))
        elif kind in ("S1", "S2", "DT") and len(p) == 3:
            out.append((kind, int(p[1]), int(p[2])))
        else:
            raise ValueError(f"bad scenario {tok!r} (S0 | S1:<off>:<P> | S2:<off>:<P> | S3 | S5 | DT:<off>:<P>)")
    kinds = [k for k, _, _ in out]
    if "S5" in kinds and ("S3" not in kinds or kinds.index("S5") < kinds.index("S3")):
        raise ValueError("S5 continues the S3 session: list S3 before S5")
    return out


def scenario_need(a):
    """(canonical tokens, prose tokens) this run's h2h / longctx modes read"""
    need_c, need_p = 0, 0

    def take(src, end):
        nonlocal need_c, need_p
        if src == "prose":
            need_p = max(need_p, end)
        else:
            need_c = max(need_c, end)

    if "h2h" in a.modes:
        for kind, off, P in a.scenarios:
            if kind == "S0":
                take("canonical", 700000 + 2048)
            elif kind in ("S1", "S2", "DT"):
                take("canonical", off + P)
            elif kind == "S3":
                src, n, off0 = a.session_first
                take(src, off0 + n)
                for src, n, off, _ in a.session_deltas:
                    take(src, off + n)
            elif kind == "S5":
                for src, n, off, _ in a.session_switch:
                    take(src, off + n)
                src, n, off = a.session_fresh
                take(src, off + n)
    if "longctx" in a.modes:
        take("canonical", max(a.long_prompt) + a.long_suffix * (a.long_decode_repeats + 1))
    return need_c, need_p


def session_tokens(a, gen_tokens):
    """The largest context one S3 / S5 session reaches (every turn's input + G), for the cache check"""
    src, n, off = a.session_first
    tot = n + gen_tokens
    for _, n, _, g in a.session_deltas:
        tot += n + (g or gen_tokens)
    for _, n, _, g in a.session_switch:
        tot += n + (g or gen_tokens)
    return tot


# ------------------------------------------------------------------------------------------------------------------
# tier arithmetic (pure)
# ------------------------------------------------------------------------------------------------------------------

def tier_summary(tr, ntok):
    """Per-job tier numbers from ctx.last_tier ({prefill, decode, total} counter deltas); ntok = decode steps"""
    if not tr:
        return None
    d, pf = tr.get("decode") or {}, tr.get("prefill") or {}
    used = d.get("policy_hits", 0) + d.get("policy_admits", 0) + d.get("policy_transients", 0)
    ntok = max(1, ntok)
    return {"decode_hit_rate": round(d.get("policy_hits", 0) / used, 5) if used else None,
            "decode_lookups": used, "decode_misses": used - d.get("policy_hits", 0),
            "decode_misses_per_token": round((used - d.get("policy_hits", 0)) / ntok, 3),
            "decode_admits": d.get("policy_admits", 0), "decode_transients": d.get("policy_transients", 0),
            "decode_h2d_mb_per_token": round(d.get("h2d_bytes", 0) / ntok / 1e6, 3),
            "decode_d2h_mb_per_token": round(d.get("d2h_bytes", 0) / ntok / 1e6, 3),
            "decode_ssd_mb_per_token": round(d.get("ssd_bytes", 0) / ntok / 1e6, 3),
            "decode_ssd_reads": d.get("ssd_reads", 0),
            "prefill_staged_gb": round((pf.get("layer_h2d_bytes", 0) + pf.get("h2d_bytes", 0)) / 1e9, 3),
            "prefill_d2h_gb": round(pf.get("d2h_bytes", 0) / 1e9, 3),
            "prefill_ssd_gb": round(pf.get("ssd_bytes", 0) / 1e9, 3), "prefill_ssd_reads": pf.get("ssd_reads", 0),
            "prefill_layer_plans": pf.get("layer_plans", 0)}


def timeline_windows(t_first, points, n_first = 1):
    """points = [(t, n_tokens, tier snapshot or None)] taken every N tokens after the first token (t_first,
    n_first); -> per window: tokens, seconds, tok/s, tier hit rate over the window (None without a tier)"""
    out = []
    prev_t, prev_n, prev_s = t_first, n_first, None
    for t, n, s in points:
        w = {"n_end": n, "tokens": n - prev_n, "s": round(t - prev_t, 5),
             "tps": round((n - prev_n) / (t - prev_t), 3) if t > prev_t else None}
        if s is not None and prev_s is not None and "error" not in s and "error" not in prev_s:
            h = s.get("policy_hits", 0) - prev_s.get("policy_hits", 0)
            u = h + (s.get("policy_admits", 0) - prev_s.get("policy_admits", 0)) + \
                (s.get("policy_transients", 0) - prev_s.get("policy_transients", 0))
            w["hit_rate"] = round(h / u, 5) if u else None
            w["h2d_mb"] = round((s.get("h2d_bytes", 0) - prev_s.get("h2d_bytes", 0)) / 1e6, 2)
        out.append(w)
        prev_t, prev_n, prev_s = t, n, s
    return out


def recovery(windows, ref_tps, ref_hit, tol_tps = 0.05, tol_hit = 0.01):
    """S5: the worst window vs the reference (T6's mean), and the tokens after which every later window stays
    within tol of it (decode tok/s; hit rate in absolute points). None when never recovered."""
    if not windows:
        return None
    tps = [w.get("tps") for w in windows]
    hits = [w.get("hit_rate") for w in windows]
    out = {"ref_tps": ref_tps, "ref_hit": ref_hit}
    if ref_tps and any(t is not None for t in tps):
        worst = min(t for t in tps if t is not None)
        out["worst_tps"] = worst
        out["collapse_pct"] = round(100 * (1 - worst / ref_tps), 2)
        rec = None
        for i in range(len(windows)):
            if all(t is not None and t >= (1 - tol_tps) * ref_tps for t in tps[i:]):
                rec = windows[i - 1]["n_end"] if i > 0 else 0
                break
        out["recovered_tps_after_tokens"] = rec
    if ref_hit is not None and any(h is not None for h in hits):
        worst = min(h for h in hits if h is not None)
        out["worst_hit"] = worst
        rec = None
        for i in range(len(windows)):
            if all(h is not None and h >= ref_hit - tol_hit for h in hits[i:]):
                rec = windows[i - 1]["n_end"] if i > 0 else 0
                break
        out["recovered_hit_after_tokens"] = rec
    return out


def devtime_summary(tokens, groups):
    """tokens = [(wall_s, [(group key, device, ms)])] for the measured decode steps; groups = {key: (kind, layer)}.
    -> per device: module ms per token, per-layer ms (mean / median / p90 over layer-tokens); wall and residual"""
    if not tokens:
        return None
    by_dev = {}
    lay = {}
    walls = []
    for wall, rows in tokens:
        walls.append(1000 * wall)
        per_dev = {}
        for key, dev, ms in rows:
            per_dev[dev] = per_dev.get(dev, 0.0) + ms
            kind, layer = groups.get(key, ("other", None))
            if kind == "block":
                lay.setdefault(dev, []).append(ms)
        for dev, ms in per_dev.items():
            by_dev.setdefault(dev, []).append(ms)
    n = len(tokens)
    out = {"tokens": n, "wall_ms_per_token": round(float(np.mean(walls)), 4),
           "wall_ms_per_token_p50": round(float(np.median(walls)), 4), "devices": {}}
    tot = 0.0
    for dev, v in sorted(by_dev.items()):
        m = float(np.mean(v))
        tot += m
        d = {"module_ms_per_token": round(m, 4)}
        lv = lay.get(dev)
        if lv:
            nl = len(lv) // n if n else 0
            d.update({"blocks": nl, "layer_ms_mean": round(float(np.mean(lv)), 4),
                      "layer_ms_p50": round(float(np.median(lv)), 4),
                      "layer_ms_p90": round(float(np.percentile(lv, 90)), 4)})
        out["devices"][str(dev)] = d
    out["residual_ms_per_token"] = round(out["wall_ms_per_token"] - tot, 4)
    return out


# ------------------------------------------------------------------------------------------------------------------
# GPU-side helpers
# ------------------------------------------------------------------------------------------------------------------

def run_job(B, ctx, generator, ids, max_new, label, gen_chunk, timeline = 0):
    """B.run_gen_job with timeline points: every `timeline` new tokens a perf_counter stamp and the tier's host
    counters (no device sync). Returns (t0, stamps, toks, eos, points, job)."""
    from exllamav3 import Job
    from exllamav3.generator.sampler import CustomSampler, SS_Argmax
    torch = ctx.torch
    job = Job(input_ids = torch.as_tensor(ids, dtype = torch.long)[None], max_new_tokens = max_new,
              stop_conditions = [], sampler = CustomSampler([SS_Argmax()]), decode_special_tokens = True,
              max_rq_tokens = gen_chunk, identifier = label)
    stamps, toks, points = [], [], []
    eos = None
    last = 0
    nxt = timeline if timeline else None
    perf = time.perf_counter
    snap = B.tier_snapshot
    ts0 = snap(ctx)
    ts1 = None
    t0 = perf()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        res = generator.iterate()
        t = perf()
        n = job.new_tokens
        if n != last:
            stamps.append((t, n))
            if last == 0:
                if ts0 is not None:
                    ts1 = snap(ctx)
                if nxt is not None:
                    nxt = 1 + timeline
            elif nxt is not None and n >= nxt:
                points.append((t, n, snap(ctx) if ts0 is not None else None))
                nxt += timeline
            last = n
        for r in res:
            ti = r.get("token_ids")
            if ti is not None:
                toks.append(ti)
            if r.get("eos"):
                eos = r
    if ts0 is not None:
        try:
            for td in ctx.config.expert_tiers.devices.values():
                if td.handle is not None:
                    td.handle.drain()
        except Exception:
            pass
        ts2 = snap(ctx)
        ctx.last_tier = {"prefill": B.tier_delta(ts0, ts1), "decode": B.tier_delta(ts1, ts2), "total": ts2}
    else:
        ctx.last_tier = None
    return t0, stamps, toks, eos, points


def new_generator(ctx, gen_chunk, rc_mb):
    from exllamav3 import Generator
    nd = getattr(ctx.args, "ngram_draft", None)
    return Generator(model = ctx.model, cache = ctx.cache, tokenizer = ctx.tokenizer, max_batch_size = 1,
                     max_chunk_size = gen_chunk, recurrent_cache_size = rc_mb * 1024 ** 2, cpu_cache_size = 0,
                     num_draft_tokens = nd[1] if nd else None, ngram_match_min = nd[0] if nd else 0)


def generator_state(generator):
    d = {}
    try:
        rcc = getattr(generator, "recurrent_cache", None)
        d["recurrent_cache"] = None if rcc is None else {
            "entries": len(rcc), "bytes_gib": round(rcc.current_size / GIB, 4), "max_gib": round(rcc.max_size / GIB, 3)}
    except Exception as e:
        d["recurrent_cache"] = {"error": repr(e)}
    try:
        cs = generator.get_cache_stats()
        d["cache_stats"] = {k: cs.get(k) for k in ("max_tokens", "used_tokens", "cached_tokens", "alloc_pages",
                                                    "alloc_cached_pages", "hit_rate")}
    except Exception:
        pass
    return d


class Runner:
    """One process's scenario runner: token sources, the job recorder, the printed line"""

    def __init__(self, B, ctx, rec, marks, canon, prose, pcie):
        self.B, self.ctx, self.rec, self.marks = B, ctx, rec, marks
        self.src = {"canonical": canon, "prose": prose}
        self.pcie = pcie
        self.a = ctx.args
        self.jobs = []

    def tokens(self, src, n, off):
        arr = self.src[src]
        if arr is None or off + n > len(arr):
            raise ValueError(f"{src} ids: {off + n} tokens needed, {0 if arr is None else len(arr)} available")
        return np.asarray(arr[off:off + n], dtype = np.int64)

    def job(self, generator, ids, G, scen, turn, meta, gchunk, timeline = None):
        B, ctx = self.B, self.ctx
        torch = ctx.torch
        label = f"{scen}{'' if turn is None else '_t' + str(turn)}"
        tl = self.a.timeline if timeline is None else timeline
        pr = B.Probe(ctx).start()
        t0, st, tk, eos, pts = run_job(B, ctx, generator, ids, G, label, gchunk, tl)
        ph = pr.stop()
        P = len(ids)
        m = B.gen_metrics(P, G, t0, st, tk, eos, torch)
        gen_ids = m.pop("_ids", None)
        eos = tk = None
        cached = m.get("job_cached_tokens") or 0
        new_in = P - cached
        m.update({"scenario": scen, "turn": turn, **meta, "cached_tokens": cached, "prefilled_tokens": new_in,
                  "delta_prefill_tps": round(new_in / m["ttft_s"], 2) if m.get("ttft_s") else None})
        ntok = max(1, (m.get("new_tokens") or 1) - 1)
        m["tier"] = ctx.last_tier
        m["tier_summary"] = tier_summary(ctx.last_tier, ntok)
        if pts and st:
            m["timeline"] = timeline_windows(st[0][0], pts, st[0][1])
        m["phase"] = ph
        if self.pcie is not None:
            m["pcie"] = self.pcie.summarize(ph["t_start"], ph["t_end"])
        m.update(generator_state(generator))
        self.jobs.append(m)
        self.marks.append((label, ph["t_start"], ph["t_end"]))
        ts = m["tier_summary"] or {}
        print(f"== {label} P={P} cached={cached} G={G}: TTFT {m.get('ttft_s')} s, prefill {m.get('prefill_tps')} "
              f"(delta {m.get('delta_prefill_tps')}) tok/s, decode {m.get('decode_tps')} tok/s (p50 "
              f"{m.get('tpot_ms_p50')} ms, stalls {m.get('stall_count')}), hit {ts.get('decode_hit_rate')}, "
              f"miss/token {ts.get('decode_misses_per_token')}, H2D {ts.get('decode_h2d_mb_per_token')} MB/token, "
              f"staged {ts.get('prefill_staged_gb')} GB, SSD {ts.get('prefill_ssd_gb')}/{ts.get('decode_ssd_mb_per_token')}, "
              f"rows {ph.get('ngram_row_cache')}, cgroup {(ph.get('cgroup_end') or {}).get('current')} GiB"
              f"{' PROVISIONAL' if ph['provisional'] else ''}", flush = True)
        return m, gen_ids


def devtime(runner, ids, lead, n, gchunk):
    """H7: n decode steps after `lead` tokens, CUDA events around every forwarded module (per-module device time)"""
    B, ctx = runner.B, runner.ctx
    torch = ctx.torch
    from exllamav3 import Job
    from exllamav3.generator.sampler import CustomSampler, SS_Argmax
    model = ctx.model
    blocks = {id(b): i for i, b in enumerate(model._blocks())}
    groups, hooked = {}, []
    cur = {"on": False, "rows": None}
    for module, instance, idx in model.fwd_modules:
        if id(module) in groups or "forward" in module.__dict__:
            continue
        key = id(module)
        groups[key] = ("block", blocks[key]) if key in blocks else (type(module).__name__, idx)
        orig = module.forward

        def wrapped(x, params, *args, _orig = orig, _m = module, _key = key, **kw):
            if not cur["on"]:
                return _orig(x, params, *args, **kw)
            dev = getattr(_m, "device", None)
            if dev is None or torch.device(dev).type != "cuda":
                return _orig(x, params, *args, **kw)
            dev = torch.device(dev)
            e0 = torch.cuda.Event(enable_timing = True)
            e0.record(torch.cuda.current_stream(dev))
            y = _orig(x, params, *args, **kw)
            e1 = torch.cuda.Event(enable_timing = True)
            e1.record(torch.cuda.current_stream(dev))
            cur["rows"].append((_key, dev.index, e0, e1))
            return y
        module.forward = wrapped
        hooked.append(module)
    toks = []
    generator = new_generator(ctx, gchunk, runner.a.recurrent_cache_mb)
    try:
        job = Job(input_ids = torch.as_tensor(ids, dtype = torch.long)[None], max_new_tokens = lead + n + 2,
                  stop_conditions = [], sampler = CustomSampler([SS_Argmax()]), decode_special_tokens = True,
                  max_rq_tokens = gchunk, identifier = "devtime")
        generator.enqueue(job)
        perf = time.perf_counter
        plain = []                  # wall per decode step without the hooks (tokens 2 .. lead - 1)
        ts_a = ts_b = None
        while generator.num_remaining_jobs():
            k = job.new_tokens
            on = lead <= k < lead + n
            if on and ts_a is None:
                ts_a = B.tier_snapshot(ctx)
            if not on and k >= lead + n and ts_b is None:
                ts_b = B.tier_snapshot(ctx)
            cur["on"], cur["rows"] = on, ([] if on else None)
            t = perf()
            generator.iterate()
            if on:
                toks.append((perf() - t, cur["rows"]))
            elif 2 <= k < lead:
                plain.append(perf() - t)
            cur["on"] = False
    finally:
        cur["on"] = False
        for m in hooked:
            try:
                del m.forward
            except AttributeError:
                pass
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        del generator
        gc.collect()
    rows = [(w, [(key, dev, e0.elapsed_time(e1)) for key, dev, e0, e1 in rs]) for w, rs in toks]
    s = devtime_summary(rows, groups)
    if s is not None:
        s["lead_tokens"] = lead
        s["unhooked_wall_ms_per_token"] = round(1000 * float(np.mean(plain)), 4) if plain else None
        td = B.tier_delta(ts_a, ts_b) if ts_a is not None and ts_b is not None else None
        if td:
            used = td.get("policy_hits", 0) + td.get("policy_admits", 0) + td.get("policy_transients", 0)
            s["tier"] = {"hit_rate": round(td.get("policy_hits", 0) / used, 5) if used else None,
                         "misses_per_token": round((used - td.get("policy_hits", 0)) / max(1, len(rows)), 3),
                         "h2d_mb_per_token": round(td.get("h2d_bytes", 0) / max(1, len(rows)) / 1e6, 2),
                         "ssd_reads": td.get("ssd_reads", 0)}
        s["prompt_tokens"] = len(ids)
        s["modules_hooked"] = len(hooked)
    return s


# ------------------------------------------------------------------------------------------------------------------
# modes
# ------------------------------------------------------------------------------------------------------------------

def mode_h2h(B, ctx, rec, canon, prose, marks, pcie):
    a = ctx.args
    r = Runner(B, ctx, rec, marks, canon, prose, pcie)
    out = {"scenarios": [list(s) for s in a.scenarios], "gen_tokens": a.gen_tokens, "chunk": a.chunk,
           "gen_chunk": a.gen_chunk, "recurrent_cache_mb": a.recurrent_cache_mb, "timeline": a.timeline,
           "session_first": a.session_first, "session_deltas": a.session_deltas, "session_switch": a.session_switch,
           "session_fresh": a.session_fresh, "jobs": r.jobs, "settings": B.current_settings(ctx)}
    rec["h2h"] = out
    G = a.gen_tokens
    sess = None                      # (generator, input ids, generated ids) of the S3 session
    t6 = None
    for kind, off, P in a.scenarios:
        if kind == "S0":
            g = new_generator(ctx, a.gen_chunk, a.recurrent_cache_mb)
            r.job(g, r.tokens("canonical", 2048, 700000), 512, "S0", None,
                  {"source": "canonical", "offset": 700000, "kind": "S0"}, a.gen_chunk)
            del g
            gc.collect()
        elif kind in ("S1", "S2"):
            g = new_generator(ctx, a.gen_chunk, a.recurrent_cache_mb)
            r.job(g, r.tokens("canonical", P, off), G, f"{kind}-{P // 1024}K@{off}", None,
                  {"source": "canonical", "offset": off, "kind": kind}, a.gen_chunk)
            del g
            gc.collect()
        elif kind == "S3":
            g = new_generator(ctx, a.gen_chunk, a.recurrent_cache_mb)
            src, n, o = a.session_first
            inp = r.tokens(src, n, o)
            m, gen = r.job(g, inp, G, "S3", 1, {"source": src, "offset": o, "delta": n, "kind": "S3"}, a.gen_chunk)
            prev = np.concatenate([inp, gen.astype(np.int64)]) if gen is not None else inp
            turns = [m]
            for t, (src, n, o, g_) in enumerate(a.session_deltas, start = 2):
                inp = np.concatenate([prev, r.tokens(src, n, o)])
                m, gen = r.job(g, inp, g_ or G, "S3", t, {"source": src, "offset": o, "delta": n, "kind": "S3"},
                               a.gen_chunk)
                prev = np.concatenate([inp, gen.astype(np.int64)]) if gen is not None else inp
                turns.append(m)
            last = turns[-1]
            t6 = {"tps": last.get("decode_tps"), "hit": (last.get("tier_summary") or {}).get("decode_hit_rate")}
            sess = (g, prev, len(turns))
        elif kind == "S5":
            g, prev, t = sess
            for src, n, o, g_ in a.session_switch:
                t += 1
                inp = np.concatenate([prev, r.tokens(src, n, o)])
                m, gen = r.job(g, inp, g_ or G, "S5", t, {"source": src, "offset": o, "delta": n, "kind": "S5"},
                               a.gen_chunk)
                m["recovery_vs_last_S3_turn"] = recovery(m.get("timeline"), t6 and t6["tps"], t6 and t6["hit"])
                prev = np.concatenate([inp, gen.astype(np.int64)]) if gen is not None else inp
            sess = None
            del g
            gc.collect()
            src, n, o = a.session_fresh
            g = new_generator(ctx, a.gen_chunk, a.recurrent_cache_mb)
            m, _ = r.job(g, r.tokens(src, n, o), G, "S5b", None, {"source": src, "offset": o, "kind": "S5b"},
                         a.gen_chunk)
            m["recovery_vs_last_S3_turn"] = recovery(m.get("timeline"), t6 and t6["tps"], t6 and t6["hit"])
            del g
            gc.collect()
        elif kind == "DT":
            if a.devtime <= 0:
                continue
            pr = B.Probe(ctx).start()
            try:
                s = devtime(r, r.tokens("canonical", P, off), 16, a.devtime, a.gen_chunk)
                err = None
            except Exception as e:
                s, err = None, repr(e)
            ph = pr.stop()
            out["devtime"] = {"offset": off, "prompt_tokens": P, "summary": s, "error": err, "phase": ph}
            marks.append(("devtime", ph["t_start"], ph["t_end"]))
            print(f"== devtime: {json.dumps(s) if s else err}", flush = True)
    if sess is not None:
        del sess
    gc.collect()
    out["summary"] = summarize_jobs(r.jobs)


def mode_longctx(B, ctx, rec, canon, marks, pcie):
    """S4: prompt P1 (timed), then --long-decode-repeats jobs of P1 + a 64-token suffix (prefix reuse; decode at
    ~P1), then the same for P2 (the prefill of P2 - P1 timed through the reuse), on ONE Generator"""
    a = ctx.args
    r = Runner(B, ctx, rec, marks, canon, None, pcie)
    rec["longctx"] = {"prompts": a.long_prompt, "suffix": a.long_suffix, "repeats": a.long_decode_repeats,
                      "gen_tokens": a.gen_tokens, "jobs": r.jobs, "settings": B.current_settings(ctx)}
    g = new_generator(ctx, a.gen_chunk, a.recurrent_cache_mb)
    try:
        for P in a.long_prompt:
            base = r.tokens("canonical", P, 0)
            r.job(g, base, a.gen_tokens, f"S4-{P // 1024}K", 0, {"source": "canonical", "offset": 0, "kind": "S4"},
                  a.gen_chunk)
            for k in range(1, a.long_decode_repeats + 1):
                suf = r.tokens("canonical", a.long_suffix, P + a.long_suffix * (k - 1))
                r.job(g, np.concatenate([base, suf]), a.gen_tokens, f"S4-{P // 1024}K", k,
                      {"source": "canonical", "offset": 0, "kind": "S4", "suffix": a.long_suffix}, a.gen_chunk)
    finally:
        del g
        gc.collect()
    rec["longctx"]["summary"] = summarize_jobs(r.jobs)


def summarize_jobs(jobs):
    """Per scenario label: n, decode / prefill / TTFT medians and the tier's decode hit rate"""
    out = {}
    for m in jobs:
        k = m["scenario"] + ("" if m.get("turn") is None else f"_t{m['turn']}")
        e = out.setdefault(k, {"n": 0, "decode_tps": [], "prefill_tps": [], "delta_prefill_tps": [], "ttft_s": [],
                               "hit": [], "cached_tokens": []})
        e["n"] += 1
        e["decode_tps"].append(m.get("decode_tps"))
        e["prefill_tps"].append(m.get("prefill_tps"))
        e["delta_prefill_tps"].append(m.get("delta_prefill_tps"))
        e["ttft_s"].append(m.get("ttft_s"))
        e["hit"].append((m.get("tier_summary") or {}).get("decode_hit_rate"))
        e["cached_tokens"].append(m.get("cached_tokens"))
    return out


# ------------------------------------------------------------------------------------------------------------------
# H8: PCIe throughput from NVML, in a separate process (no Python thread in the measured process)
# ------------------------------------------------------------------------------------------------------------------

class PcieLog:
    def __init__(self, path, period_ms = 100):
        import subprocess, sys
        self.path = path
        self.proc = None
        try:
            self.proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "pcie_log.py"), path, str(period_ms)],
                                         stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        except Exception as e:
            print(f" !! PCIe logger not started: {e!r}", flush = True)
        self._rows = None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except Exception:
                self.proc.kill()

    def rows(self):
        out = []
        try:
            with open(self.path) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) != 4 or p[0] == "t":
                        continue
                    try:
                        out.append((float(p[0]), int(p[1]), float(p[2]), float(p[3])))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def summarize(self, t0, t1):
        """Mean TX (GPU -> host) / RX (host -> GPU) KB/s per NVML index over [t0, t1], and the implied MB. NVML
        samples a 20 ms window per read: an estimate, not a count."""
        rows = [r for r in self.rows() if t0 <= r[0] <= t1]
        by = {}
        for t, i, tx, rx in rows:
            by.setdefault(i, []).append((tx, rx))
        out = {}
        for i, v in sorted(by.items()):
            tx = float(np.mean([x[0] for x in v]))
            rx = float(np.mean([x[1] for x in v]))
            d = {"samples": len(v), "tx_kbs": round(tx, 1), "rx_kbs": round(rx, 1),
                 "tx_mb": round(tx * (t1 - t0) / 1000, 2), "rx_mb": round(rx * (t1 - t0) / 1000, 2)}
            out[str(i)] = d
        return out
