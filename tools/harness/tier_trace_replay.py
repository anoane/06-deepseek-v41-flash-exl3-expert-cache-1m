#!/usr/bin/env python3
"""T1 (recipe PLAN.md v2, section 3.2): replay an EXL3_MOE_TIER_TRACE through the tier's reference policy
(exllamav3/model/expert_tier_policy.py, which the native core reproduces bit for bit) under other splits, pool sizes
and policies. The routing in the trace does not depend on the tier's state (the tier never changes the logits), so
for the PRO's cache layers of a split s (trace layers s..39 of a config-a trace) the replay is a counterfactual of
what that tier would have hit, admitted, copied and demoted on the same text (the CMP layers of a real split compute
on another GPU: routing there can differ at the chaotic floor, which the replay does not model).

usage: tier_trace_replay.py --src <tree> --trace <a.jsonl[.gz]> --result <result line json or results.jsonl + --label>
                            --variants <variants.json> [--out out.json]
variants.json: [{"name": ..., "split": s, "slots": S, "ram_slots": R | "all", "pins": 0, "disk_off": bool,
                 "policy": {PolicyConfig fields: policy, admit, evict, spare, halflife, prefill_inc, ...}}, ...]
Per variant and per job of the traced process (job boundaries from the result line's tier record counters): decode
hits / admits / transients, misses per decode token, h2d-from-RAM and d2h (demotion) copies, layer-mode staging.
"""
import argparse, gzip, importlib.util, json, sys, time


def load_policy(src):
    spec = importlib.util.spec_from_file_location("expert_tier_policy_ref", f"{src}/exllamav3/model/expert_tier_policy.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m              # dataclasses look their module up while the class is built
    spec.loader.exec_module(m)
    return m


def read_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        hdr = json.loads(f.readline())
        recs = []
        lines = f.readlines()
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            j = json.loads(line)
        except ValueError:
            # the writer's stdio buffer is not flushed when the process ends with os._exit (the harness does):
            # only a truncated LAST line is tolerated
            if n == len(lines) - 1:
                hdr["truncated_last_line"] = True
                break
            raise
        recs.append((j["seq"], j["lc"], j["mode"], j["tokens"], [(x[0], x[1], x[2]) for x in j["e"]]))
    return hdr, recs


def job_bounds(res):
    """[(label, first record index, end index, decode tokens, measured decode counters)] from a result line's h2h /
    longctx jobs: the tier's record counter at the job's end minus its prefill + decode records"""
    out = []
    for sec in ("h2h", "longctx"):
        for m in (res.get(sec) or {}).get("jobs", []):
            tr = m.get("tier") or {}
            tot, pf, dec = tr.get("total") or {}, tr.get("prefill") or {}, tr.get("decode") or {}
            if "records" not in tot:
                continue
            end = tot["records"]
            start = end - pf.get("records", 0) - dec.get("records", 0)
            lab = m["scenario"] + ("" if m.get("turn") is None else f"_t{m['turn']}")
            out.append({"label": lab, "start": start, "end": end, "prefill_end": start + pf.get("records", 0),
                        "decode_tokens": max(1, (m.get("new_tokens") or 1) - 1),
                        "measured": {k: dec.get("policy_" + k, 0) for k in ("hits", "admits", "transients", "d2h",
                                                                              "h2d_ram", "ssd", "retires")}})
    return out


def run_variant(P, hdr, recs, v, jobs):
    E = hdr["experts"]
    split = v.get("split", 0)
    L = hdr["layers"] - split
    S = v["slots"]
    kw = dict(v.get("policy", {}))
    # ram_slots "all": every cached expert the pool does not hold at the cold fill (held = S - spare), as the
    # engine sizes the RAM tier with disk experts=off
    R = L * E - (S - kw.get("spare", 8)) if v.get("ram_slots") == "all" else v["ram_slots"]
    kw.setdefault("disk_off", bool(v.get("disk_off", False)))
    cfg = P.PolicyConfig(**kw)
    core = P.TierCore(cfg, L, E, S, max(0, R), v.get("pins", 0))
    core.cold_fill(P.round_robin_order(L, E))
    # per job snapshots of the counters at the job's first record, its prefill end and its end
    marks = {}
    for j in jobs:
        marks.setdefault(j["start"], []).append((j["label"], "start"))
        marks.setdefault(j["prefill_end"], []).append((j["label"], "pf_end"))
        marks.setdefault(j["end"], []).append((j["label"], "end"))
    snaps = {}
    t0 = time.time()
    for i, (seq, lc, mode, tokens, ents) in enumerate(recs):
        for lab, what in marks.get(i, ()):
            snaps[(lab, what)] = dict(core.c)
        core.heat.tokens = tokens
        if lc < split:
            continue
        lc2 = lc - split
        base = lc * E
        ids = []
        for key, kind, cnt in ents:
            ids += [key - base] * cnt
        if mode == P.LAYER:
            core.layer_call(lc2, ids)
        else:
            core.call(lc2, ids, mode)
        if len(core.log) > 256:
            core.log.clear()
    for lab, what in marks.get(len(recs), ()):
        snaps[(lab, what)] = dict(core.c)
    out = {"name": v.get("name"), "split": split, "layers": L, "slots": S, "ram_slots": R, "policy": kw,
           "seconds": round(time.time() - t0, 1), "jobs": []}
    for j in jobs:
        a, m, b = snaps.get((j["label"], "start")), snaps.get((j["label"], "pf_end")), snaps.get((j["label"], "end"))
        if a is None or b is None or m is None:
            out["jobs"].append({"label": j["label"], "error": "job outside the trace"})
            continue
        d = {k: b.get(k, 0) - m.get(k, 0) for k in b}
        p = {k: m.get(k, 0) - a.get(k, 0) for k in m}
        used = d.get("hits", 0) + d.get("admits", 0) + d.get("transients", 0)
        n = j["decode_tokens"]
        out["jobs"].append({"label": j["label"], "decode_tokens": n,
                            "decode_hit_rate": round(d.get("hits", 0) / used, 5) if used else None,
                            "decode_misses_per_token": round((used - d.get("hits", 0)) / n, 3),
                            "decode_admits": d.get("admits", 0), "decode_transients": d.get("transients", 0),
                            "decode_d2h_per_token": round(d.get("d2h", 0) / n, 3),
                            "decode_ssd_per_token": round(d.get("ssd", 0) / n, 3),
                            "prefill_staged_experts": p.get("prefill_h2d", 0) + p.get("prefill_ssd", 0),
                            "prefill_ssd_experts": p.get("prefill_ssd", 0),
                            "measured_hit_rate": round(j["measured"]["hits"] / max(1, j["measured"]["hits"] +
                                                       j["measured"]["admits"] + j["measured"]["transients"]), 5)
                            if split == 0 else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required = True)
    ap.add_argument("--trace", required = True)
    ap.add_argument("--result", required = True, help = "a results.jsonl (with --label) or one result line's JSON file")
    ap.add_argument("--label", default = None)
    ap.add_argument("--variants", required = True)
    ap.add_argument("--out", default = None)
    a = ap.parse_args()
    P = load_policy(a.src)
    res = None
    for line in open(a.result):
        j = json.loads(line)
        if a.label is None or (j.get("label") == a.label and j.get("status") == "ok"):
            res = j
    if res is None:
        raise SystemExit(f"no ok result line for {a.label}")
    hdr, recs = read_trace(a.trace)
    jobs = job_bounds(res)
    vs = json.load(open(a.variants))
    out = {"trace": a.trace, "header": hdr, "records": len(recs), "jobs": jobs, "variants": []}
    for v in vs:
        r = run_variant(P, hdr, recs, v, jobs)
        out["variants"].append(r)
        print(json.dumps({"name": r["name"], "seconds": r["seconds"],
                          "hit": {x["label"]: x.get("decode_hit_rate") for x in r["jobs"]}}), flush = True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent = 1)


if __name__ == "__main__":
    main()
