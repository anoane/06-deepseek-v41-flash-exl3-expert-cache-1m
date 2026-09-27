#!/usr/bin/env python3
"""M2: dNLL between two quality npz files with a 95% moving-block bootstrap interval.

usage: dnll_ci.py <run.npz> <reference.npz> [more references ...] [--block 1024] [--iters 2000] [--seed 0] [--json]

For each reference: positions compared (min of both lengths), float32-bitwise equal log-probs, top-1 agreement,
dNLL = mean(-lp_run) - mean(-lp_ref) over the compared positions, and its 95% interval from a moving-block
bootstrap of the per-position differences (blocks of --block positions, drawn with replacement, n / block blocks
per resample: keeps the within-block correlation of neighbouring positions). "confident flips" = positions whose
top-1 differs where the run's own top-1 probability was above 0.5 (needs top1_lp in the run's file).
Anchors store actual_lp as float64 made from float32 and top_ids [n, 5]; run files store float32 actual_lp,
int32 top1 and float32 top1_lp.
"""
import json, sys

import numpy as np


def load(path):
    z = np.load(path)
    lp = z["actual_lp"].astype(np.float32)
    t1 = z["top1"] if "top1" in z.files else z["top_ids"][:, 0]
    t1lp = z["top1_lp"].astype(np.float32) if "top1_lp" in z.files else None
    return lp, t1.astype(np.int64), t1lp


def compare(run, ref, block, iters, seed):
    lp, t1, t1lp = run
    rlp, rt1, _ = ref
    n = min(len(lp), len(rlp))
    a, b = lp[:n], rlp[:n]
    d = (-a.astype(np.float64)) - (-b.astype(np.float64))
    bitwise = int((a.view(np.int32) == b.view(np.int32)).sum())
    agree = t1[:n] == rt1[:n]
    rng = np.random.default_rng(seed)
    nb = max(1, n // block)
    starts_max = max(1, n - block + 1)
    cs = np.concatenate([[0.0], np.cumsum(d)])
    means = np.empty(iters)
    for i in range(iters):
        s = rng.integers(0, starts_max, nb)
        means[i] = (cs[np.minimum(s + block, n)] - cs[s]).sum() / (np.minimum(s + block, n) - s).sum()
    lo, hi = np.quantile(means, [0.025, 0.975])
    out = {"compared": n, "lp_bitwise_equal": bitwise, "top1_agree_frac": round(float(agree.mean()), 6),
           "dnll": float(d.mean()), "ci95": [float(lo), float(hi)], "ci_contains_0": bool(lo <= 0 <= hi),
           "abs_dlp_mean": float(np.abs(d).mean())}
    if t1lp is not None:
        conf = np.exp(t1lp[:n].astype(np.float64)) > 0.5
        out["confident_flips"] = int((conf & ~agree).sum())
    return out


def main():
    args = sys.argv[1:]
    opts = {"--block": 1024, "--iters": 2000, "--seed": 0}
    as_json = "--json" in args
    if as_json:
        args.remove("--json")
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = int(args[i + 1])
            del args[i:i + 2]
    if len(args) < 2:
        raise SystemExit(__doc__)
    run = load(args[0])
    res = {}
    for p in args[1:]:
        res[p] = compare(run, load(p), opts["--block"], opts["--iters"], opts["--seed"])
    if as_json:
        print(json.dumps({"run": args[0], "results": res}))
        return
    print(f"run {args[0]}")
    for p, r in res.items():
        print(f"  vs {p}: {r['lp_bitwise_equal']}/{r['compared']} bitwise, top-1 {r['top1_agree_frac']:.4f}, dNLL "
              f"{r['dnll']:+.6f} [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"
              f"{'' if r['ci_contains_0'] else ' (CI excludes 0)'}"
              f"{', confident flips ' + str(r['confident_flips']) if 'confident_flips' in r else ''}")


if __name__ == "__main__":
    main()
