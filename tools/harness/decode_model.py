#!/usr/bin/env python3
"""Stage-0 decode model: per arm, least squares of decode ms/token = T0 + t_miss * (tier misses per decode token) over
the h2h jobs (S0 and S1-128K left out: 2K / 128K contexts), then t_pro = T0(a) / 40 and, for a split s,
s * t_cmp_eff = T0(c_s) - (40 - s) * t_pro (t_cmp_eff includes the per-token crossing).
usage: decode_model.py <results.jsonl> ... """
import json, sys
import numpy as np

rows = {}
for p in sys.argv[1:]:
    for l in open(p):
        j = json.loads(l)
        if j.get("status") != "ok" or not j.get("h2h") or "trace" in j["label"]:
            continue
        arm = j["label"].rsplit("-", 1)[0] if j["label"][-1].isdigit() else j["label"]
        for m in j["h2h"]["jobs"]:
            ts = m.get("tier_summary") or {}
            if m["scenario"] in ("S0",) or m["scenario"].startswith("S1-128K") or not m.get("decode_tps"):
                continue
            if ts.get("decode_misses_per_token") is None:
                continue
            rows.setdefault(arm, []).append((ts["decode_misses_per_token"], 1000.0 / m["decode_tps"], m["scenario"]))
fit = {}
for arm, r in sorted(rows.items()):
    x = np.array([a for a, _, _ in r]); y = np.array([b for _, b, _ in r])
    A = np.vstack([np.ones_like(x), x]).T
    (t0, tm), res, *_ = np.linalg.lstsq(A, y, rcond = None)
    pred = A @ np.array([t0, tm])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fit[arm] = (t0, tm)
    print(f"{arm}: n {len(r)}, T0 {t0:.2f} ms/token, t_miss {tm:.3f} ms/miss, R^2 {r2:.3f}, "
          f"misses/token {x.min():.1f}-{x.max():.1f}, ms/token {y.min():.1f}-{y.max():.1f}")
if "A1M" in fit:
    tpro = fit["A1M"][0] / 40
    print(f"t_pro = T0(a) / 40 = {tpro:.3f} ms per layer-token")
    for arm, (t0, tm) in fit.items():
        if arm.startswith("C"):
            s = int("".join(c for c in arm[1:] if c.isdigit())[:2])
            print(f"{arm}: split {s}: t_cmp_eff = (T0 - (40 - s) * t_pro) / s = {(t0 - (40 - s) * tpro) / s:.3f} ms per CMP layer-token")
