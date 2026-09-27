#!/usr/bin/env python3
"""Stage-1 decode model over several splits (replaces stage 0's one-split model, which had to put the per-token
crossing into t_cmp).

  config a:        ms/token = A0 + t_miss_a * misses                     -> t_pro = A0 / 40
  family c, split s: ms/token = B0 + d * s + t_miss_c * misses           -> d = t_cmp - t_pro (per CMP layer),
                                                                             t_cross = B0 - 40 * t_pro
Jobs: h2h decode jobs with tier counters, S0 and S1-128K left out (2K / 128K contexts), as in decode_model.py.
usage: decode_model2.py <results.jsonl> ...
"""
import json, re, sys, math


def lstsq(X, y):
    """least squares by the normal equations (small, well-conditioned problems; no numpy on every host)"""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for i in range(k):
        piv = max(range(i, k), key = lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        for r in range(k):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * z for x, z in zip(M[r], M[i])]
    return [M[i][k] / M[i][i] for i in range(k)]


def arm_of(label):
    a = re.sub(r"-\d+$", "", re.sub(r"-rerun$", "", label))
    return "C12" if a == "C12-base" else a


rows = {}
for p in sys.argv[1:]:
    for l in open(p):
        j = json.loads(l)
        if j.get("status") != "ok" or not j.get("h2h") or "trace" in j["label"]:
            continue
        arm = arm_of(j["label"])
        if not (arm == "A1M" or re.match(r"^C\d+$", arm)):
            continue
        for m in j["h2h"]["jobs"]:
            ts = m.get("tier_summary") or {}
            if m["scenario"] == "S0" or m["scenario"].startswith("S1-128K") or not m.get("decode_tps"):
                continue
            if ts.get("decode_misses_per_token") is None:
                continue
            rows.setdefault(arm, []).append((ts["decode_misses_per_token"], 1000.0 / m["decode_tps"]))


def fit(X, y):
    coef = lstsq(X, y)
    pred = [sum(c * x for c, x in zip(coef, r)) for r in X]
    my = sum(y) / len(y)
    ss = sum((a - b) ** 2 for a, b in zip(y, pred))
    r2 = 1 - ss / sum((a - my) ** 2 for a in y)
    return coef, r2, math.sqrt(ss / len(y))


a = rows.get("A1M", [])
(A0, tma), r2a, rmsa = fit([[1.0, m] for m, _ in a], [t for _, t in a])
tpro = A0 / 40
print(f"a: n {len(a)}: {A0:.2f} ms + {tma:.3f} ms/miss (R^2 {r2a:.3f}, rms {rmsa:.2f} ms) -> t_pro = {tpro:.3f} ms per layer-token")
cs = []
for arm, r in rows.items():
    if arm == "A1M":
        continue
    s = int(arm[1:])
    for m, t in r:
        cs.append((s, m, t))
(B0, d, tmc), r2c, rmsc = fit([[1.0, s, m] for s, m, _ in cs], [t for _, _, t in cs])
print(f"c: n {len(cs)} over splits {sorted(set(s for s, _, _ in cs))}: {B0:.2f} ms + {d:.3f} ms per CMP layer + "
      f"{tmc:.3f} ms/miss (R^2 {r2c:.3f}, rms {rmsc:.2f} ms)")
print(f"   -> t_cmp - t_pro = {d:.3f} ms; with t_pro from a: t_cmp = {tpro + d:.3f} ms, fixed per-token cost of the split "
      f"(crossing + second device) = {B0 - 40 * tpro:.2f} ms")
for arm in sorted(rows, key = lambda x: (x != "A1M", -int(x[1:]) if x != "A1M" else 0)):
    r = rows[arm]
    print(f"   {arm}: n {len(r)}, misses/token mean {sum(m for m, _ in r) / len(r):.1f}, "
          f"ms/token mean {sum(t for _, t in r) / len(r):.2f}")
