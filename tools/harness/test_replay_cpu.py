"""CPU check of tier_trace_replay.py: a synthetic trace written by the reference core itself replays to the same
counters (split 0, same sizes), and a split drops the first layers."""
import json, os, random, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier_trace_replay as T
SRC = sys.argv[1]
P = T.load_policy(SRC)
L, E, S, R = 6, 32, 60, 40
cfg = P.PolicyConfig(disk_off = False)
core = P.TierCore(cfg, L, E, S, R)
core.cold_fill(P.round_robin_order(L, E))
rnd = random.Random(1)
td = tempfile.mkdtemp(dir = "/root/rnd/release/recipe")
tr = os.path.join(td, "t.jsonl")
lines = [json.dumps({"format": "exl3-tier-trace", "version": 1, "device": 0, "layers": L, "experts": E, "slots": S,
                     "pins": 0, "ram_slots": R, "spare": 8})]
jobs, nrec, tokens = [], 0, 0
tot = {"records": 0}
for job in range(3):
    start = nrec
    for lc in range(L):                                   # one layer-mode prefill pass
        ids = [rnd.randrange(E) for _ in range(200)]
        core.seq += 0
        acts = core.layer_call(lc, ids)
        rec = core.log[-1][0]
        cnt = {}
        for e in ids:
            cnt[e] = cnt.get(e, 0) + 1
        order = []
        for e in ids:
            if e not in order:
                order.append(e)
        lines.append(json.dumps({"seq": core.seq, "lc": lc, "mode": P.LAYER, "tokens": tokens,
                                 "e": [[lc * E + e, 2, cnt[e], -1, -1, -1] for e in order]}))
        nrec += 1
    pf_end = nrec
    c0 = dict(core.c)
    for t in range(40):
        tokens += 1
        core.tick(1)
        for lc in range(L):
            hot = [rnd.randrange(8) for _ in range(4)] + [rnd.randrange(E) for _ in range(2)]
            rec, acts = core.call(lc, hot, P.DECODE)
            lines.append(json.dumps({"seq": rec.seq, "lc": lc, "mode": P.DECODE, "tokens": core.heat.tokens,
                                     "e": [[x.key, x.kind, x.cnt, x.dst, x.vkey, x.vslot] for x in rec.entries]}))
            nrec += 1
    c1 = dict(core.c)
    dec = {"policy_" + k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    dec["records"] = nrec - pf_end
    jobs.append({"scenario": f"J{job}", "turn": None, "new_tokens": 41,
                 "tier": {"total": {"records": nrec}, "prefill": {"records": pf_end - start}, "decode": dec}})
open(tr, "w").write("\n".join(lines) + "\n")
res = os.path.join(td, "r.jsonl")
open(res, "w").write(json.dumps({"label": "x", "status": "ok", "h2h": {"jobs": jobs}}) + "\n")
var = os.path.join(td, "v.json")
json.dump([{"name": "same", "split": 0, "slots": S, "ram_slots": R, "policy": {}},
           {"name": "split2-all", "split": 2, "slots": 40, "ram_slots": "all", "disk_off": True, "policy": {}}], open(var, "w"))
out = os.path.join(td, "o.json")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_trace_replay.py"),
                "--src", SRC, "--trace", tr, "--result", res, "--variants", var, "--out", out], check = True)
o = json.load(open(out))
same = o["variants"][0]["jobs"]
for j, m in zip(same, jobs):
    d = m["tier"]["decode"]
    exp = d["policy_hits"] / (d["policy_hits"] + d["policy_admits"] + d["policy_transients"])
    assert abs(j["decode_hit_rate"] - round(exp, 5)) < 1e-9, (j, exp)
    assert j["measured_hit_rate"] == j["decode_hit_rate"], j
sp = o["variants"][1]
assert sp["layers"] == 4 and sp["ram_slots"] == 4 * E - 32 and all(j.get("decode_hit_rate") is not None for j in sp["jobs"]), sp
print("replay reproduces the reference counters per job:", [j["decode_hit_rate"] for j in same])
print("split 2 (disk off, RAM = all):", [j["decode_hit_rate"] for j in sp["jobs"]])
import shutil; shutil.rmtree(td)
print("all replay checks passed")
