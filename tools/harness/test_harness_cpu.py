"""CPU-only checks of the harness's new pieces (no torch, no model, no GPU)."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_bench as B

ok = 0
def check(cond, what):
    global ok
    if not cond:
        raise SystemExit(f"FAIL: {what}")
    ok += 1
    print(f"ok  {what}")

# 1. latency list + stall flag
lat = [0.050] * 100 + [0.400] + [0.051] * 50
d = B.latency_fields(lat)
check(len(d["tpot_ms_list"]) == 151 and d["stall_count"] == 1 and d["stalls"][0]["step"] == 100, "stall found at step 100")
check(abs(d["decode_tps_excl_stalls"] - 150 / (0.05 * 100 + 0.051 * 50)) < 1e-2, "decode excl. stalls")
check(B.latency_fields([])["stall_flag"] is False, "empty latency list")

# 2. gen_metrics with synthetic stamps (no tokens, so no torch)
t0 = 100.0
stamps = [(t0 + 1.0 + 0.05 * i, i + 1) for i in range(20)]
m = B.gen_metrics(4096, 20, t0, stamps, [], None, None)
check(m["ttft_s"] == 1.0 and m["new_tokens"] == 20 and m["stall_count"] == 0 and len(m["tpot_ms_list"]) == 19, "gen_metrics fields")

# 3. argument parsing: vary, legacy, max-output-size rules
def parse(argv):
    sys.argv = ["dsv41_bench.py", "--label", "t"] + argv
    return B.parse_args()
a = parse(["--modes", "speed", "--repeats", "8", "--vary", "numerics=deepseek:index/precise/deepseek:index/precise",
           "--vary", "xdev_bf16=0/0/1/1"])
check(a.vary == {"numerics": ["deepseek:index", "precise", "deepseek:index", "precise"], "xdev_bf16": [False, False, True, True]},
      "vary parsed")
check(a.max_output_size == 32 and a.max_output_size_source == "serving-default" and a.recurrent_cache_mb == 4096, "speed defaults")
a = parse(["--modes", "legacy", "--reserve-mb", "-1"])
check(a.recurrent_cache_mb == 512 and a.cache_tokens == 1048576 and a.reserve_mb < 0 and a.legacy_decode_ctx == [24, 20000],
      "legacy defaults")
a = parse(["--modes", "speed,quality"])
check(a.max_output_size == 4096 and a.max_output_size_source == "quality-chunk", "quality chunk output size")
for bad in (["--modes", "speed,quality", "--max-output-size", "32"], ["--modes", "speed", "--vary", "bogus=1/2"],
            ["--modes", "quality", "--vary", "numerics=precise"], ["--modes", "speed", "--vary", "xdev_bf16=2"]):
    try:
        parse(bad)
        check(False, f"refused {bad}")
    except SystemExit as e:
        check(e.code == 2, f"refused {bad}")

# 4. legacy prompts from the vLLM reference file
a = parse(["--modes", "legacy", "--legacy-prefill", "8192,33600", "--legacy-decode-ctx", "24,20000"])
short, long_ids = B.legacy_prompts(a)
check(len(short) == 24 and len(long_ids) == 33600, "legacy prompts (24 / 33600)")
ref = json.load(open(B.DEFAULT_LEGACY_REF))
base = next(c for c in ref["cases"] if c["n"] == 20000)["prompt_ids"]
check(long_ids[:20000] == base and long_ids[20000:33600] == base[:13600], "long prompt = the 20000 case repeated")

# 5. pinned cores of our own tree, and per-thread attribution on two cores (one own, one foreign)
own = subprocess.Popen(["taskset", "-c", "22", sys.executable, "-c", "import time\nt=time.time()\nwhile time.time()-t<2.0: pass"])
foreign = subprocess.Popen(["setsid", "-f", "taskset", "-c", "23", sys.executable, "-c",
                            "import time\nt=time.time()\nwhile time.time()-t<2.5: pass"])
foreign.wait()           # setsid -f: the busy child is reparented, so it is not in our tree
time.sleep(0.2)
check(22 in B.pinned_cores(B.own_tree()), "pinned core 22 found in own tree")
class A:
    moe_cores = "auto"
class Ctx(B.Ctx):
    pass
ctx = Ctx(A())
ctx.moe_cores = [22, 23]
pr = B.Probe(ctx).start()
time.sleep(1.5)
ph = pr.stop()
own.wait()
print(json.dumps({k: ph[k] for k in ("moe_cores", "moe_core_foreign_direct", "moe_core_foreign_residual", "moe_core_foreign_max",
                                     "moe_core_residual_excess", "moe_cores_contended", "provisional_reasons",
                                     "tree_read_mib", "tree_majflt", "wall_s")}))
print(json.dumps(ph["foreign_threads_top"][:3]))
check(ph["moe_core_foreign_direct"][1] > 0.5, "foreign busy loop on core 23 attributed to core 23")
# core 22 carried our own busy loop (~1 core); other jobs share cores 16-23, so allow their share
check(ph["moe_core_foreign_direct"][0] < 0.5 and ph["moe_core_foreign_residual"][0] < 0.5,
      "own busy loop on core 22 not counted foreign")
check(ph["moe_cores_contended"] and ph["provisional"], "contended core makes the phase provisional")

# 6. load monitor
mon = B.HostMonitor(period = 0.1).start()
time.sleep(0.5)
r = mon.stop()
check(r["samples"] >= 3 and r["min_mem_available_gib"] > 0, "load monitor samples")
# 7. recipe (harness version 3): cgroup memory, n-gram row cache counters, tier sizing record
cg = B.cgroup_mem()
check(cg is not None and cg.get("current") is not None and cg["current"] > 0, f"cgroup memory read ({cg})")
class RC:
    def __init__(self, h, m):
        self.h, self.m = h, m
    def stats(self):
        return {"slots": 8, "bytes": 1000, "hits": self.h, "misses": self.m}
class E:
    def __init__(self, rc):
        self.row_cache = rc
class Blk:
    def __init__(self, e):
        self.engram = e
class FakeModel:
    def __init__(self, blocks):
        self.blocks = blocks
    def _blocks(self):
        return self.blocks
c2 = Ctx(A())
c2.model = FakeModel([Blk(E(RC(5, 3))), Blk(None), Blk(E(None)), Blk(E(RC(2, 1)))])
r = c2.row_cache_counters()
check(r == {"hits": 7, "misses": 4, "bytes": 2000, "caches": 2}, f"row cache counters summed ({r})")
c2.model = FakeModel([Blk(None)])
check(c2.row_cache_counters() == {"hits": 0, "misses": 0, "bytes": 0, "caches": 0}, "no row cache: zero caches")
import types
dev = types.SimpleNamespace(device = "cuda:1", cached = 7168, slots = 5784, spare = 8, held = 5776, staging_slots = 512,
                            pool_bytes = 5784 * 13 * 2**20, staging_bytes = 512 * 13 * 2**20)
sz = types.SimpleNamespace(devices = [dev], cached = 7168, hot = 0, tier_slots = 4976, ram_bytes = 61 * 2**30,
                           static_bytes = 0, ngram_bytes = 0, slab_slots = 8, slab_bytes = 2**20, disk_only = 0,
                           left_unpinned = None, text = "sizing text")
c2.config = types.SimpleNamespace(expert_tiers = types.SimpleNamespace(sizing = sz))
ts = B.tier_sizing(c2)
check(ts["devices"][0]["slots"] == 5784 and ts["tier_slots"] == 4976 and ts["ram_gib"] == 61.0 and ts["disk_only"] == 0,
      "tier sizing record")
c2.config = types.SimpleNamespace()
check(B.tier_sizing(c2) is None, "no tier: no sizing record")
print(f"all {ok} checks passed")
