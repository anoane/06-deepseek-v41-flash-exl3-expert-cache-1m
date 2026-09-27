"""CPU-only checks of the stage-0 harness additions (h2h_modes.py, engram_drop.py, pcie_log.py, the new arguments).
No torch, no model, no GPU work (NVML is only queried)."""
import json, os, struct, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h2h_modes as H
import engram_drop as ED
import dsv41_bench as B

ok = 0
def check(cond, what):
    global ok
    if not cond:
        raise SystemExit(f"FAIL: {what}")
    ok += 1
    print(f"ok  {what}")

# 1. spans, turns, scenarios
check(H.parse_span("16384@400000") == ("canonical", 16384, 400000), "span canonical")
check(H.parse_span("prose:8192@100000") == ("prose", 8192, 100000), "span prose")
t = H.parse_turns("1024,4096,512,8192,2048@420000")
check([x[1:3] for x in t] == [(1024, 420000), (4096, 421024), (512, 425120), (8192, 425632), (2048, 433824)],
      "consecutive deltas from one offset")
t = H.parse_turns("prose:4096@0:512,canonical:2048@460000")
check(t == [("prose", 4096, 0, 512), ("canonical", 2048, 460000, None)], "explicit turns with G")
sc = H.parse_scenarios("full")
check([k for k, _, _ in sc] == ["S0", "S1", "S2", "S1", "S2", "S1", "S3", "S5", "DT"] and sc[5] == ("S1", 200000, 131072),
      "full set")
check([k for k, _, _ in H.parse_scenarios("short")] == ["S0", "S1", "S2", "S3", "S5", "DT"], "short set")
for bad in ("S5,S3", "S1:1", "S9", "S5"):
    try:
        H.parse_scenarios(bad)
        check(False, f"refused {bad}")
    except ValueError:
        check(True, f"refused {bad}")

# 2. arguments through the harness parser
def parse(argv):
    sys.argv = ["dsv41_bench.py", "--label", "t"] + argv
    return B.parse_args()
a = parse(["--modes", "h2h"])
check(a.cache_tokens == 1048576 and a.timeline == 32 and a.devtime == 32 and a.scenarios == sc, "h2h defaults (1M cache)")
nc, npr = H.scenario_need(a)
check(nc == 700000 + 2048 and npr == 100000 + 8192, f"token needs ({nc}, {npr})")
st = H.session_tokens(a, 256)
check(st == 16384 + 256 + sum([1024, 4096, 512, 8192, 2048]) + 5 * 256 + 4096 + 512 + 2048 + 256, f"session tokens {st}")
a = parse(["--modes", "longctx", "--long-prompt", "1040384,524288"])
check(a.long_prompt == [524288, 1040384] and H.scenario_need(a)[0] == 1040384 + 64 * 4, "longctx prompts sorted, need")
for bad in (["--modes", "h2h", "--cache-tokens", "65536"],                        # 128K scenario does not fit
            ["--modes", "longctx", "--long-prompt", "1048576"],                  # P + suffix + G > 1M
            ["--modes", "h2h", "--scenarios", "S5"], ["--modes", "h2h", "--session-first", "bogus:1@2"]):
    try:
        parse(bad)
        check(False, f"refused {bad}")
    except SystemExit as e:
        check(e.code == 2, f"refused {bad}")
a = parse(["--modes", "h2h", "--scenarios", "S1:100000:32768", "--cache-tokens", "65536"])
check(a.scenarios == [("S1", 100000, 32768)], "a 32K scenario fits a 64K cache")

# 3. tier summary, timeline windows, recovery
tr = {"decode": {"policy_hits": 900, "policy_admits": 60, "policy_transients": 40, "h2d_bytes": 100 * 13_000_000,
                 "d2h_bytes": 0, "ssd_bytes": 0, "ssd_reads": 0},
      "prefill": {"layer_h2d_bytes": 50_000_000_000, "h2d_bytes": 1_000_000_000, "ssd_bytes": 0, "layer_plans": 28}}
ts = H.tier_summary(tr, 255)
check(ts["decode_hit_rate"] == 0.9 and ts["decode_misses_per_token"] == round(100 / 255, 3) and ts["prefill_staged_gb"] == 51.0,
      "tier summary")
check(H.tier_summary(None, 10) is None, "no tier: no summary")
snap = lambda h, a_, t_: {"policy_hits": h, "policy_admits": a_, "policy_transients": t_, "h2d_bytes": 0}
pts = [(1.0 + 0.032 * k, 1 + 32 * k, snap(30 * k, 2 * k, 0)) for k in range(1, 5)]
w = H.timeline_windows(1.0, pts, 1)
check(len(w) == 4 and w[0]["tokens"] == 32 and abs(w[0]["tps"] - 1000.0) < 1e-6 and w[0].get("hit_rate") is None and
      abs(w[1]["hit_rate"] - 30 / 32) < 1e-9, "timeline windows (first window has no tier delta)")
wins = [{"n_end": 33, "tps": 20.0, "hit_rate": 0.80}, {"n_end": 65, "tps": 28.0, "hit_rate": 0.90},
        {"n_end": 97, "tps": 30.5, "hit_rate": 0.95}, {"n_end": 129, "tps": 31.0, "hit_rate": 0.955}]
rv = H.recovery(wins, 31.0, 0.96)
check(rv["worst_tps"] == 20.0 and rv["collapse_pct"] == round(100 * (1 - 20 / 31), 2) and
      rv["recovered_tps_after_tokens"] == 65 and rv["recovered_hit_after_tokens"] == 65, f"recovery {rv}")
check(H.recovery([{"n_end": 33, "tps": 10.0}], 31.0, None)["recovered_tps_after_tokens"] is None, "never recovered")
# 4. device-time summary: 2 devices, 3 blocks on dev 0 and 2 on dev 1, plus a head
groups = {1: ("block", 0), 2: ("block", 1), 3: ("block", 2), 4: ("block", 3), 5: ("block", 4), 6: ("Head", 45)}
toks = [(0.010, [(1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0), (4, 1, 0.5), (5, 1, 0.7), (6, 1, 0.3)])] * 4
d = H.devtime_summary(toks, groups)
check(d["devices"]["0"]["blocks"] == 3 and d["devices"]["0"]["layer_ms_mean"] == 1.0 and d["devices"]["1"]["blocks"] == 2 and
      abs(d["devices"]["1"]["module_ms_per_token"] - 1.5) < 1e-9 and abs(d["residual_ms_per_token"] - 5.5) < 1e-9,
      f"device time summary {json.dumps(d)}")
# 5. engram ranges from the real model's headers (read only)
rng = ED.engram_ranges(ED.DEFAULT_MODEL)
names = sorted(k for rs in rng.values() for k, _, _ in rs)
check(names == ["layers.1.engram.embed.scale", "layers.1.engram.embed.weight", "layers.14.engram.embed.scale",
                "layers.14.engram.embed.weight"], f"engram tensors {names}")
tot = sum(e - s for rs in rng.values() for _, s, e in rs)
check(tot > 100 * 2**30, f"engram tables {tot / 2**30:.1f} GiB")
# 6. drop() on a fake model dir: a written file is resident, then dropped
with tempfile.TemporaryDirectory(dir = "/root/rnd/release/recipe") as td:
    data = np.random.default_rng(0).integers(0, 255, 8 << 20, dtype = np.uint8).tobytes()
    hdr = json.dumps({"layers.1.engram.embed.weight": {"dtype": "U8", "shape": [len(data)], "data_offsets": [0, len(data)]}}).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    fn = os.path.join(td, "m-00001.safetensors")
    with open(fn, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + data)
        f.flush()
        os.fsync(f.fileno())
    json.dump({"weight_map": {"layers.1.engram.embed.weight": "m-00001.safetensors", "x.weight": "m-00001.safetensors"}},
              open(os.path.join(td, "model.safetensors.index.json"), "w"))
    with open(fn, "rb") as f:
        f.read()
    r = ED.drop(td)
    check(r["bytes"] == len(data) and r["resident_before"] >= len(data) - 4096 and r["resident_after"] <= 2 * 4096,
          f"fadvise drop of a resident range (partial edge pages stay) ({r['resident_before']} -> {r['resident_after']})")
# 7. PCIe logger: runs, samples, stops on SIGTERM; the summary per NVML index
with tempfile.TemporaryDirectory(dir = "/root/rnd/release/recipe") as td:
    pl = H.PcieLog(os.path.join(td, "p.csv"), 100)
    t0 = time.time()
    time.sleep(1.2)
    pl.stop()
    s = pl.summarize(t0, time.time())
    check(set(s) == {"0", "1"} and all(v["samples"] >= 8 for v in s.values()) and pl.proc.returncode == 0,
          f"PCIe log: 2 GPUs sampled, clean exit ({ {k: v['samples'] for k, v in s.items()} }, rc {pl.proc.returncode})")
# 8. prose ids: no 64-token window shared with the canonical ids (make_prose_ids.py's record)
rep = json.load(open("/root/rnd/release/recipe/data/prose_ids.npy.json"))
check(rep["windows_64_in_canonical"] == 0 and rep["tokens"] > 2_000_000, "prose ids disjoint from canonical at 64 tokens")
# 9. a job the generator reaped: its exception is recorded
try:
    raise RuntimeError("expert tier: cache layer 3 of a prefill pass was called from another thread")
except RuntimeError as e:
    err = e
m = B.gen_metrics(32768, 256, 0.0, [], [], {"eos": True, "error": err, "serial": 7}, None)
check(m["job_error"].startswith("RuntimeError: expert tier") and "Traceback" in m["job_error_traceback"] and
      m["error"].startswith("no tokens produced (job error: RuntimeError"), "reaped job's exception recorded")
print(f"all {ok} checks passed")
