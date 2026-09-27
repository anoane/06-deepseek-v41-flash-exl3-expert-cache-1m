# Method: how the recipe search measures

**Status: early.** This describes the method of the whole search; only stages 0, 1 and part of stage 2 have run
(see [CHECKPOINT.md](../CHECKPOINT.md)).

## 1. The question and the rules

The search looks for the fastest way to serve DeepSeek-V4.1-Flash EXL3 3.0 bpw at a 1M context on this machine
(RTX PRO 6000 + CMP 170HX, 128 GB of the host's 160 GB), **decode first, then prefill**, at the best quality level.
Two families are compared:

- **Family c** (the recipe): the CMP holds whole layers `0..s-1` with every expert resident; the PRO runs layers
  `s..39` with `experts=cache` (a VRAM expert cache) over a RAM tier that holds every other expert; `disk
  experts=off`, so **no expert is read from the SSD after the load**. Engram tables stay on the SSD.
- **Config a** (the reference): the PRO alone, `*=cuda:0 experts=cache; ram experts=102GiB`. `ram experts=all` would
  need ~131 GB, so the largest tier within 120 GB of pinned + resident memory is used, and ~1,486 experts are read
  from the SSD on demand.

Binding rules: FP16 KV cache only (no KV quantization, ever); a 1,048,576-token cache on every candidate; host RAM
<= 128 GB for the serving process (enforced by a memory cgroup, below); zero expert SSD reads for family c
(`tier_ssd_reads_after_load` = 0 is checked on every run).

## 2. The harness

Files: [`tools/harness/`](../tools/harness/). The measuring program is `dsv41_bench.py` (harness version 3 in windows
0a and 0b, 4 in window 0c and stage 1, 5 in stage 2; each version only adds options). It drives the exllamav3
`Generator` directly (greedy argmax, no stop conditions), **not** TabbyAPI, so every arm sees the same token ids in
the same order. Modes:

| Mode | What it runs |
|---|---|
| `speed` | fixed-length prompts (`gen_P<P>_r<i>`; `--vary` changes one setting per repeat inside one process) and a raw forward (`raw_P<P>`) |
| `legacy` | the shape of an older record (prefill at 8K / 33.6K, decode at 24 and 20K tokens of context) |
| `quality` | 64K-token teacher-forced scoring against stored anchors (NLL, top-1 agreement, bitwise equality) |
| `h2h` | the head-to-head scenarios S0-S5 (below), `h2h_modes.py` |
| `longctx` | S4: one 512K prompt, then short suffixes on the same generator (prefix reuse) |

One process per configuration: load the model at a 1M FP16 cache, drop the engram tables' pages from the page cache
(`engram_drop.py`, `posix_fadvise(DONTNEED)` over their byte ranges, so every arm starts equally cold), run the
scenarios, write one JSON record (`results.jsonl`) and a log.

Around it:

| File | Role |
|---|---|
| `run_batch.sh` | runs batch lines `label | env | args` one process each, with a clean environment (`env -i` plus a base), `taskset` on cores 0-15, power caps 300 / 150 W, temperature (85 C) and MemAvailable guards, a cool-down to 60 C between lines, per-line timeouts |
| `driver.sh` | one GPU window: the master batch plus an append-only requeue file, a time budget, a `minavail` gate before each line (`#@ minavail=<GiB>`), automatic re-queue (once) of a provisional line; a memory kill or cgroup OOM stops the window |
| `window.sh`, `wait_idle.sh`, `gpu_window.sh` | idle gate (cores 16-23 below 25% busy over 60 s), one GPU job at a time on the host, the serving endpoint stopped during a window (it was held down for the whole search) |
| `py_memwatch.sh`, `memwatch.sh` | memory watchdog on the measured process only: if MemAvailable drops below 8 GiB it sends SIGTERM then SIGKILL to that one PID (pinned by its start time, never a pattern), and leaves a `.KILLED` marker |
| `py_mem.sh` | runs the harness inside `systemd-run --scope -p MemoryMax=119G -p MemorySwapMax=0` (111G for the old-production lines): 119 GiB = 127.8 GB, so the kernel enforces the 128 GB rule, and the load's host-memory check sees the cgroup limit |
| `h2h_tables.py`, `stage_tables.py`, `stage1_tables.py`, `stage2_tables.py` | the per-stage tables in `docs/stages/tables/` |
| `tier_trace_replay.py`, `mk_variants.py`, `replay_summary.py` | T1: replays a recorded tier trace through the reference tier policy |
| `decode_model.py`, `decode_model2.py` | the decode-time fit (ms per token vs CMP layers and misses) |
| `dnll_ci.py` | 95% moving-block bootstrap of dNLL, top-1 agreement and confident flips between two quality runs |
| `make_prose_ids.py`, `engram_drop.py`, `pcie_log.py` | the prose token source, the engram page drop, NVML PCIe counters |
| `test_*_cpu.py` | CPU tests of the harness logic |

The swap-0 cgroup runs at its limit through page cache (engram rows, model files), which is reclaimable; the memory
events show `oom_kill` = 0 on every line. **The watchdog never fired and no cgroup OOM happened in stages 0-2.**

### Idle gating

Each timed phase records foreign CPU time on the harness cores 0-15, the load on cores 16-23 and kernel reclaim
(kswapd / kcompactd CPU). A line with foreign load or reclaim inside a timed phase is **provisional**: re-queued once,
never tabulated. The flag was extended after run `A-old-3`, whose decode was cut to 12 tok/s by a page-cache flood
from a stray `find` over the root file system that the first version of the gauges did not flag; that run is excluded.

## 3. Scenarios

Token sources: **canonical ids** (1,048,560 tokens of one Python code base, the same text as the quality anchors) and
**prose ids** (wikitext-2 train, tokenized with the model's tokenizer, 2,424,283 tokens, no 64-token window shared with
the canonical ids; regenerate with `make_prose_ids.py`). Offsets are fixed and disjoint.

| Id | What | Prompt | Generated | Generator |
|---|---|---|---|---|
| S0 | first request after the load (tier and engram pages cold) | canonical @700,000, 2,048 tokens | 512 | fresh |
| S1-32K | novel text, cold | canonical @100,000 and @140,000, 32,768 tokens each | 256 | fresh each |
| S2-32K | the same request again (warm tier; full re-prefill, no prefix reuse) | the S1 ids once more | 256 | fresh |
| S1-128K | novel text at 128K (= S4 at 128K) | canonical @200,000, 131,072 tokens | 256 | fresh |
| S3 | one chat session: T1 = 16,384 tokens @400,000; T2-T6 append the previous input + its generated ids + a code delta of 1,024 / 4,096 / 512 / 8,192 / 2,048 tokens | exact token append, so the prefix is reused (recurrent-state checkpoints and pages) | 256 per turn | shared |
| S5 | topic switch, continuing S3: T7 = a 4,096-token prose delta, T8 = a 2,048-token code delta; then S5b, a fresh 8,192-token prose conversation | prose @0, canonical @460,000, prose @100,000 | 512 / 256 / 256 | shared, then fresh |
| S4-512K | long context: a 524,288-token prompt timed, then 3 decode repeats of 256 on 64-token suffixes | canonical @0 | 256 x 3 | shared |
| S4-1M | continues S4-512K to 1,040,384 tokens | canonical | 256 x 3 | shared (stage 4; **not run yet**) |
| DT | per-device decode time: CUDA events around every forwarded module for 32 decode tokens (H7) | a warm S1 text | 32 | fresh |

Scenario sets: `full` = S0, S1-32K x2, S2-32K x2, S1-128K, S3, S5, DT; `short` = S0, S1-32K, S2-32K, S3, S5, DT.
A timeline records a timestamp and the tier counters every 32 generated tokens (windowed decode and hit rate: the S5
collapse and recovery).

## 4. Metrics

| Metric | Definition |
|---|---|
| prefill tok/s | prompt tokens / time to first token (S3 / S5 turns: new rows / TTFT) |
| decode tok/s | generated tokens after the first / their time |
| TTFT | time to first token, seconds |
| TPOT p50 / p90 | per-token decode latency percentiles |
| hit rate, misses / token | expert-cache decode hits, and misses per decoded token |
| H2D / D2H MB per token | host-to-device (promotions) and device-to-host (demotions) bytes per decoded token |
| SSD reads after load | expert reads from the SSD after the load (must be 0 for family c) |
| RSS, cgroup peak, MemAvailable min | host memory of the serving process |
| VRAM | torch peak per device, and nvidia-smi minimum free memory per GPU over the run |
| temperature, power | nvidia-smi maxima per GPU |

**DECODE_score** = the mean of five decode means: S1-32K cold (both texts), S2-32K warm, S1-128K cold, S3 turns 2-6,
S5 turns 7-8. **PREFILL_score** = the mean of prefill S1-32K, S2-32K and S1-128K. Each is averaged over every job of
every process of a configuration; a configuration missing a part gets no score (it is still listed). The scores rank
configurations; the per-scenario columns say where a configuration wins.

## 5. Quality

- **64K teacher-forced lines** (`--modes quality --quality-tokens 65536`) against stored anchors: the serving anchor
  and the numerics anchor (both `deepseek:index`), the old layout on this tree (W1-c), config a (W1-a), and the recipe's
  own `RQ-4096`, `A1M-Q`, `Q-C11`. Reported: NLL, top-1 agreement, dNLL with a 95% moving-block bootstrap interval.
- **The floor F**: the spread over function-preserving changes (C12 at quality chunks 4096 / 2048 / 2047, C12 vs a):
  top-1 >= 0.9949, |dNLL| <= 0.00081.
- **Bitwise anchors**: under `EXL3_STABLE_ARITHMETIC=1`, C12's 64K log-probabilities are bitwise equal to the stored
  anchors for `deepseek:index` and `precise` (65,535 / 65,535 positions).
- **Greedy-token equality** for bit-exact knobs: the SHA-256 of each job's generated tokens is compared with the
  reference run's.
- Limitation: teacher-forced scoring runs the prefill path; decode-only arithmetic is covered by greedy equality only.

## 6. Decision rule

A knob wins when **both** of its processes beat the reference on its target metric by more than max(2%, the
reference's own spread) and do not lose decode by more than that spread; ties keep the default. Function-changing
knobs are reported as quality trades, never adopted silently. Most stage-2 arms have one process so far, so **no
stage-2 knob has formally won yet**; the second processes were queued for the ones that looked better.

## 7. The staged plan

| Stage | Content | Status |
|---|---|---|
| 0 | baselines (old production vs the same layout on the new tree), harness bring-up, C12 at 1M, config a at 1M, the tier trace and its offline replay (T1), the quality floor, the stable anchors | **done** ([STAGE_0.md](stages/STAGE_0.md)) |
| 1 | the splits C12 / C11 / C10 / C8 head-to-head with config a, static slices (`stream` / `cpu hot=208`), 512K, quality of the new splits | **done** ([STAGE_1.md](stages/STAGE_1.md)) |
| 2 | one-factor sweeps on the stage-1 winners C12 (decode) and C11 (prefill): pipeline / chunking, BF16 crossing, engram RAM, pool and staging, allocator, tier policy, heat file, prefill rows, numerics, config a at larger chunks | **partial**: 37 runs ([STAGE_2_PARTIAL.md](stages/STAGE_2_PARTIAL.md)) |
| 3 | 3-4 combinations of the stage-2 winners, 2 processes each + a 64K quality line | not run |
| 4 | top 3 + config a: 3 processes each, S4-512K and S4-1M, memory / VRAM headroom at 1M, quality, a tabby dry run | not run |

The full plan, with the knob inventory, the ranges and the expected effects written before the measurements, is
[plan/PLAN.md](plan/PLAN.md) (a working document, kept as written; the user decisions at its top override it).

### The offline replay (T1)

The tier never changes routing, so a recorded trace of routing decisions can be replayed through the reference tier
policy (`exllamav3/model/expert_tier_policy.py`) for any pool size and policy: an exact counterfactual for hit rates
and copies. Config a's trace (`results/raw/traces/a1m-short-tier-trace.jsonl.gz`) reproduced a's own measured hit
rates exactly and, cut to layers 12-39, predicted C12's measured hit rates within 0.002. It prices copies, not time:
its speed predictions used stage 0's decode model, which stage 1 corrected.

## 8. Caveats (read before using a number)

1. **Partial.** Stage 2 is mostly one process per arm; stages 3 and 4 have not run; no configuration has been measured
   at a 1M-token prompt (S4-1M) yet; the BF16 crossing's quality lines did not run.
2. **Workload.** One code corpus plus wikitext prose. Hit rates, and so decode speed, depend on the workload; S1
   (novel text), S2 (same text) and S3 (session) are reported separately for that reason.
3. **Harness, not the API.** The harness drives the generator. TabbyAPI adds templating, tokenization and HTTP; the
   serving numbers in the README (35.3 tok/s warm, 46.7 on a repeated prompt, ~8 tok/s on the first request after a
   load) are separate measurements through the API.
4. **Spread.** The tier's decisions are identical from process to process, so the spread between processes is timing
   only (~0.6% decode for C12); it understates the variation across workloads.
5. **Grouping.** The leaderboard in the auto-generated RESULTS.md groups runs that differ only in the allocator; use
   the first table of [results/LEADERBOARD.md](../results/LEADERBOARD.md).
6. **Config a** at `ram experts=102GiB` reads experts from the SSD (~1.9 TB per full test process), so it is a
   reference, not a family-c candidate; it ran in the same 119G cgroup.
7. **Old production** (`OLD`) and its layout on the new tree (`L0`) have one head-to-head process each; their other
   runs are 32K speed runs.
8. **Load-time eligibility.** A pipelined load reports "eligible" even where the first long prefill will run out of CMP
   memory (split 12 under cudaMallocAsync); only a real long prompt proves a sub-chunk size.
9. **Code under test.** `dsv41-all` `be33b0e1` in stage 0 and `5e55c062` from stage 1 on (the difference is a
   Python-only recurrent-cache fix; same native build, sha256 `823174af46fb`). `C12-E2F` ran on a Python-only fix of
   the n-gram auto sizing that is not on the published tip yet.
