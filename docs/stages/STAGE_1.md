# Recipe stage 1: the family-c splits head-to-head with config a at a 1M cache (PLAN.md v2, section 5)

> Published copy of the stage report as written during the search. Paths in it are the test machine's recipe folder; in
> this repository the per-run records are in `results/raw/`, the logs in `results/logs/`, the batch files in `configs/`,
> the generated tables in `docs/stages/tables/`, and the harness in `tools/harness/`.

Status: DONE (2026-09-26 23:06 to 2026-09-27 02:13 UTC, one GPU window, 186 min). Nothing was deployed.
`ds41-exl3.service` and its config were not touched; the endpoint stays down by
`/root/rnd/gpu-windows/HOLD_ENDPOINT_DOWN` (the user's request). No code defect of `dsv41-all` turned up, so the
branch was not changed and nothing was pushed.

Code under test: `dsv41-all` tip `5e55c062` (54 commits of the plan + the recurrent-cache fix, on upstream dev `988fa662`
v1.5.2); remote `dsv41-all` = `5e55c062` = local. Source `/root/rnd/release/all/src-5e55c0627ece` (checked file by file
against `git archive 5e55c062`: identical), native `native-5e55c0627ece` -> `native-b8105f2d6a05` (sha256 `823174af46fb`,
sm_80 + sm_120a; the tip's diff to the stage-0 tree `be33b0e1` is Python only: `exllamav3/cache/recurrent.py` + its test,
active only when one recurrent checkpoint is larger than the whole recurrent cache, which no stage-1 line does).
Harness `recipe/bench2` (version 4), batch `bench2/batches/stage1.batch`, output `stage1/w1/`.

Records: every result line of the window in `stage1/stage1_runs.jsonl` (21 lines); one flat row per job, pooled with
window 0c's A1M / C12 / L0 / OLD lines, in `stage1/stage1_h2h_jobs.jsonl` (299 rows); generated tables in
`stage1/tables_1.md` (`bench2/stage1_tables.py`); the decode model in `stage1/decode_model2.txt`
(`bench2/decode_model2.py`); quality bootstrap intervals in `stage1/quality_ci.txt` (`bench2/dnll_ci.py`). Per-line logs,
memwatch logs and nvidia-smi CSVs in `stage1/w1/`.

## 1. The answer: does c beat a?

1M FP16 cache on both, the same prompts, the same 119G swap-0 cgroup, engram pages dropped at each process start, each
arm's processes interleaved in mirrored order (a c12 c11 c10 c8 | c8 c10 c11 c12 a, then one more a / c12 pair).
a = the PRO alone, `*=cuda:0 experts=cache; ram experts=102GiB` (+1,486 experts on the SSD only), 5 processes (2 from
window 0c, 3 here). c = the CMP holds whole layers 0..s-1, the PRO runs `experts=cache` for s..39 over a RAM tier of every
other expert, `disk experts=off`: **C12** plain (chunk 4096; its pipeline OOMs at 1M, stage 0), 4 processes (1 from 0c);
**C11 / C10 / C8** with the pipeline (4096-row sub-chunks, 16384-token generator chunk), 2 processes each. Values are
means over every job (tok/s); in brackets the change against a.

| Scenario | a decode | C12 decode | C11 decode | a prefill | C12 prefill | C11 prefill | Decode winner | Prefill winner |
|---|---|---|---|---|---|---|---|---|
| S0 first request (2K) | 32.7 | 35.2 (+8%) | 34.9 (+7%) | 411 | 563 (+37%) | 556 (+35%) | **c** (C12 +8%) | **c** (C12 +37%) |
| S1 cold 32K (2 texts) | 30.7 | 35.8 (+17%) | 34.1 (+11%) | 1,208 | 1,109 (-8%) | 1,504 (+25%) | **c** (C12 +17%) | **c with the pipeline** (C11 +25%); C12 plain loses 8% |
| S2 same prompt again | 37.6 | 38.9 (+3.5%) | 37.8 (+1%) | 1,320 | 1,137 (-14%) | 1,525 (+16%) | **c by a hair** (C12 +3.5%; process ranges touch: a 37.0-38.4, C12 38.3-39.7) | **C11** +16%; C12 plain loses 14% |
| S1 cold 128K (= S4 at 128K) | 33.0 | 35.9 (+9%) | 36.4 (+10%) | 1,270 | 1,123 (-12%) | 1,517 (+19%) | **c** (C11 +10%) | **C11** +19%; C12 plain loses 12% |
| S3 turn 1 (16K) | 31.7 | 36.5 (+15%) | 36.3 (+14%) | 1,156 | 1,086 (-6%) | 1,450 (+25%) | **c** (C12 +15%) | **C11** +25% |
| S3 turns 2-6 (0.5K-8K deltas) | 34.3 | 37.4 (+9%) | 37.0 (+8%) | 543 | 680 (+25%) | 709 (+31%) | **c** (C12 +9%) | **c** (C11 +31%; TTFT 4.45 s vs a 5.87 s) |
| S5 prose turn in the session | 26.8 | 32.7 (+22%) | 30.2 (+13%) | 576 | 738 (+28%) | 758 (+32%) | **c** (C8 35.3, +32%) | **c** (C11 +32%) |
| S5 back to code | 28.7 | 36.2 (+26%) | 36.0 (+25%) | 677 | 815 (+20%) | 810 (+20%) | **c** (C12 +26%) | **c** (C12 +20%) |
| S5b fresh prose conversation | 27.6 | 30.3 (+10%) | 30.9 (+12%) | 953 | 983 (+3%) | 1,187 (+25%) | **c** (C11 +12%) | **C11** +25% |
| S4 512K prefill (1 process each) | - | - | - | 1,235 (424 s) | 1,057 (-14%, 496 s) | 1,422 (+15%, 369 s) | - | **C11** +15%; C12 plain loses 14% |
| S4 decode at ~512K (3 repeats) | 36.8 | 38.3 (+4%) | 37.9 (+3%) | - | - | - | **c, marginal** (C12 +4%, one process each) | - |

**Plain answer.** Family c beats config a in decode in every scenario; the margin is large on new or switched text
(S1 +17%, S5 +22-32%) and in multi-turn chat (S3 +9-15%), and small where a's cache is warm (S2 +3.5%, at the edge of
the run-to-run spread; decode at ~512K +4%, one process each). In prefill, **c with the pipeline (C11) beats a in every
scenario** (+15-25% on long prompts, +31% on chat deltas); **C12 without the pipeline loses long-prompt prefill** to a
(-6 to -14%) and wins only the short ones (S0, S3/S5 deltas). No single c split wins everything: C12 has the best
decode (1-5% over C11 in S1-S3, 8% on the prose turn; C11 is ahead at 128K and in S5b), C11 the best prefill (+33-36% over
C12 on long prompts, +4% on chat deltas). Config a at 1M itself: decode 26.8-37.6 tok/s depending on the scenario, prefill 1,156-1,320 tok/s on 16K-128K
prompts, 1,235 at 512K, 1.9 TB of SSD expert reads per H2H-full process.

Everything else measured in this window, per split (mean decode / prefill over S1-S3, see 3.1): C10 is between C11 and
C8; C8 is the slowest c in decode except on the prose turn. Static hot slices on the PRO instead of the cache (C12-S
`stream hot=208`, C12-C `cpu hot=208`) decode at half the cache's speed (17-24 tok/s): the dynamic cache is what makes
family c competitive.

## 2. What ran

One window `recipe-stage1` (gpu_window, idle gate passed at 0.7% on cores 16-23), 21 lines, all `ok`, none provisional
or re-queued by the driver, 186 min of the 240 budgeted (plan: 4.3 h in 3 windows).

| # | Lines | What |
|---|---|---|
| 1A | A1M-3, C12-2, C11-1, C10-1, C8-1, C8-2, C10-2, C11-2, C12-3, A1M-4 | H2H-full (S0, S1-32K x2, S2-32K x2, S1-128K, S3, S5, H7), mirrored |
| 1C | C12-S4, C10-S4, C8-S4, C11-S4 | S4-512K (canonical @0, 524,288-token prompt timed, 3 decode repeats by prefix reuse); a = 0c's A1M-S4 |
| 1B | Q-C10, Q-C11, Q-C8 | 64K teacher-forced quality, chunk 4096 |
| 1B | C12-S-1, C12-C-1 | H2H-short, C12 layout with a static slice instead of the cache |
| requeue | A1M-5, C12-4 | the extra a / best-c pair of 1C (S2 decode was within noise after 1A) |

Memory and safety: every line under `py_memwatch.sh` (8 GiB floor, kills only its own PID) in a swap-0 cgroup of 119G;
**the watchdog never fired, no cgroup OOM kill, no kernel OOM line**; thermal guard on, caps 300 / 150 W (restored to
400 / 200 W after the batch), GPU peaks 70 C (PRO) / 60 C (CMP). The driver waited for MemAvailable >= expected RSS +
~20 GiB before each line.

## 3. Results

### 3.1 Every arm (mean over jobs; hit = tier decode hit rate; misses, H2D and D2H per decode token)

| Scenario | Arm | Decode tok/s | Prefill tok/s | TTFT s | Hit | Misses / tok | H2D MB / tok | D2H MB / tok |
|---|---|---|---|---|---|---|---|---|
| S1-32K | a | 30.73 (n10) | 1,208 | 27.14 | 0.798 | 48.5 | 644 | 46 |
| S1-32K | C12 | 35.83 (n8) | 1,109 | 29.55 | 0.923 | 13.0 | 173 | 40 |
| S1-32K | C11 | 34.05 (n4) | 1,504 | 21.78 | 0.907 | 16.2 | 215 | 47 |
| S1-32K | C10 | 34.56 (n4) | 1,474 | 22.23 | 0.905 | 17.0 | 226 | 45 |
| S1-32K | C8 | 34.20 (n4) | 1,420 | 23.07 | 0.892 | 20.8 | 276 | 45 |
| S1-32K | C12-S / C12-C | 17.54 / 21.39 (n1) | 978 / 977 | 33.5 | - | - | ~1 GB/token (NVML, stream) | - |
| S2-32K | a | 37.56 (n10) | 1,320 | 24.83 | 0.880 | 28.9 | 384 | 20 |
| S2-32K | C12 | 38.88 (n8) | 1,137 | 28.83 | 0.962 | 6.4 | 85 | 5 |
| S2-32K | C11 | 37.81 (n4) | 1,525 | 21.48 | 0.956 | 7.7 | 102 | 6 |
| S2-32K | C10 / C8 | 38.50 / 37.70 (n4) | 1,498 / 1,446 | 21.9 / 22.7 | 0.950 / 0.941 | 9.0 / 11.4 | 120 / 151 | 6 / 8 |
| S1-128K | a / C12 / C11 / C10 / C8 | 33.01 / 35.87 / 36.40 / 35.94 / 35.26 | 1,270 / 1,123 / 1,517 / 1,493 / 1,445 | 103.2 / 116.7 / 86.4 / 87.8 / 90.7 | 0.833 / 0.914 / 0.921 / 0.911 / 0.902 | 40.2 / 14.4 / 13.7 / 16.0 / 18.8 | 533 / 191 / 182 / 212 / 250 | 27 / 12 / 10 / 10 / 13 |
| S3 t1 | a / C12 / C11 / C10 / C8 | 31.70 / 36.46 / 36.25 / 34.00 / 35.05 | 1,156 / 1,086 / 1,450 / 1,417 / 1,370 | 14.2 / 15.1 / 11.3 / 11.6 / 12.0 | 0.804 / 0.913 / 0.906 / 0.875 / 0.886 | 47.0 / 14.6 / 16.4 / 22.4 / 21.8 | 624 / 194 / 218 / 298 / 290 | 30 / 8 / 12 / 14 / 16 |
| S3 t2-t6 | a / C12 / C11 / C10 / C8 | 34.31 / 37.42 / 37.03 / 36.94 / 35.78 | 543 / 680 / 709 / 695 / 670 (new rows / TTFT) | 5.87 / 4.72 / 4.45 / 4.54 / 4.71 | 0.838 / 0.928 / 0.917 / 0.916 / 0.899 | 38.8 / 12.2 / 14.5 / 15.1 / 19.3 | 515 / 161 / 192 / 201 / 256 | 26 / 8 / 10 / 9 / 13 |
| S5 prose | a / C12 / C11 / C10 / C8 | 26.77 / 32.73 / 30.23 / 29.52 / 35.26 | 576 / 738 / 758 / 747 / 710 | 8.0 / 6.2 / 6.1 / 6.2 / 6.5 | 0.744 / 0.861 / 0.813 / 0.806 / 0.901 | 61.3 / 23.3 / 32.5 / 34.9 / 19.1 | 814 / 309 / 431 / 464 / 253 | 40 / 15 / 20 / 23 / 12 |
| S5 back to code | a / C12 / C11 / C10 / C8 | 28.71 / 36.16 / 35.95 / 33.84 / 35.06 | 677 / 815 / 810 / 783 / 789 | 4.2 / 3.5 / 3.5 / 3.6 / 3.6 | 0.748 / 0.907 / 0.900 / 0.875 / 0.891 | 60.6 / 15.6 / 17.4 / 22.4 / 21.0 | 804 / 207 / 232 / 298 / 279 | 39 / 11 / 12 / 15 / 14 |
| S5b | a / C12 / C11 / C10 / C8 | 27.64 / 30.32 / 30.93 / 29.37 / 27.51 | 953 / 983 / 1,187 / 1,161 / 1,114 | 8.6 / 8.3 / 6.9 / 7.1 / 7.4 | 0.765 / 0.808 / 0.827 / 0.800 / 0.743 | 56.4 / 32.3 / 30.1 / 36.1 / 49.4 | 749 / 429 / 399 / 479 / 656 | 37 / 21 / 20 / 22 / 35 |
| S4 512K prefill | a / C12 / C11 / C10 / C8 | (first 256 tokens) 27.4 / 30.9 / 30.9 / 30.4 / 30.3 | 1,235 / 1,057 / 1,422 / 1,403 / 1,360 | 424 / 496 / 369 / 374 / 385 | 0.829 / 0.896 / 0.895 / 0.890 / 0.877 | 41.1 / 17.5 / 18.3 / 19.8 / 23.7 | 546 / 232 / 243 / 262 / 314 | 220 / 126 / 127 / 133 / 147 |
| S4 decode ~512K | a / C12 / C11 / C10 / C8 | 36.76 / 38.31 / 37.90 / 37.26 / 37.68 (3 repeats) | - | 0.53 / 0.41 / 0.37 / 0.37 / 0.39 | 0.912 / 0.972 / 0.965 / 0.961 / 0.955 | 21.1 / 4.8 / 6.2 / 7.0 / 8.7 | 279 / 63 / 82 / 92 / 115 | 15 / 25 / 25 / 26 / 23 |

Processes are very repeatable (the tier's decisions are identical from process to process; the spread is timing only):
a's per-process decode means in S1-32K 30.6-30.8, C12's 35.6-36.2; a's S2 37.0-38.4, C12's 38.3-39.7 (the one scenario
where the ranges touch). Per-process values: `tables_1.md`, "Per process".

SSD: every c line read **zero** expert bytes after the load (`tier_ssd_reads_after_load` = 0); a read 144,635-144,782
experts (~1.9 TB) per H2H-full process. S0 (first request after the load, cold tier) is included above.

S5 collapse and recovery (per 32-token window, against the last S3 turn): a's worst window ~15.1 tok/s (-57%), decode not
back within 5% inside the 512-token prose turn (hit back after 449 tokens); C12's worst 23.2-23.4 (-36 to -38%), back
after 225 tokens; C8 recovers fastest (97 tokens); returning to code, C12 dips 7-11% and is back after 33-65 tokens, a
dips 34-37% and is not back within 256.

### 3.2 Host memory and VRAM at 1M (per process; nvidia-smi peaks over the whole line)

| Arm | Tier: VRAM pool / RAM slots (GiB) | Max RSS GiB | Min MemAvailable GiB | PRO max used / min free MiB | CMP max used / min free MiB |
|---|---|---|---|---|---|
| a | 5,660 / 8,222 (102.0) + 1,486 SSD-only | 107.96 (115.9 GB) | 29.9-30.4 | 94,394 / 3,493 | - |
| C12 | 5,818 / 4,942 (61.3) | 67.19 | 71.3-72.9 | 93,860 / 4,027 | 64,736 / 800 |
| C11 | 5,803 / 5,341 (66.3) | 72.09 | 66.7-68.0 | 94,062 / 3,825 | 60,490 / 5,046 |
| C10 | 5,798 / 5,730 (71.1) | 76.91 | 61.7-62.3 | 93,976 / 3,911 | 55,412 / 10,124 |
| C8 | 5,769 / 6,527 (81.0) | 86.79 | 51.6-51.9 | 94,092 / 3,795 | 44,584 / 20,952 |
| C12-S / C12-C | (static 208 per layer; 61 GiB shared arena) | 67.95 / 67.98 | 70.7-70.8 | 85,376 / 12,511 | 64,704 / 832 |

The cgroups ran at their 119G limit through page cache (engram rows, model files), which is reclaimable
(`events_oom_kill` 0 everywhere). a's pinned + resident stays below 120 GB; every c is far below it (C12 leaves ~52 GiB
of RAM unpinned inside the 128 GB envelope for an engram row cache, C11 ~48, C10 ~43, C8 ~33). C11's pipeline leaves 5.0 GB free on the CMP at 1M
(user decision 3: recorded).

### 3.3 Where decode time goes (corrects stage 0)

A fit over 130 c jobs from four splits (`decode_model2.txt`): ms/token = 21.47 + **0.195 ms per CMP layer** + 0.274 ms
per tier miss (R^2 0.91); a: 17.20 + 0.315 ms per miss (R^2 0.96), so t_pro = 0.430 ms per layer-token. Hence **the CMP
is only ~0.2 ms per layer slower than the PRO (t_cmp ~0.63 ms), and splitting across the two GPUs costs a fixed ~4.3 ms
per token** (the x1 crossing and the second device's launches). Stage 0 had one split and put that fixed cost into
t_cmp (0.99 ms). H7 agrees: CMP blocks p50 0.81-0.89 ms, PRO blocks p50 0.60-0.63 ms. Consequence: every layer moved from
the CMP to the PRO saves ~0.2 ms but adds ~1.6-2 misses per token (the PRO pool stays ~5,800 slots) = ~0.5 ms, so
**C12 is the decode-best feasible split**, and the T1 replay's prediction (C8 2-4% faster than C12) had the direction
wrong because it used stage 0's t_cmp; its hit-rate predictions were right (misses/token S1 predicted 11.5 / 14.5 / 17.9
for C12 / C10 / C8, measured 13.0 / 17.0 / 20.8). c beats a in decode because a's pool covers 40 layers (misses 29-61 per
token) against c's 28-32 (5-35).

### 3.4 Static slices instead of the cache (C12 layout, 1 process each, H2H-short)

| Arm | S0 | S1-32K | S2-32K | S3 t1 / t2-t6 | S5 prose / back / fresh | Prefill S1 / S2 |
|---|---|---|---|---|---|---|
| C12 (cache, reference) | 35.2 | 35.8 | 38.9 | 36.5 / 37.4 | 32.7 / 36.2 / 30.3 | 1,109 / 1,137 |
| C12-S `stream hot=208` | 17.1 | 17.5 | 18.9 | 19.5 / 18.2 | 15.5 / 17.2 / 16.6 | 978 / 1,001 |
| C12-C `cpu hot=208` | 20.2 | 21.4 | 23.5 | 23.7 / 24.4 | 21.4 / 23.4 / 21.2 | 977 / 1,006 |

Both lose ~40-50% of decode and ~10-12% of prefill against the cache with the same resident expert count (5,824 vs
5,818 slots): the stream arm re-copies every non-resident expert it touches (~1 GB of H2D per decode token on the PRO,
NVML), the CPU arm computes them on the CPU worker. Not candidates.

### 3.5 Quality (64K teacher-forced, chunk 4096; `quality_ci.txt`: 95% moving-block bootstrap)

| Line | NLL | top-1 agreement (min over references) | dNLL vs C12 (RQ-4096) | vs a (A1M-Q) | vs serving anchor | vs n64k2 anchor |
|---|---|---|---|---|---|---|
| Q-C11 | 0.250370 | 0.9949 | -0.00039 | +0.00007 | -0.00059 | +0.00041 |
| Q-C10 | 0.250833 | 0.9949 | +0.00007 | +0.00054 | -0.00013 | +0.00087 |
| Q-C8 | 0.249976 | 0.9950 | -0.00079 | -0.00032 | -0.00098 | +0.00001 |
| (floor: C12 at chunks 4096 / 2048 / 2047, a) | 0.250152-0.250762 | >= 0.9949 | | | | up to +0.00080 |

All three splits keep top-1 >= 0.9949 against every reference (the floor's own minimum), and their NLLs (0.24998-0.25083)
extend the range of the floor lines (0.25015-0.25076) by less than 0.0002 on either side. Two single comparisons sit just
outside the stage-0 floor F (|dNLL| <= 0.00081): C10 vs n64k2 +0.00087, C8 vs the serving anchor -0.00098 (lower NLL).
The floor itself has comparisons whose bootstrap interval excludes 0 (RQ-4096 vs RQ-2047 +0.00061, vs n64k2 +0.00080),
so these are the chaotic floor, not a split effect with a sign: no split has a systematic quality cost (C11, the prefill
candidate, is the closest to a: +0.00007). Splits 11 / 10 / 8 change where layers 8-11 compute (sm_80 -> sm_120), so no
bitwise gate applies (plan 3.4).

## 4. Adaptations of the plan (written reasons)

1. **One window instead of three** (1A, 1B, 1C in one batch, BUDGET_MIN 240): the endpoint is held down by the user's hold
   file, so splitting only adds idle gates and restarts; the driver's append-only requeue file carried the conditional
   lines decided after 1A. Used 186 min.
2. **Code under test `5e55c062`**, not `be33b0e1`: the branch moved after the plan was written (rebased onto upstream
   dev `988fa662` and the recurrent-cache fix added and pushed). Its only difference from the stage-0 tree is Python
   code that runs only when a checkpoint exceeds the recurrent cache; native build unchanged (same sha256), source
   verified against `git archive`.
3. **C12 runs plain**, and the plan's separate "C12-plain" arm is C12 itself: C12's pipelined prefill OOMs on the CMP at
   1M (stage 0, 3.2). C11 / C10 / C8 run the plan's template (pipeline 4096 / generator chunk 16384).
4. **C11 added** in 1A, 1C and the 1B quality lines (user decision 3).
5. **Static-slice arms at `hot=208`**, not a probed fit maximum: the experts are 12.70 MiB each (safetensors headers,
   every layer), not the 11.25 MiB the plan's "~220" assumed, so the fit maximum is ~230 and each refused probe costs a
   ~3.5 min CMP load; 28 x 208 = 5,824 resident experts = C12's 5,818-slot pool, the same resident count for a policy
   comparison. They ran without the pipeline (C12 layout).
6. **C2 not run** (neither the fit probe nor a process): its condition (decode rising 12 -> 10 -> 8) failed (S1-32K
   35.8 / 34.6 / 34.2, S3 turns 37.4 / 36.9 / 35.8, and 3.3 explains why), and its ~116 GiB of pinned RAM sits at the
   edge of the host rule for no expected gain.
7. **Extra a / C12 pair appended** (A1M-5, C12-4) because S2 decode was within noise after 1A (plan 1C). C12 was chosen as
   best-c for decode. With it, S2 is +3.5% for C12 with the process ranges touching (a 37.0-38.4, C12 38.3-39.7).
8. Quality lines ran at chunk 4096 = the pipeline sub-chunk (one sub-chunk per call, plan 3.4), with the pipeline armed as
   in serving, against six references (the four stage-0 anchors, C12's RQ-4096, A1M-Q). S4 at 1M is stage 4 (plan).

## 5. Incidents, defects, open items

- None in this window: no memory kill, no OOM, no provisional line, no job error, no load refusal.
- **Upstream dev moved** after the branch was based: `git ls-remote` now gives `5783a936` ("Loader: Reuse file streams
  across threads (Linux only)", 15 lines in `stloader.cpp`) on top of `988fa662`. `dsv41-all` does not touch that file
  (a test merge is conflict-free), but it is native code: the next push of `dsv41-all` must rebase onto it, rebuild, and
  re-run the CPU gates and the GPU anchor check (binding rule). Nothing was pushed in this stage, so it was not done here.
- Open from stage 0, unchanged: C12's pipeline at 1M fails at run time although the load reports it eligible (a follow-up
  could refuse it at load or fall back to the plain path); `EXL3_MOE_FUSED_DET=0` OOMs the CMP at 1M; the trace file's
  last line is cut when the harness exits.

## 6. Consequences for stage 2 (to be confirmed there, not decided here)

- **cB = C12** by the plan's rule (best mean decode over S1 / S2 / S3; C12 leads C11 by 5% in S1, 3% in S2, 1% in S3).
  **C11 is the prefill-balanced runner-up** (+33-36% long-prompt prefill over C12 for a 1-5% decode cost in S1-S3; with
  C10 / C8 the only splits that win prefill against a in every scenario): it should be carried as K4 and gets the 2.1 sub-chunk sweep (user decision 3).
  Which of the two becomes the recipe is a decode-vs-prefill choice for the user (decode first, per the priorities,
  gives C12; C12's long-prompt prefill stays 6-14% behind a unless its pipeline fits).
- 2.1 on C12 can only use 2048-row sub-chunks (no faster in stage 0); a chunk-size change on C12 must re-check the CMP's
  0.8 GB at 1M. On C11 / C10: sub-chunk 8192 (C10 has 10 GB free on the CMP; C11 5 GB).
- 2.3 (engram RAM): every c has 32-52 GiB of the envelope left for `ram ngram=`.
- 2.5: `admit=always` (and `evict=lfu`) on both C12 and a, as stage 0 said; the S5 prose turn is where C12 loses most to
  C8 (hit 0.861 vs 0.901), which is where a policy change could show.
- The static slices (3.4) and C2 are closed.
