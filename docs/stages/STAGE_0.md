# Recipe stage 0: baselines, harness additions, config a at 1M (PLAN.md v2, section 4)

> Published copy of the stage report as written during the search. Paths in it are the test machine's recipe folder; in
> this repository the per-run records are in `results/raw/`, the logs in `results/logs/`, the batch files in `configs/`,
> the generated tables in `docs/stages/tables/`, and the harness in `tools/harness/`.

Status: DONE (2026-09-26, 19:31-22:53 UTC). Nothing was deployed; `ds41-exl3.service` and its config were not
touched (the endpoint is held down by `/root/rnd/gpu-windows/HOLD_ENDPOINT_DOWN`, the user's request of ~22:05 CEST).
Code under test: `dsv41-all` = `be33b0e1` (tree = validated `b8105f2d`), 54 commits on upstream dev `988fa662`
(`git ls-remote` of upstream dev = `988fa662` = the branch base; remote `dsv41-all` = `be33b0e1` = local). Source
`/root/rnd/release/all/src-b8105f2d6a05`, native sha256 `823174af46fb` (sm_80 + sm_120a); OLD = `wt-dsv41-crnd` +
JIT `4fa4d0749921`. No code defect of `dsv41-all` was found, so the branch was not changed and nothing was pushed.

Records: every result line of the three windows in `stage0/stage0_runs.jsonl` (34 lines), one flat row per h2h job
in `stage0/stage0_h2h_jobs.jsonl`, generated tables `stage0/tables_0a.md`, `tables_0b.md`, `tables_0c.md`
(`tables_0c_a_vs_c12.md`: a vs C12 only), T1 replay `stage0/t1/replay.json`, `replay2.json`, `pred_a.md`, `pred_c.md`,
trace `stage0/w0c/traces/a1m-short.jsonl.gz`. Per-line logs, memwatch logs, nvidia-smi CSVs in `stage0/w0{a,b,c}/`.

## 1. The answer so far: does c beat a?

1M FP16 cache on both, same prompts, same 119G swap-0 cgroup, engram pages dropped at each process start. c =
**C12 plain** (CMP layers 0-11 whole, PRO 12-39 `experts=cache` over a RAM tier of every other expert, `disk
experts=off`, chunk 4096, **no pipeline**: see 3.2). a = PRO alone, `*=cuda:0 experts=cache; ram experts=102GiB`
(+1,486 experts SSD-only). a: 2 processes, c: 1 process (stage 1 adds processes; a's two processes made identical
tier decisions, so a's spread is timing only: <= 1.3 tok/s decode, <= 3% prefill).

| Scenario | Decode tok/s a / c | Decode winner | Prefill tok/s a / c | Prefill winner |
|---|---|---|---|---|
| S0 first request (2K) | 33.3 / 35.0 | c +5% | 410 / 561 | c +37% |
| S1 cold 32K (2 texts) | 30.7 / 35.9 | **c +17%** | 1,206 / 1,104 | **a +9%** |
| S1 cold 128K (= S4 at 128K) | 33.0 / 35.9 | c +9% | 1,260 / 1,116 | a +13% |
| S2 warm, same prompt again | 37.8 / 38.6 | tie (+2%, inside a's spread 37.2-38.5) | 1,311 / 1,134 | **a +16%** |
| S3 turn 1 (16K) | 31.7 / 36.2 | c +14% | 1,148 / 1,067 | a +8% |
| S3 turns 2-6 (deltas 0.5K-8K) | 34.3 / 37.4 | **c +9%** | 544 / 678 new rows/s (TTFT 5.86 / 4.74 s) | **c +25%** |
| S5 prose turn in the session | 26.7 / 32.6 | c +22% | 575 / 743 | c +29% |
| S5 back to code | 28.6 / 35.9 | c +25% | 674 / 816 | c +21% |
| S5b fresh prose conversation | 27.6 / 30.1 | c +9% | 952 / 969 | tie |
| S4 512K (a only; c in stage 1) | a 36.8 at ~512K (3 repeats), 512K prefill 1,235 tok/s (424 s) | - | - | - |

Plain answer: **config c (C12) beats config a in decode in every scenario but S2, where they tie** (a 37.8 vs c
38.6); the margin is largest on new or switched text (S1 +17%, S5 +22-25%) and in multi-turn sessions (S3 +9%).
**Config a wins prefill on long cold prompts** (S1 +9%, S2 +16%, 128K +13%); c wins the short-delta prefill that
multi-turn chat pays (S3 turns +25%, S5 +21-29%, TTFT 4.7 vs 5.9 s) and the first request after load. After a
domain switch c's decode is back within 5% of its pre-switch speed after 225 tokens; a's was not within 512. Today's
production layouts are far behind both: L0 (dsv41-all, CPU experts 12-22) 22.8-23.3 decode, 876-991 prefill; OLD
(the serving tree) 20.9-22.3 decode, 965-1,099 prefill.

The pre-registered expectation (a wins warm decode by 20-50%) was wrong: the CMP's layer time is ~1 ms as expected,
but a's pool must cover 40 layers (37% of 15,360 experts in VRAM, plus SSD reads) while c's covers 28 (54% of 10,752),
so a misses 3-4x as often (S1: 48.5 vs 13.0 misses per token) and each miss costs ~0.3 ms.

Caveats: c has 1 process per scenario so far; c ran without the pipeline (it OOMs at split 12, 3.2), which is also
why a still wins long prefill: the pipelined splits 10 and 8 prefilled 16K at 1,443 and 1,370 tok/s in their fits
(3.2) against a's 1,206-1,311 at 32K. c at 512K / 1M and splits 10 / 8 in all scenarios are stage 1.

## 2. What ran

| Window | Label | Time (UTC) | GPU min | Lines | Harness |
|---|---|---|---|---|---|
| 0a (reused, not re-run) | recipe-stage0a | 19:31-20:42 | 70 | 12 (OLD vs L0 x3, B-np, B-det, D-rc, C-legacy x2, E-L0) | bench/ (v3) |
| 0b | recipe-stage0b | 20:42-21:43 | 61 | 11 master + 3 appended diagnostics | bench/ (v3); diagnostics via bench2 |
| 0c | recipe-stage0c | 21:43-22:53 | 69 | 8 | bench2/ (v4) |

New GPU time 2.2 h in 2 windows (plan: 3.4 h, 2 windows). Every line ran under `py_memwatch.sh` (8 GiB floor, kills
only its own PID) in a `systemd-run --scope` swap-0 cgroup (119G for a / family c, 111G for L0 / OLD); **the
watchdog never fired, no cgroup OOM**, no line was provisional (foreign load) or auto-requeued, thermal guard on (max 70 C PRO / 60 C CMP),
caps 300 / 150 W. Host memory per process (max RSS / min MemAvailable over the run, GiB): a 107.9 (115.9 GB <= 120 GB)
/ 30.7, C12 67.2 / 72.9, C10 76.8 / 62.6, C8 86.7 / 52.6, L0 58.9 / 75.3, OLD 58.8 / 76.1. The cgroups run at
their limit (119.0 GiB peak) because page cache (engram rows, the model files) fills them; that part is
reclaimable (`memory.events max` counts, `oom_kill` 0).

## 3. Results

### 3.1 Head-to-head detail (window 0c; mean [min-max] over jobs)

| Scenario | Arm | Decode tok/s | Prefill tok/s (S3/S5: new rows / TTFT) | TTFT s | Tier hit | Misses / token | H2D MB / token | D2H MB / token |
|---|---|---|---|---|---|---|---|---|
| S0 | a | 33.27 [33.20-33.34] | 410 | 5.00 | 0.894 | 25.3 | 336 | 127 |
| S0 | C12 | 35.05 | 561 | 3.65 | 0.935 | 10.9 | 145 | 77 |
| S1-32K | a | 30.73 [29.11-32.43] n4 | 1,206 [1,186-1,226] | 27.18 | 0.798 | 48.5 | 644 | 46 |
| S1-32K | C12 | 35.88 [35.31-36.44] n2 | 1,104 [1,101-1,106] | 29.69 | 0.923 | 13.0 | 173 | 40 |
| S1-32K | L0 | 22.93 | 966 | 33.92 | - | - | - | - |
| S1-32K | OLD | 21.36 | 1,075 | 30.49 | - | - | - | - |
| S2-32K | a | 37.80 [37.22-38.54] n4 | 1,311 [1,301-1,323] | 24.99 | 0.880 | 28.9 | 384 | 20 |
| S2-32K | C12 | 38.56 [38.39-38.74] n2 | 1,134 | 28.90 | 0.962 | 6.4 | 85 | 5 |
| S2-32K | L0 | 23.21 | 991 | 33.06 | - | - | - | - |
| S2-32K | OLD | 21.63 | 1,097 | 29.88 | - | - | - | - |
| S1-128K | a | 33.03 [32.66-33.39] | 1,260 | 104.0 | 0.833 | 40.2 | 533 | 27 |
| S1-128K | C12 | 35.85 | 1,116 | 117.5 | 0.914 | 14.4 | 191 | 12 |
| S1-128K | L0 / OLD | 22.78 / 20.87 | 985 / 1,093 | 133.1 / 119.9 | - | - | - | - |
| S3 t1 | a / C12 | 31.69 / 36.16 | 1,148 / 1,067 | 14.27 / 15.36 | 0.804 / 0.913 | 47.0 / 14.6 | 624 / 194 | 30 / 8 |
| S3 t2-t6 | a | 34.30 [32.44-35.97] n10 | 544 [251-865] | 5.86 [3.94-10.22] | 0.838 | 38.8 | 515 | 26 |
| S3 t2-t6 | C12 | 37.39 [36.74-38.22] n5 | 678 [399-923] | 4.74 [2.56-9.43] | 0.928 | 12.2 | 161 | 8 |
| S3 t2-t6 | L0 / OLD | 23.12 / 21.78 | 590 / 626 | 5.37 / 5.04 | - | - | - | - |
| S5 prose turn | a / C12 / L0 / OLD | 26.74 / 32.55 / 23.12 / 22.23 | 575 / 743 / 687 / 744 | 8.01 / 6.20 / 6.71 / 6.19 | 0.744 / 0.861 | 61.3 / 23.3 | 814 / 309 | 40 / 15 |
| S5 back to code | a / C12 / L0 / OLD | 28.63 / 35.90 / 23.19 / 21.95 | 674 / 816 / 684 / 726 | 4.18 / 3.45 / 4.12 / 3.88 | 0.748 / 0.907 | 60.6 / 15.6 | 804 / 207 | 39 / 11 |
| S5b fresh prose | a / C12 / L0 / OLD | 27.60 / 30.06 / 23.09 / 22.26 | 952 / 969 / 876 / 965 | 8.61 / 8.45 / 9.35 / 8.49 | 0.765 / 0.808 | 56.4 / 32.3 | 749 / 429 | 37 / 21 |
| S4 512K prefill | a | 27.35 (the first 256 tokens) | 1,235 | 424.5 | 0.829 | 41.1 | 546 | 220 |
| S4 decode at ~512K | a | 36.76 [36.64-36.93] n3 | (64-token suffix, prefix reused) | 0.53 | 0.912 | 21.1 | 279 | 15 |

SSD: C12 read **zero** expert bytes after the load in every line (`tier_ssd_reads_after_load` = 0); a read 1.93 TB
per H2H-full process (layer-mode staging of the 1,486 SSD-only experts, ~170 GB per 32K prefill) and 2.6 TB for
the 512K prefill; a's decode read 0.6-20 MB per token from the SSD. Prefix reuse worked in every session turn
(cached = the previous turn's input + its tokens, to the page), recurrent stashes ~15 MB each (15 after 8 turns,
0.23 GiB of the 4 GiB cache).

S5 collapse / recovery, per 32-token window against the last S3 turn: a's worst window 15.1-15.2 tok/s (-57%), hit
rate back within 1 pp after 449 tokens, decode not back within 5% inside the 512-token prose turn; C12's worst window
23.4 (-37%), decode back after 225 tokens, hit after 257; returning to code, C12 dipped 9% and was back after 65
tokens, a 34-36% and not back within 256. L0 / OLD do not collapse (no expert cache): within 4%.

### 3.2 Window 0b: C12 bring-up at 1M, pipeline x tier, anchors, quality floor

- **C12 loads at 1M** in 193 s: pool 5,818 slots (71.9 GiB), staging 768 slots (9.5 GiB), RAM tier 4,942 slots
  (61.3 GiB), 53.0 GiB of the 119G cgroup left unpinned, RSS 67.1 GiB, CMP 640-800 MiB free (nvidia-smi), PRO ~4 GB
  free, zero SSD reads after the load. Plain 32K (T-fit): prefill 1,137 / 1,140 tok/s, decode 38.4 / 38.4 (hit
  0.943 / 0.955); 128K plain: prefill 1,131, decode 20.1 over only 16 tokens right after the prefill (p50 28.7 ms).
- **Pipeline x tier at split 12: CUDA OOM on the CMP.** Every pipelined call longer than 2 sub-chunks failed within
  3.6 s (T-fit 32K x2, T-pipe-128k, DIAG-C12-pipe): `torch.OutOfMemoryError` on device 0 (the CMP) in stage 1's
  engram forward. The job was reaped by the generator (the generator stays usable), and the v3 harness recorded only
  "no tokens produced"; the bench2 harness now records the reaped job's exception. Cause: capacity, not a logic
  error: stage 1 of sub-chunk k+1 runs while sub-chunk k's products still live on the CMP, and the CMP has < 1 GB
  free with layers 0-11 and a 1M cache; the load's autosplit does not budget the pipeline's second sub-chunk. With
  2048-row sub-chunks it runs (DIAG-C12-pipe2048: 1,095 tok/s at 16K), no faster than plain. **C10 / C8 have 10.4 /
  21.4 GB free on the CMP and pipeline fine**: 16K pipelined prefill 1,443 / 1,370 tok/s (one job each, right after
  the load). So stage 0c ran C12 **plain** (the plan's C12 line specified the pipeline; written reason in the batch).
- **Split fits at 1M** (load + one pipelined 16K job): C14 refused at load, cleanly, before any RAM was pinned
  ("layers.12 does not fit on cuda:0", cgroup 10.4 GiB): 13 whole layers do not fit either. C10: pool 5,798, RAM
  5,730 slots (71.1 GiB), 43.2 GiB unpinned, RSS 76.8 GiB. C8: pool 5,769, RAM 6,527 (81.0 GiB), 33.3 GiB unpinned,
  RSS 86.7 GiB. The pool is ~5,800 slots at every split (the PRO's free VRAM barely changes); RAM grows 10 GiB per
  2 layers.
- **Stable anchors on C12 (tier + CMP layout): bitwise** 65,535 / 65,535 for `deepseek:index` (NLL 0.249963) and
  `precise` (0.250939) against `n64k2_deepseek-index.npz` / `n64k_place_precise.npz`.
- **Quality floor F** (C12 serving numerics, 64K teacher-forced, chunks 4096 / 2048 / 2047, NLL 0.250762 / 0.250370
  / 0.250152): against each other, W1-a, W1-c and the serving anchors top-1 >= 0.9949, |dNLL| <= 0.00081. **Config a
  at a 1M cache is bitwise equal to W1-a** (A1M-Q, 65,535 / 65,535, NLL 0.250298). C12 vs a: dNLL +0.00047, top-1
  0.9953: inside F. L0 on this tree is bitwise equal to W1-c (0a E-L0).
- T12-stable-speed (optional): the stable profile on C12 prefills 915 tok/s and decodes 9.4 tok/s (no native
  graphs): an anchor profile only, as expected.

### 3.3 Window 0a (reused): OLD vs L0 at 32K, the decode-gap decomposition

| Arm | Decode tok/s | Prefill tok/s | Note |
|---|---|---|---|
| OLD (serving tree, 1M) | 21.58, 21.56 (n2) | 1,112 [1,105-1,116] | A-old-3 (12.02) excluded: see 5 |
| L0 (dsv41-all, same layout) | 23.15 [23.03-23.24] (n3) | 1,000 [999-1,001] | |
| B-np: L0 numerics x BF16 crossing (1 process, 2 repeats per cell) | deepseek:index 23.24 / 23.15, precise 23.83 / 23.88 (crossing FP32 / BF16) | FP32 995-998, BF16 1,101-1,104 | BF16 crossing +10.6% prefill, 0 decode; precise +2.7% decode |
| C-legacy (the 23.4 record's shape) OLD / new | 22.88 / 25.12 (24-token ctx), 22.65 / 23.71 (20K ctx) | 8K: 811 / 736, 33.6K: 958 / 978 | |
| E-L0 64K quality | NLL 0.250437, bitwise 65,535 / 65,535 vs W1-c | | |

The "decode gap" is gone: on this tree L0 decodes faster than the serving tree (23.2 vs 21.6), and OLD's prefill
lead (1,112 vs 1,000) is its BF16 residual crossing (on by default in OLD; `EXL3_DSV41_XDEV_BF16=1` gives L0 1,101).
B-det (`EXL3_MOE_FUSED_DET=0`) and D-rc (recurrent cache 0 vs 4096) produced no numbers: see 5.

### 3.4 Where decode time goes (H7 and a fit over the h2h jobs)

Least squares of decode ms/token against tier misses per token over the h2h jobs (S0 and 128K left out): a: 16.97 ms
+ 0.320 ms per miss (26 jobs, R^2 0.96); C12: 23.78 ms + 0.279 ms per miss (13 jobs, R^2 0.90). So **t_pro = 0.424
ms** per PRO layer-token at a hit, **t_cmp = 0.99 ms** per CMP layer-token (crossing included), **t_miss = 0.28-0.32
ms** (RAM; a's includes SSD-backed misses). H7 (CUDA events around every module, 32 decode steps, warm 32K text) agrees:
C12's CMP blocks 0.958 ms mean (p50 0.81), its PRO blocks 0.639 ms (with 11 misses/token), 0.57 ms residual
(crossing + host); a's blocks 1.006 ms mean with 78.8 misses/token in that step window (hooked 40.9 vs unhooked 40.5
ms/token: the events cost ~1%); L0's CMP blocks 1.14 ms, its PRO blocks p50 0.47 / p90 2.06 ms (the CPU layers).

### 3.5 T1: the offline replay of a's trace (`stage0/t1`)

The trace of A1M-trace (H2H-short, 147,492 tier records) replayed through the reference policy reproduces a's own
measured decode hit rates **exactly** for S0 / S1 / S2 (0.894 / 0.780 / 0.877) and within 0.02 for the session
turns, and a's trace cut to layers 12-39 predicts C12's measured S1 / S2 hit rates (0.932 / 0.984 predicted vs 0.932 /
0.982 measured at @100000). Predicted decode (model 3.4; tok/s S1 / S2 / S3 turns 2-6; misses per token S1):

| Variant | S1 / S2 / S3-follow tok/s | S1 misses / token | S1 D2H experts / token |
|---|---|---|---|
| a (measured 30.7 / 37.8 / 34.3) | 29.5 / 37.8 / 33.9 | 52.9 | 4.6 |
| c12 (measured 35.9 / 38.6 / 37.4) | 37.1 / 40.7 / 36.5 | 11.5 | 5.3 |
| c10 | 37.5 / 41.6 / 36.9 | 14.5 | 5.4 |
| c8 | 37.7 / 42.3 / 37.1 | 17.9 | 5.5 |
| c2 (pool of c8) | 34.6 / 41.6 / 37.0 | 38.6 | 5.2 |
| c12 / c10 / c8 `admit=always` | 38.6 / 42.1 / 38.8; 39.6 / 44.2 / 39.7; 40.7 / 46.5 / 40.5 | 7.7 / 9.4 / 11.0 | 7.7 / 9.4 / 11.0 |
| c12 `evict=lfu` | 37.5 / 39.7 / 38.7 (S5 prose 17.3 misses vs 27.5) | 10.4 | 5.5 |
| c12 +384 slots (single staging) | 37.7 / 41.2 / 37.4 | 9.8 | 5.2 |
| a `admit=always` | 40.9 / 43.7 / 40.8 | 23.4 | 23.4 |

Reading: the split barely moves decode (12 -> 8 is +2-4% predicted, inside one process's spread), so the split
choice is about prefill (pipeline eligibility) and RAM (10 GiB per 2 layers). `halflife`, `EXL3_MOE_HEAT_PREFILL`,
`spare` and `lazy-exclusive` change nothing on this text (identical counters). **`admit=always` is the largest lever
the replay sees, for a as well as c**, but the replay does not price its extra device-to-host demotions (one per
admission: c12 ~100 MB/token, a ~310 MB/token), so it is a stage-2 GPU measurement, and a must get it too for a fair
final comparison.

## 4. Adaptations of the plan (written reasons)

1. Harness copy `recipe/bench2` = `recipe/bench` + `patch_h2h.py` (literal edits, each asserted once): windows 0a / 0b
   ran from `bench/`, whose files must not change under a running window. Harness version 4. New files:
   `h2h_modes.py` (modes `h2h`, `longctx`: H1-H3, H5, H7), `engram_drop.py` (H4), `pcie_log.py` (H8), `make_prose_ids.py`
   (H6: 2,424,283 wikitext-2 tokens, no 64-token window shared with the canonical ids), `tier_trace_replay.py` +
   `mk_variants.py` + `replay_summary.py` (T1), `h2h_tables.py` (M1), `decode_model.py`; tests `test_h2h_cpu.py` (31
   checks), `test_replay_cpu.py`, and the existing `test_harness_cpu.py` (24) all pass.
2. The scenario CLI is one list (`--scenarios full|short|S0,S1:<off>:<P>,...`, `--session-*`) instead of the plan's
   `--prompt-offset` / `--modes session` flags; same prompts and offsets as section 3.1. `--prewarm` is not used in
   h2h (S1 is cold, engram pages included).
3. H7 uses CUDA events around every forwarded module (per-layer device time, read after the job) instead of
   torch.profiler / CUPTI: no profiler stays attached to the measured process; the fetch kernel's own share is not
   split out, t_miss comes from the fit over jobs (3.4). H8 (NVML PCIe) ran on L0 / OLD only (arms without tier
   counters).
4. 0b ran the existing `stage0b.batch` with the old harness (no H7 in T-fit; C12's H7 is 0c's DT). The split fits are
   a pipelined 16K speed job (not load-only), so they also tested the pipeline per split.
5. Three diagnostic lines were appended to 0b's requeue file (DIAG-C12-pipe, DIAG-L0-det0, DIAG-C12-pipe2048,
   ~12 min) to capture the swallowed exceptions (5); they ran the bench2 harness through `bench2/py_mem_b2.sh`.
6. 0c's C12 line ran without the pipeline (3.2). The plan's C12 = pipeline 4096 / 16384 is infeasible at 1M.
7. The provisional check gained a kernel-reclaim flag (kswapd / kcompactd CPU > 5 s in a timed phase), after
   A-old-3 (5).

## 5. Incidents, defects and open items

- **A-old-3 (0a) is contaminated and excluded**: a `find /` over the 5.8 TB root filesystem of the VM was started by
  mistake at ~19:58:40 (killed by PID at ~20:00:45); the page-cache / dentry flood made `kcompactd0` run 17 s inside
  A-old-3's decode phase (12.02 tok/s instead of 21.6). The foreign-load gauges did not flag it (hence item 4.7). Not
  re-run (the running window's lines were not to be re-run); OLD has 2 clean 0a samples plus OLD-base in 0c.
- For ~15 s around 20:07:35 `recipe/bench/dsv41_bench.py` was overwritten with the bench2 version by a mistaken shell
  redirection and then restored; every 0a / 0b line recorded the original harness (sha256 `64886a42b6b3`), so no
  measurement was affected.
- **Pipelined prefill at split 12 with a 1M cache OOMs on the CMP** (3.2). Not a logic defect (capacity; the load
  reports "eligible" and the first long prefill fails, the job is reaped cleanly). Options for the user: accept it
  (use splits 10 / 8 for the pipeline), or a follow-up change that budgets one more sub-chunk on the first GPU when
  `EXL3_DSV41_PIPELINE=1` (it would refuse split 12 at 1M at load instead of failing at runtime) or falls back to the
  plain path on OOM. Not changed on `dsv41-all`.
- **B-det / DIAG-L0-det0**: `EXL3_MOE_FUSED_DET=0` on L0 at 1M fails every 32K prefill with CUDA OOM on the CMP (the
  atomic path's extra scratch; the 8K warm-up passes). A diagnostic, non-default knob; capacity, not logic.
- **D-rc**: a Generator with `recurrent_cache_size = 0` crashes at its first checkpoint (`KeyError: 'dictionary is
  empty'` in `RecurrentCache.put`, which pops from an empty cache while one stash exceeds max_size). This is upstream
  exllamav3 code, untouched by `dsv41-all`, so it was not changed here; tabby's `sysmem_recurrent_cache: 0` would hit
  it. The plan's "0 = the no-reuse bound" cannot be measured; stage 2.6 uses 1024 / 4096.
- The tier trace file is not flushed when the harness ends with `os._exit` (the trace's last line was truncated; the
  replay tolerates only a truncated last line). Harness-side; for future captures the harness can unload the tiers
  before exiting.
- **Config a at 1M fits**: pool 5,660 slots (vs ~5,900 at a 64K cache), 1,486 experts SSD-only, RSS 107.9 GiB, 0.23
  GiB of recurrent stashes after 8 turns; min MemAvailable 30.7 GiB.

## 6. Consequences for stage 1 (to be confirmed there, not decided here)

- C10 and C8 with the pipeline (4096 / 16384) are the prefill candidates (fits: 1,443 / 1,370 tok/s at 16K) at a
  predicted decode equal to C12's (3.5); C12 stays plain. C2 is predicted slower in S1 (fewer PRO pool hits per
  layer) and needs ~116 GiB RSS: not worth its window unless 1A shows decode rising 12 -> 8.
- The c-vs-a decode verdict needs more c processes (stage 1A: 2 processes each, mirrored) and c at 512K / 1M (1C).
- Stage 2.5 should test `admit=always` (and `evict=lfu`) on both c and a, with their D2H demotion cost measured.
