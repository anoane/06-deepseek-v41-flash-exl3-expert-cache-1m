# Where the recipe search stopped (checkpoint of 2026-09-27, 08:22 UTC)

The search was paused so the endpoint could serve again. The line that was running (`C12-HR`, 08:13-08:24 UTC)
finished normally; every line after it is recorded as `skipped` ("paused by the user"). Nothing was lost: every
finished run's record, log and batch line is in this repository.

## Done

| Stage | Window (UTC) | Runs | Report |
|---|---|---|---|
| 0a: old production vs its layout on the new tree, decode-gap decomposition, numerics x BF16 crossing, recurrent cache 0, legacy shape, quality of the old layout | 2026-09-26 19:31-20:42 | 12 (11 ok, `D-rc` error) | [docs/stages/STAGE_0.md](docs/stages/STAGE_0.md) |
| 0b: C12 bring-up at 1M, pipeline x tier, stable anchors, the quality floor, split fits (C14 / C10 / C8), diagnostics | 20:42-21:43 | 14 (13 ok, `C14-fit` refused) | same |
| 0c: the head-to-head scenarios, config a at 1M, the tier trace for the offline replay, config a at 512K | 21:43-22:53 | 8 | same |
| 1: C12 / C11 / C10 / C8 vs config a, 512K, quality of the new splits, static slices | 23:06-02:13 | 21 | [docs/stages/STAGE_1.md](docs/stages/STAGE_1.md) |
| 2 (partial): one-factor sweeps | 2026-09-27 02:33-08:24 | 37 (36 ok, `C12-K8` refused) | [docs/stages/STAGE_2_PARTIAL.md](docs/stages/STAGE_2_PARTIAL.md) |

## Not done yet

**Stage 2, 16 lines never started:**

- 9 master lines (plan items 2.7 and the optional 2.10-2.13): `C12-NG`, `C12-HP`, `C12-SP0`, `C12-CW1`, `C12-CW0`,
  `C12-KS2`, `C12-FR0`, `C12-FC`, `C12-PR4K`;
- 7 queued follow-ups: `Q-C12-X`, `Q-C11-X` (64K quality of the BF16 crossing), `C12-H1-S4` (512K safety run of the
  1024 MiB headroom), `C12-R3` (closing C12 reference), `C12-ES-P4-2`, `C12-ES-P4-S4`, `C12-ES-P4G32` (second process,
  512K run and 32K generator chunk of the best C12 arm).

Their exact lines are in [configs/stage2.batch](configs/stage2.batch) and
[configs/stage2-requeue.batch](configs/stage2-requeue.batch); [docs/ATTEMPTS.md](docs/ATTEMPTS.md) lists them as
`skipped`.

**Stage 3** (combinations): 3-4 candidates from the stage-2 winners, two processes each plus a 64K quality line.
Candidates written at the pause: C12 or C11 pipelined with expandable segments, the BF16 crossing (if its quality
check passes), precise numerics (if that trade is accepted) and a larger generator chunk; C10 pipelined 8192 with the
BF16 crossing (serving now); plus the bit-exact extras (heat file, `EXL3_MOE_TIER_PREFILL_ROWS=1024`, `admit=always`).

**Stage 4** (confirmation): the top 3 of stage 3 and config a, three processes each, interleaved; S4-512K and
S4-1M (decode at ~512K and ~1M, the 1M memory and VRAM headroom table); 64K quality per arm; a TabbyAPI dry run of the
winner on a side port.

**Also open:** the recurrent-cache sweep (plan item 2.6) was not in the batch; the n-gram `auto` sizing fix used by
`C12-E2F` is not on the published branch yet; a pipelined load at split 12 under cudaMallocAsync reports "eligible"
and then runs out of CMP memory on the first long prompt (the engine could refuse it at load or fall back to the plain
path).

## How the search resumes

1. Stop the temporary endpoint for the GPU windows (the search runs with the endpoint held down).
2. Run the 16 remaining stage-2 lines. The driver runs every line of the master file it is given, so give it a batch
   that holds only those lines (the 9 `optional=1` lines at the end of `configs/stage2.batch` and the last 7 lines of
   `configs/stage2-requeue.batch`, each with its `#@` directive), started with
   `tools/harness/window.sh <label> <that batch> <outdir>` (see [configs/README.md](configs/README.md)).
3. Write the stage-2 report, then run stages 3 and 4.
4. Regenerate `results/` with `tools/collect_results.py` and `tools/master_table.py`, then
   `python3 tools/make_attempts.py`, and update this repository (README TL;DR, KNOBS measured columns, serving/).

## The endpoint at the pause

Serving the temporary configuration `C10-P8X` ([serving/DEPLOY.md](serving/DEPLOY.md)), reversible by removing one
systemd drop-in. The original unit and config were not changed.
