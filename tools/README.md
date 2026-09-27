# Tools

Copied as they ran on the test machine (a few comments and one heading reworded for publication, in
`collect_results.py`, `harness/window.sh` and `harness/test_harness_cpu.py`). Their default paths
(`/root/rnd/release/...`, the harness's `data/` folder, the stored quality anchors) are that machine's; pass your own
on the command line or edit the constants at the top.
Nothing here needs network access.

## Results tables

| File | What it does |
|---|---|
| [collect_results.py](collect_results.py) | reads every `results.jsonl` record and run log of the search (and of three earlier campaigns) and writes `RESULTS.md`: best-so-far, leaderboard, master table, head-to-head, every run in full. Idempotent; it ran every 5 minutes during the search. `python3 collect_results.py <out.md>` |
| [master_table.py](master_table.py) | the master table (one row per run x scenario, every configuration knob and metric; `RESULTS.csv`) and the leaderboard; imported by `collect_results.py` |
| [make_attempts.py](make_attempts.py) | this repository's `docs/ATTEMPTS.md` and `results/LEADERBOARD.md`, from `results/RESULTS.csv`, `results/RESULTS.md.gz` and `configs/*.batch`. Run from the repository root: `python3 tools/make_attempts.py` |

## The measuring harness ([harness/](harness/))

The version that ran stage 2 (harness version 5; the earlier windows ran versions 3 and 4, which differ only by the
options added since). See [docs/METHOD.md](../docs/METHOD.md) for what it measures.

| File | Role |
|---|---|
| `dsv41_bench.py` | the measuring program: loads the model at the requested cache size and placement, runs one mode (`speed`, `legacy`, `quality`, `h2h`, `longctx`), writes one JSON record and a log |
| `h2h_modes.py` | the head-to-head scenarios S0-S5, sessions with exact token append, the timeline, long context, per-device decode timing |
| `engram_drop.py` | drops the engram tables' pages from the page cache before each run |
| `make_prose_ids.py` | builds the prose token source (wikitext-2 train with the model's tokenizer) and checks it shares no 64-token window with the code source |
| `pcie_log.py` | NVML PCIe throughput per GPU |
| `run_batch.sh` | runs a batch file, one process per line, with guards (power caps, temperature, MemAvailable, timeouts, CPU affinity) |
| `driver.sh` | one GPU window: master batch + append-only requeue file, time budget, `minavail` gate, automatic re-queue of provisional lines, stop on a memory kill |
| `window.sh`, `wait_idle.sh`, `wait_results.sh` | idle gate and detached launch of a window |
| `gpu_window.sh` | serializes GPU work on the host and stops the serving unit while a window runs |
| `memwatch.sh`, `py_memwatch.sh` | the host-memory watchdog on the measured process only (8 GiB MemAvailable floor, SIGTERM then SIGKILL to that one PID) |
| `py_mem.sh`, `py_mem_b2.sh`, `py_mem_serve.sh` | run the harness's Python inside a `systemd-run --scope` memory cgroup (`MEMMAX`, swap 0), with the new tree's or the old serving tree's Python |
| `h2h_tables.py`, `make_tables.py`, `stage_tables.py`, `stage1_tables.py`, `stage2_tables.py`, `tier_table.py` | the per-stage tables (`docs/stages/tables/`) |
| `tier_trace_replay.py`, `mk_variants.py`, `replay_summary.py` | T1: replay of an `EXL3_MOE_TIER_TRACE` recording through the reference tier policy for other pool sizes and policies |
| `decode_model.py`, `decode_model2.py` | least-squares decode-time model (ms per token vs CMP layers and tier misses) |
| `dnll_ci.py` | 95% moving-block bootstrap of dNLL and top-1 agreement between two 64K quality runs |
| `quick.py` | quick look at a results file |
| `test_harness_cpu.py`, `test_h2h_cpu.py`, `test_replay_cpu.py` | CPU tests of the harness logic |

The harness needs the exllamav3 fork (`dsv41-all`) on `PYTHONPATH`, the model folder, and the token sources
(`canonical_ids.npy`: 1,048,560 tokens of one Python code base, not included; `prose_ids.npy`: rebuild with
`make_prose_ids.py`). The 64K quality anchors it compares against are not included (tens of MB each).

Each record stores the environment variables whose names start with `EXL3_`, `EXLLAMA`, `CUDA_`, `OMP_`, `MKL_`,
`PYTORCH_`, `TORCH_`, `HF_` or `TOKENIZERS_`, plus `PYTHONPATH`. That prefix list includes `HF_TOKEN`: unset it (or
any other secret under those prefixes) before a run whose results you will share. The runs published here had only
`HF_HUB_OFFLINE=1` from that group.
