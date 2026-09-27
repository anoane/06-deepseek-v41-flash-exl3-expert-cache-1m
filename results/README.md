# Results (partial: stages 0, 1 and 37 stage-2 runs)

| File | What | Size |
|---|---|---|
| [LEADERBOARD.md](LEADERBOARD.md) | one row per configuration with the scores and per-scenario means; first the allocator-aware table (use this one), then the best-so-far, leaderboard and head-to-head sections of RESULTS.md as generated | ~70 KB |
| [RESULTS.csv](RESULTS.csv) | the master table: one row per run x scenario (1,093 rows x 806 columns): every configuration knob, the placement's parts, the tier's sizing and counters, every metric, memory, VRAM, quality, temperatures and power. It also holds three earlier campaigns (the `campaign` column): validation of the engine branch, the expert-tier development runs, earlier benchmark windows | 3.0 MB |
| [RESULTS.md.gz](RESULTS.md.gz) | the auto-generated report the search refreshed every 5 minutes (27 MB uncompressed): every run in full, with its configuration, every measurement line and its complete JSON record | 3.1 MB |
| [raw/](raw/) | the raw per-run JSON records of the recipe search, one gzipped file per GPU window, plus the flat per-job tables of stages 0 and 1 and the offline-replay outputs | 2.0 MB |
| [raw/traces/a1m-short-tier-trace.jsonl.gz](raw/traces/a1m-short-tier-trace.jsonl.gz) | config a's expert-tier trace (147,492 records) that the offline replay (T1) used | 8.5 MB |
| [heat/c12-r1.exl3moe](heat/c12-r1.exl3moe) | the heat file run `C12-R1` wrote and run `C12-HR` loaded (`EXL3_MOE_HEAT_FILE`, safetensors-based `.exl3moe` format, layers 12-39) | 87 KB |
| [logs/](logs/) | per-run logs, memory-watchdog logs, and the window / driver / batch logs of every window | 1.7 MB |

Not included: the per-run nvidia-smi and PCIe CSVs (100-450 KB each, about 23 MB in total), the 64K quality `.npz`
arrays (0.8 MB each) and the stored anchors, the token sources, the dry runs.

## Record format (raw/*.jsonl.gz)

One JSON object per run: `label`, `run_id`, `status` (`ok`, `error`, `skipped`), `error`, `env` (the environment the
run saw), `args` (the harness arguments), `host` (kernel, Python, torch, CUDA), `build` (source tree and native build
hashes), `load` (time, VRAM, host memory, cgroup), `tier_sizing` / `tier_load` / `tier_end` (the expert tier),
`h2h` (per scenario: prompt, cached tokens, TTFT, prefill, decode, TPOT, the tier and engram counters of the prefill
and the decode, the timeline, per-device times), `quality` (NLL, per-anchor comparisons), `peaks` (VRAM, RSS, cgroup,
temperatures, power), `tier_ssd_reads_after_load`, `provisional` and the idle-gate flags.

Sanitized for publication: the host name is `ai-vm-01` and the GPU UUIDs are placeholders; nothing else was changed.
