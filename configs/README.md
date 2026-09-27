# Batch files: the exact configuration of every attempt

Every run of the recipe search was one line of one of these files. They are copied as they ran (paths are the test
machine's: `/root/rnd/release/...` for the harness and source trees, `/root/workspace/...` for the old production tree
and the model). [docs/ATTEMPTS.md](../docs/ATTEMPTS.md) joins every line with its measurements.

## Format (`tools/harness/run_batch.sh`, `tools/harness/driver.sh`)

```
# comment
#@ est=<minutes> timeout=<seconds> minavail=<GiB> [optional=1]     # applies to the next line (driver)
<label> | <environment assignments> | <harness arguments>
```

- The environment field is shell words; every line runs with a clean environment (`env -i` plus a small base) and
  these assignments. `BENCH_PYTHON` / `MEMWATCH_PY` put the process under the memory watchdog and its cgroup
  (`MEMMAX`, swap 0); `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` is set on every line except the ones that test
  expandable segments (`*-ES*`) and the `C-legacy-*` lines.
- The arguments go to `tools/harness/dsv41_bench.py` (`--modes`, `--scenarios`, `--chunk` = load chunk, `--gen-chunk` =
  generator chunk, `--cache-tokens 1048576`, `--expect-native-sha` = the extension build the line must run on, ...).
- `minavail` = the MemAvailable the driver waits for before starting the line (expected RSS + 16-20 GiB).
- A window runs one master file; lines appended to its requeue file while it runs are drained before the first
  `optional=1` line and at the end.

## Files

| File | Window | Lines | What |
|---|---|---|---|
| [stage0a.batch](stage0a.batch) | 0a | 12 | old production (`A-old-*`) vs its layout on the new tree (`A-new-*`) x3, numerics x BF16 crossing (`B-np`), atomic MoE (`B-det`), recurrent cache 0 (`D-rc`), the older record's shape (`C-legacy-*`), 64K quality of the old layout (`E-L0`) |
| [stage0b.batch](stage0b.batch) | 0b | 11 | C12 bring-up at 1M with and without the pipeline (`T-fit`, `T-pipe-128k`), stable anchors (`T12-stable-*`), the quality floor (`RQ-*`), split fits (`C14-fit`, `C10-fit`, `C8-fit`), the stable profile's speed |
| [stage0b-original.batch](stage0b-original.batch) | - | - | the first version of 0b before the fits were added (not run as such) |
| [stage0b-requeue.batch](stage0b-requeue.batch) | 0b | 3 | diagnostics appended during 0b (`DIAG-C12-pipe`, `DIAG-L0-det0`, `DIAG-C12-pipe2048`) |
| [stage0b-diag.batch](stage0b-diag.batch), [stage0b-diag2.batch](stage0b-diag2.batch) | - | - | the same diagnostics as written before appending |
| [stage0c.batch](stage0c.batch), [mk0c.sh](mk0c.sh) | 0c | 8 | the head-to-head scenarios for config a, C12, L0, OLD; config a's 64K quality, tier trace and 512K run; `mk0c.sh` wrote the batch |
| [stage1.batch](stage1.batch) | 1 | 19 | C12 / C11 / C10 / C8 vs config a mirrored, 512K per split, 64K quality of C11 / C10 / C8, static slices |
| [stage1-requeue.batch](stage1-requeue.batch) | 1 | 2 | the extra a / C12 pair (`A1M-5`, `C12-4`) |
| [stage2.batch](stage2.batch), [mk_stage2.py](mk_stage2.py) | 2 | 30 | round 1 (21 arms, one process each) and 9 optional arms (not run) |
| [stage2-requeue.batch](stage2-requeue.batch) | 2 | 23 | `C12-E2F`, then round 2 (second processes and follow-ups); the last 7 did not run |
| [stage2-round2.batch](stage2-round2.batch), [mk_round2.py](mk_round2.py), [mk_round2b.py](mk_round2b.py) | - | - | round 2 as generated, before it was appended to the requeue file |

## Label conventions

`A1M*` = config a at a 1M cache; `C<s>*` = family c with the CMP holding layers `0..s-1`; `-P<n>` = pipeline
sub-chunk n (P25 = 2560, P3 = 3072, P4 = 4096, P8 = 8192); `-X` = BF16 crossing; `-ES` = expandable segments;
`-K<n>` = load chunk n x 1024; `-G32` = generator chunk 32768; `-T` = the unpatched tabby shape; `-AA` = `admit=always`;
`-LFU` = `evict=lfu`; `-SS` = single staging; `-H<n>` = headroom n x 1024 MiB; `-E2` / `-E3` = engram RAM; `-HR` = heat-file
restart; `-PR<n>K` = prefill rows; `-NP` = precise numerics; `-S4` = 512K long context; `Q-*` = 64K quality;
`-R<n>` = a reference process; `-S` / `-C` = static `stream` / `cpu` slices; `OLD` / `A-old` = the old production;
`L0` / `A-new` = its layout on the new tree.
