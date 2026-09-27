# DeepSeek-V4.1-Flash EXL3 at 1M context: a GPU expert cache on an RTX PRO 6000 + CMP 170HX (early recipe)

> **EARLY / PARTIAL: work in progress.** This is a snapshot of a recipe search that is **paused, not finished**.
>
> - **Ran:** stage 0 (baselines, 34 runs), stage 1 (the split points against the PRO-alone reference, 21 runs) and
>   **37 of the stage-2 runs** (one-factor knob sweeps; 36 finished, 1 refused at load).
> - **Did not run yet:** 16 stage-2 lines (9 planned arms and 7 queued follow-ups, among them the 64K quality check
>   of the BF16 crossing and the second process of the best C12 arm), **stage 3** (combinations) and **stage 4**
>   (confirmation with three processes per arm, 1M-token prompts, quality). See [CHECKPOINT.md](CHECKPOINT.md).
> - **Most stage-2 numbers come from one process.** Expect them to move.
> - **What is serving now is a temporary pick**, not the result of the search.
>
> The repository will be updated with the full configuration runs when the search ends.

A 204 GiB decoder at **1,048,576 tokens** of FP16 context, with every routed expert either in GPU memory or in
pinned DDR5, **no expert ever read from the SSD after the load**, and a mining GPU holding whole layers:

```
RTX PRO 6000 (96 GB, sm_120)   layers 10-39 with their KV, head; GPU expert cache: 5,704 slots (70.5 GiB, 8 spare)
CMP 170HX    (64 GB, sm_80)    layers 0-9 whole with their KV; PCIe Gen2 x1 (~0.4 GB/s): activations only
host DDR5                      RAM tier: the other 5,824 experts of layers 10-39, 72.2 GiB pinned
SSD                            engram tables (~189 GiB), read at runtime; zero expert reads
```

| Serving now (temporary, `C10-P8X`) | |
|---|---|
| Context | **1,048,576** tokens, FP16 KV |
| Decode (through the API) | **35.3 tok/s** warm on a new prompt, **46.7** on a repeated prompt; ~8 on the first request after a load |
| Prefill (harness) | **1,883 tok/s** at 32K, 1,918 at 128K; TTFT 17.4 s for a 32K prompt |
| Load | ~3 min |
| Host memory | RSS 77.6 GiB after the load, 80.9 GiB after 75 min of serving (72.2 GiB of it the pinned RAM tier) |
| Old production on the same machine | 21.5 tok/s decode, 1,088 tok/s prefill |

---

## Hardware, model, software

| | |
|---|---|
| Machine | Proxmox VM, 24 vCPUs of an AMD Ryzen 9 9950X3D, 160 GB DDR5 (the recipe may use **128 GB**; the rest belongs to another VM), **no swap** |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell Workstation, 97,887 MiB, sm_120, PCIe 5.0 x16; power cap 300 W while serving |
| GPU 2 | NVIDIA CMP 170HX unlocked to 64 GB ([repo 01](https://github.com/anoane/01-proxmox-dual-gpu-passthrough)), sm_80, in a PCIe 3.0 **x1** slot, link Gen2 x1, **~0.38-0.4 GB/s**; power cap 150 W |
| Storage | Samsung 9100 PRO NVMe (VirtIO SCSI single + iothread); holds the ~189 GiB engram tables read at runtime |
| Guest | Ubuntu 24.04.4, kernel 6.8.0-139, NVIDIA driver 610.43.02, torch 2.13.0+cu132, CUDA 13.2, Python 3.12 |
| Model | [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw) + DeepSeek's FP8 engram shards 47-48 linked into the folder (see [repo 05](https://github.com/anoane/05-deepseek-v41-flash-exl3-1m) for how) |
| Engine | exllamav3 fork, branch [`dsv41-all`](https://github.com/anoane/exllamav3-anemone/tree/dsv41-all), tip `5e55c062`: upstream `dev` v1.5.2 + 55 commits (DeepSeek-V4.1 support, explicit placement, stable arithmetic, precision options, a disk engine, the expert tier) |
| Server | TabbyAPI `e632af41` + two small patches ([serving/](serving/)) |

## The rules of this recipe

1. **FP16 KV cache only.** No KV quantization, ever (it destroys quality).
2. **1M context is mandatory**: every candidate holds a 1,048,576-token cache.
3. **Host RAM <= 128 GB** for the serving process (enforced in every test by a swap-0 memory cgroup: 119 GiB, 111 GiB
   for the old layout's lines).
4. **Engram tables stay on the SSD.**
5. **Family c** (the recipe): the CMP holds whole layers, the PRO runs the rest with a GPU expert cache over a RAM tier
   that holds every other expert, and **zero expert reads from the SSD** after the load. **Config a** (the PRO alone,
   expert cache over 102 GiB of RAM plus the SSD) is the reference.
6. Priorities: decode first, then prefill, at the best quality; anything that changes the numerics is reported as a
   trade, never adopted silently.

## TL;DR: best so far (1M FP16 cache, same prompts, same 128 GB envelope)

Decode score = mean decode tok/s of five scenarios (cold 32K, the same 32K again, cold 128K, chat turns 2-6, topic
switch); prefill score = mean prefill tok/s of cold 32K, warm 32K and cold 128K ([docs/METHOD.md](docs/METHOD.md)).
Measured with the harness (the generator directly), not through the API.

| Configuration | What it is | Procs | Decode score | Prefill score | Dec. cold 32K | Dec. same 32K | Dec. chat turns | Dec. topic switch | Prefill 32K | Prefill 128K | TTFT 32K | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `C12-ES` | C12 plain, expandable-segments allocator | 2 | **38.6** | 1,131 | 37.8 | 41.6 | 39.7 | 36.0 | 1,117 | 1,133 | 29.3 s | bit-exact vs C12 |
| `C12-ES-P4` | C12 pipelined (4096 / 16384), expandable segments | 1 | **38.3** | 1,547 | 37.7 | 41.1 | 39.4 | 35.7 | 1,536 | 1,547 | 21.3 s | bit-exact vs C12 |
| `C11-X` | C11 pipelined (4096 / 16384) + BF16 crossing | 1 | 36.5 | 1,772 | 35.6 | 38.9 | 37.2 | 35.0 | 1,744 | 1,767 | 18.8 s | lossy: check pending |
| **`C10-P8X`** (serving) | C10 pipelined (8192 / 32768) + BF16 crossing | 1 | 35.3 | **1,912** | 34.7 | 38.1 | 36.5 | 32.9 | 1,883 | 1,918 | 17.4 s | lossy: check pending |
| `C12` | C12 plain (the stage-2 reference) | 7 | 36.5 | 1,126 | 35.9 | 39.1 | 37.4 | 34.4 | 1,113 | 1,126 | 29.4 s | NLL inside the floor |
| `C11` | C11 pipelined (4096 / 16384) | 3 | 35.7 | 1,516 | 34.4 | 37.9 | 36.9 | 33.1 | 1,506 | 1,516 | 21.8 s | 64K NLL 0.250370 |
| `C10` | C10 pipelined (4096 / 16384) | 2 | 35.5 | 1,489 | 34.6 | 38.5 | 36.9 | 31.7 | 1,474 | 1,493 | 22.2 s | 64K NLL 0.250833 |
| config a | PRO alone, cache over 102 GiB RAM + SSD | 6 | 32.7 | 1,267 | 30.6 | 37.6 | 34.3 | 27.8 | 1,209 | 1,270 | 27.1 s | reads ~1.9 TB of experts from the SSD per test process |
| config a, chunk 16K | the same, 16384-row chunks | 1 | 32.8 | 2,167 | 29.3 | 36.7 | 33.2 | 31.9 | 2,026 | 2,154 | 16.2 s | same |
| L0 | the old layout on the new engine (CPU experts 12-22) | 1 | 23.0 | 981 | 22.9 | 23.2 | 23.1 | 23.2 | 966 | 985 | 33.9 s | |
| **old production** | the old tree and layout | 1 | **21.5** | **1,088** | 21.4 | 21.6 | 21.8 | 22.1 | 1,075 | 1,093 | 30.5 s | |

`C<s>` = the CMP holds layers `0..s-1`, the PRO runs `s..39` with the expert cache over a RAM tier of every other expert
(`EXL3_PLACEMENT='0-<s-1>=cuda:0; <s>-39=cuda:1 experts=cache; ram experts=all; disk experts=off'`). "Procs" = the
head-to-head processes behind the scores (C12 includes `C12-HR`, the heat-file restart with identical tokens; config a
includes the trace-capture run `A1M-trace`; L0 and the old production also have 32K speed runs that agree).
The full table, 46 configurations x 46 columns: [results/LEADERBOARD.md](results/LEADERBOARD.md). Every run:
[docs/ATTEMPTS.md](docs/ATTEMPTS.md).

At 512K (one process each): prefill C11 1,422, C10 1,403, config a 1,235, C12 plain 1,057 tok/s; decode at ~512K of
context 37.3-38.3 for family c and 36.8 for config a.

### What the numbers say so far

1. **Family c beats config a in decode in every scenario** (stage 1, C12 against a: +3.5% on a warm repeat, +9-15%
   in chat, +17% on new text, +22-26% after a topic switch). Config a's cache has to cover 40 layers (48.5 misses per token on new text,
   some served from the SSD); family c's covers 28-30 layers (13-17 misses) and never touches the SSD.
2. **The split**: a CMP layer is only ~0.2 ms slower than a PRO layer, but each layer moved to the PRO costs ~0.5 ms of
   extra cache misses, so **12 CMP layers is the decode-best split**; 13 or more whole layers do not fit on the CMP at
   1M. Splits 11 and 10 leave CMP room for the pipelined prefill.
3. **Pipelined prefill** is the prefill lever of family c (+33-37% on long prompts), bounded by the CMP's free memory
   at 1M; it needs several sub-chunks per generator call, which TabbyAPI cannot express without the
   [generator-chunk patch](serving/tabby-generator-chunk.patch) (the unpatched shape lost 36%).
4. **The allocator matters**: expandable segments instead of cudaMallocAsync gave C12 +5.5% decode with identical
   tokens, and freed enough CMP memory for C12's 4096-row pipeline (+37% prefill, identical tokens). Not yet tried
   through TabbyAPI.
5. **The BF16 residual crossing** adds 12-17% prefill for no decode change; it changes the numerics and its 64K check
   has not run. **Precise attention numerics** decode 3-6% faster at a known NLL cost against the default; not adopted.
6. **Tier policy knobs move decode by 2% or less**; `admit=always` shrinks the collapse after a topic switch (23% vs
   37%) and nearly halves its recovery (129 vs 225 tokens); a heat file makes the first request after a load 3.5%
   (decode) and 13% (prefill) faster. **Static hot slices** instead of the cache (`stream` / `cpu hot=208`) lose
   40-50% of decode.
7. **Quality**: no split (8, 10, 11, 12) shows a systematic cost on 64K teacher-forced tokens: NLL 0.24998-0.25083
   against a run-to-run floor of 0.25015-0.25076, top-1 agreement >= 0.9949 everywhere. Every stage-2 knob that only
   moves bytes gave identical greedy tokens.

Details: [docs/stages/STAGE_0.md](docs/stages/STAGE_0.md), [STAGE_1.md](docs/stages/STAGE_1.md),
[STAGE_2_PARTIAL.md](docs/stages/STAGE_2_PARTIAL.md).

## What is serving now, and how to deploy or revert it

`C10-P8X`: CMP layers 0-9, PRO layers 10-39 with the expert cache, pipelined prefill with 8192-row sub-chunks and a
32768-token generator chunk, the BF16 crossing, FP16 KV, 1M. It was the best family-c prefill with a good decode at
the pause. **Temporary**: its BF16 crossing has not had its quality check, and the stage-2 arms `C12-ES` /
`C12-ES-P4` decode faster (not yet tried through TabbyAPI).

- Deploy: copy [serving/ds41-exl3-c10.yml](serving/ds41-exl3-c10.yml) and the drop-in
  [serving/zz-temp-c10.conf](serving/zz-temp-c10.conf), apply the two tabby patches, `systemctl daemon-reload &&
  systemctl restart ds41-exl3.service`. Exact steps, the expected load log, memory and the warm-up note:
  [serving/DEPLOY.md](serving/DEPLOY.md). Every setting explained: [serving/ENV.md](serving/ENV.md).
- Revert: delete the drop-in, `systemctl daemon-reload && systemctl restart ds41-exl3.service` (the original unit and
  config were never changed).
- Send one short request after every restart: the first request after a load decoded at ~8 tok/s.

## How to read the results

- **Start with** [results/LEADERBOARD.md](results/LEADERBOARD.md) (first table: one row per configuration) and the
  per-stage reports. [docs/ATTEMPTS.md](docs/ATTEMPTS.md) has one row per run with its exact knobs, key numbers and
  status; the run label is the key to everything else.
- **Scenarios** (S0 first request, S1 cold 32K / 128K, S2 the same prompt again, S3 a six-turn chat with prefix reuse,
  S5 a topic switch, S4 512K): decode differs a lot between them because the expert cache's hit rate does. Compare a
  configuration with its reference scenario by scenario, not only by score.
- **Processes**: a score from one process is a pointer, not a verdict; the tier makes identical decisions in every
  process, so the spread between processes (~0.6% decode) is timing only.
- **Quality class** of every knob (bit-exact, last-bit numerics, lossy, forbidden): [docs/KNOBS.md](docs/KNOBS.md).
- The raw per-run records, logs and batch lines are there to re-derive any number: [results/](results/),
  [configs/](configs/), [tools/](tools/).

## Documents

| Document | What |
|---|---|
| [CHECKPOINT.md](CHECKPOINT.md) | where the search stopped, what remains, how it resumes |
| [docs/KNOBS.md](docs/KNOBS.md) | every knob for this machine: what it does, values, defaults, expected and measured effect, recommendation, quality class |
| [docs/METHOD.md](docs/METHOD.md) | the harness, scenarios, metrics, idle gating, memory watchdog, quality gates, the staged plan, caveats |
| [docs/ATTEMPTS.md](docs/ATTEMPTS.md) | every attempt, one row per run (generated) |
| [docs/stages/](docs/stages/) | the stage reports (0, 1, 2 partial) and their generated tables |
| [docs/plan/PLAN.md](docs/plan/PLAN.md) | the plan as written before the measurements: the knob inventory with ranges and expected effects |
| [serving/](serving/) | the temporary config, drop-in, base unit, tabby patches, env summary, deploy / revert steps |
| [results/](results/) | leaderboard, master CSV, the generated report, raw per-run records, logs, the tier trace, a heat file |
| [configs/](configs/) | every batch file of the search: the exact configuration of every attempt |
| [tools/](tools/) | the results generators and the measuring harness |

## Licence and credits

MIT for this repository's own files ([LICENSE](LICENSE)). The two files in `serving/*.patch` modify
[TabbyAPI](https://github.com/theroyallab/tabbyAPI) and are under its licence. The engine is a fork of
[exllamav3](https://github.com/turboderp-org/exllamav3); the quantized model is Mia-AiLab's; the model and its engram
tables are DeepSeek's. The CMP 170HX unlock comes from [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker).

Sanitized: the host name is `ai-vm-01`, GPU UUIDs are placeholders, no domain, address, key or token appears.
Hardware, byte counts and measured numbers are real.
