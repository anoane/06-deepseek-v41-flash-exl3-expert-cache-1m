# Every knob, for this machine and this model

**Status: early.** Knobs marked "not measured" or "did not run" are in the plan but had not run when the search was paused
(stage 2 is 37 runs in; stages 3 and 4 have not run). This page will be updated with the full search.

Scope: DeepSeek-V4.1-Flash EXL3 3.0 bpw at a 1,048,576-token FP16 cache, served by TabbyAPI on the exllamav3 fork
branch `dsv41-all` (tip `5e55c062`), on one RTX PRO 6000 Blackwell (96 GB, sm_120, PCIe 5.0 x16) plus one CMP 170HX
(64 GB, sm_80, PCIe Gen2 **x1**, about 0.38-0.4 GB/s), 160 GB of DDR5 of which the recipe may use 128 GB, and the
engram tables on a Samsung 9100 PRO. The engine's own documentation is authoritative for syntax and edge cases:
`doc/placement.md`, `doc/expert_tiers.md`, `doc/env_vars.md` and `doc/disk_engine.md` in
[the fork](https://github.com/anoane/exllamav3-anemone/tree/dsv41-all). This page says what each knob means **here**.

Device numbering: every family-c run uses `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0`, so **`cuda:0` is the
CMP** and **`cuda:1` is the PRO** (nvidia-smi still calls the PRO 0 and the CMP 1). Config a uses
`CUDA_VISIBLE_DEVICES=0`, so there `cuda:0` is the PRO.

## How to read the entries

Each knob has: what it does, its values and default, what to expect on this machine, what was measured so far (with
the run labels of [ATTEMPTS.md](ATTEMPTS.md), where you find every number), a recommendation, and a quality class:

| Class | Meaning | How it was checked |
|---|---|---|
| **bit-exact** | changes where bytes live, memory, I/O or scheduling; the logits do not change | greedy tokens equal to the reference run in every job (`tokens_sha256_12`), or bitwise 64K anchors |
| **numerics, last-bit** | another arithmetic path of the same function (chunking, kernels, which GPU runs a layer) | 64K teacher-forced NLL and top-1 within the run-to-run floor F (below); greedy tokens may differ |
| **numerics, lossy** | another rounding or precision contract: a quality trade | needs its own 64K quality line; reported as a trade, never adopted silently |
| **forbidden** | breaks a recipe rule (FP16 KV only, 1M context, RAM <= 128 GB, zero expert SSD reads for family c) | not run |

The floor F (stage 0, C12 at quality chunks 4096 / 2048 / 2047, against config a and four anchors): top-1 agreement
>= 0.9949 and |dNLL| <= 0.00081 on 64K teacher-forced tokens. The model is chaotic: a last-bit change flips near-ties
and the flips grow, so "greedy tokens differ" alone is not a quality verdict.

Speed evidence: `+x%` is against the layout's own reference process in the same window (C12 plain under
cudaMallocAsync for the C12 arms, C11 pipelined 4096 for the C11 arms, config a for the A1M arms). Decode is the
per-process mean of S1-32K, S2-32K and the S3 follow-up turns; prefill is S1-32K and S1-128K, unless a line says
otherwise. Most stage-2 arms ran **one** process; the run-to-run spread of the C12 reference was about 0.6% in decode
and 1-3% in prefill (`docs/stages/tables/tables_2_partial.md`).

---

## 1. Placement (`EXL3_PLACEMENT`)

TabbyAPI has no placement option: set `EXL3_PLACEMENT` in the service environment. It is read when the model object
is built. The family-c baseline:

```
EXL3_PLACEMENT='0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off'   # C12
EXL3_PLACEMENT='0-9=cuda:0; 10-39=cuda:1 experts=cache; ram experts=all; disk experts=off'    # C10 (serving now)
```

### 1.1 The split point (first PRO layer) and the CMP rules

- **What it does.** Layers `0..s-1` load on the CMP with every routed expert resident (`experts=vram`); layers
  `s..39` load on the PRO. Only activations cross the x1 link: 80 KiB per token of FP32 residual streams (4 x 5120),
  top-k selections and, for a split inside a kv group, the pool-replica rows.
- **Free cuts and the one split.** DeepSeek-V4.1-Flash shares state across layers: kv groups 2-7, 8-13, 14-19, 20-39
  (one compressed-KV pool each, owned by the first layer), index sources 2, 8, 14, 20, 24, 28, 32, 36. A device change
  at layer 1, 2, 8, 14 or 20 is a *free cut* (nothing shared across it). At most **one** change may fall anywhere
  else; the first PRO layer then owns a **replica** of its group's pool (512 MiB at 1M for splits 9-13, on the PRO)
  and the top-k selection of layer 8 crosses once per forward. Splits 10, 11 and 12 are all of this kind
  (`replicas 10<-8`, `11<-8`, `12<-8` in the load log). Two such changes are refused when the model is built.
- **What fits at 1M.** A 3.0 bpw layer with every expert resident is about 4.9 GiB (384 experts x 12.7 MiB plus
  about 0.15 GiB of attention). Measured at a 1M FP16 cache with 4096-row chunks:

| Split | CMP holds | CMP min free at 1M (nvidia-smi) | PRO pool (slots) | RAM tier | Peak RSS | Pipeline at 4096 |
|---|---|---|---|---|---|---|
| C14 | 0-13 | refused at load (`layers.12 does not fit on cuda:0`, run `C14-fit`) | - | - | - | - |
| C12 | 0-11 | 0.8 GB (cudaMallocAsync), 1.3 GB (expandable segments) | 5,818 | 4,942 slots, 61.3 GiB | 67.2 GiB | OOM under cudaMallocAsync; fits under expandable segments |
| C11 | 0-10 | 5.0 GB | 5,803 | 5,341 slots, 66.3 GiB | 72.1 GiB | yes |
| C10 | 0-9 | 10.1 GB (8.6-8.9 GB with 8192-row chunks) | 5,798 (5,704 at chunk 8192) | 5,730 slots, 71.1 GiB (5,824 slots, 72.2 GiB at chunk 8192) | 76.9 GiB (~78 at chunk 8192) | yes, also 8192 |
| C8 | 0-7 | 21.0 GB | 5,769 | 6,527 slots, 81.0 GiB | 86.8 GiB | yes |

- **Expected behaviour.** Decode time per token (fit over 130 family-c jobs, stage 1): about 21.5 ms + 0.195 ms per
  CMP layer + 0.274 ms per expert-cache miss; config a: 17.2 ms + 0.315 ms per miss. A CMP layer is only ~0.2 ms
  slower than a PRO layer, but every layer moved to the PRO adds ~1.6-2 misses per token (the PRO pool stays ~5,800
  slots whatever the split), about 0.5 ms. So **C12 is the decode-best feasible split**; smaller splits buy CMP room
  (larger pipeline sub-chunks) and cost RAM (about 5 GiB of RAM tier per layer moved).
- **Measured.** Stage 1, 1M, mean decode S1 / S2 / S3 turns: C12 35.8 / 38.9 / 37.4, C11 34.1 / 37.8 / 37.0,
  C10 34.6 / 38.5 / 36.9, C8 34.2 / 37.7 / 35.8 (runs `C12-2`..`C12-4`, `C11-1`, `C11-2`, `C10-1`, `C10-2`, `C8-1`,
  `C8-2`). Prefill (C11 / C10 / C8 pipelined, C12 plain): 1,504 / 1,474 / 1,420 / 1,109 at 32K.
- **Quality.** Numerics, last-bit: moving layers 8-11 from sm_80 to sm_120 changes their arithmetic. 64K NLL: C11
  0.250370, C10 0.250833, C8 0.249976 against the floor lines 0.250152-0.250762, top-1 >= 0.9949 (`Q-C11`, `Q-C10`,
  `Q-C8`): no split has a systematic quality cost.
- **Recommendation.** C12 for decode; C11 or C10 when prefill matters (they pipeline without allocator tricks). Not
  C14/C16 (do not fit) or C2 (predicted slower in S1, needs ~116 GiB of pinned RAM: not run).

### 1.2 `experts=` per layer range

| Mode | Where the experts live / who computes | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|
| `vram` (default) | the layer's GPU | the only mode for the CMP layers | every C run | CMP layers | bit-exact |
| `cache` | the GPU's expert cache (VRAM pool) over the RAM tier over the SSD; computed on the GPU from VRAM | the family-c mode for the PRO layers; refused on the CMP (its link is below `EXL3_MOE_TIER_MIN_LINK_GBS` = 2.0 GB/s) | C12/C11/C10/C8, config a | PRO layers | bit-exact (the cache only changes where bytes come from) |
| `stream [hot=k]` | RAM; every selected expert is copied to the GPU per call; `hot=k` resident per layer | a static resident slice | `C12-S-1` (`stream hot=208`, same resident count as C12's pool): decode 17.5-18.9, **-52%**, ~1 GB of H2D per decode token | no | numerics, last-bit (other expert kernels; greedy tokens 1/12 equal to C12) |
| `cpu [hot=k]` | RAM; the CPU worker computes the non-resident experts | CPU compute; the old production used 11 CPU layers | `C12-C-1` (`cpu hot=208`): decode 21.4-24.4, **-39%** | no | numerics, last-bit (CPU arithmetic; tokens 1/12 equal) |
| `split cpu=k` | the last k experts of every layer on the CPU worker | same family as `cpu hot=E-k` | not measured | no | numerics, last-bit |

GPU-computed (`stream`, `cache`) and CPU-computed (`cpu`, `split`) layers cannot share a GPU. The pipeline needs no
RAM-held experts before the split, so a family-c layout (CMP layers `vram`) is always eligible.

### 1.3 Layer-rule attributes: `hot=`, `prefetch=`, `profile=`, `embed=` / `head=`

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `hot=<k>` / `<p>%` on `cache` | pins k experts per layer in VRAM (taken from the pool), chosen by a profile or the heat file | 0 .. E-1 (0) | pins are pool slots that never move; with a heat file they would be the hottest experts | not measured | leave 0 until a heat-file run shows a gain | bit-exact |
| `prefetch=` | disk read-ahead of experts (`layer[:D]` at prefill, `router[:D]` at decode) | `auto` (= `layer:1` when the disk holds experts), `off`, `layer`, `router` | an explicit `layer` / `router` is **refused with `disk experts=off`** (nothing to read); `auto` then reads nothing | config a runs `auto` | family c: leave unset | bit-exact |
| `profile=` | seeds hot pins, cold-fill order and heat from a routing profile | names / paths with weights (none) | no DeepSeek-V4.1 profile exists; profiles do not transfer across domains | not measured | use the heat file (5.3) instead | bit-exact |
| `embed=` / `head=` | device of the modules before layer 0 / after layer 39 | first / last layer's device | moving `hc_expand` to the PRO would send 4 FP32 streams over x1 into layer 0 | not measured | keep defaults | bit-exact |

---

## 2. The GPU expert cache: `cuda:<n> cache= spare= evict= admit=`

Written as a storage rule of the placement, e.g. `...; cuda:1 admit=always; ram experts=all; disk experts=off`.

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `cache=` | size of the PRO's VRAM pool shared by every `cache` layer | `auto`, a size, `0` (`auto` = everything left after modules, KV, staging, margins and `EXL3_MOE_TIER_HEADROOM_MB`) | auto gives ~5,700-5,800 slots (70.5-71.9 GiB) at 1M on every split | pool per run in ATTEMPTS / tables_2_partial "Processes" | `auto`; make room with staging / headroom instead | bit-exact |
| `spare=` | free slots kept ready so a promotion never waits for a demotion | >= 1 with `disk experts=off` (8) | offline replay: no change on this text | replay only (`docs/stages/tables/t1_replay.log`) | 8 | bit-exact |
| `evict=` | which slot a promotion replaces | `lru`, `lfu` (`lru`) | LFU keeps long-lived experts; helps the prose turn | `C12-LFU`: decode -1.6%, prefill 0, S3 TTFT -2.5%, S5 prose turn collapse 31% vs 37% | `lru` | bit-exact (tokens 15/15 equal) |
| `admit=` | which decode misses enter the pool | `adaptive`, `heat`, `always` (`adaptive`) | `always` admits every miss: more hits, one device-to-host demotion per admission (about 100 MB per decode token for C12, replay estimate) | `C12-AA` x2: decode +0.7% (37.4 / 38.4 per process), S5 collapse 23% vs 37% and recovery after 129 vs 225 tokens; `A1M-AA`: -4.4% | a candidate for chat with topic switches; not a clear win on average | bit-exact (15/15) |

---

## 3. The RAM tier and the RAM budgets

### 3.1 `ram experts= ngram= pagecache= policy= demote= evict=`

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `ram experts=` | RAM for the routed experts: the RAM tier behind the cache | `all`, `auto`, a size, `0` | family c needs `all` (with `disk experts=off` anything smaller is refused): 61.3 GiB (C12) to 81.0 GiB (C8), pinned; config a: `all` would need ~131 GB, so it runs `102GiB` and leaves ~1,486 experts on the SSD only | every run | family c: `all`; config a: `102GiB` under the 128 GB rule | bit-exact |
| `ram ngram=` | RAM for the engram tables: whole scale tables first (5.7 GiB), the rest a direct-mapped row cache | `0`, a size, `auto`, `all` (`0`) | `all` = 189 GiB (does not fit); every family c leaves 33-53 GiB of the 128 GB unpinned for it | `C12-E2` (`auto`, which sized 96 GiB on 5e55c062: a sizing bug), `C12-E2F` (a Python-only fix, 30.8 GiB: decode +0.2%, prefill +1.4% / +0.3%), `C12-E3` (`40GiB` + `disk io=direct`: decode +0.5%, prefill +1.4% / +0.2%) | within noise so far; leave `0` (the page cache already holds hot rows) | bit-exact (15/15) |
| `ram pagecache=` | RAM that `auto` budgets leave unpinned | `auto` = max(8 GiB, MemTotal/8), a size | 20 GiB here | not varied | default | bit-exact |
| `ram policy=` | RAM tier vs VRAM cache: `exclusive` frees the RAM copy on promotion | `exclusive`, `lazy-exclusive`, `inclusive` (`exclusive`) | `inclusive` needs a RAM copy of the pool too (> 128 GB): refused | replay: `lazy-exclusive` no change | `exclusive` | bit-exact |
| `ram demote=` | what happens to a VRAM victim | `heat`, `swap`, `all`, `off` (`heat`) | with `disk experts=off` every victim returns to RAM unconditionally (`off` is refused) | n/a in family c | default | bit-exact |
| `ram evict=` | RAM tier victims | `lfu`, `lru` (`lfu`) | no RAM victims when RAM holds every non-VRAM expert | n/a in family c | default | bit-exact |

### 3.2 Budgets and caps: `-er`, `-der`, `-ngr` and their variables

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `-er` / `--expert_ram` / `EXL3_EXPERT_RAM` | cap on RAM for the main model's routed experts; a larger request is refused before anything loads | a size (no cap) | a safety cap equal to the tier size | not varied | set it in the final recipe (e.g. the C10 tier, 73 GiB) | bit-exact |
| `-der` / `--draft_expert_ram` | the same for a draft model or MTP head | a size (no cap) | no draft / MTP in this recipe | - | unset | bit-exact |
| `-ngr` / `--ngram_ram [SIZE]` / `EXL3_NGRAM_RAM` | cap / default of the n-gram (engram) RAM | `0`, a size, `all` (`0`) | tabby has no flag: use the placement's `ram ngram=` or the variable | see `ram ngram=` | `0` | bit-exact |
| `EXL3_NGRAM_STREAM` | older spelling: `0` means every table in RAM | 1 | 189 GiB: does not fit | - | unset | bit-exact |
| `EXL3_HOST_MEM_RESERVE_MB` | host memory kept free by every load-time check | MiB (2048) | the recipe's hard cap is the 119G cgroup of the test runs | - | default | bit-exact |

---

## 4. The disk rule: `disk experts= ngram= io=`

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `disk experts=` | where cached experts are read at runtime | `model`, `off`, a checked copy dir (`model`) | **`off` is the family-c rule** (zero expert SSD reads after load, checked on every run: `tier_ssd_reads_after_load` = 0); config a keeps `model` and read ~1.9 TB per test process | every run | family c: `off` | bit-exact |
| `disk ngram=` | where engram rows are read | `model`, `off` (needs `ram ngram=all`), a dir (`model`) | engrams stay on the SSD (rule) | - | `model` | bit-exact |
| `disk io=` | page-cache mode of the engine's reads | `auto` (extents O_DIRECT, rows buffered), `direct`, `buffered` | `direct` makes the row cache the only row cache | `C12-E3` (with `ngram=40GiB`) | `auto` | bit-exact |

---

## 5. Expert-tier tuning: `EXL3_MOE_TIER_*` and `EXL3_MOE_HEAT_*`

Read at load; a typo under these prefixes is refused with a hint; the load prints the ones that differ from default.

### 5.1 Pool room and staging

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `EXL3_MOE_TIER_STAGING` | VRAM for layer-mode prefill staging: `double` overlaps the next layer's copies with this layer's compute | `double`, `single` (`double`) | `single` gives +384 pool slots (+4.75 GiB) | `C12-SS` x2: decode +1.2%, prefill **-6.7% / -6.3%**, S3 TTFT +3% | `double` (decode first, but the prefill cost is larger than the decode gain) | bit-exact (15/15) |
| `EXL3_MOE_TIER_HEADROOM_MB` | VRAM kept free on the cache GPU after sizing the pool | MiB (4096) | 2048: +161 slots, 1024: +242 slots; 1024 OOMed once on the PRO alone at 16K in earlier work | `C12-H2` x2 (2048): decode +0.3%, PRO min free 1,983 MiB; `C12-H1` (1024): decode -0.1%, PRO min free 961 MiB; the 512K safety run `C12-H1-S4` did not run | default until a 512K run shows the cut is safe; the gain is within noise | bit-exact (15/15) |
| `EXL3_MOE_TIER_MIN_LINK_GBS` | refuses an expert cache on a GPU whose host link is slower than this | GB/s (2.0) | keeps the CMP (0.38 GB/s) out of the tier: correct | - | default | - |
| `EXL3_MOE_TIER_COMPACT` | one aligned copy per promotion | 0 / 1 (1) | - | - | 1 | bit-exact |

### 5.2 Routed vs layer mode

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `EXL3_MOE_TIER_PREFILL_ROWS` | rows from which a prefill call stages whole layers (layer mode); below it only touched experts are copied (routed mode) | 2 .. 2^20 (256) | even 512 rows touch almost all 384 experts of a layer; chat deltas are 0.5-8K rows | `C12-PR1K` x2 (1024): S3 follow-up TTFT **-8%**, decode +0.2%; `C12-PR2K` (2048): TTFT -9%, decode -2%; `C12-PR4K` did not run | 1024 is a stage-3 candidate for multi-turn chat | bit-exact (12/12) |
| `EXL3_MOE_TIER_DECODE_ROWS` | rows up to which a call may admit into the pool | 1 .. 8 (8) | batch 1 | not varied | default | bit-exact |

### 5.3 Heat and admission

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `EXL3_MOE_HEAT_FILE` | writes the tier's heat at unload and between generations; a later load seeds the cold fill and the heat from it | path (unset) | without it the cold fill is a round robin across layers (the load log says so) | `C12-HR` (seeded from `C12-R1`'s file, `results/heat/c12-r1.exl3moe`): first request after load (S0) decode **+3.5%**, prefill **+12.9%**, hit 0.955 vs 0.935; later scenarios -0.4% (the live heat takes over) | a candidate for serving (warm restarts); set it to a writable path | bit-exact (15/15) |
| `EXL3_MOE_HEAT_HALFLIFE` | decode tokens per halving of the heat | tokens (256) | - | replay: no change on this text | default | bit-exact |
| `EXL3_MOE_HEAT_PREFILL` | heat of one prefill assignment | 0 .. 1 (0.0625) | - | replay: no change | default | bit-exact |
| `EXL3_MOE_TIER_ADMIT_P` / `_ADAPT_EVERY` / `_ADMIT_PMIN` | pin / adapt the admission probability of `admit=adaptive` | 0..1 (unset) / 2048 / 0.05 | - | not varied (see `admit=always`) | default | bit-exact |
| `EXL3_MOE_TIER_SAMPLE` | RAM entries sampled when choosing the coldest | 16 | no RAM victims in family c | - | default | bit-exact |
| `EXL3_MOE_TIER_DEMOTE` | master switch of demotion | 0 / 1 (1) | `0` is refused with `disk experts=off` | - | 1 | bit-exact |

### 5.4 SSD-side tier knobs (config a only)

`EXL3_MOE_TIER_RAM_ADMIT` (`always`), `EXL3_MOE_TIER_REFILL` (1), `EXL3_MOE_TIER_REFILL_INFLIGHT` (2),
`EXL3_MOE_TIER_DISK_SLAB` (`auto`): they govern experts read from the SSD, which family c never does. Config a ran
the defaults. Class: bit-exact.

### 5.5 Host-side threads and memory

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `EXL3_MOE_TIER_SPIN_US` | tier thread spin while a pass is active | -1, 0, N (-1) | host-bound decode loop | `C12-SP0` did not run | default | bit-exact |
| `EXL3_MOE_TIER_AFFINITY` | CPU list of the tier thread | unset | keep it off the main thread's core | not run | default | bit-exact |
| `EXL3_MOE_TIER_HUGEPAGE` | transparent huge pages for the RAM tier | 0 / 1 (0) | slower load (compaction), maybe fewer IOMMU/TLB misses | `C12-HP` did not run | default | bit-exact |
| `EXL3_MOE_TIER_DETERMINISTIC` / `_VERIFY` | apply every call before the next / check invariants | 0 | debug; timing only | - | 0 | bit-exact |
| `EXL3_MOE_TIER_TRACE` | writes every tier record (JSON lines) | path | feeds the offline replay (`tools/harness/tier_trace_replay.py`) | `A1M-trace` (trace in `results/raw/traces/`) | tool only | bit-exact |
| `EXL3_MOE_PROFILE_DIR` | where `profile=` names are looked up | unset | - | - | unset | bit-exact |

---

## 6. DeepSeek-V4.1 switches (`EXL3_DSV41_*`)

### 6.1 `EXL3_DSV41_PIPELINE` and `EXL3_DSV41_PIPELINE_CHUNK` (pipelined prefill)

- **What it does.** A prefill call longer than one sub-chunk runs as consecutive sub-chunks: the calling thread runs
  the CMP layers of sub-chunk k+1 while a worker thread runs the PRO layers of sub-chunk k. Without it the two GPUs
  take turns.
- **Values.** `EXL3_DSV41_PIPELINE`: 0 (default) / 1. `EXL3_DSV41_PIPELINE_CHUNK`: tokens (4096); must be <= the load's
  `max_chunk_size` (tabby `chunk_size`), or the load is refused.
- **The generator chunk.** The pipeline only engages when one generator call holds several sub-chunks, so the
  generator's `max_chunk_size` must be larger than the sub-chunk (4 sub-chunks per call: 16384 for 4096, 32768 for
  8192). TabbyAPI passes its one `chunk_size` to the load and to the generator, hence the
  [`TABBY_EXL3_GENERATOR_CHUNK` patch](../serving/tabby-generator-chunk.patch).
- **Memory.** While the PRO works on sub-chunk k the CMP holds sub-chunk k+1's working set and k's products (320 MiB of
  FP32 streams at 4096 rows, half with the BF16 crossing). The autosplit does not reserve it. At split 12 under
  cudaMallocAsync the CMP has 0.8 GB free at 1M and **every pipelined call longer than two sub-chunks ran out of CMP
  memory** (`T-fit`, `T-pipe-128k`, `DIAG-C12-pipe`); the load still says "eligible" and the failed job is reaped
  cleanly.
- **Measured** (prefill against the same layout's plain or reference run, 1M):

| Run | Layout | Sub-chunk / generator chunk | Prefill 32K / 128K | Note |
|---|---|---|---|---|
| `C12` reference | C12 plain | - / 4096 | 1,113 / 1,126 | 7 processes |
| `DIAG-C12-pipe2048` | C12 | 2048 / 16384 | 1,095 at 16K | fits, no faster than plain |
| `C12-P25` | C12 | 2560 / 10240 | +8.5% / +12.2% | |
| `C12-P3` | C12 | 3072 / 12288 | +19.2% / +23.8% | 1 of 15 jobs hit a CMP OOM |
| `C12-ES-P3` | C12, expandable segments | 3072 / 12288 | +19.6% / +24.1% | no OOM |
| `C12-ES-P4` | C12, expandable segments | 4096 / 16384 | **+37.1% / +36.7%** (1,536 / 1,547) | tokens equal to plain C12 (15/15) |
| `C11-1`, `C11-2`, `C11-R1` | C11 | 4096 / 16384 | 1,506 / 1,516 (+33-36% over C12 plain) | |
| `C11-P8` | C11 | 8192 / 16384 | -1.0% vs C11 | only 2 sub-chunks per call |
| `C11-G32` | C11 | 4096 / 32768 | +6.2% / +7.3% vs C11 | 8 sub-chunks per call |
| `C11-T` | C11 | 2048 / 4096 | **-36%** vs C11 | what unpatched tabby at `chunk_size: 4096` allows |
| `C10-P8` | C10, chunk 8192 | 8192 / 32768 | 1,647 / 1,646 (+11.7% / +10.2% vs C10 at 4096) | |

- **Decode.** Unchanged within noise (the pipeline only touches prefill calls).
- **Quality.** Numerics, last-bit: a pipelined prefill is a plain prefill in sub-chunk-sized calls, so it matches a
  plain run at that chunk; against another chunk size the greedy tokens differ (C11-P8 4/15 equal, C12-P3 3/15).
  With the sub-chunk equal to the plain chunk (C12-ES-P4 vs C12 at 4096) the tokens were identical.
- **Recommendation.** On. The sub-chunk is bounded by the CMP's free memory: 4096 on C11 (and on C12 only under
  expandable segments), 8192 on C10. Use 4 or more sub-chunks per generator call.

### 6.2 `EXL3_DSV41_XDEV_BF16` (BF16 residual crossing)

- **What it does.** The 4 FP32 residual streams cross the device change in BF16 (40 instead of 80 KiB per token);
  each GPU keeps computing in FP32. On the x1 link a 4096-row chunk's crossing drops from ~0.84 s to ~0.42 s.
- **Values.** 0 (default) / 1. The old production tree had it on by default.
- **Measured.** L0 plain (`B-np`): prefill +10.6%, decode 0. C12 plain (`C12-X`): prefill **+12.0% / +12.3%**, decode
  +0.1%. C11 pipelined (`C11-X`): prefill **+15.6% / +16.8%** (1,744 / 1,767), decode +1.7%. C10 at 8192 (`C10-P8X`
  vs `C10-P8`): prefill 1,883 / 1,918 vs 1,647 / 1,646 (+14% / +17%), decode -2%.
- **Quality.** Numerics, **lossy** (rounds the streams): greedy tokens differ (C12-X 3/15 equal). Its 64K quality lines
  (`Q-C12-X`, `Q-C11-X`) were queued and **did not run**. The only earlier comparison (one 20K run with other options
  on) is not evidence either way.
- **Recommendation.** The largest single prefill lever after the pipeline; adopt only after its 64K check. The
  temporary serving config uses it (a known, listed trade).

### 6.3 `EXL3_DSV41_NUMERICS` and `EXL3_DSV41_NUMERICS_STRICT`

- **What it does.** How attention rounds three operands DeepSeek's reference code rounds on purpose: the lightning
  indexer's Q/K (MXFP4), the sliding-window KV and the compressed KV. The roundings are computed in FP16 and stored
  widened: they reproduce a contract's values, not its memory savings.
- **Values.** `precise` (FP16 everywhere), `deepseek[:index,window,compressed]`, `vllm[:parts]`; default
  `deepseek:index`.
- **Measured.** `precise` decodes faster: +2.7% on L0 (`B-np`, 23.24 / 23.15 vs 23.83 / 23.88), +6.1% on C12
  (`C12-NP`, one short process). Prefill unchanged. 64K NLL on the stable anchors (C12): `deepseek:index` 0.249963,
  `precise` 0.250939 (`T12-stable-dsidx`, `T12-stable-precise`); earlier numerics work found that the KV parts
  (`window`, `compressed`) raise the 64K NLL.
- **Quality.** Numerics, lossy (another rounding contract). The default has the better NLL of the two.
- **Recommendation.** Keep `deepseek:index`. `precise` is a listed decode-for-quality trade, undecided.
- **`EXL3_DSV41_NUMERICS_STRICT=1`** raises on a block that cannot be rounded instead of storing it unrounded; it reads
  its checks back to the host on every rounding call (10-11 host syncs per token under `deepseek:index`). Debug only.

### 6.4 The other V4.1 switches

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `EXL3_DSV41_ENGRAM_PREFETCH` | hashes and reads the engram rows of a prefill chunk on a worker thread before block 0 (layer 1's rows overlap block 0, layer 14's blocks 0-13) | 0 / 1 (1) | layer 1 is on the CMP, so its overlap is one block | not varied | 1 | bit-exact |
| `EXL3_DSV41_ROUTER_BIAS` | FP32 router selection bias from a sidecar file instead of the checkpoint's FP16 | path (unset) | needs the original FP32 biases | earlier numerics work: no measured gain | unset | numerics, lossy |
| `EXL3_DSV41_FUSED_COMPRESS` | one fused FP32 Triton kernel for the rate-2 compressor pooling | 0 / 1 (0) | the rate-2 kv sources, layers 2, 8, 14 (at splits 10-12: 2 and 8 on the CMP, 14 on the PRO) | `C12-FC` did not run | unset until measured | numerics, last-bit |
| `EXL3_DSV41_ACTIVATION` | `silu` (engine) or `reference` (DeepSeek's clamp-then-SiLU in FP32) | `silu`, `reference` (`silu`) | - | earlier numerics work: no consistent gain | `silu` | numerics, lossy |
| `EXL3_DSV41_ABLATE` | disables parts of the model | tests only | - | - | never | changes the model |
| `EXL3_DSA_DEBUG_BOUNDS` | bounds checks in the sparse attention | debug | - | - | never | bit-exact, slow |

---

## 7. Arithmetic, kernels and the build option

| Knob | What it does | Values (default) | On this machine | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|
| `EXL3_STABLE_ARITHMETIC` | row-count-independent arithmetic (implies `EXL3_HGEMM_FIXED_ROWS=128` and `EXL3_MOE_FUSED_PREFILL=1`; disables native graphs) | 0 / 1 (0) | the anchor profile: C12's 64K anchors are bitwise equal to the reference under it | `T12-stable-speed`: prefill 915, **decode 9.4**; earlier -23% prefill | never for serving | numerics, last-bit (a validation profile) |
| `EXL3_HGEMM_FIXED_ROWS` | cuBLAS GEMMs in fixed 128-row tiles | 0 / 128 (0) | part of the stable profile | with the stable profile only | 0 | numerics, last-bit |
| `EXL3_MOE_FUSED_PREFILL` | every routed expert of a prefill chunk through the fused kernel | 0 / 1 (0) | - | earlier: prefill -9% on the production layout (904 vs 997) | 0 | numerics, last-bit |
| `EXL3_HGEMM_FP32_REDUCTION` | forbids cuBLAS FP16 split-K reductions for FP16 outputs | 0 / 1 (1) | the sm_80 projection was not slower with 1 (0.161 vs 0.174 ms) | `C12-FR0` did not run | 1 | numerics, last-bit |
| `EXL3_HGEMM_F16ACC` | FP16 accumulation probe | auto (the PRO probes 1.00x: off) | - | - | auto | numerics, last-bit |
| `EXL3_FP32_LOGITS` | FP32 output logits instead of FP16 | 0 / 1 (0) | - | earlier: no measured benefit | 0 | numerics, lossy (more precise) |
| `EXL3_MOE_FUSED_DET` / `EXL3_MOE_RECON_DET` | deterministic per-assignment output slots (1) vs atomics (0) | 1 | `0` needs extra scratch | `B-det`, `DIAG-L0-det0`: **every 32K prefill ran out of CMP memory at 1M** | 1 | numerics (0 is non-reproducible) |
| `EXL3_MOE_COOP_WIDE` / `EXL3_MOE_COOP_KSPLIT` | tile geometry of the fused decode MoE kernel | unset | the CMP's decode share is the decision variable | `C12-CW1`, `C12-CW0`, `C12-KS2` did not run | unset until measured | numerics, last-bit |
| `EXL3_MOE_MTILE`, `_TILE_N`, `_FUSED_ROWS(_WIDE)`, `_BATCH_RECON`, `_RECON_*` | prefill tier boundaries of the MoE kernels | defaults | - | not varied | defaults | numerics, last-bit |
| `EXL3_BC_DSA`, `EXL3_BC_ATTN`, `EXL3_BC_MLA`, `EXL3_MOE_SHARED_COOP`, `EXL3_GEMV`, `EXL3_INT8_GEMV`, `EXL3_FUSED_SAMPLER` | native decode paths | defaults (on) | decode-only paths the teacher-forced gate does not cover | not varied | defaults | numerics, last-bit |
| `EXLLAMA_NO_WHOLE_K_MOE` (build) | leaves the whole-K fused-MoE instances out of the extension | set = leave out (unset) | only `EXL3_STABLE_ARITHMETIC` uses them; they are 185 MiB of 1,982 MiB of objects and 13% of the compile time of an sm_80 + sm_120a build | build-time only | set it for a serving-only build; leave it unset to keep the anchor profile | no runtime effect on serving |

---

## 8. Allocator and loading

### 8.1 `cudaMallocAsync` vs expandable segments

- **What it does.** The PyTorch CUDA allocator backend. TabbyAPI's `cuda_malloc_async: true` (its default) sets
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`; the temporary drop-in also sets it explicitly. With `PYTORCH_CUDA_ALLOC_CONF` unset,
  exllamav3 turns on expandable segments (`EXL3_EXPANDABLE_SEGMENTS`, default 1).
- **Measured.** `C12-ES` x2 (expandable segments, C12 plain): decode **+5.5%** (39.5 / 40.0 per process vs 37.6),
  prefill 0, tokens equal (15/15), CMP min free 1.32 GB instead of 0.80-0.86 GB. That extra CMP room is what let
  `C12-ES-P4` run the 4096 pipeline at split 12 (+37% prefill, +4.7% decode, tokens equal). Config a: `A1M-ES` +2.4%
  decode. H7 device time shows the gain on the CMP side (CMP blocks 0.86-0.87 ms mean instead of 0.96).
- **Quality.** Bit-exact (tokens equal).
- **Recommendation.** The strongest candidate of stage 2. In tabby: `cuda_malloc_async: false` and no
  `PYTORCH_CUDA_ALLOC_CONF` in the unit. **Not yet tried through tabby**; `C12-ES-P4`'s second process, its 512K run
  and its 32K generator chunk were queued and did not run.

### 8.2 Other loading knobs

| Knob | What it does | Values (default) | On this machine | Recommendation | Class |
|---|---|---|---|---|---|
| `EXL3_EXPANDABLE_SEGMENTS` | expandable segments for all torch allocations unless `PYTORCH_CUDA_ALLOC_CONF` is set | 1 | see 8.1 | default | bit-exact |
| `EXL3_LOAD_ARENA` | slab allocation of small weight tensors at load | 1 | fewer allocator segments for 15,360 experts | 1 | bit-exact |
| `EXL3_AUTOSPLIT_WORSTCASE`, `_PREPARE`, `_MARGIN_MB` | the loader's transient measurement and margin | 1, 1, 256 | the margin trades pool room against OOM; it does not budget the pipeline's second sub-chunk | defaults | bit-exact |
| `EXLLAMA_NO_P2P_COPY` | forces copies through host memory | probe | CMP <-> PRO go through the host either way | default | bit-exact |
| tabby `gpu_split` / `gpu_split_auto` / `autosplit_reserve` | per-GPU budgets | auto | under a placement they only cap memory, they never move a layer | `gpu_split_auto: true` | bit-exact |

---

## 9. Disk engine (`EXL3_DISK_*`): engram rows only in family c

In family c the disk engine reads only engram rows (two layers, 384,006,168 rows x 256 FP8 bytes = 91.5 GiB each, plus
2.86 GiB of scales); config a also reads experts. The `auto` choices were decided on this host (VirtIO SCSI single +
iothread) before the recipe search; the recipe did not sweep them.

| Knob | What it does | Values (default) | On this machine | Recommendation | Class |
|---|---|---|---|---|---|
| `EXL3_DISK_BACKEND` | reader backend | `auto` (= `io_uring` here), `original`, `pread`, `odirect`, `io_uring` | `io_uring` measured best | `auto` | bit-exact |
| `EXL3_DISK_DIRECT` | O_DIRECT for rows / extents | `auto` (io_uring: extents), `none`, `rows`, `extents`, `all` | set through `disk io=` | `auto` | bit-exact |
| `EXL3_DISK_KEEPALIVE_MS` / `_IDLE_S` | keep-alive reads so the first read after idle is not slow | `auto` (5) / 30 | the first read after 150+ ms idle took ~14 ms without it | defaults | bit-exact |
| `EXL3_DISK_QD` / `_RESERVE0` / `_ENGRAM_QD` | queue depth, depth reserved for demand reads, engram depth | 128 / 96 / 128 | a cold 4,096-row prefill chunk gathers ~50-60K rows per engram layer | defaults | bit-exact |
| `EXL3_DISK_SQPOLL` (+ `_IDLE_MS`, `_CPU`) | kernel submission polling | 0 | steals a vCPU in a guest | 0 | bit-exact |
| `EXL3_DISK_IOPOLL` | polled completions | 0 | refused on virtio-scsi | 0 | bit-exact |
| `EXL3_DISK_REGISTER` | registered files / buffers | `auto` (all) | the READ_FIXED rule of the branch is a correctness fix (an earlier revision hung the VM) | `auto` | bit-exact |
| `EXL3_DISK_WINDOW_EXPERT` / `_PREFETCH` / `_REFILL` | byte windows per request class | 8M, 8M/0, 8M/0 | expert windows matter for config a only | defaults | bit-exact |
| `EXL3_DISK_HOLD_ARM_MS`, `_REFILL_AGE_MS`, `_CHUNK`, `_ALIGN`, `_THREADS`, `_SPIN_US`, `_URING_ASYNC`, `_IOWQ_WORKERS`, `_AFFINITY`, `_FADVISE`, `_EXTENT_DONTNEED` | engine internals | see the engine's `doc/disk_engine.md` | engine smoke sweeps: within noise or worse | defaults | bit-exact |
| `EXL3_DISK_VERBOSE`, `_TRACE`, `_FAULT_*` | diagnostics and fault injection | 0 | - | never in serving | - |

---

## 10. CPU MoE knobs (the old production and the `cpu` arms only)

Family c computes no expert on the CPU. These knobs mattered for the old production (`OLD`, `L0`) and the `C12-C`
comparison arm: `EXL3_MOE_CPU_OFFLOAD` / `-mcl` (old production 11), `-mcs`, `-mcm` / `EXL3_MOE_CPU_MODE` (all three
are refused next to a placement), `EXL3_MOE_CPU_THREADS` / `-mct` (12; at most 16 here), `EXL3_MOE_CPU_WSLOTS` (2; old
production 4), `EXL3_MOE_CPU_WSLOT_MB` (32), `EXL3_MOE_CPU_SLOTS` / `_SLOT_ROWS` (4 / 64), `EXL3_MOE_CPU_STAGE_THREADS`
(4), `EXL3_MOE_STREAM_T` (bandwidth-scaled 16), `_STREAM_FUSED_T` (256), `_STREAM_MIN_ROWS` (32),
`_STREAM_BATCH_EXPERTS` (24), `_STREAM_BATCH_RECON` (1), `EXL3_MOE_PINNED_ARENA` (0; old production 1),
`EXL3_MOE_ARENA_HUGE` (reserved hugetlb pages: excluded), `EXL3_MOE_ARENA_HUGEPAGE` (1), `EXL3_MOE_CPU_SWIZZLE` (1),
`EXL3_MOE_CPU_MAX_ISA`, `EXL3_MOE_CPU_PIN` (1), `EXL3_MOE_MEMOPS` (1), `EXL3_MOE_CPU_SWAP` (dynamic placement for `hot=`
/ `split` layers).

Measured: the old layout decodes 21.4-23.2 tok/s whichever tree runs it (`A-old-*`, `A-new-*`, `L0-base`,
`OLD-base`); `C12-C` (`cpu hot=208`) 21.4-24.4. Recommendation: not part of the recipe.

---

## 11. TabbyAPI settings

| Setting | What it does | Values | Old production | Serving now | Measured | Recommendation | Class |
|---|---|---|---|---|---|---|---|
| `chunk_size` | the load's `max_chunk_size` (sizes each GPU's prefill workspace) and, unpatched, also the generator chunk | >= 256, multiples of 256 | 4096 | 8192 | 8192 at split 12: refused at load (`C12-K8`: layer 11 does not fit on the CMP); fine at C10 | >= the pipeline sub-chunk; 8192 only at split <= 10 | numerics, last-bit |
| `TABBY_EXL3_GENERATOR_CHUNK` (env, [patch](../serving/tabby-generator-chunk.patch)) | the generator's `max_chunk_size`, larger than the load chunk | tokens (unset = `chunk_size`) | unset | 32768 | without it the pipeline gets one sub-chunk per call: `C11-T` -36% prefill | 4 x the sub-chunk | numerics, last-bit |
| `max_seq_len` / `cache_size` | context length and KV cache size | tokens | 1,048,576 | 1,048,576 | - | 1,048,576 (rule) | - |
| `max_batch_size` | concurrent sequences | 1 | 1 | 1 | - | 1 | - |
| `sysmem_recurrent_cache` | MiB of host RAM for recurrent-state checkpoints (prefix reuse across turns) | MiB | 4096 | 4096 | stashes are ~15 MB each (15 stashes, 0.23 GiB after 8 turns); `0` crashed on the stage-0 tree `be33b0e1` (`D-rc`: an upstream `KeyError`, fixed on `5e55c062`); the planned 1024 / 4096 sweep (2.6) did not run | 4096 | bit-exact |
| `sysmem_kv_cache` | host-RAM KV cache for evicted sequences | MiB | 0 | 0 | a 1M VRAM cache at batch 1 never evicts a live conversation | 0 | bit-exact |
| `cuda_malloc_async` | sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` | bool (true) | true | true | see 8.1: expandable segments +5.5% decode | candidate: false (with no `PYTORCH_CUDA_ALLOC_CONF` in the unit) | bit-exact |
| `tool_format` | tool-call parser | `deepseek_v41` ([patch](../serving/tabby-deepseek_v41-toolformat.patch)) | deepseek_v41 | deepseek_v41 | - | deepseek_v41 | - |
| `reasoning`, `start_in_reasoning`, `template_vars_default` | chat template / reasoning handling | - | on, always, `reasoning_effort: high` | same | - | as served | - |
| n-gram drafting (`ngram_match_min`, `num_draft_tokens`, `dynamic_draft`) | speculative drafting from the prompt | off | off | off | `C12-NG` did not run | not measured | bit-exact under greedy |
| `realtime_process_priority` | raises the process priority | false | false | false | not run | default | bit-exact |
| `-mcl` / `-mcs` / `-mcm` (as `cpu_moe_*` settings) | the old CPU-offload switches | - | set via env in the old unit | unset | refused next to a placement | unset | - |

---

## 12. Forbidden by the recipe rules

| Knob | Why |
|---|---|
| `cache_mode` / `-cq` (Q4, Q6, Q8, `k,v` bits) | **FP16 KV only.** KV quantization destroys quality (every numerics comparison that rounded the KV raised the 64K NLL). It would only shrink the compressed NoPE lanes (~1 GiB on the PRO, ~75 more pool slots, ~1%). |
| `EXL3_QC_STAGING`, `EXL3_DSA_QC_STAGE*` | only apply with a quantized cache |
| any `cache_size` below 1,048,576 | 1M context is mandatory |
| `disk experts=model` / `<dir>` for family c | zero expert SSD reads after load |
| `ram ngram=all` / `disk ngram=off` | 189 GiB: over the 128 GB host rule |
| `ram policy=inclusive` | a RAM copy of the pool too: over 128 GB |
| `experts=cache`, `stream` or `cpu` on the CMP | the rule that the CMP holds whole layers. `cache` is also refused by the engine (its 0.38 GB/s link is below `EXL3_MOE_TIER_MIN_LINK_GBS` = 2.0); that check does not cover `stream` / `cpu`, which would push every selected expert or every expert call's activations over the x1 link |
| splits 13-16 with whole layers | do not fit on the CMP at 1M |

## 13. Host settings (fixed, listed for completeness)

Power caps 300 W (PRO) / 150 W (CMP) and the thermal guard (the unit's `ExecStartPre`, restored to 400 / 200 W on
stop); the test harness ran on cores 0-15 of 24 vCPUs; transparent huge pages `madvise`, shmem THP `never`; swap 0
(pinned memory is unreclaimable, so the RAM tier must fit); VirtIO SCSI single with an iothread for the SSD. None of
these were varied by the search.
