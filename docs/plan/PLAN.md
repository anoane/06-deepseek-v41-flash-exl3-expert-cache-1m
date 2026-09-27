> Published copy of the plan as written before the measurements (version 2, 2026-09-26), including the user decisions
> at its top. It is a working document: paths are the test machine's, estimates are estimates, and the stage reports
> say where the plan was adapted. The knob reference with measured effects is [../KNOBS.md](../KNOBS.md).

# USER DECISIONS (binding, 2026-09-26 ~23:50 CEST; override anything below)
1. **No KV-cache quantization, ever.** FP16 KV only: no cache_mode / -cq / Q4/Q6/Q8 / k_bits,v_bits variants in any stage
   (the user: it destroys quality). EXL3_QC_STAGING and EXL3_DSA_QC_STAGE* are therefore irrelevant.
2. **1M context is a strict requirement.** Every recipe candidate must load and serve with a 1,048,576-token cache (FP16).
   A configuration that cannot hold 1M is disqualified as a recipe (config a may still be reported as a reference, labelled with
   the largest cache it fits). Measure speed at 1M cache size (prompts of 32K/128K/long contexts inside the 1M cache).
3. **Test 11 CMP layers too (C11)**, pipelined with full-size 4096-row sub-chunks (and 8192 if it fits), next to C10 and C12:
   EXL3_PLACEMENT='0-10=cuda:0; 11-39=cuda:1 experts=cache; ram experts=all; disk experts=off' (split 11 is inside kv group 8-13:
   layer 11 gets a replica of layer 8's pool, allowed: one split inside a group). Include C11 in stage 1A (S1-S5 vs a), 1C (S4-512K)
   and the stage 2.1 sub-chunk sweep; record CMP free VRAM at 1M under the pipeline.

# DeepSeek-V4.1-Flash serving recipe, family c, and the c-vs-a head-to-head: knobs and staged plan

Status: PLAN ONLY (version 2, 2026-09-26). It replaces version 1 (kept as `PLAN-v1-20260926.md` next to this file).
What version 1 got right is kept: the constraints, the knob inventory, the harness and quality methodology, the
stage-0 windows 0a/0b. What is new: the head-to-head against config a in scenarios S1-S5, the split-point question
(10 / 12 / 14 / 16), config a at the real 1M cache, and the stage structure 0-4 requested for this round.

Code under test: `dsv41-all` tip `be33b0e1` (tree identical to `b8105f2d`), source
`/root/rnd/release/all/src-b8105f2d6a05`, native `/root/rnd/release/all/native-b8105f2d6a05/exllamav3_ext.so`
(sha256 `823174af46fb...`, sm_80 + sm_120a). Only branch `dsv41-all` is used. Nothing here changes the branch, the
production unit `ds41-exl3.service` or its config, and nothing is deployed.

## 0. Goal, definitions, hard constraints

**Family c** = RTX PRO 6000 + CMP 170HX + host RAM <= 128 GB TOTAL for the serving process (cgroup
`MemoryMax=119G` = 127.8 GB, swap 0; RSS + pinned + page cache charged to it) + engram tables on the SSD + ZERO
routed-expert SSD reads after the load + 1M context + batch 1 (tabby). The CMP holds WHOLE layers
(`experts=vram`); `experts=cache` is never placed on the CMP (x1 link, 0.38 GB/s; refused below
`EXL3_MOE_TIER_MIN_LINK_GBS`=2.0 anyway). The PRO runs `experts=cache` over a RAM tier that holds every non-VRAM
expert (`ram experts=all; disk experts=off`).

**Config a** (the reference) = the PRO alone (`CUDA_VISIBLE_DEVICES=0`), `*=cuda:0 experts=cache; ram experts=102GiB`
(tier defaults `policy=exclusive demote=heat`, `prefetch=auto` = `layer:1`), the SSD backs the ~1,250-1,500
experts that fit neither VRAM nor RAM. Its documented form `ram experts=all; disk experts=off` needs 117.5 GiB of
tier alone (~131 GB total) and is refused under the memory rule; `ram experts=102GiB` is the largest RAM tier whose
pinned + resident stays <= 120 GB (RSS 107.6 GiB measured). In the head-to-head it runs in the SAME 119G cgroup as
c, so both have the same 128 GB host envelope; a's RSS stays <= 120 GB by its `ram experts=`.

Priorities (binding): maximize decode first, then prefill; quality at the best level (64K NLL / top-1 vs the anchors
within the run-to-run floor). Every function-changing knob is reported as a quality trade, never adopted silently.

Machine: nvidia-smi 0 = RTX PRO 6000 (sm_120, 96 GB, x16, ~50 GB/s pinned H2D), 1 = CMP 170HX (sm_80, 64 GB, gen2
x1, ~0.38 GB/s). `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0`: cuda:0 = CMP, cuda:1 = PRO. 160 GiB RAM,
swap 0, 24 vCPUs (harness on cores 0-15; other jobs on 16-23). Disk: 43 GB free (traces must stay small).

Rules for every GPU line: only inside `/root/rnd/gpu_window.sh` (`setsid nohup`, one window per batch, <= ~108 min),
thermal guard on, caps 300 / 150 W (the serving caps, `run_batch.sh` default), kill by PID only; every process
under `aux/py_memwatch.sh` (kills its own PID when MemAvailable < 8 GiB, writes `*.KILLED`) inside a
`systemd-run --scope` cgroup with swap 0: `MEMMAX=111G` (<= 120 GB) for OLD / L0 / stable lines, `MEMMAX=119G`
(<= 128 GB) for family c AND for config a in the head-to-head (a's pinned + resident <= 120 GB by construction); the
driver waits for `MemAvailable >= expected RSS + 16 GiB` before each line (`#@ minavail=`).

## 1. Where we stand (measured) and what the numbers already imply

| Configuration | Decode tok/s | Prefill tok/s (32K) | Host RSS | Source |
|---|---|---|---|---|
| OLD = production (old tree `wt-dsv41-crnd`, CMP 0-11, PRO 12-39, CPU experts 12-22, 1M) | 23.4-23.7 (record, 24-token context) | ~1,000-1,045 | ~58.6 GiB | 2026-09-23 records |
| L0 = same layout on dsv41-all (`0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1`), 1M | 23.2 | 1,000 | 58.7 GiB | VALIDATION.md |
| a, 64K cache: r0 (novel text after a warm-up on other text) | ~31 (hit 0.856, 0.6 SSD reads/token) | 1,289-1,341 | 107.6 GiB | VALIDATION.md, W2 |
| a, 64K cache: r1 (same prompt again, fresh generator) | 38.5-43.5 (hit 0.961) | 1,302-1,397 | 107.6 GiB | VALIDATION.md, W2 |
| window 0a (running now, gpu_window pid 2271596) | OLD vs L0 x3 at 32K + decode-gap decomposition | | | `stage0/w0a/` |

Never loaded yet: any family-c tier layout (T12 etc.), config a at a 1M cache, any S3 / S5 measurement.

**Split feasibility (the user's 10 / 12 / 14 / 16).** A V4.1-Flash 3.0 bpw layer with every routed expert resident
is ~4.9 GiB (384 x 13.3 MB experts + ~0.15 GiB attention). The CMP at 1M with layers 0-11 used 64,558 of 65,536 MiB
(chunk 4096, version-1 smoke). So:

| Split (first PRO layer) | CMP holds | CMP weights | Fits the CMP? | Replica | Cached layers on the PRO | RAM tier (`all`) | Expected RSS | Left in 128 GB for engram rows + page cache |
|---|---|---|---|---|---|---|---|---|
| 16 | 0-15 | ~78 GiB | **no** (+~14 GiB over) | - | 24 | - | - | - |
| 14 (free cut) | 0-13 | ~69 GiB | **no** (+~4-5 GiB over) | none | 26 | - | - | - |
| 12 | 0-11 | ~59 GiB | yes (~1 GB free) | 12 <- 8 (512 MiB at 1M, PRO) | 28 | 4,976 slots = 61.7 GiB | ~68 GiB | ~51 GiB |
| 10 | 0-9 | ~49 GiB | yes (~10 GB free: room for 8K chunks) | 10 <- 8 (512 MiB) | 30 | ~5,744 = ~71 GiB | ~78 GiB | ~41 GiB |
| 8 (free cut) | 0-7 | ~39 GiB | yes (~20 GB free) | none | 32 | ~6,512 = ~81 GiB | ~87 GiB | ~32 GiB |
| 2 (free cut) | 0-1 | ~10 GiB | yes | none | 38 | ~8,816 = ~109 GiB | ~116 GiB | ~3 GiB (recurrent cache must shrink to 1 GiB; engram rows uncached) |

(Pool ~5,780 slots at every split, from the doc's (c) estimate; RAM tier = cached experts - pool; RSS = tier +
~6.5 GiB base.) **14 and 16 cannot hold whole layers on the CMP**; the only way to put 14-16 layers there is
`experts=cpu hot=<k>` on the CMP layers (CPU computes the rest), which breaks "whole layers", makes the pipeline
ineligible (RAM-held experts before the split) and reintroduces CPU expert compute. The plan replaces them with the
free cut 8 (and 2 as a conditional boundary arm) and records one load refusal at 14 as evidence (stage 1).

**Where decode time goes (the decision variable for c vs a).** From a's r1 (43.5 tok/s = 23 ms/token, 40 PRO
layers, ~9.4 misses/token): PRO time t_pro ~0.5 ms per layer-token without misses; a miss costs t_miss ~0.3-0.36
ms (a's r0 - r1: +9 ms for +25 misses). For c at split s with CMP time t_cmp per layer-token:

```
T_c(s) = s * t_cmp + (40 - s) * t_pro + misses_c(s) * t_miss + t_cross (~0.5 ms: 80 KiB streams + top-k over x1)
T_a    = 40 * t_pro + misses_a * t_miss
```

With guessed hit rates (c12 warm ~0.975, novel ~0.91; to be replaced by the exact trace replay, T1): c12 beats a in
warm decode only if t_cmp < ~0.6 ms, and on novel text if t_cmp < ~1.0 ms. L0's 23.2 tok/s (CMP 12 layers + 17 PRO
layers + 11 CPU layers at ~0.9 GB/token of DRAM reads) suggests t_cmp ~1-2 ms. **Pre-registered expectation**: a
wins warm decode (S2/S3) by ~20-50% at C12 (less at C8); novel-text decode (S1) is close (within ~10%); the
best c for decode is the smallest CMP share (split 8, or 2 if it fits); c's likely wins are prefill (pipelined CMP || PRO, and the PRO stages 28-32 instead of
40 layers per pass), short-delta TTFT in S3, zero expert SSD reads and ~30-50 GB more RAM for engram rows. Stage 0
measures t_cmp directly (H7) so this is tested, not assumed.

**Where prefill time goes.** In layer mode every non-pool expert of every cache layer is copied once per forward
pass. a at chunk 4096: ~9,470 x 13.3 MB = ~126 GB H2D per pass = ~2.5 s at 50 GB/s (+ ~20 GB of SSD reads per
pass) for 4,096 rows, a bound of ~1,640 tok/s: a's measured ~1,390 tok/s is close to it, and staging per token
scales with 1/chunk. c12 stages ~66 GB per pass (~1.3 s); its limit is the CMP stage plus the residual crossing (320 MiB per 4,096 rows in
FP32 = ~0.84 s over x1), hence the pipeline and `EXL3_DSV41_XDEV_BF16`. Consequences: chunk / pipeline sub-chunk is
the biggest prefill lever for BOTH configs (so the final comparison gives a the same chunk choice), and short
deltas (S3) pay a floor of about one pass of staging (~1.3 s c12, ~2.5 s + SSD for a) because even 512 rows touch
~all 384 experts of a layer (384 x (1 - e^-8)).

## 2. Knob inventory for family c (dsv41-all)

Class: **N** numerically neutral (logits bitwise equal; tier state, I/O, memory, scheduling), **A** arithmetic path
changes that keep the function (last-bit / chaotic-floor level), **L** function-changing (lossy or another rounding
contract). Rel.: H swept, M swept if time, L listed (not swept), F fallback / comparison arms only, X not applicable
or excluded (reason given).

### 2.1 Placement (`EXL3_PLACEMENT` / `--placement` / `config.infer_params.placement`; tabby: env only)

| Knob | Range | Default | Cls | Rel. | Notes for family c |
|---|---|---|---|---|---|
| split point (first PRO layer) | free cuts 1, 2, 8, 14, 20, or ONE split inside a kv group (pool replica) | - | A | H | feasible whole-layer CMP: 2, 8, 10, 12 (table above); 14 / 16 do not fit. CMP must come first for the pipeline |
| `experts=` per range | `vram`, `cache`, `stream [hot=k]`, `cpu [hot=k]`, `split cpu=k` | `vram` | N/A | H | CMP: `vram` only. PRO: `cache` (family c); `stream hot=k` / `cpu hot=k` as comparison arms; one kind of RAM-held experts per GPU |
| `hot=<k>` / `<p>%` | 0 .. E-1 (cache, stream, cpu) | 0 | N | M | cache: pins from the heat file / profile, taken from the pool; stream / cpu: the resident slice (fit-probed max ~220 per layer on the PRO at split 12) |
| `prefetch=` | `auto`, `off`, `layer[:D]`, `router[:D]` | `auto` | N | X (c) | refused with `disk experts=off` (auto = nothing). Config a keeps `auto` (`layer:1`) |
| `profile=` | names / paths with weights | none | N | L | no V4.1 profile on the box; profiles do not transfer across domains; the heat file is the in-domain seed |
| `embed=` / `head=` | `cuda:<n>` | first / last layer's device | A | X | moving hc_expand to the PRO would send 4 FP32 streams over x1 into layer 0 |
| `cuda:1 cache=` | `auto`, a size >= (top-k + spare) slots | `auto` | N | L | auto is the maximum; room is moved with STAGING / HEADROOM instead |
| `cuda:1 spare=` | 1 .. (0 refused with disk off) | 8 | N | M | offline replay first |
| `cuda:1 evict=` | `lru`, `lfu` | `lru` | N | M | offline replay first |
| `cuda:1 admit=` | `adaptive`, `heat`, `always` | `adaptive` | N | M | offline replay first (adaptive admits a decode miss with probability p in [0.05, C/(C+R)]; the rest are transients) |
| `ram experts=` | `all`, `auto`, size, `0` | cap, else auto | N | fixed | family c: `all` (with `disk experts=off` anything smaller is refused); `auto` = `all` only while the host has room (fails at split 2) |
| `ram ngram=` | `0`, size, `all`, `auto` | `0` | N | H | engram row cache: whole scale tables first (5.7 GiB), the rest a direct-mapped row cache (~264 B/row vs a 4 KiB page per row in the page cache); `all` = 189 GiB, excluded; room per split in the table above |
| `ram pagecache=` | `auto` (= max(8 GiB, MemTotal/8) = 20 GiB), size | `auto` | N | H | what `auto` budgets leave unpinned (the rows' page-cache tier) |
| `ram policy=` | `exclusive`, `lazy-exclusive`, `inclusive` | `exclusive` | N | M | inclusive would need a RAM copy of the pool too (> 128 GB): excluded. lazy-exclusive keeps duplicates, fewer D2H on churn (replay) |
| `ram demote=` | `heat`, `swap`, `all`, `off` | `heat` | N | X | with `disk experts=off` every victim returns to RAM unconditionally (`off` refused) |
| `ram evict=` | `lfu`, `lru` | `lfu` | N | X | no RAM victims when RAM holds every non-VRAM expert |
| `disk experts=` | `model`, `off`, `<dir>` | `model` | N | fixed | `off` = the zero-expert-SSD-read requirement (checked on every c line: `tier_ssd_reads_after_load` = 0) |
| `disk ngram=` | `model`, `off` (needs `ngram=all`), `<dir>` | `model` | N | fixed | engrams stay on the SSD |
| `disk io=` | `auto` (rows buffered, extents O_DIRECT), `direct`, `buffered` | `auto` | N | H | `direct` makes the row cache the only row cache (no page-cache use inside the 128 GB) |

### 2.2 CLI / Python / tabby budgets and generator settings

| Knob | Where | Range | Now (serving) | Cls | Rel. | Notes |
|---|---|---|---|---|---|---|
| `-er` / `--expert_ram` / `EXL3_EXPERT_RAM` | cap | size | none | N | final recipe | safety cap = the tier's size |
| `-ngr` / `--ngram_ram [SIZE]` / `EXL3_NGRAM_RAM` | cap (placement `ram ngram=` is the request) | 0, size, `all`, (auto via placement) | 0 | N | H | tabby has no flag: env or the placement's `ram` rule |
| `chunk_size` (tabby) = load `max_chunk_size` = generator chunk = `max_rq_tokens` | yml | >= 256, x256 | 4096 | A | H | staging per token ~1/chunk; CMP transients (8192 only at split <= 10); tabby cannot set the generator chunk separately (the pipeline wants it larger: user decision, 14) |
| `cache_size` / `max_seq_len` | yml | tokens | 1,048,576 | - | fixed | 1M |
**FORBIDDEN by the user (no KV quantization; FP16 KV only) -- do not run:** | `cache_mode` | yml | FP16, Q4, Q6, Q8, `k,v` bits | FP16 | L | L | shrinks only the compressed NoPE lanes (~1 GiB on the PRO = ~+75 slots) minus the QC staging buffer; refused by the stable profile and by any `compressed` numerics part; KV rounding raised the 64K NLL in every numerics comparison. Fit probe only; speed + quality only if >= +2% pool |
| `sysmem_recurrent_cache` (Generator `recurrent_cache_size`) | yml | MiB | 4096 | N | H (S3) | checkpoints for prefix reuse (V4.1 state has an SWA ring, compressor carry, engram ring): 0 makes every turn re-prefill; RAM inside the 128 GB; stash of the CMP layers crosses x1 |
| `sysmem_kv_cache` (`cpu_cache_size`) | yml | MiB | 0 | N | X | a 1M VRAM cache at batch 1 never evicts a live conversation; RAM budget |
| `max_batch_size` | yml | 1 | 1 | - | fixed | |
| `cuda_malloc_async` vs expandable segments (`EXL3_EXPANDABLE_SEGMENTS`) | yml / env | bool | true | N | M | pool room (auto sizing reads free memory), long-context fragmentation |
| `gpu_split` / `autosplit_reserve` / `gpu_split_auto` | yml | GiB / MB | auto | N | L | budgets only cap memory under a placement |
| `output_chunking` | yml | bool | true | N | L | |
| draft: `ngram_match_min`, `num_draft_tokens`, `dynamic_draft` | yml | | off | N (greedy) | L (opt.) | n-gram drafting on code conversations; verification rows (<= 8) stay decode calls |
| `realtime_process_priority` | yml | bool | false | N | L (opt.) | host-bound decode loop |
| `cpu_moe_offload_layers` (`-mcl`), `-mcs`, `-mcm` | yml / CLI | | unset | - | X | refused next to a placement |

### 2.3 Expert tier tuning (`EXL3_MOE_TIER_*`, `EXL3_MOE_HEAT_*`; read at load, typos refused)

| Variable | Range | Default | Cls | Rel. | Notes |
|---|---|---|---|---|---|
| `EXL3_MOE_TIER_STAGING` | `double`, `single` | `double` | N | H | single: +384 pool slots (+4.75 GiB, +6.6% at split 12), loses the layer-mode copy / compute overlap (prefill cost) |
| `EXL3_MOE_TIER_HEADROOM_MB` | MiB | 4096 | N | H | 3072 / 2048: +80 / +160 slots; 1024 OOMed at a 16K prefill on the PRO alone: any cut needs a 512K+ safety run |
| `EXL3_MOE_TIER_PREFILL_ROWS` | 2 .. 2^20 (> DECODE_ROWS) | 256 | N | M | layer mode from here; routed mode below copies only touched experts but with the SMs and no overlap. S3 deltas decide |
| `EXL3_MOE_TIER_DECODE_ROWS` | 1 .. 8 | 8 | N | L | batch 1 (n-gram drafting uses up to ~5 rows) |
| `EXL3_MOE_HEAT_HALFLIFE` | 1 .. 2^30 tokens | 256 | N | M | S3 stability vs S5 recovery: replay |
| `EXL3_MOE_HEAT_PREFILL` | 0 .. 1 | 0.0625 | N | M | replay |
| `EXL3_MOE_HEAT_FILE` | path | unset | N | H (serving) | warm restart: S0 (first request after load) |
| `EXL3_MOE_TIER_ADMIT_P` / `_ADAPT_EVERY` / `_ADMIT_PMIN` | 0..1 / 1..2^30 / 1e-4..1 | unset / 2048 / 0.05 | N | M | replay; ADMIT_P pins p for A/B |
| `EXL3_MOE_TIER_SAMPLE` | 1 .. 4096 | 16 | N | L | RAM victim sampling (no RAM victims with full coverage) |
| `EXL3_MOE_TIER_SPIN_US` / `_AFFINITY` | -1, 0, N us / CPU list | -1 / unset | N | L (opt.) | tier thread reaction on misses; keep it off the main thread's core |
| `EXL3_MOE_TIER_HUGEPAGE` | 0 / 1 | 0 | N | L (opt.) | THP for the RAM tier: slower load (compaction), maybe fewer IOMMU / TLB misses for the fetch kernel |
| `EXL3_MOE_TIER_COMPACT` | 0 / 1 | 1 | N | L | keep 1 (one aligned copy per promotion) |
| `EXL3_MOE_TIER_DEMOTE` | 0 / 1 | 1 | N | X | 0 refused with `disk experts=off` |
| `EXL3_MOE_TIER_RAM_ADMIT`, `_REFILL`, `_REFILL_INFLIGHT`, `_DISK_SLAB` | | always, 1, 2, auto | N | X (c) | only with SSD-backed experts (config a keeps defaults) |
| `EXL3_MOE_TIER_MIN_LINK_GBS` | GB/s | 2.0 | - | X | keeps the CMP out of the tier (correct) |
| `EXL3_MOE_TIER_DETERMINISTIC` / `_VERIFY` | 0 / 1 | 0 | - | X | debug; timing only |
| `EXL3_MOE_TIER_TRACE` | path | unset | N | tool | feeds the offline replay T1 (capture lines only, excluded from speed tables) |
| `EXL3_MOE_PROFILE_DIR` | dirs | unset | N | L | |

### 2.4 DeepSeek-V4.1 numerics, pipeline, crossing, engram

| Variable | Range | Default | Cls | Rel. | Notes |
|---|---|---|---|---|---|
| `EXL3_DSV41_NUMERICS` | `precise`, `deepseek[:parts]`, `vllm[:parts]` | `deepseek:index` | L | H (speed only) | default has the best 64K/128K NLL and reference agreement; precise = quality trade; since D12 no host read per rounding call; measure the speed cost only |
| `EXL3_DSV41_PIPELINE` + `_PIPELINE_CHUNK` | 0/1; sub-chunk <= load `max_chunk_size` | 0; 4096 | A | H | CMP stage \|\| PRO stage; needs generator chunk > sub-chunk (4 sub-chunks per call ~1.6x ideal); eligible for every family-c layout (no RAM-held experts before the split); tier + pipeline never ran together (0b) |
| `EXL3_DSV41_XDEV_BF16` | 0 / 1 | 0 | L | H | halves the streams' crossing (0.84 -> 0.42 s per 4,096 rows over x1); decode effect negligible; OLD production runs it ON; quality unmeasured in isolation: needs a 64K line |
| `EXL3_DSV41_ENGRAM_PREFETCH` | 0 / 1 | 1 | N | L | layer 1's rows overlap block 0 only (on the CMP), layer 14's blocks 0-13; keep 1 |
| `EXL3_DSV41_FUSED_COMPRESS` | 0 / 1 | 0 | A | L (opt.) | speed never measured |
| `EXL3_DSV41_ACTIVATION`, `EXL3_DSV41_ROUTER_BIAS`, `EXL3_FP32_LOGITS` | | silu, unset, 0 | L | X | measured without gain |
| `EXL3_STABLE_ARITHMETIC` (implies `EXL3_HGEMM_FIXED_ROWS=128`, `EXL3_MOE_FUSED_PREFILL=1`) | 0 / 1 | 0 | A | anchors | validation profile only: prefill 796 vs 1,040 tok/s (-23%), decode much slower (no native graphs); used for the bitwise anchor check of split 12; its speed cost on T12 measured once (0b, optional) |
| `EXL3_MOE_FUSED_PREFILL` alone | 0 / 1 | 0 | A | X | measured -9% prefill |
| `EXL3_DSV41_NUMERICS_STRICT`, `_ABLATE`, `EXL3_DSA_DEBUG_BOUNDS` | | 0 | - | X | debug / tests |

### 2.5 Disk engine (engram rows only in family c; doc/disk_engine.md)

| Variable | Range | Default | Cls | Rel. | Notes |
|---|---|---|---|---|---|
| `EXL3_DISK_BACKEND` | auto (= io_uring here), original, pread, odirect, io_uring | auto | N | L | decided on this host 2026-09-25 |
| `EXL3_DISK_DIRECT` | none, rows, extents, all | auto (io_uring: extents) | N | H | via `disk io=` |
| `EXL3_DISK_KEEPALIVE_MS` / `_IDLE_S` | 0, 1..1000 / 1..86400 | auto (5) / 30 | N | L (opt.) | first read after 150+ ms idle ~14 ms without it |
| `EXL3_DISK_QD` / `_RESERVE0` / `_ENGRAM_QD` | 1..4096 | 128 / 96 / 128 | N | L | a cold 4,096-row prefill chunk gathers ~50-60K rows per engram layer |
| `EXL3_DISK_SQPOLL` (+`_IDLE_MS`, `_CPU`) | 0/1 | 0 | N | L (opt.) | steals a core in a guest |
| `EXL3_DISK_REGISTER` | none, files, buffers, all | auto (all) | N | X | the READ_FIXED rule of commit 46 is a correctness fix |
| `EXL3_DISK_IOPOLL` | 0/1 | 0 | - | X | refused on virtio-scsi |
| `EXL3_DISK_WINDOW_*`, `_HOLD_ARM_MS`, `_REFILL_AGE_MS`, `_CHUNK`, `_ALIGN`, `_THREADS`, `_SPIN_US`, `_URING_ASYNC`, `_IOWQ_WORKERS`, `_AFFINITY`, `_FADVISE`, `_EXTENT_DONTNEED` | see env_vars.md | 8M, 8M/0, 8M/0, 50, 250, auto, auto, 24, 50, 0, 0, unset, auto, 1 | N | L | engine smoke sweeps: within noise or worse; no expert reads in family c |

### 2.6 Kernels, allocator, host

| Variable | Default | Cls | Rel. | Notes |
|---|---|---|---|---|
| `EXL3_HGEMM_FP32_REDUCTION` | 1 | A | L (opt.) | 0 = cuBLAS may reduce in FP16; with 1 the sm_80 projection was not slower (0.161 vs 0.174 ms); comparisons only |
| `EXL3_HGEMM_F16ACC` | auto (probe: PRO 1.00x, off) | A | L | keep auto |
| `EXL3_MOE_COOP_WIDE` / `_COOP_KSPLIT` | unset | A | L (opt.) | fused decode MoE tile geometry: the CMP's decode share is the decision variable, so one probe on the CMP layers (H7 timing) |
| `EXL3_MOE_MTILE`, `_TILE_N`, `_FUSED_ROWS(_WIDE)`, `_BATCH_RECON`, `_RECON_*` | defaults | A | L | prefill tier boundaries |
| `EXL3_MOE_FUSED_DET` / `_RECON_DET` | 1 | A | X | 0 = atomic, non-reproducible (0a decomposition only) |
| `EXL3_BC_DSA`, `EXL3_BC_ATTN`, `EXL3_BC_MLA`, `EXL3_MOE_SHARED_COOP`, `EXL3_GEMV`, `EXL3_INT8_GEMV`, `EXL3_FUSED_SAMPLER`, `EXL3_DSA_*` | defaults | A | X | decode-only paths the teacher-forced gate does not cover |
**FORBIDDEN by the user (no KV quantization; FP16 KV only) -- do not run:** | `EXL3_QC_STAGING`, `EXL3_DSA_QC_STAGE*` | 1 | A | L | only with `cache_mode` != FP16 |
| `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` vs `EXL3_EXPANDABLE_SEGMENTS=1` | cudaMallocAsync (tabby) | N | M | fit probe |
| `EXL3_LOAD_ARENA`, `EXL3_AUTOSPLIT_WORSTCASE`, `_PREPARE`, `_MARGIN_MB` | 1, 1, 1, 256 | N | L | margin trades pool room vs OOM |
| `EXL3_HOST_MEM_RESERVE_MB` | 2048 | N | fixed | the cgroup is the hard cap |
| `EXLLAMA_NO_P2P_COPY` | probe | N | X | CMP <-> PRO bounce through host either way |
| power caps 300 / 150 W, thermal guard | fixed | - | X | policy; the CMP's 150 W cap may bound t_cmp (logged, not changed) |
| process affinity (cores 0-15), THP (`madvise`, shmem `never`) | fixed | - | X | system policy |

### 2.7 CPU MoE (comparison arms `cpu hot=k`, L0, OLD only)

`-mct` / `EXL3_MOE_CPU_THREADS` (12; <= 16 here), `EXL3_MOE_CPU_WSLOTS` (2; production 4), `EXL3_MOE_CPU_WSLOT_MB`
(32), `EXL3_MOE_CPU_SLOTS` / `_SLOT_ROWS` (4 / 64), `EXL3_MOE_CPU_STAGE_THREADS` (4), `EXL3_MOE_STREAM_T`
(bandwidth-scaled 16), `_STREAM_FUSED_T` (256), `_STREAM_MIN_ROWS` (32), `_STREAM_BATCH_EXPERTS` (24),
`_STREAM_BATCH_RECON` (1), `EXL3_MOE_PINNED_ARENA` (0; production 1; recommended with `stream`),
`EXL3_MOE_ARENA_HUGE` (excluded: reserved hugetlb pages), `EXL3_MOE_ARENA_HUGEPAGE` (1; shmem THP `never` here),
`EXL3_MOE_CPU_SWIZZLE` (1), `EXL3_MOE_CPU_MAX_ISA`, `EXL3_MOE_CPU_PIN` (1), `EXL3_MOE_MEMOPS` (1),
`EXL3_MOE_CPU_SWAP` (+interval / hysteresis, for `hot=` / `split` layers). Swept only if a CPU-expert arm reaches
stage 3.

## 3. Scenarios and the harness additions

### 3.1 Scenario definitions (identical for every arm: same token ids, same order, same generator settings)

Token sources: canonical ids (`bench/data/canonical_ids.npy`, 1,048,560 tokens of one Python code base: the
anchors' text, code domain) and a new prose source `recipe/data/prose_ids.npy` (wikitext-2 train tokenized with the
model's tokenizer, ~2.4M tokens, CPU). Offsets are fixed and disjoint:

| Id | What | Tokens | Generator | Reported |
|---|---|---|---|---|
| S0 | first request after load (the warm-up): canonical @700,000, P = 2,048, G = 512 | fresh | TTFT, decode, hit rate (cold fill round-robin or heat-file seed) |
| S1-32K | novel text: @100,000 and @140,000, P = 32,768, G = 256, no prewarm | fresh each | prefill tok/s = P / TTFT, TTFT, decode tok/s over 256, TPOT p50/p99, hit rate, H2D / D2H MB per token, staged GB, SSD bytes |
| S2-32K | the same request again (the tier runs' r1): the S1 ids once more | fresh (no prefix reuse, full re-prefill) | as S1 |
| S1-128K | novel text @200,000, P = 131,072, G = 256 (this is also S4 at 128K) | fresh | as S1 |
| S3 | one session (one Generator, recurrent cache as tabby): T1 = @400,000 P = 16,384, G = 256; T2-T6 append the previous input + its generated ids + a code delta of 1,024 / 4,096 / 512 / 8,192 / 2,048 tokens (@420,000 onward), G = 256 each | shared | per turn: cached tokens (prefix reuse confirmed), re-prefilled tokens, delta prefill tok/s = delta / TTFT, TTFT, decode tok/s, hit rate, H2D / D2H MB per token |
| S5 | continues S3: T7 = prose delta 4,096 (prose @0), G = 512; T8 = code delta 2,048 (@460,000), G = 256; then S5b: a fresh prose conversation P = 8,192 (prose @100,000), G = 256 | shared, then fresh | per 32-token window: decode tok/s and hit rate; collapse = worst window vs T6's mean; recovery = tokens (and s) until the window stays within 5% (decode) / 1 pp (hit) of T6's mean; T8 = the return cost |
| S4-512K | canonical @0, P = 524,288 (timed), then 3 decode repeats of G = 256 on 64-token suffixes (prefix reuse) | shared | 512K prefill tok/s, decode at ~512K |
| S4-1M | continues S4-512K: extend to P = 1,040,384 (prefix reuse; time the remaining ~516K), 3 decode repeats at ~1M, then the memory / VRAM snapshot | shared | prefill 512K-1M, decode at ~1M, VRAM peak per GPU, nvidia-smi min free, max RSS, cgroup peak, min MemAvailable |

Scenario sets: **H2H-full** = S0, S1-32K x2, S2-32K x2, S1-128K, S3, S5 (~8 min after the load); **H2H-short** = S0,
S1-32K x1, S2-32K x1, S3, S5 (~6 min); **long** = S4-512K (stage 1) or S4-512K + S4-1M (stage 4). At the start of
every process, after the load and before S0, H4 drops the engram tables' pages (so every arm starts with the same
cold engram page cache; no global `drop_caches`). In tabby an identical re-sent prompt would reuse its prefix: S2
deliberately measures the tier-warm full re-prefill (as the r1 runs did); prefix reuse is S3.

### 3.2 Harness additions (CPU-only, before window 0c; each with a `test_harness_cpu.py` case and a `--dry-run` pass)

Directory `recipe/bench/` (copy of the validated harness; `--expect-native-sha 823174af46fb` on new-tree lines,
`4fa4d0749921` on OLD lines).

- **H1** `--prompt-offset N[,N2...]` (one per repeat) and `--prompt-source canonical|prose`: novel-text S1 and the S2
  repeat (same offset twice).
- **H2** `--modes session --session-first 16384@400000 --session-deltas 1024,4096,512,8192,2048@420000
  --session-switch prose:4096@0,canonical:2048@460000 --session-fresh prose:8192@100000`: S3 + S5 on ONE Generator
  (exact token append: previous input ids + generated ids + delta, so the page hashes match as with a chat client
  that re-sends the raw history); per turn `job_cached_tokens`, re-prefilled tokens, recurrent-cache stash count /
  bytes, the tier and engram counters of the turn's prefill and decode.
- **H3** `--timeline 32`: tier snapshot (`tier_snapshot`, host-side counters, no GPU sync; the CPU test checks it
  does not synchronize) and a timestamp every 32 new tokens: windowed decode tok/s and hit rate (S5 collapse /
  recovery, S4 stability).
- **H4** `engram_drop.py`: `posix_fadvise(DONTNEED)` over the byte ranges of both engram layers' tables and scales
  (from the safetensors headers), called by the harness after the load (`--engram-drop`).
- **H5** `--modes longctx --long-prompt 524288[,1040384] --long-decode-repeats 3`: S4 with prefix reuse between the
  two lengths.
- **H6** `recipe/data/prose_ids.npy` + a check that no 64-token window of it occurs in the canonical ids.
- **H7** per-device decode time: over 32 decode tokens after S1 and S2, the GPU busy time per device from
  `torch.profiler` (CUPTI, kernels inside graph replays included), plus CUDA events at the first and last module of
  each device's segment as a fallback: t_cmp and t_pro per layer-token, t_cross, and the fetch kernel's share (t_miss).
- **H8** NVML PCIe throughput (`nvmlDeviceGetPcieThroughput`, TX / RX per GPU, 100 ms samples) per phase: H2D / D2H
  MB per token also for non-tier arms (L0, OLD, stream / cpu arms) and the CMP link's load.
- **T1** `tier_trace_replay.py`: an `EXL3_MOE_TIER_TRACE` JSONL (records `lc`, `mode`, `e` = [[key, kind, count,
  ...]]) replayed through `exllamav3/model/expert_tier_policy.py` on cores 16-23. Routing does not depend on tier
  state (the tier never changes logits), so a replay is an exact counterfactual for hit rate, admissions, H2D / D2H
  copies and starved transients under any pool size and policy. A trace from config a (all 40 layers cached)
  replays ANY split's PRO tier by dropping the CMP layers: predicted hit rates per split x pool (auto, +160
  headroom, +384 single staging) x policy (admit x evict x spare x halflife x heat_prefill x hot x lazy-exclusive),
  per scenario (S1 novel, S2 same text, S3 turns, S5 switch).
- **M1** `make_tables.py` / `stage_tables.py`: per-arm mean, min, max over processes, per scenario; the head-to-head
  table (4.4); **M2** `dnll_ci.py` (exists): 95% moving-block bootstrap of dNLL, top-1 agreement, confident flips.

### 3.3 Measurement protocol

- Window: `ssh <vm> 'cd /root/rnd/release/recipe && (BUDGET_MIN=108 setsid nohup /root/rnd/gpu_window.sh <label>
  /root/rnd/release/recipe/bench/driver.sh <batch> <outdir> >/dev/null 2>&1 </dev/null &)'`; `run_batch.sh`
  defaults (`--power-caps 300,150`, cores 0-15, RAM / temperature guards).
- Idle gating: the service is stopped by the window (no requests); a line whose phases saw foreign load on cores
  0-15 or > 25% on 16-23 is `provisional` and re-queued once (`AUTO_REQUEUE=1`), never tabulated.
- Interleaving: arms in a mirrored order within a window (e.g. a c12 c10 c8 | c8 c10 c12 a); head-to-head repeats
  ABAB across processes.
- Per line: everything in 3.1, plus tier sizing, `tier_ssd_reads_after_load` (must be 0 for family c), engram bytes
  and row-cache hits, max RSS, cgroup peak, min MemAvailable, VRAM peak and nvidia-smi min free per GPU, GPU
  temperature / power.
- A knob wins when both of its processes beat the reference on its target metric by more than max(2%, the
  reference's own spread) and do not lose decode by more than the reference's spread; ties keep the default.

### 3.4 Quality gates (unchanged from version 1)

- 64K teacher-forced line (`--modes quality --quality-tokens 65536`) vs `/root/rnd/proof/n64k_serving_deepseek-index.npz`,
  `/root/rnd/numerics/n64k2_deepseek-index.npz`, W1-c (L0 on this tree:
  `/root/rnd/release/all/val/w1/runs-a/20260926-144016-09-W1-c-quality_quality.npz`) and W1-a (PRO-only tier:
  `.../runs-b/20260926-145856-01-W1-a-quality_quality.npz`).
- Floor F = the spread over function-preserving perturbations (R0 at quality chunk 4096 / 2048 / 2047, R0 vs L0;
  so far across layouts top-1 >= 0.995, |dNLL| <= 0.0007).
- N-class arms: bitwise equal to their layout's reference line. A-class (a new split, chunking, kernels): within F
  (|dNLL| <= F, top-1 >= F, bootstrap CI consistent with 0). L-class: reported as trades with their numbers.
- Split 12 also gets the stable-profile anchors bitwise (layers 12-39 are PRO-computed in both layouts). Splits 8 /
  10 move layers onto the PRO: gated by F, not bitwise.
- Limitation: teacher-forced scoring runs the prefill path; decode-only arithmetic stays at defaults. Pipelined
  prefill = plain prefill in sub-chunk calls: gated at `--quality-chunk` = sub-chunk and by greedy-token equality.

### 3.5 Batch line templates

```
NEW   = BENCH_PYTHON=/root/rnd/release/all/aux/py_memwatch.sh MEMWATCH_PY=/root/rnd/release/ws-tier/bench/py_mem.sh
        PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync PYTHONDONTWRITEBYTECODE=1
OLD   = (as in stage0a.batch) MEMMAX=111G, serving venv, EXL3_DSV41_PLACEMENT='split=12;cpu=12-22' EXL3_MOE_CPU_OFFLOAD=11 ...
L0    = NEW MEMMAX=111G EXL3_PLACEMENT='0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1' EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_CPU_WSLOTS=4
A1M   = NEW MEMMAX=119G CUDA_VISIBLE_DEVICES=0 EXL3_PLACEMENT='*=cuda:0 experts=cache; ram experts=102GiB'     # config a
C<s>  = NEW MEMMAX=119G EXL3_PLACEMENT='0-<s-1>=cuda:0; <s>-39=cuda:1 experts=cache; ram experts=all; disk experts=off'
        EXL3_DSV41_PIPELINE=1 EXL3_DSV41_PIPELINE_CHUNK=4096 | --chunk 4096 --gen-chunk 16384        # s in {2, 8, 10, 12}
C12-S = NEW MEMMAX=119G EXL3_PLACEMENT='0-11=cuda:0; 12-39=cuda:1 experts=stream hot=<fit max>' EXL3_MOE_PINNED_ARENA=1 (+ pipeline)
C12-C = NEW MEMMAX=119G EXL3_PLACEMENT='0-11=cuda:0; 12-39=cuda:1 experts=cpu hot=<fit max>' EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_CPU_WSLOTS=4 (+ pipeline)
common| --cache-tokens 1048576 --max-output-size 32 --warmup-gen 512 --engram-drop --timeline 32 --expect-native-sha 823174af46fb
```

(`run_batch.sh` sets `CUDA_VISIBLE_DEVICES=1,0`; the A1M line overrides it with `0`, as the validated W2 runs did,
so the CMP is not initialized for config a. `EXL3_DSV41_PIPELINE=1` only arms the switch; `--vary pipeline=1/0`
toggles it per repeat.)

## 4. Stage 0: baselines and harness (new GPU time ~3.4 h, 2 windows; window 0a already running)

**0a (running, not re-run):** window `recipe-stage0a` (gpu_window pid 2271596), batch `bench/batches/stage0a.batch`:
OLD vs L0 interleaved x3 at 32K, B-np (numerics x BF16 crossing on L0), B-det, D-rc (recurrent cache 0 vs 4096),
C-legacy-old/new, E-L0 (64K quality bitwise vs W1-c). Results in `stage0/w0a/`; stage 0 waits for it and reuses
them (the OLD-vs-L0 table, the decode-gap decomposition, the numerics / crossing / recurrent-cache costs on L0).

**CPU preparation (no GPU, cores 16-23, ~5 h):** H1-H8, T1, M1 with tests and dry runs; `prose_ids.npy`; the batch
files of 0b-add, 0c and stage 1.

**0b (~80 min, the existing `bench/batches/stage0b.batch` + two lines):** T12 = C12 bring-up at 1M (T-fit: sizing,
VRAM left per GPU, RSS, `ssd_reads` = 0, plain vs pipelined, greedy equality; now with H7 per-device timing),
T-pipe-128k (tier x pipeline, CMP headroom), T12-stable-dsidx / -precise (bitwise anchors), RQ-4096 / -2048 / -2047
(the floor F), optional T12-stable-speed. Added: **C14-fit** (`0-13=cuda:0; 14-39=cuda:1 experts=cache; ram
experts=all; disk experts=off`, load only: the expected CMP refusal is the record for splits 14 / 16; ~5 min) and
**C10-fit / C8-fit** (load + tier sizing only, ~4 min each: pool, RAM tier, RSS, CMP free VRAM, needed for the
1M memory table).

**0c (~100 min, new batch `stage0c.batch`, needs H1-H8):** the baselines in the new scenarios, and config a at 1M:

| Line | What | Est. min |
|---|---|---|
| A1M-fit-1 | config a, 1M cache, H2H-full; prints the tier sizing at 1M (expected: pool ~5,650 slots, 3 GiB below the 64K run, SSD-only ~1,500 experts, RSS 107.6 GiB). **If the 1M load is refused**, the line bisects 786,432 / 524,288 and the plan records the largest cache a can hold | 13 |
| C12-base-1 | C12 (pipeline 4096 / 16384), H2H-full: the harness smoke for tier + session + timeline | 14 |
| L0-base | L0, H2H-full (today's c layout, CPU experts 12-22) | 14 |
| OLD-base | OLD, H2H-full (session modes on the old tree's Generator API; if it lacks one, S3 / S5 are skipped for OLD and noted) | 14 |
| A1M-fit-2 | config a again (second process) | 13 |
| A1M-Q | config a, 64K quality at a 1M cache: must be bitwise equal to W1-a (cache size changes no arithmetic) | 7 |
| A1M-trace | config a, H2H-short with `EXL3_MOE_TIER_TRACE` (capture only, excluded from speed tables; trace gzip'd, < 2 GB) | 12 |
| A1M-S4 | config a, S4-512K (1 process; ~20 GB of SSD reads per 4,096-row pass, ~2.5 TB for the 512K prefill: reads, no wear) | 14 |

After 0c (CPU): T1 replays the a trace for splits 2 / 8 / 10 / 12 x pool variants x policies (~1 h CPU): predicted
decode hit rates and miss bytes per token per scenario; with H7's t_cmp / t_pro / t_miss this predicts each split's
decode (section 1). Gate to stage 1: T12 loads at 1M under 119G with MemAvailable >= 16 GiB, `ssd_reads` = 0, both
stable anchors bitwise, no OOM at 128K, the harness passes on C12 and a. If T12 cannot load: fix the blocker
(`EXL3_MOE_TIER_STAGING=single`, headroom) or report family c as infeasible at 1M with the numbers.

## 5. Stage 1: the family-c layouts head-to-head with config a (~4.3 h, 4.0-4.8, 3 windows)

Arms (all at a 1M cache, same prompts):

| Arm | Placement | Why |
|---|---|---|
| A1M | config a | the reference |
| C12 | `0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off` (+ pipeline 4096 / 16384) | the user's first candidate; densest pool share (54%) |
| C10 | `0-9=cuda:0; 10-39=cuda:1 experts=cache; ...` | 2 fewer CMP layers, CMP room for 8K sub-chunks, replica 10 <- 8 |
| C11 | `0-10=cuda:0; 11-39=cuda:1 experts=cache; ...` (+ pipeline 4096 / 16384, 8192 probe) | added by the user: 1 fewer CMP layer than C12, room for full-size pipeline sub-chunks, replica 11 <- 8 |
| C8 | `0-7=cuda:0; 8-39=cuda:1 experts=cache; ...` | free cut, no replica; the decode-favoured split if t_cmp > t_pro |
| C2 (conditional) | `0-1=cuda:0; 2-39=cuda:1 experts=cache; ...` + `sysmem_recurrent_cache` 1024 | the a-like end: runs only if decode improves 12 -> 10 -> 8 AND its fit probe shows RSS <= 115 GiB |
| C12-plain | C12 without the pipeline, chunk 4096 (what tabby runs today without a patch) | the tabby-as-is reference |
| C12-S | PRO `experts=stream hot=<k>` (static resident slice, CPU_SWAP) | does the dynamic cache beat a static hot slice? |
| C12-C | PRO `experts=cpu hot=<k>` | CPU compute of the cold experts instead of streaming |
| L0, OLD | from 0c | today's references |

| Window | Lines | Est. min |
|---|---|---|
| 1A | H2H-full, mirrored: A1M, C12, C10, C8, C8, C10, C12, A1M (2 processes per arm; with 0c, a has 4) | 100 |
| 1B | 64K quality C10, C8 (gate F); C12-S fit + 1 process, C12-C fit + 1 process, C12-plain 1 process (H2H-short); C2 fit probe (+1 process if its condition holds) | 75-90 |
| 1C | S4-512K for C12, C10, C8 (1 process each); A1M-S4 from 0c; a second A1M / best-c pair if 1A leaves the decision within noise | 40-70 |

Samples per arm: S1-32K 4, S2-32K 4, S1-128K 2, S3 and S5 2 sessions, S4-512K 1 (a: 1 + 0c). Stage-1 winner cB:
the best mean decode over S1 / S2 / S3 (decode first), prefill (S1-32K, S3 delta TTFT) as the tie-break, subject to
`ssd_reads` = 0, the 128 GB envelope with MemAvailable >= 16 GB, quality within F. Output: the first head-to-head
table (a vs every c, per scenario) and the measured t_cmp / t_pro / t_miss model vs the T1 prediction.

## 6. Stage 2: one-factor sweeps on cB, ordered by expected gain (core ~7.6 h, 5 windows; options ~3 h)

Two processes per arm (H2H-short unless noted), mirrored against cB's own processes (the reference). Expected gains
are estimates for ordering, not promises.

| # | Knob(s) | Arms | Cls | Expected | Est. min |
|---|---|---|---|---|---|
| 2.1 | pipeline sub-chunk / generator chunk / load chunk | P2048 / g4096 (tabby as is); P4096 / g16384 (cB); P8192 / g16384 (split <= 10, fit probe first); plain chunk 8192 | A | prefill +10-40% (staging per token ~1/sub-chunk, CMP \|\| PRO overlap); decode ~0 unless the pool shrinks | 68 |
| 2.2 | `EXL3_DSV41_XDEV_BF16` | in-process `--vary xdev_bf16=0/1/1/0` in S1 + one 64K quality line | L | plain prefill +5-12%, pipelined 0-8% (only if the CMP stage is the bottleneck); decode ~0 | 28 |
| 2.3 | engram RAM and io | E0 = cB; E2 `ram ngram=auto`; E3 `ram ngram=auto pagecache=8GiB; disk io=direct` (room per split, section 1) | N | novel-text decode +2-4% (2 gathers per token at 0.27 vs ~0.75 ms), cold prefill +3-8% (layer 1's rows overlap only block 0) | 63 |
| 2.4 | pool and staging | fit probes: `EXL3_MOE_TIER_STAGING=single`, `EXL3_MOE_TIER_HEADROOM_MB=3072` / `2048`, `cuda:1 spare=4` (cache_mode q8 REMOVED: user forbids KV quantization), expandable segments; speed for probes giving >= +2% slots; one S4-512K safety run for a headroom cut | N (q8: L) | decode +1-3% (hit rate); single staging costs prefill | 85 |
| 2.5 | tier policy | T1-ranked top 2 (admit / evict / spare / halflife / heat_prefill / hot pins / lazy-exclusive) on the GPU; heat-file pair (process 1 writes `EXL3_MOE_HEAT_FILE`, process 2 loads it: S0) | N | decode 0-3%, S5 recovery, S0 +20-40% (a: 31 cold vs 43 warm) | 58 |
| 2.6 | recurrent cache | `--vary recurrent_cache_mb=1024/4096/4096/1024` in S3 (0 only as the no-reuse bound, from 0a D-rc) | N | S3 turn TTFT (per-request fixed cost vs checkpoint coverage), RAM | 21 |
| 2.7 | `EXL3_MOE_TIER_PREFILL_ROWS` | 1024, 4096 (S3 plus 128- and 512-token deltas) | N | S3 short-delta TTFT +-; long prefill unaffected | 42 |
| 2.8 | numerics speed | `--vary numerics=deepseek:index/precise/precise/deepseek:index` (report only: precise is a quality trade) | L | decode 0-3% | 21 |
| 2.9 | fairness arm for a | config a at the chunk 2.1 picks for cB (if it is not 4096), fit + 2 processes | A | a's prefill +30-80% at 8192 (it is staging-bound), pool -1-2 GiB | 27 |
| 2.10 (opt.) | host side | `EXL3_MOE_TIER_SPIN_US=0`, `EXL3_MOE_TIER_AFFINITY`, `EXL3_DISK_KEEPALIVE_MS=0` / `2`, `EXL3_DISK_SQPOLL=1`, `EXL3_MOE_TIER_HUGEPAGE=1`, realtime priority (1 process each) | N | <= 2% each | 55 |
| 2.11 (opt.) | CMP decode kernels | `EXL3_MOE_COOP_WIDE=0` / `1`, `EXL3_MOE_COOP_KSPLIT=2` (H7: CMP time) + one quality line | A | t_cmp +-? (the decision variable) | 39 |
| 2.12 (opt.) | GEMM / compressor | `EXL3_HGEMM_FP32_REDUCTION=0`, `EXL3_DSV41_FUSED_COMPRESS=1` (1 process + quality each) | A | ~0 | 36 |
| 2.13 (opt.) | n-gram drafting | Generator `ngram_match_min` 2 / 3, `num_draft_tokens` 4 on S3 (greedy tokens must equal no-draft) | N | S3 decode 0-30% on code, workload-dependent | 21 |
| 2.14 (opt.) | KV q8 | only if 2.4's probe shows >= +2% pool: 2 processes + 64K quality | L | decode +0-1%, NLL cost | 28 |

## 7. Stage 3: combinations (~2.4 h, 2 windows)

Three or four candidates from the stage-2 winners, e.g. **K1** exact (cB + every N-class winner: engram RAM / io,
pool / staging, policy, heat file, recurrent cache, PREFILL_ROWS; pipeline setting deployable on tabby as is);
**K2** K1 with the pipeline setting that needs the tabby generator-chunk change; **K3** K2 + `EXL3_DSV41_XDEV_BF16=1`
(only if 2.2 passed F; a listed quality trade); **K4** the runner-up split with the same winners if stage 1 left it
within noise of cB. Each: 2 processes H2H-full + one 64K quality line (K1 bitwise vs cB at the same chunking; K3
within F). Watch the interactions: sub-chunk x `PREFILL_ROWS` x pool (staging vs transients), engram row cache x
recurrent cache x page cache inside the 119G cgroup.

## 8. Stage 4: confirmation, the final head-to-head at 1M (~6.6 h, 4 windows)

Arms: the top 3 of stage 3 (covering decode-best, prefill-best, balanced) + config a (at the chunk 2.9 chose,
otherwise as validated).

1. H2H-full, 3 processes per arm, interleaved: a K1 K2 K3 | K3 K2 K1 a | a K1 K2 K3 (~150 min).
2. Long: S4-512K + S4-1M, 3 processes for a and the top c, 1 for the other two (~185 min): 512K and 1M prefill,
   decode at ~512K and ~1M, VRAM peak and nvidia-smi min free on both GPUs, max RSS, cgroup peak, min MemAvailable
   (the 1M memory / VRAM headroom table). a at 1M reads ~4.5 TB from the SSD per 1M prefill (reads only).
3. 64K quality per arm (bitwise vs its stage-3 line; vs anchors, F, bootstrap CI) (~28 min).
4. (optional, ~25 min, no deployment) tabby dry run of the winner on port 1921 with a copy of `ds41-exl3.yml`, under
   `systemd-run --scope -p MemoryMax=119G -p MemorySwapMax=0` + memwatch: short chat, a 32K prompt, 4 follow-up
   turns through the real chat template (S3 as a client sees it), a tool call. The production unit is untouched.

### 8.1 Deliverables

`recipe/HEAD2HEAD.md`: per scenario (S0, S1-32K, S1-128K = S4-128K, S2-32K, S3 per turn, S5 collapse / recovery,
S4-512K, S4-1M) the table a vs K1-K3 (and L0 / OLD from stages 0-1): decode tok/s and prefill tok/s as mean
[min-max] of 3, TTFT, hit rate, H2D / D2H MB per token, RSS, cgroup peak, VRAM per GPU; and one plain verdict line
per scenario, e.g. "S2 decode: a 42.1 vs c 33.0 tok/s: a wins by 28%". Config a at 1M is the reference row (or, if
1M was refused, the largest cache a holds, stated). `recipe/RECIPE.md`: the winning placement string, env, tabby
yml deltas, a systemd drop-in proposal (`MemoryMax=119G`, `MemorySwapMax=0`, `EXL3_MOE_HEAT_FILE`,
`EXL3_EXPERT_RAM` cap), the quality table, the memory / VRAM headroom at 1M, every quality trade listed separately.
Nothing is applied to `ds41-exl3.service`.

## 9. GPU-hour estimate

Per-process costs: C12 load ~4 min (CMP weights over x1 dominate; C10 ~3.5, C8 ~3), a ~3, L0 / OLD ~3.7; teardown
and RAM release ~1; H2H-full ~8 min after the load, H2H-short ~6; S4-512K process ~13-14; S4-512K + 1M process
~22-25; 64K quality ~7; fit probe ~4-5; window stop / restart overhead ~9 min each.

| Stage | Content | GPU hours | Windows |
|---|---|---|---|
| 0 | 0a (running, not counted: ~1.6 h); 0b T12 bring-up + fits (~1.4); 0c baselines in S1-S5 + config a at 1M + trace (~1.7); overheads | 3.4 | 2 (+0a) |
| 1 | layouts C12 / C10 / C8 (+C2, C12-plain, C12-S, C12-C) vs a in S1-S5, quality of new splits, S4-512K | 4.3 (4.0-4.8) | 3 |
| 2 | core sweeps 2.1-2.9 | 7.6 | 5 |
| 2 opt. | 2.10-2.14 | ~3.0 | 2 |
| 3 | 3-4 combinations + quality | 2.4 | 2 |
| 4 | top 3 + a: H2H-full x3, S4-512K / 1M, quality (+ optional tabby dry run 0.4) | 6.6 | 4 |
| Total | | ~24 core (~27.5 with options) | 16 (18) |

Endpoint downtime is about the GPU time. CPU work (cores 16-23): ~5 h preparation + ~1 h T1 replay + tables.

## 10. Decision logic for "can c beat a, especially in decode?"

- After 0b / 0c: t_cmp, t_pro, t_miss (H7) and the T1 hit-rate predictions give each split's predicted decode. If
  t_cmp >= ~1.0 ms, c cannot beat a in warm decode at any feasible split (only C2, if it fits, can tie); the plan
  continues (prefill, S3 TTFT and the zero-SSD / RAM-headroom properties still decide the family-c recipe) and says
  so in the stage-1 report instead of waiting for stage 4.
- Stage 1 answers it per scenario with 2-4 samples; stage 4 confirms with 3 processes and the final knobs on both
  sides (a gets the shared serving knobs: chunk, recurrent cache, allocator, engram budget within its envelope).

## 11. Risks and stop conditions

- The C12 CMP is within ~1 GB of full at 1M; the pipeline adds first-GPU memory the autosplit does not reserve (0b
  T-pipe-128k). C10 / C8 relieve it.
- Tier + pipeline never ran together; the tier refuses a layer-mode pass driven from two threads (0b).
- Commits 51 / 52 are experimental (a fetch-kernel race was fixed, 48 loaded runs clean): every family-c line checks
  `ssd_reads` = 0, `device_error` = 0; quality lines are bitwise-gated; any tier error stops the stage.
- Config a at 1M: ~3 GiB less pool than at 64K and ~4.5 TB of SSD reads per 1M prefill; if its 1M load is refused,
  the head-to-head runs at the largest cache it holds, and that is stated as its limit.
- C2 sits at the 128 GB edge (RSS ~116 GiB): fit probe first, memwatch, `#@ minavail=132`; skipped, never forced.
- Hit rates are workload-dependent (one code corpus + prose for S5; profiles do not transfer across domains): the
  report separates novel (S1), same-text (S2) and session (S3) numbers and never merges them.
- Disk: 43 GB free: traces compressed and deleted after T1; no npz beyond the quality lines.
- Abort a window on the thermal guard (85 C), the RAM guards, a memwatch kill or a cgroup OOM; the driver never
  re-runs a line killed for memory.

## 12. Not planned, and why

- Splits 14 / 16 with whole layers (do not fit the CMP: section 1); `experts=cache` / `stream` / `cpu` on the CMP
  (x1 link; CPU-only prefill); the head or hc_expand on the other device.
- `ram ngram=all` / `disk ngram=off` (189 GiB), `policy=inclusive` (> 128 GB), `disk experts=model` for family c
  (expert SSD reads), `ram experts=` below `all` with `disk experts=off` (refused).
- `EXL3_STABLE_ARITHMETIC` or `EXL3_MOE_FUSED_PREFILL` for serving (-23% / -9% prefill measured).
- MTP / speculative draft models: no V4.1 draft model class in dsv41-all.
- `EXL3_MOE_ARENA_HUGE`, shmem THP, power caps, Proxmox storage / NVMe passthrough: system policy (listed for the
  user).
- `EXL3_DSV41_ACTIVATION`, `_ROUTER_BIAS`, `EXL3_FP32_LOGITS`: measured without gain.

## 13. Decisions for the user (not made by this plan)

- The pipeline with 4,096-row sub-chunks needs a generator chunk above the load chunk; tabby passes one
  `chunk_size` to `model.load(max_chunk_size)`, the generator and `max_rq_tokens`. Options: a small tabby option
  (e.g. `generator_chunk_size`) or sub-chunks of 2,048 with `chunk_size: 4096` (2.1 measures both).
- Adopting any L-class knob (BF16 crossing, KV quantization, numerics other than `deepseek:index`).
- If a wins decode: whether the recipe should be family c (zero expert SSD reads, prefill / S3 wins, RAM for engram
  rows) or config a (decode), with the numbers of section 8.1.
- Switching production to the dsv41-all tree and a tier layout (a unit restart with new env and config).
