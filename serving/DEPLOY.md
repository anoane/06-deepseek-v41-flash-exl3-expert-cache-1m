# Deploying the temporary serving configuration (C10-P8X)

**This is a temporary configuration**, picked from the partial recipe search so the endpoint could serve while the
search is paused. It is the best family-c *prefill* measured so far with a good decode, but its BF16 crossing has not
had its 64K quality check, and stage-2 arms with better decode exist (see the README). It will be replaced when the
search ends.

What it is:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0            # cuda:0 = CMP 170HX, cuda:1 = RTX PRO 6000
EXL3_PLACEMENT='0-9=cuda:0; 10-39=cuda:1 experts=cache; ram experts=all; disk experts=off'
EXL3_DSV41_PIPELINE=1  EXL3_DSV41_PIPELINE_CHUNK=8192             # pipelined prefill, 8192-row sub-chunks
EXL3_DSV41_XDEV_BF16=1                                            # BF16 residual crossing (numerics, lossy)
TABBY_EXL3_GENERATOR_CHUNK=32768                                  # 4 sub-chunks per generator call (tabby patch)
tabby: chunk_size 8192, max_seq_len = cache_size = 1048576 (FP16 KV), max_batch_size 1, cuda_malloc_async true
```

The CMP holds layers 0-9 with every expert resident. The PRO runs layers 10-39 with a GPU expert cache over a RAM tier
that holds every other expert; no expert is read from the SSD after the load. The engram tables stay on the SSD.

## Files

| File | Where it goes | What it is |
|---|---|---|
| [`ds41-exl3-c10.yml`](ds41-exl3-c10.yml) | `/root/serve/ds41-exl3-c10.yml` | the TabbyAPI config (`chunk_size: 8192`, 1M, FP16 KV) |
| [`zz-temp-c10.conf`](zz-temp-c10.conf) | `/etc/systemd/system/ds41-exl3.service.d/zz-temp-c10.conf` | the systemd drop-in: clears the unit's environment and command, sets the new tree, the placement and the switches |
| [`ds41-exl3.service`](ds41-exl3.service) | `/etc/systemd/system/ds41-exl3.service` | the base unit (the old production); unchanged; its power caps (`ExecStartPre` 300 / 150 W) still apply |
| [`ds41-exl3.yml`](ds41-exl3.yml) | `/root/serve/ds41-exl3.yml` | the old production config (`chunk_size: 4096`); unchanged; used again on revert |
| [`tabby-generator-chunk.patch`](tabby-generator-chunk.patch) | TabbyAPI checkout | `TABBY_EXL3_GENERATOR_CHUNK`: a generator chunk larger than the load chunk (one hunk; inert when the variable is unset) |
| [`tabby-deepseek_v41-toolformat.patch`](tabby-deepseek_v41-toolformat.patch) | TabbyAPI checkout | `tool_format: deepseek_v41` (the DSML tool-call spelling of V4.1), used by both configs |
| [`ENV.md`](ENV.md) | - | every variable and config field of this deployment, explained |

Paths are this machine's (`/root/rnd/...` for the source trees, `/root/workspace/ds41-serve-venv` for the Python
environment); adapt them.

## Prerequisites

1. **The engine**: the exllamav3 fork, branch `dsv41-all` at `5e55c062`
   (<https://github.com/anoane/exllamav3-anemone/tree/dsv41-all>), with its native extension built for both cards:
   `TORCH_CUDA_ARCH_LIST="8.0;12.0a"` (sm_80 for the CMP, sm_120a for the PRO). This machine uses torch
   2.13.0+cu132, CUDA 13.2, Python 3.12, and puts the source tree and the built extension on `PYTHONPATH` (see the
   drop-in).
2. **The model**: <https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw>, plus DeepSeek's FP8 engram shards
   47 and 48 linked into the same folder and merged into its index (the EXL3 repo does not carry the engram tables;
   see repo 05, `tools/make_engram_index.py` and `tools/link_engram_into_pack.py`). About 393 GiB (422 GB) on disk:
   204 GiB of EXL3 weights and the ~189 GiB engram tables, which are read at runtime, so keep them on a fast NVMe.
3. **TabbyAPI** at `e632af41` (2026-08-13) with both patches: `patch -p1 < tabby-generator-chunk.patch` and
   `patch -p1 < tabby-deepseek_v41-toolformat.patch` in the TabbyAPI checkout.
4. **The CMP 170HX unlocked to 64 GB** (repo 01) and visible as the second GPU. The recipe assumes the x1 link: the
   CMP must hold whole layers (`experts=vram`), never an expert cache.
5. **Host memory**: about 78-81 GiB for the process (72.2 GiB of it pinned) plus page cache for the engram rows. The
   machine has no swap, and pinned memory cannot be reclaimed, so nothing else may need that RAM while it serves.

## Deploy

```bash
# 1. config and drop-in
install -m 644 ds41-exl3-c10.yml /root/serve/ds41-exl3-c10.yml
install -d /etc/systemd/system/ds41-exl3.service.d
install -m 644 zz-temp-c10.conf /etc/systemd/system/ds41-exl3.service.d/zz-temp-c10.conf

# 2. (re)start
systemctl daemon-reload
systemctl restart ds41-exl3.service
journalctl -u ds41-exl3.service -f          # the load takes about 3 minutes
```

### Expected load log (this machine, 2026-09-27; paths shortened)

```
INFO:     Configured backend: cudaMallocAsync
INFO:     exllamav3 version: 1.5.2
 !! Replaced 4 tensors from <model dir>                       (engram q/k weights present in two shards; four !! Overriding lines before it)
INFO:     Using tool format: deepseek_v41
INFO:     Loading model:
INFO:     Loading with autosplit
 -- RAM budget (text): n-gram tables 0 (every row streams from disk); 139.6 GiB available
 -- expert tiers: cuda:1 cache auto (sized at the end of the load, 8 spare, evict=lru admit=adaptive; staging 768 slots (9.5 GiB, double))
    ram experts 0 MiB pinned (static 0 MiB + tier sized at the end of the load (experts=all)), policy exclusive demote=heat, evict=lfu
    ngram 0 MiB (every row streams from disk)
    disk experts=off io=auto: experts on disk only counted at the end of the load; 139.6 GiB of RAM left unpinned
 -- expert tiers: cuda:1 cache 5704 slots (70.5 GiB, 8 spare, evict=lru admit=adaptive; staging 768 slots (9.5 GiB, double))
    ram experts 72.2 GiB pinned (static 0 MiB + tier 5824 slots), policy exclusive demote=heat, evict=lfu
    ngram 0 MiB (every row streams from disk)
    disk experts=off io=auto: 0 experts on disk only; 59.5 GiB of RAM left unpinned
 -- expert cache cuda:1: cold fill of 5696 VRAM and 5824 RAM experts, 142.9 GiB read in 20.5 s; 72.2 GiB of RAM page-locked in 2.8 s; seeded from a round robin across layers (no profile or heat file)
 -- DSV41 placement [0-9=cuda:0; 10-39=cuda:1 experts=cache; ram experts=all; disk experts=off]: cpu embed | cuda:0 (NVIDIA Graphics Device) hc_expand,layers.0-9 | cuda:1 (NVIDIA RTX PRO 6000 Blackwell Workstation Edition) layers.10-39,hc_head,norm,head | replicas 10<-8 | top-k cross 8->10-13 | candidates cross none
 -- DSV41 pipelined prefill: eligible, 12 modules on cuda:0, 33 on cuda:1, sub-chunk 8192
INFO:     Model successfully loaded.
INFO:     Uvicorn running on http://127.0.0.1:1920
```

Check in it: `0 experts on disk only` (no expert SSD reads), `replicas 10<-8` (the split inside kv group 8-13), and
`pipelined prefill: eligible ... sub-chunk 8192`. If the pipeline line says "not pipelinable", the prefill runs plain
(slower, still correct). The CMP shows as "NVIDIA Graphics Device".

### Check it answers

```bash
curl -s http://127.0.0.1:1920/v1/models
```

The service listens on 127.0.0.1 with tabby's own authentication off; this machine's reverse proxy terminates TLS and
checks the bearer token. Do not bind it to 0.0.0.0 with `disable_auth: true`.

### Warm up after every (re)start

The first request after a load was slow on this machine: about **8 tok/s** of decode, against **35.3 tok/s** warm on
a new prompt and **46.7 tok/s** on a repeated one. Send one short request after each restart before real traffic. The
cause was not isolated; the expert cache starts from a round-robin cold fill (the load line above says "no profile or
heat file"), and in the harness a heat file from an earlier run (`EXL3_MOE_HEAT_FILE`, run `C12-HR`) made the first
request 3.5% faster in decode and 12.9% faster in prefill. The heat file is not in this deployment yet.

## Measured

| | This deployment (through TabbyAPI) | The same configuration in the harness (`C10-P8X`, 1 process) |
|---|---|---|
| Load | ~3 min | 169 s |
| Decode | 35.3 tok/s warm, new prompt; 46.7 on a repeated prompt; ~8 on the first request after load | 34.7 (cold 32K), 38.1 (same 32K again), 34.6 (128K), 36.5 (chat turns), 32.9 (topic switch) |
| Prefill | not measured through the API | 1,883 (32K), 1,934 (32K again), 1,918 (128K) tok/s; TTFT 17.4 s at 32K |
| Host memory | RSS 77.6 GiB after the load, 80.9 GiB after 75 min and ~100 requests (RAM tier 72.2 GiB pinned) | peak RSS 77.8 GiB |
| VRAM | PRO 90.5 GB, CMP 52.6 GB used after the load; 94,044 / 56,384 MiB used (nvidia-smi) after 75 min of serving, close to the harness peak | nvidia-smi min free over the run: 3,653 MiB (PRO) / 8,936 MiB (CMP) |
| Quality | the BF16 crossing is a numerics change; its 64K check has not run. The same layout without it: 64K NLL 0.250833 (`Q-C10`), inside the run-to-run floor | |

For comparison, the old production on the same machine: 21.5 tok/s decode, 1,088 tok/s prefill in the harness
(run `OLD-base`).

## Revert to the old production

```bash
rm /etc/systemd/system/ds41-exl3.service.d/zz-temp-c10.conf
systemctl daemon-reload
systemctl restart ds41-exl3.service
```

This goes back to the unit's own environment (the old tree, `EXL3_DSV41_PLACEMENT=split=12;cpu=12-22`,
`EXL3_MOE_CPU_OFFLOAD=11`) and `ds41-exl3.yml` (`chunk_size: 4096`), which the temporary setup never changed. The
tabby generator-chunk patch can stay: with `TABBY_EXL3_GENERATOR_CHUNK` unset it does nothing.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `placement: layers.N does not fit on cuda:0 ...` at load | too many layers or too large a `chunk_size` for the CMP at 1M (split 12 cannot take `chunk_size: 8192`; 13 whole layers never fit) |
| a `ValueError` at load naming `EXL3_DSV41_PIPELINE_CHUNK` and `max_chunk_size` | the sub-chunk is larger than tabby's `chunk_size` |
| a request ends with `OutOfMemoryError ... device 0` during a long prompt | the pipeline's second sub-chunk does not fit on the CMP (seen at split 12 under cudaMallocAsync); use a smaller sub-chunk or a smaller split |
| `placement: cuda:0 reaches host memory at 0.38 GB/s ...` | an expert cache was placed on the CMP; keep its layers `experts=vram` |
| `disk experts=off needs every cached expert in VRAM or RAM` | `ram experts=` smaller than `all`, or not enough host RAM |
| load fails with a host-memory message | something else holds RAM; the tier needs ~72 GiB of pinned memory at this split |
