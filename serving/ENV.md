# The temporary deployment, setting by setting

Everything the service runs with, from the drop-in ([`zz-temp-c10.conf`](zz-temp-c10.conf)), the base unit
([`ds41-exl3.service`](ds41-exl3.service)) and the tabby config ([`ds41-exl3-c10.yml`](ds41-exl3-c10.yml)). Details and
measurements for each knob: [../docs/KNOBS.md](../docs/KNOBS.md).

## Environment (drop-in)

The drop-in starts with `Environment=` (empty), which clears everything the base unit set (the old tree's
`EXL3_DSV41_PLACEMENT`, `EXL3_MOE_CPU_OFFLOAD`, `EXL3_MOE_PINNED_ARENA`, `EXL3_MOE_CPU_WSLOTS`), and `ExecStart=`
(empty) before the new command.

| Variable | Value | What it does | Class |
|---|---|---|---|
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | stable GPU numbering | - |
| `CUDA_VISIBLE_DEVICES` | `1,0` | `cuda:0` = CMP 170HX (first stage), `cuda:1` = RTX PRO 6000 | - |
| `CUDA_HOME`, `PATH` | `/usr/local/cuda`, with `/usr/local/cuda/bin` | the toolkit for the JIT parts (Triton) | - |
| `PYTHONPATH` | the `dsv41-all` 5e55c062 source tree, its native build, the TabbyAPI checkout | which engine runs | - |
| `HF_HUB_OFFLINE` | `1` | no hub access at runtime | - |
| `TOKENIZERS_PARALLELISM` | `false` | quiet tokenizer threads | - |
| `EXL3_PLACEMENT` | `0-9=cuda:0; 10-39=cuda:1 experts=cache; ram experts=all; disk experts=off` | CMP layers 0-9 whole; PRO layers 10-39 with the expert cache over a RAM tier of every other expert; no expert SSD reads | bit-exact vs other tier settings; the split itself is last-bit |
| `EXL3_DSV41_PIPELINE` | `1` | pipelined prefill across the split | last-bit |
| `EXL3_DSV41_PIPELINE_CHUNK` | `8192` | sub-chunk rows; must be <= tabby `chunk_size` | last-bit |
| `EXL3_DSV41_XDEV_BF16` | `1` | residual streams cross the x1 link in BF16 | **lossy, quality check pending** |
| `TABBY_EXL3_GENERATOR_CHUNK` | `32768` | generator chunk (the tabby patch): 4 sub-chunks per call | last-bit |
| `PYTORCH_CUDA_ALLOC_CONF` | `backend:cudaMallocAsync` | the allocator (tabby's `cuda_malloc_async: true` sets the same) | bit-exact; expandable segments measured faster (KNOBS 8.1), not deployed yet |

Not set, so at their defaults: `EXL3_DSV41_NUMERICS` (`deepseek:index`), every `EXL3_MOE_TIER_*` / `EXL3_MOE_HEAT_*`
(no heat file), every `EXL3_DISK_*` (`io_uring`), `EXL3_EXPERT_RAM` (no cap), the tier words of the placement
(`cuda:1 cache=auto spare=8 evict=lru admit=adaptive`, `ram policy=exclusive demote=heat evict=lfu ngram=0`,
`disk io=auto`).

## From the base unit (unchanged)

| Setting | Value |
|---|---|
| `ExecStartPre` | `nvidia-smi -i 0 -pl 300`, `nvidia-smi -i 1 -pl 150` (power caps: PRO 300 W, CMP 150 W) |
| `ExecStopPost` | restores 400 / 200 W |
| `WorkingDirectory` | the TabbyAPI checkout |
| `Restart` / `RestartSec` | `on-failure` / 30 s |
| `TimeoutStartSec` / `TimeoutStopSec` | 1200 / 120 s |
| `KillSignal` | `SIGINT` |

No memory cgroup limit is set on the service (the recipe runs used `MemoryMax=119G`, `MemorySwapMax=0`). Adding the
same limit to the drop-in is a proposal of the plan, not applied.

## TabbyAPI config

| Field | Value | Note |
|---|---|---|
| `network.host` / `port` | `127.0.0.1` / `1920` | behind a reverse proxy |
| `network.disable_auth` | `true` | authentication at the proxy; never with a public bind |
| `model.max_seq_len`, `model.cache_size` | `1048576` | the 1M rule; FP16 KV (no `cache_mode`) |
| `model.max_batch_size` | `1` | one recurrent slot; requests queue |
| `model.chunk_size` | `8192` | the load chunk (old production: 4096); split 12 cannot take 8192 |
| `model.gpu_split_auto` | `true` | the placement decides the split; budgets only cap memory |
| `model.tool_format` | `deepseek_v41` | needs the tool-format patch |
| `model.reasoning`, `start_in_reasoning`, `template_vars_default` | `true`, `always`, `enable_thinking: true`, `reasoning_effort: high` | chat template behaviour, unchanged from production |
| `memory.sysmem_recurrent_cache` | `4096` (MiB) | recurrent-state checkpoints for prefix reuse (~15 MB each) |
| `memory.sysmem_kv_cache` | `0` | no host-RAM KV cache |
| `memory.cuda_malloc_async` | `true` | sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` |
