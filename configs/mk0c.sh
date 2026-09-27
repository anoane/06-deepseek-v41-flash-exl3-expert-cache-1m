#!/bin/bash
# writes /root/rnd/release/recipe/bench2/batches/stage0c.batch
set -eu
B=/root/rnd/release/recipe/bench2/batches/stage0c.batch
MW="BENCH_PYTHON=/root/rnd/release/all/aux/py_memwatch.sh MEMWATCH_PY=/root/rnd/release/recipe/bench/py_mem.sh"
MWS="BENCH_PYTHON=/root/rnd/release/all/aux/py_memwatch.sh MEMWATCH_PY=/root/rnd/release/recipe/bench/py_mem_serve.sh"
AL="PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync PYTHONDONTWRITEBYTECODE=1"
A1M="$MW $AL MEMMAX=119G CUDA_VISIBLE_DEVICES=0 EXL3_PLACEMENT='*=cuda:0 experts=cache; ram experts=102GiB'"
C12="$MW $AL MEMMAX=119G EXL3_PLACEMENT='0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off' EXL3_DSV41_PIPELINE=1 EXL3_DSV41_PIPELINE_CHUNK=4096"
L0="$MW $AL MEMMAX=111G EXL3_PLACEMENT='0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1' EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_CPU_WSLOTS=4"
OLD="$MWS MEMMAX=111G PYTHONPATH=/root/workspace/wt-dsv41-crnd:/root/.cache/torch_extensions/py312_cu132/exllamav3_ext EXL3_DSV41_PLACEMENT='split=12;cpu=12-22' EXL3_MOE_CPU_OFFLOAD=11 EXL3_MOE_PINNED_ARENA=1 EXL3_MOE_CPU_WSLOTS=4 PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
COM="--cache-tokens 1048576 --max-output-size 32 --engram-drop --timeline 32"
NEW_SHA="--expect-native-sha 823174af46fb"; OLD_SHA="--expect-native-sha 4fa4d0749921"
Q="--anchor /root/rnd/release/all/val/w1/runs-b/20260926-145856-01-W1-a-quality_quality.npz --anchor /root/rnd/proof/n64k_serving_deepseek-index.npz --anchor /root/rnd/numerics/n64k2_deepseek-index.npz --anchor /root/rnd/release/all/val/w1/runs-a/20260926-144016-09-W1-c-quality_quality.npz"
cat > "$B" <<EOB
# Recipe stage 0, window 0c (PLAN.md v2 section 4): the baselines in the head-to-head scenarios and config a at 1M.
# Harness: recipe/bench2 = recipe/bench + patch_h2h.py (modes h2h / longctx, engram page drop, PCIe log, H7 device time).
# Tree: dsv41-all be33b0e1 (tree = b8105f2d), src-b8105f2d6a05 + native sha256 823174af46fb; OLD = wt-dsv41-crnd + 4fa4d0749921.
# H2H-full = S0, S1-32K @100000 / @140000, S2-32K of each, S1-128K @200000, S3 (16K + 5 deltas), S5 (prose turn, code turn,
# fresh prose conversation), DT (H7 per-module device time, 32 decode steps on the warm @100000 text). 1M FP16 cache,
# chunk 4096 (C12: pipeline sub-chunk 4096, generator chunk 16384), max_output_size 32, engram pages dropped after the load.
# Memory: every line under py_memwatch (8 GiB floor, kills its own PID) in a swap-0 cgroup: 119G (127.8 GB) for config a
# and family c (a's pinned + resident <= 120 GB by ram experts=102GiB, RSS 107.6 GiB measured), 111G for L0 / OLD.
# minavail = expected RSS + 16 GiB.

#@ est=13 timeout=1500 minavail=124
A1M-1 | $A1M | --modes h2h --scenarios full $COM $NEW_SHA
#@ est=14 timeout=1500 minavail=88
C12-base-1 | $C12 | --modes h2h --scenarios full --chunk 4096 --gen-chunk 16384 $COM $NEW_SHA
#@ est=14 timeout=1500 minavail=75
L0-base | $L0 | --modes h2h --scenarios full $COM --pcie-log 100 $NEW_SHA
#@ est=14 timeout=1500 minavail=75
OLD-base | $OLD | --modes h2h --scenarios full $COM --pcie-log 100 $OLD_SHA
#@ est=13 timeout=1500 minavail=124
A1M-2 | $A1M | --modes h2h --scenarios full $COM $NEW_SHA
#@ est=7 timeout=900 minavail=124
A1M-Q | $A1M | --modes quality --quality-tokens 65536 --quality-chunk 4096 --cache-tokens 1048576 $Q $NEW_SHA
#@ est=12 timeout=1200 minavail=124
A1M-trace | $A1M EXL3_MOE_TIER_TRACE=/root/rnd/release/recipe/stage0/w0c/traces/a1m-short.jsonl | --modes h2h --scenarios short $COM $NEW_SHA
#@ est=16 timeout=1800 minavail=124
A1M-S4 | $A1M | --modes longctx --long-prompt 524288 --long-decode-repeats 3 $COM $NEW_SHA
EOB
echo written "$B"; grep -c '^[A-Z]' "$B"
