#!/bin/bash
# As bench/py_mem.sh (swap-0 memory cgroup of MEMMAX), but runs the bench2 harness (patch_h2h.py: h2h modes, the
# reaped-job exception record) when run_batch.sh of recipe/bench passes bench/dsv41_bench.py: lets a line appended to
# a running bench/ window's requeue file use the bench2 harness. Use as MEMWATCH_PY behind py_memwatch.sh (exec keeps
# the PID the watchdog watches).
a1=$1
[ "$a1" = /root/rnd/release/recipe/bench/dsv41_bench.py ] && { shift; set -- /root/rnd/release/recipe/bench2/dsv41_bench.py "$@"; }
exec systemd-run --scope -q -p MemoryMax=${MEMMAX:-111G} -p MemorySwapMax=0 /root/rnd/release/venv/bin/python "$@"
