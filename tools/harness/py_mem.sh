#!/bin/bash
# The harness's Python (the dsv41-all tree's venv) inside a memory cgroup of MEMMAX (default 111G = 119 GB), swap 0:
# the kernel enforces the RAM budget of a configuration and the load's host-memory check sees it. Use it as
# MEMWATCH_PY behind /root/rnd/release/all/aux/py_memwatch.sh (every exec keeps the PID the watchdog watches).
exec systemd-run --scope -q -p MemoryMax=${MEMMAX:-111G} -p MemorySwapMax=0 /root/rnd/release/venv/bin/python "$@"
