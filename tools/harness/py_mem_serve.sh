#!/bin/bash
# H5: py_mem.sh with the SERVING venv's Python, for the lines that run the older serving tree (OLD): the harness
# inside a memory cgroup of MEMMAX (default 111G = 119 GB), swap 0. Use it as MEMWATCH_PY behind
# /root/rnd/release/all/aux/py_memwatch.sh, so the watchdog is on the process's own PID (every exec keeps it).
exec systemd-run --scope -q -p MemoryMax=${MEMMAX:-111G} -p MemorySwapMax=0 /root/workspace/ds41-serve-venv/bin/python "$@"
