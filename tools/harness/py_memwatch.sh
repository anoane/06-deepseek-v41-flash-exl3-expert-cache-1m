#!/bin/bash
# py_memwatch.sh: a BENCH_PYTHON interpreter for run_batch.sh that puts memwatch.sh on the
# configuration's own process. It starts memwatch.sh on $$ and then execs the real interpreter, so
# the watched PID becomes the python itself (exec keeps the PID and its start time; taskset and
# "systemd-run --scope" in py_mem.sh exec as well, so the chain stays one process).
#
#   MEMWATCH_PY   the interpreter to exec (default /root/rnd/release/venv/bin/python; use
#                 /root/rnd/release/ws-tier/bench/py_mem.sh to keep the MEMMAX cgroup as well)
#   MEMWATCH_LOG  log file (default <--outdir>/<--run-id>_memwatch.log from the harness arguments)
#   MEMWATCH_MIN_KB / MEMWATCH_GRACE / ...  passed through to memwatch.sh
#
# batch line example:
#   my-config | BENCH_PYTHON=/root/rnd/release/all/aux/py_memwatch.sh MEMWATCH_PY=/root/rnd/release/ws-tier/bench/py_mem.sh MEMMAX=120G EXL3_PLACEMENT='...' | --modes speed ...
set -u
out=""; rid=""; prev=""
for a in "$@"; do
    case "$prev" in
        --outdir) out=$a ;;
        --run-id) rid=$a ;;
    esac
    prev=$a
done
log=${MEMWATCH_LOG:-${out:-/root/rnd/release/all/logs}/${rid:-py-$$}_memwatch.log}
# double fork: the watchdog is not a descendant of the watched process (run_batch.sh would count it as a
# leftover), and $$ inside the subshell is still this shell's PID
( setsid /root/rnd/release/all/aux/memwatch.sh $$ "$log" </dev/null >/dev/null 2>&1 & )
exec "${MEMWATCH_PY:-/root/rnd/release/venv/bin/python}" "$@"
