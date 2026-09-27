#!/bin/bash
# wait_results.sh <outdir> <n> [max_min]: returns when <outdir>/results.jsonl has >= n lines, WINDOW_DONE exists, or after max_min
d=$1; n=$2; max=${3:-120}; t0=$(date +%s)
while :; do
  c=$(wc -l < "$d/results.jsonl" 2>/dev/null || echo 0)
  [ "${c:-0}" -ge "$n" ] && { echo "lines $c"; exit 0; }
  [ -f "$d/WINDOW_DONE" ] && { echo "window done, lines $c"; exit 0; }
  [ $(( $(date +%s) - t0 )) -ge $((max * 60)) ] && { echo "timeout, lines $c"; exit 1; }
  sleep 30
done
