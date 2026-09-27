#!/bin/bash
# wait_idle.sh [max_min]: idle gate before a GPU window. Returns when cores 16-23 were below 25% busy (mean) over
# a 60 s sample, or after max_min minutes (default 60) with a warning. Prints one line per sample.
set -u
MAX=${1:-60}
t0=$(date +%s)
snap() { awk '/^cpu(1[6-9]|2[0-3]) / {b=$2+$3+$4+$7+$8+$9; t=b+$5+$6; B+=b; T+=t} END {print B, T}' /proc/stat; }
while :; do
  read -r b0 t0j < <(snap); sleep 60; read -r b1 t1j < <(snap)
  pct=$(awk -v a="$b0" -v b="$b1" -v c="$t0j" -v d="$t1j" 'BEGIN {printf "%.1f", (d > c) ? 100 * (b - a) / (d - c) : 0}')
  echo "[idle $(date -Is)] cores 16-23 busy ${pct}% over 60 s; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  awk -v p="$pct" 'BEGIN {exit !(p < 25)}' && { echo "[idle] OK"; exit 0; }
  if [ $(( $(date +%s) - t0 )) -ge $((MAX * 60)) ]; then echo "[idle] WARNING: still busy after $MAX min, starting anyway"; exit 0; fi
done
