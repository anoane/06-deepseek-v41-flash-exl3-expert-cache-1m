#!/bin/bash
# window.sh <label> <master.batch> <outdir>: one recipe GPU window. Idle gate first (outside the window, so the
# endpoint stays up while waiting), then /root/rnd/gpu_window.sh runs driver.sh (budget BUDGET_MIN, default 108)
# with the recipe guards. Launch detached:
#   ssh <vm> '(setsid nohup /root/rnd/release/recipe/bench3/window.sh <label> <batch> <outdir> >/dev/null 2>&1 </dev/null &)'
set -u
B=/root/rnd/release/recipe/bench3
label=${1:?label}; batch=$(readlink -f "${2:?batch}"); out=${3:?outdir}
mkdir -p "$out"
{
  echo "[window $(date -Is)] $label: $batch -> $out"
  "$B/wait_idle.sh" "${IDLE_MAX_MIN:-60}"
  echo "[window $(date -Is)] launching gpu_window"
} >> "$out/window.log" 2>&1
BUDGET_MIN=${BUDGET_MIN:-108} RB_OPTS="--min-free-gb 16 --run-min-free-gb 8 --max-temp 85" \
  /root/rnd/gpu_window.sh "$label" "$B/driver.sh" "$batch" "$out"
rc=$?
echo "[window $(date -Is)] gpu_window rc=$rc; endpoint $(systemctl is-active ds41-exl3.service)" >> "$out/window.log"
touch "$out/WINDOW_DONE"
exit $rc
