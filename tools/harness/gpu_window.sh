#!/bin/bash
# Serialize GPU work on this host.
#   usage: /root/rnd/gpu_window.sh <label> <command> [args...]
# Waits for /tmp/dsv41_gpu.lock, stops ds41-exl3.service (if active) while the command runs, and restores it only
# when no other gpu_window invocation is queued (tokens in /root/rnd/gpu-windows/queue). Exit code = the command's.
# Log: /root/rnd/gpu-windows/<time>-<label>.log (the command's own output goes there too).
set -u
label=${1:?label}; shift
[ $# -gt 0 ] || { echo "usage: $0 <label> <command> [args...]" >&2; exit 2; }
Q=/root/rnd/gpu-windows/queue
mkdir -p "$Q"
tok="$Q/$$-$(date +%s%N)"
safe=$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')
log="/root/rnd/gpu-windows/$(date +%Y%m%d-%H%M%S)-$safe.log"
touch "$tok"
exec 9>/tmp/dsv41_gpu.lock
flock 9
run() {
  echo "START $label $(date -Is) pid $$ cmd: $*"
  if systemctl is-active --quiet ds41-exl3.service; then
    echo "stopping ds41-exl3.service"
    systemctl stop ds41-exl3.service
    touch /root/rnd/gpu-windows/.endpoint_was_active
  fi
  local used=0
  for i in $(seq 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$used" -lt 3000 ] && break
    sleep 2
  done
  echo "GPU MiB used before run: $used"
  "$@"
  local rc=$?
  echo "END $label rc=$rc $(date -Is)"
  rm -f "$tok"
  if [ -f /root/rnd/gpu-windows/HOLD_ENDPOINT_DOWN ]; then
    echo "HOLD_ENDPOINT_DOWN present: endpoint stays down (user request)"
  elif [ -z "$(ls -A "$Q" 2>/dev/null)" ] && [ -f /root/rnd/gpu-windows/.endpoint_was_active ]; then
    echo "queue empty: restoring ds41-exl3.service"
    systemctl start ds41-exl3.service
    for i in $(seq 90); do
      if curl -sf -m 5 http://127.0.0.1:1920/v1/models -o /dev/null; then
        echo "HEALTHY $(date -Is)"
        rm -f /root/rnd/gpu-windows/.endpoint_was_active
        break
      fi
      sleep 10
    done
  else
    echo "other GPU work queued: endpoint stays down"
  fi
  return $rc
}
run "$@" >> "$log" 2>&1
