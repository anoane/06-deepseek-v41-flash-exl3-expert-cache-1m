#!/bin/bash
# Run a batch of dsv41_bench.py configurations sequentially, one process per configuration.
# Meant to run INSIDE one GPU window (the caller wraps it), e.g.
#   setsid nohup /root/rnd/gpu_window.sh my-batch /root/rnd/release/bench/run_batch.sh batches/my.batch >/dev/null 2>&1 &
#
# usage: run_batch.sh <batch file> [options]
#   -t, --timeout SEC        per-config timeout (default 3600); SIGINT, then SIGTERM +90 s, SIGKILL +150 s
#   -o, --out DIR            output directory (default runs/<time>-<batch name>)
#   --power-caps W0,W1|none  power caps set with nvidia-smi -i 0 / -i 1 for the whole batch (default 300,150,
#                            the serving caps); the limits found at start are restored at exit
#   --min-free-gb N          abort the batch when MemAvailable is below N GiB between configs (default 8)
#   --run-min-free-gb N      stop the running config when MemAvailable falls below N GiB (default 4)
#   --max-temp C             stop the running config and abort the batch above C degrees (default 85)
#   --cool-c C               before each config wait until every GPU is at or below C (default 60) ...
#   --cool-max-s S           ... for at most S seconds (default 180)
#   --cores LIST             taskset CPU list for the configs (default 0-15)
#   --dry-run                pass --dry-run to every config (no model, no GPU work: plumbing test)
#   --no-window-check        do not refuse when ds41-exl3.service is active / GPU memory is in use
#
# Batch file: one config per line, `label | env assignments | CLI args`; blank lines and lines starting with #
# are skipped. The env and args fields are shell words (quote values with spaces or ';'). Every config runs
# with a clean environment: `env -i` + BASE below + the line's assignments (which override BASE). Two
# assignments are taken by this script instead of passed on: BENCH_PYTHON=<interpreter> runs that line with
# another Python (e.g. the serving venv for the older serving tree, together with its PYTHONPATH), and
# BENCH_CORES=<list> overrides --cores for that line.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/root/rnd/release/venv/bin/python
BENCH="$HERE/dsv41_bench.py"
# recipe harness: dsv41-all's validated tip be33b0e1 (tree identical to b8105f2d) and its clean native build
SRC=${WS_SRC:-/root/rnd/release/all/src-b8105f2d6a05}
NATIVE=${WS_NATIVE:-/root/rnd/release/all/native-b8105f2d6a05}

TIMEOUT=3600; OUT=""; CAPS="300,150"; MIN_FREE=8; RUN_MIN_FREE=4; MAX_TEMP=85; COOL_C=60; COOL_MAX=180
CORES=0-15; DRY=0; WINCHECK=1; BATCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    -t|--timeout) TIMEOUT=$2; shift 2;;
    -o|--out) OUT=$2; shift 2;;
    --power-caps) CAPS=$2; shift 2;;
    --min-free-gb) MIN_FREE=$2; shift 2;;
    --run-min-free-gb) RUN_MIN_FREE=$2; shift 2;;
    --max-temp) MAX_TEMP=$2; shift 2;;
    --cool-c) COOL_C=$2; shift 2;;
    --cool-max-s) COOL_MAX=$2; shift 2;;
    --cores) CORES=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    --no-window-check) WINCHECK=0; shift;;
    -h|--help) sed -n 2,25p "$0"; exit 0;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) BATCH=$1; shift;;
  esac
done
[ -n "$BATCH" ] && [ -f "$BATCH" ] || { echo "usage: $0 <batch file> [options]" >&2; exit 2; }
BATCH=$(readlink -f "$BATCH")
bname=$(basename "$BATCH"); bname=${bname%.*}
[ -n "$OUT" ] || OUT="$HERE/runs/$(date +%Y%m%d-%H%M%S)-$bname"
mkdir -p "$OUT" "$HERE/results"
RESULTS="$OUT/results.jsonl"
ALL="$HERE/results/all.jsonl"
[ "$DRY" = 1 ] && ALL="$HERE/results/dry_runs.jsonl"
exec > >(tee -a "$OUT/batch.log") 2>&1
cp "$BATCH" "$OUT/batch.txt"

# PYTHONDONTWRITEBYTECODE=1: the source tree under test is a read-only input; without it every import writes
# __pycache__ into it (126 .pyc files landed in src-2740b68 during the harness smoke runs)
BASE=(PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8
      CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
      PYTHONPATH=$SRC:$NATIVE CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0)

log() { echo "[$(date -Is)] $*"; }
max_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'; }
avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }
avail_mib() { awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo; }
gpu_used_each() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%s%d", (NR>1?",":""), $1}'; }
snapshot() {
  log "free -g:"; free -g | sed 's/^/    /'
  log "gpus:"; nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used --format=csv,noheader | sed 's/^/    /'
  log "loadavg $(cat /proc/loadavg)"
}
descendants() {   # all descendants of $1 (children files of every task)
  local p=$1 c
  for c in $(cat /proc/"$p"/task/*/children 2>/dev/null); do echo "$c"; descendants "$c"; done
}

log "BATCH $BATCH -> $OUT (timeout $TIMEOUT s, caps $CAPS, cores $CORES, dry $DRY)"
if [ "$WINCHECK" = 1 ] && [ "$DRY" = 0 ]; then
  if systemctl is-active --quiet ds41-exl3.service; then
    log "REFUSED: ds41-exl3.service is active (run this inside /root/rnd/gpu_window.sh)"; exit 3
  fi
  u=$(gpu_used)
  if [ "$u" -ge 3000 ]; then log "REFUSED: $u MiB of GPU memory in use"; exit 3; fi
fi

PREV_CAPS=""
restore_caps() {
  if [ -n "$PREV_CAPS" ]; then
    local i=0 w
    for w in $PREV_CAPS; do nvidia-smi -i $i -pl "$w" >/dev/null && log "restored power limit gpu $i -> $w W"; i=$((i+1)); done
  fi
}
if [ "$CAPS" != none ] && [ "$DRY" = 0 ]; then
  PREV_CAPS=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%d ", $1}')
  trap restore_caps EXIT
  IFS=, read -r c0 c1 <<<"$CAPS"
  nvidia-smi -i 0 -pl "$c0" >/dev/null && nvidia-smi -i 1 -pl "$c1" >/dev/null && log "power caps set: gpu0 $c0 W, gpu1 $c1 W (were: $PREV_CAPS)"
fi

snapshot
n=0; nfail=0; ABORT=""
while IFS= read -r line <&3 || [ -n "$line" ]; do
  [ -n "$ABORT" ] && break
  case "$(echo "$line" | sed 's/^[[:space:]]*//')" in ''|'#'*) continue;; esac
  IFS='|' read -r label envpart argpart <<<"$line"
  label=$(echo "$label" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  envs=(); cli=()
  if ! eval "envs=( ${envpart:-} )" || ! eval "cli=( ${argpart:-} )"; then
    log "SKIP unparsable line: $line"; nfail=$((nfail+1)); continue
  fi
  lpy=$PY; lcores=$CORES; keep=()
  for e in "${envs[@]}"; do
    case "$e" in
      BENCH_PYTHON=*) lpy=${e#BENCH_PYTHON=};;
      BENCH_CORES=*) lcores=${e#BENCH_CORES=};;
      *) keep+=("$e");;
    esac
  done
  envs=("${keep[@]}")
  if [ ! -x "$lpy" ]; then log "SKIP line with a missing interpreter ($lpy): $line"; nfail=$((nfail+1)); continue; fi
  n=$((n+1))
  safe=$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')
  rid="$(date +%Y%m%d-%H%M%S)-$(printf '%02d' $n)-$safe"
  clog="$OUT/$rid.log"

  if [ "$DRY" = 0 ]; then
    t0=$(date +%s)
    while :; do
      mt=$(max_temp); [ "$mt" -le "$COOL_C" ] && break
      [ $(( $(date +%s) - t0 )) -ge "$COOL_MAX" ] && { log "cool-down: still ${mt}C after ${COOL_MAX}s, starting anyway"; break; }
      sleep 5
    done
  fi
  log "START [$n] $label run_id $rid"
  log "  env: ${envs[*]:-}"
  log "  args: ${cli[*]:-}"
  [ "$lpy" != "$PY" ] && log "  python: $lpy"
  [ "$lcores" != "$CORES" ] && log "  cores: $lcores"
  extra=(); [ "$DRY" = 1 ] && extra=(--dry-run)
  env -i "${BASE[@]}" "${envs[@]}" taskset -c "$lcores" "$lpy" "$BENCH" --label "$label" --run-id "$rid" \
      --results "$RESULTS" --outdir "$OUT" "${cli[@]}" "${extra[@]}" >"$clog" 2>&1 &
  pid=$!
  declare -A seen=()
  start=$(date +%s); kill_at=""; why=""
  # whole-config monitor (load included), from this 2 s loop: minimum MemAvailable, maximum GPU memory in use
  mon_min_av=$(avail_mib); mon_max_gpu=$(gpu_used); mon_max_gpus=$(gpu_used_each)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    for c in $(descendants "$pid"); do
      [ -n "${seen[$c]:-}" ] || seen[$c]=$(tr '\0' ' ' </proc/"$c"/cmdline 2>/dev/null | head -c 300)
    done
    av_now=$(avail_mib); [ "$av_now" -lt "$mon_min_av" ] && mon_min_av=$av_now
    if [ "$DRY" = 0 ]; then
      g_each=$(gpu_used_each); g_now=$(echo "$g_each" | tr ',' '\n' | awk '{s+=$1} END {print s+0}')
      [ "$g_now" -gt "$mon_max_gpu" ] && mon_max_gpu=$g_now
      mon_max_gpus=$(paste -d, <(echo "$mon_max_gpus" | tr ',' '\n') <(echo "$g_each" | tr ',' '\n') | awk -F, '{printf "%s%d", (NR>1?",":""), ($1>$2?$1:$2)}')
    fi
    now=$(date +%s)
    if [ -z "$kill_at" ]; then
      if [ $((now - start)) -ge "$TIMEOUT" ]; then why="timeout ${TIMEOUT}s"
      elif [ "$DRY" = 0 ] && [ "$(max_temp)" -gt "$MAX_TEMP" ]; then why="GPU above ${MAX_TEMP}C ($(max_temp)C)"; ABORT="$why"
      elif [ "$(avail_gb)" -lt "$RUN_MIN_FREE" ]; then why="MemAvailable $(avail_gb) GiB < $RUN_MIN_FREE"; ABORT="$why"
      fi
      if [ -n "$why" ]; then log "STOPPING [$n] $label: $why (SIGINT pid $pid)"; kill -INT "$pid" 2>/dev/null; kill_at=$now; fi
    else
      if [ $((now - kill_at)) -ge 150 ]; then log "SIGKILL pid $pid"; kill -KILL "$pid" 2>/dev/null
      elif [ $((now - kill_at)) -ge 90 ]; then log "SIGTERM pid $pid"; kill -TERM "$pid" 2>/dev/null
      fi
    fi
  done
  wait "$pid"; rc=$?
  # descendants that outlived the config (normally the CPU-MoE worker dies with its parent): by PID, only if
  # the cmdline is still the one recorded while it was our descendant
  for c in "${!seen[@]}"; do
    if kill -0 "$c" 2>/dev/null && [ "$(tr '\0' ' ' </proc/"$c"/cmdline 2>/dev/null | head -c 300)" = "${seen[$c]}" ]; then
      log "leftover descendant $c (${seen[$c]:0:80}): SIGTERM"; kill -TERM "$c" 2>/dev/null; sleep 10
      kill -0 "$c" 2>/dev/null && { log "leftover $c: SIGKILL"; kill -KILL "$c" 2>/dev/null; }
    fi
  done
  unset seen
  # result line: keep the bench's own, or record the failure; mirror to the global results file
  log "  monitor: min MemAvailable $mon_min_av MiB, max GPU memory used $mon_max_gpu MiB (per nvidia-smi index: $mon_max_gpus)"
  "$PY" - "$RESULTS" "$ALL" "$rid" "$label" "$rc" "$clog" "${why:-}" "$mon_min_av" "$mon_max_gpu" "$mon_max_gpus" <<'PYEOF'
import json, sys, os, datetime
res, all_, rid, label, rc, clog, why = sys.argv[1:8]
mon = {"min_mem_available_gib": round(int(sys.argv[8]) / 1024, 2), "max_gpu_used_mib": int(sys.argv[9]),
       "max_gpu_used_mib_each": [int(x) for x in sys.argv[10].split(",") if x.strip()], "period_s": 2}
line = None
if os.path.exists(res):
    for l in open(res):
        try:
            j = json.loads(l)
        except ValueError:
            continue
        if j.get("run_id") == rid:
            line = j
if line is None:
    tail = []
    try:
        tail = [l.rstrip() for l in open(clog, errors = "replace")][-40:]
    except OSError:
        pass
    err = next((l for l in reversed(tail) if "Error" in l or "error" in l or "Killed" in l), tail[-1] if tail else "no output")
    line = {"schema": 1, "run_id": rid, "label": label, "status": "error", "rc": int(rc),
            "error": (f"{why}: " if why else "") + err[:1000], "log": clog, "log_tail": tail[-15:],
            "t_end": datetime.datetime.now().isoformat(timespec = "seconds"), "written_by": "run_batch"}
    with open(res, "a") as f:
        f.write(json.dumps(line) + "\n")
else:
    line["rc"] = int(rc)
    if why:
        line["stopped_by_batch"] = why
line["batch_monitor"] = mon
with open(all_, "a") as f:
    f.write(json.dumps(line) + "\n")
st = line.get("status")
print(f"RESULT {rid}: status {st} rc {rc}" + (f" error {line.get('error')!s:.300}" if st != "ok" else ""))
PYEOF
  [ "$rc" = 0 ] || nfail=$((nfail+1))
  log "END [$n] $label rc=$rc ($(( $(date +%s) - start )) s)"

  if [ "$DRY" = 0 ]; then
    for i in $(seq 60); do [ "$(gpu_used)" -lt 3000 ] && break; sleep 2; done
    [ "$(gpu_used)" -ge 3000 ] && log "WARNING: $(gpu_used) MiB GPU memory still in use after the config exited"
  fi
  snapshot
  mt=$(max_temp); av=$(avail_gb)
  if [ -z "$ABORT" ] && [ "$DRY" = 0 ] && [ "$mt" -gt "$MAX_TEMP" ]; then ABORT="GPU at ${mt}C > ${MAX_TEMP}C"; fi
  if [ -z "$ABORT" ] && [ "$av" -lt "$MIN_FREE" ]; then ABORT="MemAvailable ${av} GiB < ${MIN_FREE} GiB"; fi
done 3<"$BATCH"

[ -n "$ABORT" ] && log "BATCH ABORTED: $ABORT"
log "BATCH DONE: $n configs, $nfail failed; results $RESULTS (mirrored to $ALL)"
[ -z "$ABORT" ] && [ "$nfail" = 0 ]
