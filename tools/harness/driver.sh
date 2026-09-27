#!/bin/bash
# Driver for one GPU window: runs every config line of a master batch file through run_batch.sh, one call per
# line, all into one output directory, within a time budget.
#   usage: driver.sh <master.batch> <outdir> [--dry-run]
#   env:   BUDGET_MIN (default 108)  minutes from the driver's start after which no config may still be running:
#                                    a config is skipped when elapsed + its estimate would pass it
#          AUTO_REQUEUE (default 1)  append a config whose result is provisional (foreign load on cores 0-15 or on a
#                                    CPU-MoE-pinned core, or cores 16-23 more than 25% busy) or that was skipped for
#                                    RAM to the requeue file, once, as <label>-rerun
#          RB_OPTS                   extra run_batch.sh options for every line (recipe: --min-free-gb 16
#                                    --run-min-free-gb 8 --max-temp 85; power caps default 300,150)
# Recipe additions: a line whose process memwatch killed (<run_id>_memwatch.log.KILLED) or that a memory cgroup OOM
# killed (kernel log since the line started) stops the window: no later line starts, nothing is re-queued.
# Master file = run_batch.sh format. A line "#@ est=<min> timeout=<s> minavail=<GiB> optional=1" applies to the
# next config line. The master file is read as it is run, but never edit it while the driver runs: add configs to
# the requeue file instead.
#
# Requeue file: <outdir>/requeue.batch, created empty at start, APPEND-ONLY (same format as the master, "#@"
# lines included; e.g. `cat more.batch >> <outdir>/requeue.batch`). The driver runs what was appended (it keeps its
# read offset in <outdir>/requeue.offset and never rewrites the file) at two points: before the first master line
# marked optional=1, and after the last master line (repeatedly, until nothing new is left or the budget is spent).
# Skipped configs get a status "skipped" result line; every decision is in <outdir>/driver.log.
set -u
B=$(cd "$(dirname "$0")" && pwd)
MASTER=$(readlink -f "${1:?master batch}"); OUT=${2:?outdir}; DRY=${3:-}
BUDGET_MIN=${BUDGET_MIN:-108}
AUTO_REQUEUE=${AUTO_REQUEUE:-1}
mkdir -p "$OUT/lines"
OUT=$(readlink -f "$OUT")
exec > >(tee -a "$OUT/driver.log") 2>&1
cp "$MASTER" "$OUT/master.batch"
RQ="$OUT/requeue.batch"; RQOFF="$OUT/requeue.offset"
touch "$RQ"; [ -f "$RQOFF" ] || echo 0 > "$RQOFF"
T0=$(date +%s)
log() { echo "[driver $(date -Is)] $*"; }
avail_gib() { awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }
kv() { echo "$1" | tr ' ' '\n' | sed -n "s/^$2=\([0-9]*\)$/\1/p" | head -1; }
skipline() {
  python3 - "$OUT/results.jsonl" "$1" "$2" <<'PYEOF'
import json, sys, datetime
res, label, why = sys.argv[1:4]
with open(res, "a") as f:
    f.write(json.dumps({"schema": 1, "label": label, "status": "skipped", "error": why,
                        "t_end": datetime.datetime.now().isoformat(timespec = "seconds"), "written_by": "driver"}) + "\n")
PYEOF
}
# verdict on the latest result line of a label: "provisional: <why>" / "ok" / "missing"
verdict() {
  python3 - "$OUT/results.jsonl" "$1" <<'PYEOF'
import json, sys
res, label = sys.argv[1:3]
last = None
try:
    for l in open(res):
        try:
            j = json.loads(l)
        except ValueError:
            continue
        if j.get("label") == label:
            last = j
except OSError:
    pass
if last is None:
    print("missing"); sys.exit()
why = []
if last.get("status") == "ok":
    if last.get("provisional"):
        why.append("provisional")
    if last.get("moe_cores_contended"):
        why.append("CPU-MoE core contended")
    if last.get("cores_16_23_contended"):
        why.append("cores 16-23 contended")
print(("provisional: " + ", ".join(why)) if why else last.get("status", "?"))
PYEOF
}
requeue_line() {    # <directive> <line> <reason>: append once, as <label>-rerun
  local dir=$1 line=$2 why=$3 label rlabel
  label=$(echo "${line%%|*}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  case "$label" in *-rerun) log "not re-queueing $label again ($why)"; return;; esac
  rlabel="$label-rerun"
  if grep -qF -- "$rlabel |" "$RQ" 2>/dev/null; then log "$rlabel already queued"; return; fi
  { echo "# auto-requeued $(date -Is): $label $why"; echo "$dir"; echo "$rlabel |${line#*|}"; } >> "$RQ"
  log "REQUEUED $label as $rlabel ($why)"
}

n=0; stop=""
# run one config line; $1 = directive line ("#@ ..."), $2 = config line
run_one() {
  local dir=$1 line=$2 est tmo minav label el av f before rc v
  est=$(kv "$dir" est); tmo=$(kv "$dir" timeout); minav=$(kv "$dir" minavail)
  est=${est:-10}; tmo=${tmo:-1500}; minav=${minav:-0}
  n=$((n+1))
  label=$(echo "${line%%|*}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [ -n "$stop" ]; then
    log "SKIP [$n] $label: $stop"; skipline "$label" "$stop"; return
  fi
  el=$(( ($(date +%s) - T0) / 60 ))
  av=$(avail_gib)
  # the previous config's pinned arena is released a few seconds after its process exits: wait for it before
  # judging the floor
  if [ -z "$DRY" ] && [ "$minav" -gt 0 ]; then
    for w in $(seq 60); do [ "$av" -ge "$minav" ] && break; sleep 2; av=$(avail_gib); done
  fi
  if [ $((el + est)) -gt "$BUDGET_MIN" ]; then
    why="budget: elapsed $el min + estimate $est min > $BUDGET_MIN min"; log "SKIP [$n] $label: $why"; skipline "$label" "$why"
    return
  fi
  if [ -z "$DRY" ] && [ "$minav" -gt 0 ] && [ "$av" -lt "$minav" ]; then
    why="MemAvailable $av GiB < floor $minav GiB before load"; log "SKIP [$n] $label: $why"; skipline "$label" "$why"
    [ "$AUTO_REQUEUE" = 1 ] && requeue_line "$dir" "$line" "(skipped: $why)"
    return
  fi
  f="$OUT/lines/$(printf '%02d' $n)-$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_').batch"
  printf '%s\n' "$line" > "$f"
  before=0; [ -f "$OUT/batch.log" ] && before=$(wc -l < "$OUT/batch.log")
  log "RUN [$n] $label (elapsed $el min, est $est min, timeout $tmo s, MemAvailable $av GiB, loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
  local lstart; lstart=$(date '+%Y-%m-%d %H:%M:%S'); touch "$OUT/.line_start"
  "$B/run_batch.sh" "$f" -t "$tmo" -o "$OUT" ${RB_OPTS:-} $DRY
  rc=$?
  if [ -z "$DRY" ]; then
    local killed oom
    killed=$(find "$OUT" -maxdepth 1 -name '*_memwatch.log.KILLED' -newer "$OUT/.line_start" 2>/dev/null | head -1)
    oom=$(journalctl -k --since "$lstart" --no-pager 2>/dev/null | grep -ciE 'memory cgroup out of memory|oom-kill|killed process' || true)
    if [ -n "$killed" ] || [ "${oom:-0}" -gt 0 ]; then
      stop="not started: memory kill during $label (memwatch marker '${killed:-none}', kernel OOM lines ${oom:-0})"
      log "MEMORY KILL: $stop"
    fi
  fi
  v=$(verdict "$label")
  log "DONE [$n] $label rc=$rc result: $v (elapsed $(( ($(date +%s) - T0) / 60 )) min)"
  if [ -f "$OUT/batch.log" ] && tail -n +$((before + 1)) "$OUT/batch.log" | grep -q "BATCH ABORTED"; then
    stop="not started: batch aborted after $label ($(tail -n +$((before + 1)) "$OUT/batch.log" | grep 'BATCH ABORTED' | tail -1 | sed 's/.*BATCH ABORTED: //'))"
    log "$stop"
  fi
  case "$v" in provisional*) [ "$AUTO_REQUEUE" = 1 ] && requeue_line "$dir" "$line" "($v)";; esac
}
# run what was appended to the requeue file since the last drain (whole lines only: an append still in progress
# is taken next time); each drained span is copied to <outdir>/requeue.chunk.* and run from the copy
drain_requeue() {
  local off new cf dir t line k=0
  while :; do
    off=$(cat "$RQOFF")
    new=$(python3 -c 'import sys; d = open(sys.argv[1], "rb").read(); o = int(sys.argv[2]); i = d.rfind(b"\n", o); print(i + 1 if i >= 0 else o)' "$RQ" "$off")
    [ "$new" -gt "$off" ] || return 0
    k=$((k + 1)); cf="$OUT/requeue.chunk.$(date +%s).$k"
    tail -c +$((off + 1)) "$RQ" | head -c $((new - off)) > "$cf"
    echo "$new" > "$RQOFF"
    log "requeue: running bytes $off..$new of $RQ ($cf)"
    dir=""
    while IFS= read -r line <&4 || [ -n "$line" ]; do
      t=$(echo "$line" | sed 's/^[[:space:]]*//')
      case "$t" in
        '#@'*) dir=$t; continue;;
        ''|'#'*) continue;;
      esac
      run_one "${dir:-#@ est=10}" "$line" </dev/null; dir=""
    done 4<"$cf"
  done
}

log "START master $MASTER -> $OUT, budget $BUDGET_MIN min, auto-requeue $AUTO_REQUEUE, requeue file $RQ, dry '$DRY'"
log "free -g before: $(free -g | awk '/^Mem:/ {print "used", $3, "avail", $7}'); loadavg $(cat /proc/loadavg)"
dir=""; optional_seen=0
while IFS= read -r line <&3 || [ -n "$line" ]; do
  t=$(echo "$line" | sed 's/^[[:space:]]*//')
  case "$t" in
    '#@'*) dir=$t; continue;;
    ''|'#'*) continue;;
  esac
  if [ "$optional_seen" = 0 ] && [ "$(kv "${dir:-}" optional)" = 1 ]; then
    optional_seen=1
    log "first optional config reached: draining the requeue file first"
    drain_requeue
  fi
  run_one "${dir:-#@ est=10}" "$line" </dev/null; dir=""
done 3<"$MASTER"
log "master done: draining the requeue file"
drain_requeue
log "END: $n configs, elapsed $(( ($(date +%s) - T0) / 60 )) min"
[ -z "$stop" ]
