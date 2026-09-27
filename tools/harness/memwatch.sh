#!/bin/bash
# memwatch.sh <pid> <logfile>
#
# Host-memory watchdog for ONE process. Every 2 s it reads MemAvailable from
# /proc/meminfo; if it drops below the floor (default 8 GiB) it sends SIGTERM
# to <pid>, waits up to MEMWATCH_GRACE seconds, then SIGKILL. It never signals
# any other process: no pattern matching, no process groups, no children.
# Give it the PID of the process that actually holds the memory (the python
# worker), not a wrapper shell whose child would survive.
#
# The PID is pinned by its start time (/proc/<pid>/stat field 22), so a
# recycled PID is never signalled. The watchdog exits when the process is gone.
#
# Env overrides:
#   MEMWATCH_MIN_KB    floor in kB (default 8388608 = 8 GiB)
#   MEMWATCH_INTERVAL  poll seconds (default 2)
#   MEMWATCH_GRACE     seconds between TERM and KILL (default 10)
#   MEMWATCH_HEARTBEAT seconds between status lines (default 60)
#   MEMWATCH_OOM_ADJ   1 (default): set the target's oom_score_adj to 1000, so if
#                      the kernel OOM killer fires first it picks the run and
#                      not sshd/journald; 0 leaves it alone.
#
# Exit codes: 0 target exited on its own, 3 the watchdog killed it,
#             2 bad arguments / target not running.
set -u

PID="${1:-}"
LOG="${2:-}"
if [[ -z "$PID" || -z "$LOG" || ! "$PID" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 <pid> <logfile>" >&2
    exit 2
fi

MIN_KB="${MEMWATCH_MIN_KB:-8388608}"
INTERVAL="${MEMWATCH_INTERVAL:-2}"
GRACE="${MEMWATCH_GRACE:-10}"
HEARTBEAT="${MEMWATCH_HEARTBEAT:-60}"
OOM_ADJ="${MEMWATCH_OOM_ADJ:-1}"

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') memwatch[$$] pid=$PID $*" >> "$LOG"; }

# start time in clock ticks since boot; the comm field may contain spaces, so
# cut everything up to the last ')' first. Fails for a zombie (state Z): its
# memory is already gone and there is nothing left to kill.
starttime() {
    local s
    s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
    s="${s##*) }"
    # after ')': field 3 is state (word 1), so starttime (field 22) is word 20
    set -- $s
    [[ "$1" != "Z" && "$1" != "X" ]] || return 1
    echo "${20}"
}

same_proc() {
    local now
    now=$(starttime "$PID") || return 1
    [[ "$now" == "$START" ]]
}

mem_avail_kb() { awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo; }
rss_kb() { awk '/^VmRSS:/ {print $2; exit}' "/proc/$PID/status" 2>/dev/null; }
psi() { head -n1 /proc/pressure/memory 2>/dev/null | tr -d '\n'; }

START=$(starttime "$PID") || { echo "memwatch: pid $PID is not running" >&2; log "not running at start"; exit 2; }
CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-200)

# keep the watchdog itself out of the OOM killer's reach
echo -900 > /proc/self/oom_score_adj 2>/dev/null
if [[ "$OOM_ADJ" == "1" ]]; then
    if echo 1000 > "/proc/$PID/oom_score_adj" 2>/dev/null; then oomnote="target oom_score_adj=1000"; else oomnote="target oom_score_adj unchanged"; fi
else
    oomnote="target oom_score_adj unchanged"
fi

log "start floor=${MIN_KB}kB interval=${INTERVAL}s grace=${GRACE}s avail=$(mem_avail_kb)kB $oomnote cmd=[$CMD]"

min_avail=999999999999
last_hb=$(date +%s)
while :; do
    if ! same_proc; then
        log "target exited on its own; min MemAvailable seen ${min_avail}kB"
        exit 0
    fi
    avail=$(mem_avail_kb)
    (( avail < min_avail )) && min_avail=$avail
    now=$(date +%s)
    if (( now - last_hb >= HEARTBEAT )); then
        log "ok avail=${avail}kB min=${min_avail}kB rss=$(rss_kb)kB psi=[$(psi)]"
        last_hb=$now
    fi
    if (( avail < MIN_KB )); then
        log "TRIGGER avail=${avail}kB < floor ${MIN_KB}kB rss=$(rss_kb)kB psi=[$(psi)] cmd=[$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-200)] -> SIGTERM"
        kill -TERM "$PID" 2>/dev/null
        for ((i = 0; i < GRACE; i++)); do
            sleep 1
            same_proc || break
        done
        if same_proc; then
            log "still alive after ${GRACE}s -> SIGKILL"
            kill -KILL "$PID" 2>/dev/null
            for ((i = 0; i < 30; i++)); do
                sleep 1
                same_proc || break
            done
        fi
        if same_proc; then
            log "KILL sent but pid still present after 30s (D state?) avail=$(mem_avail_kb)kB"
        else
            log "killed; avail now $(mem_avail_kb)kB"
        fi
        # marker next to the log so a batch driver can tell the run was cut
        echo "$(date '+%F %T') pid=$PID killed by memwatch at avail=${avail}kB floor=${MIN_KB}kB" >> "${LOG}.KILLED"
        exit 3
    fi
    sleep "$INTERVAL"
done
