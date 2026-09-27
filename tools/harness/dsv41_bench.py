#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash speed + quality harness (one model load per process).

A configuration is the process environment (EXL3_*, CUDA_*, OMP_*, PYTORCH_* ...) set by the caller plus the
CLI arguments below. Modes (--modes, comma-separated, run in this order after one load):

  speed    generator path, as the serving frontend drives it: Generator + Job, greedy CustomSampler([SS_Argmax()]),
           no stop conditions, max_rq_tokens = generator chunk. Per prompt length P (--prompt-tokens, a comma list)
           and repeat: a fresh Generator, a warmup job on a different region of the canonical text (so the timed
           prompt shares no cached prefix), then the timed job: first P canonical tokens, G greedy tokens.
           Reports TTFT, prefill tok/s, decode tok/s and per-token latency percentiles.
           With --raw (default on) also a raw forward loop per P: model.prefill over P-1 tokens in --chunk
           slices, then G single-token model.forward calls with a host argmax (the engram hashes host ids, so one
           device->host sync per token is inherent there).
  quality  teacher-forced scoring of the first N canonical tokens exactly as the retained numerics drivers
           (numerics64k.py / numerics_len.py) do it: fresh job state, model.forward in --quality-chunk slices with
           {"attn_mode": "flash_attn", "recurrent_states": [state]}, FP32 log_softmax per 512-row slice, top-5 via
           torch.topk (top-1 = its first column), actual-token log-prob gathered. Saves float32 actual_lp,
           int32 top1 and float32 top1_lp to an npz, prints NLL (float64 mean of -actual_lp) and compares with
           each --anchor npz (bitwise counts, first differing position, top-1 agreement).
  legacy   the measurement shape of the retained 2026-09-23 records (bench_speed.py, sha256 3cb36b676d97, preserved
           as /root/rnd/disk-tier/engram/bench_speed.py.copy-20260923): ONE Generator for the whole run
           (recurrent cache 512 MiB), ArgmaxSampler, the vLLM reference prompts (24-token case; the 20000-token
           case repeated for longer contexts) with token 0 replaced by 1000 + k on every job (no prefix reuse), a
           warm-up job (24 tokens, 8 new), prefill jobs (--legacy-prefill, 2 new tokens each), then decode jobs
           (--legacy-decode-ctx, --gen-tokens new tokens). decode tok/s = (n - 1) / (last - first token time),
           steady = the same without the first 8 steps.

--vary KEY=V1/V2/... (repeatable) changes a setting per generator repeat (value = repeat index modulo the list),
inside one loaded process: numerics (config.dsv41_numerics, read on every forward), xdev_bf16 (the BF16 residual
crossing switch, read on every call), recurrent_cache_mb (the Generator's recurrent cache) and pipeline (the
pipelined prefill switch; the generator then gets --gen-chunk when on, --chunk when off). Write the order out
(e.g. A/B/B/A) to balance drift.

Python on the hot path: the generator loop adds one time.perf_counter() and one attribute read per iterate()
call (plus a list append when a token arrives); everything else (probes, /proc reads, nvidia-smi parsing) runs
only at phase boundaries. nvidia-smi samples come from a separate `nvidia-smi -lms` process writing a CSV file,
not from a Python thread. The raw decode loop is a host loop by construction (argmax .item() per token).

One JSON line per process is appended to --results (status ok | error | interrupted). See README.md.
"""
import argparse, datetime, hashlib, json, os, platform, signal, socket, subprocess, sys, threading, time, traceback

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "/root/serve/models/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
DEFAULT_IDS = os.path.join(BENCH_DIR, "data", "canonical_ids.npy")
DEFAULT_LEGACY_REF = "/root/ds41/ref/vllm_ref.json"
ENV_PREFIXES = ("EXL3_", "EXLLAMA", "CUDA_", "OMP_", "MKL_", "PYTORCH_", "TORCH_", "HF_", "TOKENIZERS_")
FOREIGN_PROVISIONAL = 0.5       # average cores of foreign work on cores 0-15 above which a timed phase is PROVISIONAL
MOE_CORE_FOREIGN = 0.15         # foreign cores on ONE CPU-MoE-pinned core above which a timed phase is PROVISIONAL
MOE_RESIDUAL_EXCESS = 0.35      # ... or a pinned core's unexplained busy fraction above its peers' median by this much
CONTENDED_16_23 = 25.0          # mean busy % on cores 16-23 above which a phase shares DRAM bandwidth with other work
RECLAIM_PROVISIONAL_S = 5.0      # kswapd / kcompactd CPU seconds in a timed phase above which it is PROVISIONAL
STALL_FACTOR = 5.0              # a decode step longer than this many times the run's median step is a stall
HARNESS_VERSION = 5             # 2: per-thread CPU attribution, latency lists, tree I/O, load monitor, --vary, legacy
                                # 3 (recipe): cgroup memory, n-gram row cache, tier SSD reads per phase, recurrent cache
                                # 4 (recipe stage 0): h2h / longctx modes (h2h_modes.py), engram page drop, PCIe log
                                # 5 (recipe stage 2): --ngram-draft, accepted draft tokens per job
VARY_KEYS = ("numerics", "xdev_bf16", "recurrent_cache_mb", "pipeline")
GIB = 1024 ** 3


# ----------------------------------------------------------------------------------------------------------------
# host probes (called at phase boundaries only)
# ----------------------------------------------------------------------------------------------------------------

def read_proc_stat():
    """{cpu index: (busy jiffies, total jiffies)} from /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            p = line.split()
            v = [int(x) for x in p[1:9]]          # user nice system idle iowait irq softirq steal
            total = sum(v)
            idle = v[3] + v[4]
            out[int(p[0][3:])] = (total - idle, total)
    return out


def proc_status(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "VmLck", "VmPin", "RssShmem", "RssAnon", "Threads", "Name"):
                    v = v.strip()
                    d[k] = int(v.split()[0]) * 1024 if v.endswith("kB") else v
    except OSError:
        pass
    return d


def proc_cpu_jiffies(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        p = s[s.rindex(")") + 2:].split()
        return int(p[11]) + int(p[12])              # utime + stime (fields 14, 15)
    except (OSError, ValueError, IndexError):
        return 0


def child_pids(pid):
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/children") as f:
                    out += [int(x) for x in f.read().split()]
            except OSError:
                pass
    except OSError:
        pass
    return out


def own_tree():
    """This process and its descendants (the CPU-MoE worker is a spawned child)."""
    seen, todo = [], [os.getpid()]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.append(p)
        todo += child_pids(p)
    return seen


def meminfo():
    keys = ("MemTotal", "MemAvailable", "MemFree", "Cached", "Shmem", "Mlocked", "Unevictable", "AnonHugePages",
            "ShmemHugePages", "HugePages_Total", "HugePages_Free")
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in keys:
                v = v.split()
                d[k] = int(v[0]) * (1024 if len(v) > 1 else 1)
    return d


def cgroup_mem():
    """This process's memory cgroup (v2): memory.current / memory.peak / memory.max in GiB, swap.max, and the
    oom / oom_kill counts of memory.events (None outside a cgroup v2 hierarchy)"""
    try:
        rel = None
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    rel = line[3:].strip()
        if rel is None:
            return None
        base = "/sys/fs/cgroup" + rel
        out = {"path": rel}
        for k in ("memory.current", "memory.peak", "memory.max", "memory.swap.max", "memory.swap.current"):
            try:
                v = open(os.path.join(base, k)).read().strip()
                out[k.split(".", 1)[1]] = None if v == "max" else round(int(v) / GIB, 3)
            except (OSError, ValueError):
                out[k.split(".", 1)[1]] = None
        try:
            for line in open(os.path.join(base, "memory.events")):
                k, v = line.split()
                if k in ("oom", "oom_kill", "max", "high"):
                    out["events_" + k] = int(v)
        except (OSError, ValueError):
            pass
        return out
    except Exception:
        return None


def disk_read_bytes(dev = "sda"):
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if p[2] == dev:
                    return int(p[5]) * 512
    except OSError:
        pass
    return None


def self_io_read_bytes():
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def scan_threads():
    """{tid: (pid, comm, utime + stime jiffies, last CPU)} for every thread on the host (phase boundaries only:
    one /proc/<pid>/task/<tid>/stat read per thread)."""
    out = {}
    try:
        pids = os.listdir("/proc")
    except OSError:
        return out
    for d in pids:
        if not d.isdigit():
            continue
        try:
            tids = os.listdir(f"/proc/{d}/task")
        except OSError:
            continue
        for t in tids:
            try:
                with open(f"/proc/{d}/task/{t}/stat") as f:
                    s = f.read()
                lp, rp = s.index("("), s.rindex(")")
                p = s[rp + 2:].split()
                out[int(t)] = (int(d), s[lp + 1:rp], int(p[11]) + int(p[12]), int(p[36]))
            except (OSError, ValueError, IndexError):
                pass
    return out


def tree_io_faults(pids):
    """Storage bytes read (read_bytes of /proc/<pid>/io, page-cache hits excluded) and page faults, summed over
    pids (this process and its descendants, the CPU-MoE worker included)."""
    rb, have, maj, mnr = 0, False, 0, 0
    for p in pids:
        try:
            with open(f"/proc/{p}/io") as f:
                for line in f:
                    if line.startswith("read_bytes:"):
                        rb += int(line.split()[1])
                        have = True
        except (OSError, ValueError):
            pass
        try:
            with open(f"/proc/{p}/stat") as f:
                s = f.read()
            q = s[s.rindex(")") + 2:].split()
            mnr += int(q[7])
            maj += int(q[9])
        except (OSError, ValueError, IndexError):
            pass
    return {"read": rb if have else None, "majflt": maj, "minflt": mnr}


def pinned_cores(pids):
    """Cores that a thread of pids is pinned to alone (Cpus_allowed_list = one CPU): the CPU-MoE worker pins one
    thread per core (EXL3_MOE_CPU_PIN, cores 0 .. EXL3_MOE_CPU_THREADS-1)."""
    cores = set()
    for p in pids:
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tids:
            try:
                with open(f"/proc/{p}/task/{t}/status") as f:
                    for line in f:
                        if line.startswith("Cpus_allowed_list:"):
                            v = line.split(":", 1)[1].strip()
                            if v.isdigit():
                                cores.add(int(v))
                            break
            except OSError:
                pass
    return sorted(cores)


def parse_cpu_list(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


class HostMonitor:
    """Background sampler for the LOAD phase only (the timed phases have no sampling thread): minimum
    MemAvailable, and the largest RSS each process of this tree reached. Stopped before any measurement."""

    def __init__(self, period = 0.25):
        self.period = period
        self.ev = threading.Event()
        self.th = None
        self.min_avail = None
        self.t_min = None
        self.max_rss = {}
        self.samples = 0
        self.t0 = time.time()

    def _run(self):
        while not self.ev.is_set():
            try:
                av = meminfo().get("MemAvailable")
                if av is not None and (self.min_avail is None or av < self.min_avail):
                    self.min_avail, self.t_min = av, time.time()
                for p in own_tree():
                    st = proc_status(p)
                    r = st.get("VmRSS")
                    if isinstance(r, int) and r > self.max_rss.get(p, (0, None))[0]:
                        self.max_rss[p] = (r, st.get("Name"))
                self.samples += 1
            except Exception:
                pass
            self.ev.wait(self.period)

    def start(self):
        self.th = threading.Thread(target = self._run, name = "bench-load-monitor", daemon = True)
        self.th.start()
        return self

    def stop(self):
        self.ev.set()
        if self.th is not None:
            self.th.join(5)
        return {"samples": self.samples, "period_s": self.period,
                "min_mem_available_gib": None if self.min_avail is None else round(self.min_avail / GIB, 3),
                "min_mem_available_at_s": None if self.t_min is None else round(self.t_min - self.t0, 1),
                "max_rss_gib": [{"pid": p, "name": n, "rss_gib": round(r / GIB, 3)}
                                for p, (r, n) in sorted(self.max_rss.items())]}


def latency_fields(lat):
    """Per-token decode latencies (seconds, one per generated token after the first) -> the latency list in ms and
    the stall flag: steps longer than STALL_FACTOR x the median step."""
    if not lat:
        return {"tpot_ms_list": [], "stall_count": 0, "stall_flag": False}
    arr = np.asarray(lat, dtype = np.float64)
    med = float(np.median(arr))
    thr = STALL_FACTOR * med
    idx = np.nonzero(arr > thr)[0]
    d = {"tpot_ms_list": [round(1000 * float(x), 3) for x in arr],
         "stall_threshold_ms": round(1000 * thr, 3), "stall_count": int(len(idx)), "stall_flag": bool(len(idx)),
         "stalls": [{"step": int(i), "ms": round(1000 * float(arr[i]), 3)} for i in idx[:20]],
         "stall_ms_total": round(1000 * float(arr[idx].sum()), 3) if len(idx) else 0.0}
    rest = np.delete(arr, idx)
    d["decode_tps_excl_stalls"] = round(len(rest) / float(rest.sum()), 3) if len(rest) and rest.sum() > 0 else None
    return d


def host_memory_snapshot():
    tree = own_tree()
    procs = []
    for p in tree:
        st = proc_status(p)
        if st:
            procs.append({"pid": p, "name": st.get("Name"), "rss_gib": round(st.get("VmRSS", 0) / GIB, 3),
                          "hwm_gib": round(st.get("VmHWM", 0) / GIB, 3),
                          "rss_shmem_gib": round(st.get("RssShmem", 0) / GIB, 3),
                          "vmlck_gib": round(st.get("VmLck", 0) / GIB, 3),
                          "vmpin_gib": round(st.get("VmPin", 0) / GIB, 3)})
    mi = meminfo()
    return {"procs": procs,
            "meminfo_gib": {k: round(v / GIB, 3) if not k.startswith("HugePages") else v for k, v in mi.items()}}


# ----------------------------------------------------------------------------------------------------------------
# GPU sampling: one nvidia-smi -lms child writing a CSV; parsed per phase time range at the end
# ----------------------------------------------------------------------------------------------------------------

NVSMI_FIELDS = "timestamp,index,uuid,pci.bus_id,temperature.gpu,power.draw,memory.used,clocks.sm,utilization.gpu,power.limit,memory.total"


class GpuLog:
    def __init__(self, path, period_ms = 500):
        self.path = path
        self.proc = None
        try:
            self.fh = open(path, "w")
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={NVSMI_FIELDS}", "--format=csv,noheader,nounits",
                                          "-lms", str(period_ms)], stdout = self.fh, stderr = subprocess.DEVNULL)
        except Exception as e:
            print(f" !! nvidia-smi logger not started: {e!r}", flush = True)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()               # our own child, by PID
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.fh.close()
        except Exception:
            pass

    def rows(self):
        out = []
        try:
            with open(self.path) as f:
                for line in f:
                    p = [x.strip() for x in line.split(",")]
                    if len(p) < 10:
                        continue
                    try:
                        t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                    except ValueError:
                        continue
                    def num(x):
                        try:
                            return float(x)
                        except ValueError:
                            return None
                    out.append({"t": t, "idx": int(p[1]), "uuid": p[2], "bus": p[3], "temp": num(p[4]),
                                "power": num(p[5]), "mem": num(p[6]), "sm": num(p[7]), "util": num(p[8]),
                                "plimit": num(p[9]), "mtotal": num(p[10]) if len(p) > 10 else None})
        except OSError:
            pass
        return out

    @staticmethod
    def summarize(rows, t0, t1):
        by = {}
        for r in rows:
            if t0 <= r["t"] <= t1:
                by.setdefault(r["idx"], []).append(r)
        out = {}
        for idx, rs in sorted(by.items()):
            def col(k):
                return [r[k] for r in rs if r[k] is not None]
            def mx(k):
                v = col(k)
                return max(v) if v else None
            def mean(k):
                v = col(k)
                return round(sum(v) / len(v), 1) if v else None
            out[str(idx)] = {"samples": len(rs), "temp_max_c": mx("temp"), "power_max_w": mx("power"),
                             "power_mean_w": mean("power"), "mem_max_mib": mx("mem"), "util_mean_pct": mean("util"),
                             "sm_clock_mean_mhz": mean("sm"), "power_limit_w": mx("plimit"),
                             "mem_total_mib": mx("mtotal")}
        return out


# ----------------------------------------------------------------------------------------------------------------
# phase probe
# ----------------------------------------------------------------------------------------------------------------

class Probe:
    """Snapshots at phase start/end: wall time, loadavg, per-core busy, own-tree CPU time, VRAM peaks,
    disk engine / engram prefetch counters. The foreign-load gauge: busy cores on 0-15 (sum of the per-core
    busy fractions) minus the CPU time of this process tree (which runs on 0-15 only: taskset 0-15, and the
    CPU-MoE worker pins its threads to cores 0..threads-1) / wall, as an average number of cores."""

    def __init__(self, ctx):
        self.ctx = ctx

    def _snap(self, end = False):
        c = self.ctx
        tree = own_tree()
        # At the end the per-core counters are read first, the thread scan (tens of ms) after them; at the start
        # the scan comes first, so neither end of the phase window includes the scan itself
        if end:
            s = {"t": time.time(), "pc": time.perf_counter(), "stat": read_proc_stat(),
                 "own": sum(proc_cpu_jiffies(p) for p in tree), "load": os.getloadavg(),
                 "sda": disk_read_bytes(), "io": self_io_read_bytes(), "tree_io": tree_io_faults(tree)}
            s["threads"] = scan_threads()
        else:
            th = scan_threads()
            s = {"threads": th, "tree_io": tree_io_faults(tree), "t": time.time(), "pc": time.perf_counter(),
                 "stat": read_proc_stat(), "own": sum(proc_cpu_jiffies(p) for p in tree), "load": os.getloadavg(),
                 "sda": disk_read_bytes(), "io": self_io_read_bytes()}
        s["own_pids"] = set(tree)
        s["disk"] = c.disk_counters()
        s["pref"] = c.prefetch_counters()
        s["rows"] = c.row_cache_counters()
        s["tier"] = tier_snapshot(c) if c.model is not None else None
        if end:
            s["cg"] = cgroup_mem()
        return s

    def _moe_attribution(self, a, b, busy, wall, hz):
        """Per-thread CPU attribution: every thread's CPU time over the phase, charged to the CPU it last ran on.
        For the CPU-MoE-pinned cores (ctx.moe_cores): foreign cores = other processes' threads charged there
        (direct) and the core's busy fraction minus this process tree's threads charged there (residual; also
        catches foreign threads that exited, softirq time and our own unpinned threads that migrated). A phase is
        contended when a pinned core has direct foreign load > MOE_CORE_FOREIGN, or a residual that exceeds the
        pinned cores' median residual by more than MOE_RESIDUAL_EXCESS (one foreign thread on one pinned core
        stalls every CPU-expert layer). The residual threshold is looser because this process's own unpinned
        threads (the generator loop, torch's pool) are charged to the core they last ran on, which is not
        necessarily where they ran."""
        ta, tb = a["threads"], b["threads"]
        own_pids = a["own_pids"] | b["own_pids"]
        own_c, for_c, foreign = {}, {}, []
        for tid, (pid, comm, cpu, last) in tb.items():
            prev = ta.get(tid)
            d = cpu - (prev[2] if prev is not None and prev[0] == pid else 0)
            if d <= 0:
                continue
            if pid in own_pids:
                own_c[last] = own_c.get(last, 0) + d
            else:
                for_c[last] = for_c.get(last, 0) + d
                foreign.append((d, pid, tid, comm, last))
        foreign.sort(reverse = True)
        denom = hz * wall if wall > 0 else None
        out = {"foreign_threads_top": [{"pid": p, "tid": t, "comm": cm, "cpu_s": round(d / hz, 2), "last_cpu": lc}
                                       for d, p, t, cm, lc in foreign[:8]],
               "foreign_cpu_cores_by_thread": round(sum(for_c.values()) / denom, 3) if denom else None,
               "reclaim_cpu_s": round(sum(d for d, p, t, cm, lc in foreign
                                          if cm.startswith(("kswapd", "kcompactd"))) / hz, 2)}
        moe = list(self.ctx.moe_cores or [])
        out["moe_cores"] = moe
        if not moe or not denom:
            out.update({"moe_core_foreign_direct": None, "moe_core_foreign_residual": None,
                        "moe_core_foreign_max": None, "moe_core_residual_excess": None, "moe_cores_contended": False})
            return out
        direct = [for_c.get(k, 0) / denom for k in moe]
        resid = [(busy[k][0] / busy[k][1] if k in busy and busy[k][1] > 0 else 0.0) - own_c.get(k, 0) / denom
                 for k in moe]
        med = float(np.median(resid))
        excess = [r - med for r in resid]
        out.update({"moe_core_foreign_direct": [round(x, 3) for x in direct],
                    "moe_core_foreign_residual": [round(x, 3) for x in resid],
                    "moe_core_foreign_max": round(max(direct), 3),
                    "moe_core_residual_excess": round(max(excess), 3),
                    "moe_cores_contended": bool(max(direct) > MOE_CORE_FOREIGN or max(excess) > MOE_RESIDUAL_EXCESS)})
        return out

    def start(self):
        c = self.ctx
        if not c.moe_cores and c.model is not None and c.args.moe_cores == "auto":
            # a worker that starts at the first forward instead of during the load
            c.moe_cores = pinned_cores(own_tree())
        if c.torch is not None:
            for i in range(c.torch.cuda.device_count()):
                c.torch.cuda.synchronize(i)
                try:
                    c.torch.cuda.reset_peak_memory_stats(i)
                except Exception:
                    pass
        self.a = self._snap()
        return self

    def stop(self):
        c = self.ctx
        vram = vres = None
        if c.torch is not None:
            vram, vres = [], []
            for i in range(c.torch.cuda.device_count()):
                c.torch.cuda.synchronize(i)
                try:
                    vram.append(round(c.torch.cuda.max_memory_allocated(i) / GIB, 3))
                except Exception:
                    vram.append(None)
                try:
                    vres.append(round(c.torch.cuda.max_memory_reserved(i) / GIB, 3))
                except Exception:
                    vres.append(None)
        b = self._snap(end = True)
        a = self.a
        hz = os.sysconf("SC_CLK_TCK")
        cores = sorted(b["stat"])
        busy = {}
        for k in cores:
            db = b["stat"][k][0] - a["stat"][k][0]
            dt = b["stat"][k][1] - a["stat"][k][1]
            busy[k] = (db, dt)
        pct = [round(100.0 * busy[k][0] / busy[k][1], 1) if busy[k][1] > 0 else None for k in cores]
        low = [k for k in cores if k < 16]
        wall = b["pc"] - a["pc"]
        own = b["own"] - a["own"]
        # Busy cores on 0-15 = sum of each core's busy FRACTION (its busy jiffies / its own jiffy total, i.e. the
        # core_busy_pct values), not busy jiffies / (hz * wall): when the host deschedules the vCPUs the guest loses
        # ticks (per-core jiffy totals fell 4-9 % short of wall time in the contended phases of baseline-2740b68),
        # which made the old formula under-read the foreign load by up to 0.6 cores (0.41 reported vs 0.77 implied
        # by core_busy_pct). own_tree_cpu_s is exact runtime, so the residual bias still under-reads (clean phases
        # read -0.3..+0.05 unclamped; see foreign_cores_0_15_raw).
        busy_low_cores = sum(busy[k][0] / busy[k][1] for k in low if busy[k][1] > 0)
        foreign_raw = busy_low_cores - own / hz / wall if wall > 0 else None
        foreign = None if foreign_raw is None else max(0.0, foreign_raw)
        tick_cov = sum(busy[k][1] for k in low) / (len(low) * hz * wall) if wall > 0 and low else None
        d = {"wall_s": round(wall, 4), "t_start": a["t"], "t_end": b["t"],
             "loadavg_start": [round(x, 2) for x in a["load"]], "loadavg_end": [round(x, 2) for x in b["load"]],
             "core_busy_pct": pct,
             "cores_0_15_busy_mean_pct": round(sum(p for p in pct[:16] if p is not None) / max(1, len(low)), 1),
             "cores_16_23_busy_mean_pct": round(sum(p for p in pct[16:] if p is not None) / max(1, len(pct[16:])), 1)
                                          if len(pct) > 16 else None,
             "own_tree_cpu_s": round(own / hz, 2),
             "foreign_cores_0_15": None if foreign is None else round(foreign, 3),
             "foreign_cores_0_15_raw": None if foreign_raw is None else round(foreign_raw, 3),
             "tick_coverage_0_15": None if tick_cov is None else round(tick_cov, 4),
             "provisional": bool(foreign is not None and foreign > FOREIGN_PROVISIONAL),
             "cores_16_23_contended": bool(len(pct) > 16 and
                                           sum(p for p in pct[16:] if p is not None) / len(pct[16:]) > CONTENDED_16_23),
             "vram_peak_alloc_gib": vram, "vram_peak_reserved_gib": vres,
             "sda_read_mib": None if a["sda"] is None or b["sda"] is None else round((b["sda"] - a["sda"]) / 2**20, 1),
             "proc_read_mib": None if a["io"] is None or b["io"] is None else round((b["io"] - a["io"]) / 2**20, 1)}
        # the whole process tree (the CPU-MoE worker's reads and faults included; a child that exited during the
        # phase drops out of the sum)
        ia, ib = a["tree_io"], b["tree_io"]
        d["tree_read_mib"] = None if ia["read"] is None or ib["read"] is None else round((ib["read"] - ia["read"]) / 2**20, 1)
        d["tree_majflt"] = ib["majflt"] - ia["majflt"]
        d["tree_minflt"] = ib["minflt"] - ia["minflt"]
        d.update(self._moe_attribution(a, b, busy, wall, hz))
        reasons = []
        if foreign is not None and foreign > FOREIGN_PROVISIONAL:
            reasons.append(f"foreign_cores_0_15 {foreign:.2f} > {FOREIGN_PROVISIONAL}")
        if d["moe_cores_contended"]:
            reasons.append(f"CPU-MoE core foreign load: direct max {d['moe_core_foreign_max']} (limit {MOE_CORE_FOREIGN}), "
                           f"residual excess {d['moe_core_residual_excess']} (limit {MOE_RESIDUAL_EXCESS})")
        if (d.get("reclaim_cpu_s") or 0) > RECLAIM_PROVISIONAL_S:
            reasons.append(f"kernel reclaim / compaction {d['reclaim_cpu_s']} s CPU (limit {RECLAIM_PROVISIONAL_S})")
        d["provisional"] = bool(reasons)
        d["provisional_reasons"] = reasons
        if a["disk"] is not None and b["disk"] is not None:
            d["disk_engine"] = {k: b["disk"][k] - a["disk"][k] for k in b["disk"]}
        if a["pref"] is not None and b["pref"] is not None:
            d["engram_prefetch"] = {k: b["pref"][k] - a["pref"][k] for k in b["pref"]}
        if a.get("rows") is not None and b.get("rows") is not None:
            d["ngram_row_cache"] = {k: b["rows"][k] - a["rows"][k] for k in ("hits", "misses")}
            d["ngram_row_cache"].update({"caches": b["rows"]["caches"], "bytes": b["rows"]["bytes"]})
            n = d["ngram_row_cache"]["hits"] + d["ngram_row_cache"]["misses"]
            d["ngram_row_cache"]["hit_rate"] = round(d["ngram_row_cache"]["hits"] / n, 4) if n else None
        ta, tb = a.get("tier"), b.get("tier")
        if ta is not None and tb is not None and "error" not in ta and "error" not in tb:
            d["tier_ssd_reads"] = tb.get("ssd_reads", 0) - ta.get("ssd_reads", 0)
            d["tier_ssd_bytes"] = tb.get("ssd_bytes", 0) - ta.get("ssd_bytes", 0)
        d["cgroup_end"] = b.get("cg")
        return d


# ----------------------------------------------------------------------------------------------------------------
# context: model, cache, identities
# ----------------------------------------------------------------------------------------------------------------

def sha256_file(path, n = None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n] if n else h.hexdigest()


class Ctx:
    def __init__(self, args):
        self.args = args
        self.torch = None
        self.ext = None
        self.model = None
        self.cache = None
        self.config = None
        self.tokenizer = None
        self.moe_cores = []             # CPU-MoE-pinned cores (set after load; see pinned_cores / --moe-cores)

    def disk_counters(self):
        if self.ext is None or self.model is None:
            return None
        try:
            if self.ext.disk_ngram_route() != "engine":
                return None
            s = self.ext.disk_stats(False)
            cl = s["classes"]
            return {"bytes": sum(c["bytes"] for c in cl), "ops": sum(c["ops"] for c in cl),
                    "tickets": sum(c["tickets"] for c in cl), "op_errors": sum(c["op_errors"] for c in cl),
                    "keepalive_reads": s["keepalive_reads"],
                    **{f"class{i}_bytes": c["bytes"] for i, c in enumerate(cl)}}
        except Exception:
            return None

    def row_cache_counters(self):
        """n-gram row cache hits / misses / bytes summed over the engram modules (None without any)"""
        if self.model is None:
            return None
        try:
            tot = {"hits": 0, "misses": 0, "bytes": 0, "caches": 0}
            for b in self.model._blocks():
                e = getattr(b, "engram", None)
                rc = getattr(e, "row_cache", None) if e is not None else None
                if rc is not None:
                    st_ = rc.stats()
                    tot["hits"] += int(st_.get("hits", 0))
                    tot["misses"] += int(st_.get("misses", 0))
                    tot["bytes"] += int(st_.get("bytes", 0))
                    tot["caches"] += 1
            return tot
        except Exception:
            return None

    def prefetch_counters(self):
        if self.model is None:
            return None
        try:
            tot = {"hit": 0, "miss": 0, "retired": 0}
            for b in self.model._blocks():
                e = getattr(b, "engram", None)
                if e is not None and hasattr(e, "prefetch_stats"):
                    for k in tot:
                        tot[k] += int(e.prefetch_stats.get(k, 0))
            return tot
        except Exception:
            return None


def env_snapshot():
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES) or k in ("PYTHONPATH",)}


def build_identity(args):
    d = {"bench_script": os.path.abspath(__file__), "bench_sha256_12": sha256_file(__file__, 12)}
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
    d["pythonpath"] = pp
    try:
        import exllamav3, exllamav3_ext
        src = os.path.dirname(os.path.dirname(os.path.abspath(exllamav3.__file__)))
        so = os.path.abspath(exllamav3_ext.__file__)
        d.update({"source_tree": src, "native_so": so, "native_sha256_12": sha256_file(so, 12),
                  "native_sha256": sha256_file(so)})
        # --no-optional-locks: describe --dirty refreshes the index, which would write into a read-only input's .git
        g = subprocess.run(["git", "--no-optional-locks", "-C", src, "describe", "--always", "--dirty"],
                           capture_output = True, text = True)
        d["source_git"] = g.stdout.strip() or None
        bt = os.path.join(os.path.dirname(so), "BUILD.txt")
        if os.path.exists(bt):
            d["native_build_txt"] = open(bt).read().strip()[:400]
    except Exception as e:
        d["identity_error"] = repr(e)
    return d


def device_table(torch, rows):
    out = []
    first = {}
    for r in rows:
        first.setdefault(r["idx"], r)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        uuid = str(getattr(p, "uuid", "") or "")
        bus = getattr(p, "pci_bus_id", None)
        match = None
        for idx, r in first.items():
            ru = r["uuid"].replace("GPU-", "")
            if (uuid and ru == uuid) or (bus is not None and int(r["bus"].split(":")[1], 16) == bus):
                match = idx
                break
        out.append({"torch_index": i, "name": p.name, "cc": f"{p.major}.{p.minor}", "uuid": uuid or None,
                    "pci_bus_id": bus, "total_gib": round(p.total_memory / GIB, 2), "nvsmi_index": match,
                    "power_limit_w": first[match]["plimit"] if match is not None else None})
    return out


# ----------------------------------------------------------------------------------------------------------------
# speed
# ----------------------------------------------------------------------------------------------------------------

def pct(v, q):
    return float(np.percentile(np.asarray(v, dtype = np.float64), q)) if len(v) else None


TIER_KEYS = ("policy_calls", "policy_hits", "policy_admits", "policy_admits_from_ram", "policy_transients", "policy_starved",
             "policy_retires", "policy_d2h", "policy_drops", "policy_h2d_ram", "policy_ssd", "policy_rescued",
             "policy_refills", "policy_prefill_h2d", "policy_prefill_ssd", "ssd_reads", "ssd_bytes", "h2d_bytes",
             "d2h_bytes", "d2d_bytes", "layer_h2d_bytes", "layer_plans", "layer_calls", "records", "host_records",
             "predictions", "predicted_reads", "slab_to_vram", "prefetch_issued", "prefetch_used", "prefetch_dropped",
             "prefetch_starved", "presubmitted", "slab_waits", "hazard_waits", "refill_reads", "overflow", "max_lag")


def tier_snapshot(ctx):
    """Expert tier counters summed over the cache GPUs (None without a tier)"""
    try:
        t = getattr(ctx.config, "expert_tiers", None)
        if t is None:
            return None
        out = {}
        for s in t.stats().values():
            for k in TIER_KEYS:
                if k in s:
                    out[k] = out.get(k, 0) + int(s[k])
        return out
    except Exception as e:
        return {"error": repr(e)}


def tier_sizing(ctx):
    """The expert tier's sizing as the load decided it (None without a tier)"""
    t = getattr(ctx.config, "expert_tiers", None)
    sz = getattr(t, "sizing", None) if t is not None else None
    if sz is None:
        return None
    try:
        return {"devices": [{"device": d.device, "cached": d.cached, "slots": d.slots, "spare": d.spare,
                             "held": d.held, "staging_slots": d.staging_slots,
                             "pool_gib": round(d.pool_bytes / GIB, 3), "staging_gib": round(d.staging_bytes / GIB, 3)}
                            for d in sz.devices],
                "cached": sz.cached, "hot": sz.hot, "tier_slots": sz.tier_slots,
                "ram_gib": round(sz.ram_bytes / GIB, 3), "static_gib": round(sz.static_bytes / GIB, 3),
                "ngram_gib": round(sz.ngram_bytes / GIB, 3), "slab_slots": sz.slab_slots,
                "slab_gib": round(sz.slab_bytes / GIB, 3), "disk_only": sz.disk_only,
                "left_unpinned_gib": None if sz.left_unpinned is None else round(sz.left_unpinned / GIB, 3),
                "text": sz.text}
    except Exception as e:
        return {"error": repr(e)}


def gpu_free_mib(torch):
    """cudaMemGetInfo free MiB per torch device"""
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            out.append(round(torch.cuda.mem_get_info(i)[0] / 2**20))
        except Exception:
            out.append(None)
    return out


def tier_delta(a, b):
    if a is None or b is None or "error" in a or "error" in b:
        return None
    return {k: b.get(k, 0) - a.get(k, 0) for k in b}


def run_gen_job(ctx, generator, ids, max_new, label, gen_chunk = None):
    """One job through the generator; returns wall stamps. Hot loop: iterate(), perf_counter, attribute read.
    Expert tier counters are read at the enqueue, at the first token and at the end (ctx.last_tier)"""
    from exllamav3 import Job
    from exllamav3.generator.sampler import CustomSampler, SS_Argmax
    torch = ctx.torch
    a = ctx.args
    job = Job(input_ids = torch.as_tensor(ids, dtype = torch.long)[None], max_new_tokens = max_new,
              stop_conditions = [], sampler = CustomSampler([SS_Argmax()]), decode_special_tokens = True,
              max_rq_tokens = gen_chunk or a.gen_chunk, identifier = label)
    stamps = []
    toks = []
    eos = None
    last = 0
    perf = time.perf_counter
    ts0 = tier_snapshot(ctx)
    ts1 = None
    t0 = perf()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        res = generator.iterate()
        t = perf()
        n = job.new_tokens
        if n != last:
            stamps.append((t, n))
            if last == 0 and ts0 is not None:
                ts1 = tier_snapshot(ctx)
            last = n
        for r in res:
            ti = r.get("token_ids")
            if ti is not None:
                toks.append(ti)
            if r.get("eos"):
                eos = r
    if ts0 is not None:
        try:
            for td in ctx.config.expert_tiers.devices.values():
                if td.handle is not None:
                    td.handle.drain()
        except Exception:
            pass
        ts2 = tier_snapshot(ctx)
        ctx.last_tier = {"prefill": tier_delta(ts0, ts1), "decode": tier_delta(ts1, ts2), "total": ts2}
    else:
        ctx.last_tier = None
    return t0, stamps, toks, eos


def gen_metrics(P, G, t0, stamps, toks, eos, torch):
    m = {"prompt_tokens": P, "gen_tokens_requested": G}
    if eos is not None and eos.get("error") is not None:
        err = eos["error"]
        m["job_error"] = f"{type(err).__name__}: {err}"[:2000]
        m["job_error_traceback"] = "".join(traceback.format_exception(err))[-4000:]
        print(f" !! job {eos.get('serial')} failed in the generator: {m['job_error']}\n{m['job_error_traceback']}",
              flush = True)
    if not stamps:
        m["error"] = "no tokens produced" + (f" (job error: {m['job_error']})" if "job_error" in m else "")
        return m
    t_first = stamps[0][0]
    ttft = t_first - t0
    lat = []
    for (ta, na), (tb, nb) in zip(stamps, stamps[1:]):
        lat += [(tb - ta) / (nb - na)] * (nb - na)
    ntok = stamps[-1][1]
    dec_s = stamps[-1][0] - t_first
    m.update({"new_tokens": ntok, "ttft_s": round(ttft, 4), "prefill_tps": round(P / ttft, 2),
              "decode_s": round(dec_s, 4),
              "decode_tps": round((ntok - stamps[0][1]) / dec_s, 3) if dec_s > 0 else None,
              "tpot_ms_mean": round(1000 * float(np.mean(lat)), 3) if lat else None,
              "tpot_ms_p50": round(1000 * pct(lat, 50), 3) if lat else None,
              "tpot_ms_p90": round(1000 * pct(lat, 90), 3) if lat else None,
              "tpot_ms_p99": round(1000 * pct(lat, 99), 3) if lat else None,
              "tpot_ms_max": round(1000 * max(lat), 3) if lat else None})
    m.update(latency_fields(lat))
    if eos is not None:
        for k in ("time_enqueued", "time_prefill", "time_generate", "cached_tokens", "cached_pages", "eos_reason",
                  "accepted_draft_tokens"):
            if k in eos:
                v = eos[k]
                m[f"job_{k}"] = round(v, 4) if isinstance(v, float) else v
        if eos.get("time_prefill"):
            m["job_prefill_tps"] = round(P / eos["time_prefill"], 2)
        if eos.get("time_generate") and eos.get("new_tokens", 0) > 1:
            m["job_decode_tps"] = round((eos["new_tokens"] - 1) / eos["time_generate"], 3)
    if toks:
        ids = torch.cat([t.flatten().cpu() for t in toks]).to(torch.int32).numpy()
        m["tokens_sha256_12"] = hashlib.sha256(ids.tobytes()).hexdigest()[:12]
        m["first_tokens"] = ids[:16].tolist()
        m["_ids"] = ids
    return m


def raw_prefill(ctx, x, n, state):
    """model.prefill over tokens [0, n) of x in --chunk slices with a direct job state (no page table)."""
    for a in range(0, n, ctx.args.chunk):
        b = min(a + ctx.args.chunk, n)
        ctx.model.prefill(x[:, a:b], {"attn_mode": "flash_attn", "recurrent_states": [state]})


def run_prewarm(ctx, ids, P):
    """Untimed raw prefill of the timed prompt: brings its engram rows into the OS page cache (the rows are
    read through the page cache with the io_uring backend) so the generator measurement does not depend on
    which text earlier runs happened to read. No generator exists at this point, so no page table sees it."""
    torch = ctx.torch
    x = torch.as_tensor(ids[:P], dtype = torch.long)[None]
    ctx.cache.reset_states()
    drain(ctx)
    state = ctx.cache.get_new_state()
    try:
        raw_prefill(ctx, x, P, state)
    finally:
        ctx.cache.release_state(state)
    drain(ctx)


def run_raw(ctx, ids, P, G):
    torch = ctx.torch
    model, cache = ctx.model, ctx.cache
    x = torch.as_tensor(ids[:P], dtype = torch.long)[None]
    state = cache.get_new_state()
    out = []
    stamps = []
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        perf = time.perf_counter
        t0 = perf()
        raw_prefill(ctx, x, P - 1, state)
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t_pf = perf()
        tok = x[:, P - 1:P]
        for _ in range(G):
            logits = model.forward(tok, {"attn_mode": "flash_attn", "recurrent_states": [state]})
            nxt = int(torch.argmax(logits[0, -1]).item())
            stamps.append(perf())
            out.append(nxt)
            tok = torch.tensor([[nxt]], dtype = torch.long)
            del logits
    finally:
        cache.release_state(state)
    lat = list(np.diff(stamps))
    ttft = stamps[0] - t0
    dec_s = stamps[-1] - stamps[0]
    arr = np.asarray(out, dtype = np.int32)
    return {**latency_fields(lat), "prompt_tokens": P, "new_tokens": len(out), "prefill_s": round(t_pf - t0, 4),
            "prefill_tps": round((P - 1) / (t_pf - t0), 2), "ttft_s": round(ttft, 4),
            "decode_tps": round((len(out) - 1) / dec_s, 3) if dec_s > 0 else None,
            "tpot_ms_mean": round(1000 * float(np.mean(lat)), 3) if lat else None,
            "tpot_ms_p50": round(1000 * pct(lat, 50), 3) if lat else None,
            "tpot_ms_p90": round(1000 * pct(lat, 90), 3) if lat else None,
            "tpot_ms_max": round(1000 * max(lat), 3) if lat else None,
            "tokens_sha256_12": hashlib.sha256(arr.tobytes()).hexdigest()[:12], "first_tokens": arr[:16].tolist(),
            "_ids": arr}


def drain(ctx):
    """As the retained numerics drivers: finish device work and the engram prefetch before a new state."""
    torch = ctx.torch
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    for block in ctx.model._blocks():
        e = getattr(block, "engram", None)
        if e is not None and hasattr(e, "_drain_prefetch"):
            e._drain_prefetch()
            assert not e._pending and not any(p.held for p in e._pins), "undrained engram state"
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


# ---- per-repeat settings (--vary) ----

def _mod(name):
    import importlib
    return importlib.import_module(name)


def xdev_switch():
    """(module, attribute) of the tree's BF16 residual-crossing switch: architecture/dsv41/placement.XDEV_BF16 in
    the rebuilt tree (default off), modules/dsv41_block._XDEV_BF16 in the older serving tree (default on)."""
    try:
        m = _mod("exllamav3.architecture.dsv41.placement")
        if hasattr(m, "XDEV_BF16"):
            return m, "XDEV_BF16"
    except ImportError:
        pass
    m = _mod("exllamav3.modules.dsv41_block")
    if hasattr(m, "_XDEV_BF16"):
        return m, "_XDEV_BF16"
    raise RuntimeError("this tree has no BF16 residual-crossing switch")


def current_settings(ctx):
    """The per-forward settings as they stand (recorded with every run)."""
    d = {}
    try:
        d["numerics"] = str(getattr(ctx.config, "dsv41_numerics")) if hasattr(type(ctx.config), "dsv41_numerics") else None
    except Exception:
        d["numerics"] = None
    try:
        m, attr = xdev_switch()
        d["xdev_bf16"] = bool(getattr(m, attr))
    except Exception:
        d["xdev_bf16"] = None
    try:
        d["pipeline"] = bool(getattr(_mod("exllamav3.architecture.dsv41.pipeline"), "PIPELINE"))
    except Exception:
        d["pipeline"] = None
    return d


def check_vary(ctx):
    """Refuse a --vary this tree cannot express, before anything is measured."""
    v = ctx.args.vary
    if "numerics" in v and not hasattr(type(ctx.config), "dsv41_numerics"):
        raise ValueError("--vary numerics: this tree has no dsv41_numerics setting (the older serving tree has no "
                         "attention numerics; it computes the precise contract)")
    if "numerics" in v:
        for val in v["numerics"]:
            _mod("exllamav3.architecture.dsv41.numerics").parse(val)
    if "xdev_bf16" in v:
        xdev_switch()
    if "pipeline" in v:
        pl = _mod("exllamav3.architecture.dsv41.pipeline")
        if any(v2 for v2 in v["pipeline"]) and getattr(pl, "SUB_CHUNK", None) is None:
            raise ValueError("--vary pipeline with an on value needs EXL3_DSV41_PIPELINE=1 in the environment (the "
                             "sub-chunk is read, and checked against the load, only when the switch is on at import)")


def apply_vary(ctx, rep):
    """Set this repeat's --vary values; returns (values, generator chunk, recurrent cache MiB)."""
    a = ctx.args
    vals = {k: vs[rep % len(vs)] for k, vs in a.vary.items()}
    gen_chunk, rc_mb = a.gen_chunk, a.recurrent_cache_mb
    for k, val in vals.items():
        if k == "numerics":
            ctx.config.dsv41_numerics = val
        elif k == "xdev_bf16":
            m, attr = xdev_switch()
            setattr(m, attr, bool(val))
        elif k == "pipeline":
            setattr(_mod("exllamav3.architecture.dsv41.pipeline"), "PIPELINE", bool(val))
            gen_chunk = a.gen_chunk if val else a.chunk
        elif k == "recurrent_cache_mb":
            rc_mb = int(val)
    return vals, gen_chunk, rc_mb


def mode_speed(ctx, rec, ids, gpulog_marks):
    from exllamav3 import Generator
    import gc
    torch = ctx.torch
    a = ctx.args
    runs, raws, prewarms = [], [], []
    rec["speed"] = {"prompt_tokens": a.prompt_tokens, "gen_tokens": a.gen_tokens, "chunk": a.chunk,
                    "gen_chunk": a.gen_chunk, "repeats": a.repeats, "warmup_tokens": a.warmup_tokens,
                    "warmup_gen": a.warmup_gen, "warmup_offset": a.warmup_offset,
                    "recurrent_cache_mb": a.recurrent_cache_mb, "prewarm": a.prewarm, "prewarms": prewarms,
                    "vary": a.vary, "max_output_size": a.max_output_size,
                    "serving_output_size": a.max_output_size == 32, "runs": runs, "raw": raws}
    if a.max_output_size != 32:
        print(f" !! speed measured with max_output_size {a.max_output_size}: the serving frontend loads with 32 "
              f"(run speed on its own line to measure the serving configuration)", flush = True)
    check_vary(ctx)
    defaults = current_settings(ctx)
    rec["speed"]["settings_default"] = defaults
    gen_ids = {}
    first = True
    for P in a.prompt_tokens:
        if a.prewarm:
            pr = Probe(ctx).start()
            run_prewarm(ctx, ids, P)
            ph = pr.stop()
            prewarms.append({"prompt_tokens": P, "phase": ph})
            print(f"-- prewarm P={P}: {ph['wall_s']:.1f} s, this process read {ph['proc_read_mib']} MiB from storage",
                  flush = True)
        for rep in range(a.repeats):
            vals, gchunk, rc_mb = apply_vary(ctx, rep)
            generator = Generator(model = ctx.model, cache = ctx.cache, tokenizer = ctx.tokenizer, max_batch_size = 1,
                                  max_chunk_size = gchunk, recurrent_cache_size = rc_mb * 1024**2,
                                  cpu_cache_size = 0, num_draft_tokens = None, ngram_match_min = 0)
            wn = a.warmup_tokens if first else min(a.warmup_tokens, 256)
            wg = a.warmup_gen if first else min(a.warmup_gen, 2)
            wo = a.warmup_offset + (0 if first else 4096 * (len(runs) + 1))
            pr = Probe(ctx).start()
            t0, st, tk, eos = run_gen_job(ctx, generator, ids[wo:wo + wn], wg, "warmup", gchunk)
            wph = pr.stop()
            wm = gen_metrics(wn, wg, t0, st, tk, eos, torch)
            wm.pop("_ids", None)
            eos = tk = st = None            # result dicts reference the job, which references the generator
            print(f"-- warmup P={wn} G={wg}: ttft {wm.get('ttft_s')} s, decode {wm.get('decode_tps')} tok/s", flush = True)
            label = f"gen_P{P}_r{rep}"
            pr = Probe(ctx).start()
            t0, st, tk, eos = run_gen_job(ctx, generator, ids[:P], a.gen_tokens, label, gchunk)
            ph = pr.stop()
            m = gen_metrics(P, a.gen_tokens, t0, st, tk, eos, torch)
            eos = tk = st = None
            m["repeat"] = rep
            m["fresh_generator"] = True
            m["vary"] = vals
            m["settings"] = {**current_settings(ctx), "recurrent_cache_mb": rc_mb, "gen_chunk": gchunk}
            m["warmup"] = {"tokens": wn, "gen": wg, "offset": wo, "ttft_s": wm.get("ttft_s"),
                           "decode_tps": wm.get("decode_tps"), "phase": wph}
            try:
                rcc = getattr(generator, "recurrent_cache", None)
                m["recurrent_cache"] = None if rcc is None else {
                    "entries": len(rcc), "bytes_gib": round(rcc.current_size / GIB, 4),
                    "max_gib": round(rcc.max_size / GIB, 3), **dict(getattr(rcc, "metrics", {}) or {})}
            except Exception as e:
                m["recurrent_cache"] = {"error": repr(e)}
            try:
                cs = generator.get_cache_stats()
                m["cache_stats"] = {k: cs.get(k) for k in ("max_tokens", "used_tokens", "cached_tokens", "alloc_pages",
                                                           "alloc_cached_pages", "hit_rate")}
            except Exception:
                pass
            m["phase"] = ph
            m["tier"] = getattr(ctx, "last_tier", None)
            tr = m["tier"]
            if tr and tr.get("decode") and tr.get("prefill"):
                d, pf = tr["decode"], tr["prefill"]
                used = d.get("policy_hits", 0) + d.get("policy_admits", 0) + d.get("policy_transients", 0)
                ntok = max(1, (m.get("new_tokens") or 1) - 1)
                print(f"== {label} tier: decode hit {d.get('policy_hits', 0) / max(1, used):.4f} "
                      f"({d.get('policy_hits', 0)}/{used}), SSD {d.get('ssd_bytes', 0) / ntok / 1e6:.1f} MB/token, H2D "
                      f"{d.get('h2d_bytes', 0) / ntok / 1e6:.1f} MB/token, D2H {d.get('d2h_bytes', 0) / ntok / 1e6:.1f} "
                      f"MB/token, predicted reads {d.get('predicted_reads', 0)}; prefill staged "
                      f"{(pf.get('layer_h2d_bytes', 0) + pf.get('h2d_bytes', 0)) / 1e9:.1f} GB, SSD "
                      f"{pf.get('ssd_bytes', 0) / 1e9:.2f} GB, layer plans {pf.get('layer_plans', 0)}", flush = True)
            if not vals:
                gen_ids.setdefault(P, m.pop("_ids", None))
            else:
                gen_ids.setdefault((P, json.dumps(vals, sort_keys = True)), m.pop("_ids", None))
            m.pop("_ids", None)
            runs.append(m)
            gpulog_marks.append((label, ph["t_start"], ph["t_end"]))
            print(f"== {label}{' ' + json.dumps(vals) if vals else ''}: TTFT {m.get('ttft_s')} s, prefill "
                  f"{m.get('prefill_tps')} tok/s, decode {m.get('decode_tps')} tok/s (p50 {m.get('tpot_ms_p50')} ms, "
                  f"p90 {m.get('tpot_ms_p90')} ms, stalls {m.get('stall_count')}), cached {m.get('job_cached_tokens')}, "
                  f"read {ph['proc_read_mib']} MiB (tree {ph.get('tree_read_mib')}, majflt {ph.get('tree_majflt')}), "
                  f"foreign {ph['foreign_cores_0_15']}, MoE-core foreign {ph.get('moe_core_foreign_max')}, 16-23 busy "
                  f"{ph['cores_16_23_busy_mean_pct']}%, tier SSD reads {ph.get('tier_ssd_reads')}, rows "
                  f"{ph.get('ngram_row_cache')}, cgroup {(ph.get('cgroup_end') or {}).get('current')} GiB"
                  f"{' PROVISIONAL' if ph['provisional'] else ''}", flush = True)
            first = False
            del generator
            gc.collect()
    # back to the process's own settings for the raw loop and any later mode
    if a.vary:
        if "numerics" in a.vary:
            ctx.config.dsv41_numerics = defaults["numerics"]
        if "xdev_bf16" in a.vary:
            m_, attr = xdev_switch()
            setattr(m_, attr, defaults["xdev_bf16"])
        if "pipeline" in a.vary:
            setattr(_mod("exllamav3.architecture.dsv41.pipeline"), "PIPELINE", defaults["pipeline"])
    if a.raw:
        ctx.cache.reset_states()
        for P in a.prompt_tokens:
            drain(ctx)
            label = f"raw_P{P}"
            pr = Probe(ctx).start()
            m = run_raw(ctx, ids, P, a.gen_tokens)
            ph = pr.stop()
            rid = m.pop("_ids")
            g = gen_ids.get(P)
            if g is not None:
                n = min(len(g), len(rid))
                neq = np.nonzero(g[:n] != rid[:n])[0]
                m["greedy_prefix_equal_to_generator"] = int(neq[0]) if len(neq) else n
            m["settings"] = current_settings(ctx)
            m["phase"] = ph
            raws.append(m)
            gpulog_marks.append((label, ph["t_start"], ph["t_end"]))
            print(f"== {label}: prefill {m['prefill_tps']} tok/s, TTFT {m['ttft_s']} s, decode {m['decode_tps']} tok/s "
                  f"(p50 {m['tpot_ms_p50']} ms, stalls {m.get('stall_count')}), foreign {ph['foreign_cores_0_15']}, "
                  f"MoE-core foreign {ph.get('moe_core_foreign_max')}{' PROVISIONAL' if ph['provisional'] else ''}",
                  flush = True)
    summ = {}
    for P in a.prompt_tokens:
        keys = []
        for r in runs:
            k = json.dumps(r.get("vary") or {}, sort_keys = True)
            if r["prompt_tokens"] == P and k not in keys:
                keys.append(k)
        for k in keys:
            rs = [r for r in runs if r["prompt_tokens"] == P and json.dumps(r.get("vary") or {}, sort_keys = True) == k
                  and "prefill_tps" in r]
            if not rs:
                continue
            clean = [r for r in rs if not r["phase"]["provisional"]]
            e = {"n": len(rs), "n_clean": len(clean),
                 "prefill_tps_median": round(float(np.median([r["prefill_tps"] for r in rs])), 2),
                 "ttft_s_median": round(float(np.median([r["ttft_s"] for r in rs])), 4),
                 "decode_tps_median": round(float(np.median([r["decode_tps"] for r in rs if r["decode_tps"]])), 3)
                 if any(r["decode_tps"] for r in rs) else None,
                 "decode_tps_median_clean": round(float(np.median([r["decode_tps"] for r in clean if r["decode_tps"]])), 3)
                 if any(r["decode_tps"] for r in clean) else None,
                 "tpot_ms_p50_median": round(float(np.median([r["tpot_ms_p50"] for r in rs if r["tpot_ms_p50"]])), 3)
                 if any(r["tpot_ms_p50"] for r in rs) else None,
                 "stalls": sum(r.get("stall_count", 0) for r in rs),
                 "provisional": any(r["phase"]["provisional"] for r in rs),
                 "cores_16_23_contended": any(r["phase"].get("cores_16_23_contended") for r in rs)}
            if k == "{}":
                summ[str(P)] = e
            else:
                summ[f"{P} {k}"] = dict(e, vary = json.loads(k))
    rec["speed"]["summary"] = summ


# ---- the retained records' measurement shape ----

def legacy_prompts(a):
    """bench_speed.py's prompts: the 24-token vLLM reference case, and the 20000-token case repeated up to the
    longest length asked for."""
    ref = json.load(open(a.legacy_ref))
    base = next(c for c in ref["cases"] if c["n"] == 20000)["prompt_ids"]
    short = next(c for c in ref["cases"] if c["n"] == 24)["prompt_ids"]
    need = max(a.legacy_prefill + a.legacy_decode_ctx + [24])
    long_ids = (base * (need // len(base) + 1))[:need]
    return short, long_ids


def mode_legacy(ctx, rec, ids_unused, gpulog_marks):
    """bench_speed.py (the retained 2026-09-23 records) as a mode: one shared Generator, ArgmaxSampler, token 0
    salted per job, warm-up (24 tokens, 8 new), prefill jobs (2 new tokens), decode jobs. Timing as there: TTFT =
    enqueue -> first token, decode = (n - 1) / (last - first token time), steady = without the first 8 steps."""
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler
    torch = ctx.torch
    a = ctx.args
    short, long_ids = legacy_prompts(a)
    rc_mb = a.recurrent_cache_mb
    gen = Generator(model = ctx.model, cache = ctx.cache, tokenizer = ctx.tokenizer, max_batch_size = 1,
                    max_chunk_size = a.gen_chunk, recurrent_cache_size = rc_mb * 1024**2)
    salt = [0]
    out = {"ref": a.legacy_ref, "recurrent_cache_mb": rc_mb, "gen_chunk": a.gen_chunk, "shared_generator": True,
           "settings": current_settings(ctx), "prefill": [], "decode": []}
    rec["legacy"] = out

    def run(ids, n_new, label):
        salt[0] += 1
        ids = list(ids)
        ids[0] = 1000 + salt[0]         # as bench_speed.py: defeats prefix reuse between jobs
        job = Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = n_new,
                  sampler = ArgmaxSampler(), stop_conditions = [])
        pr = Probe(ctx).start()
        stamps, last = [], None
        perf = time.perf_counter
        t_enq = perf()
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                t = r.get("token_ids")
                if t is not None and t.numel():
                    stamps += [perf()] * t.numel()
                if r.get("eos"):
                    last = r
        ph = pr.stop()
        gpulog_marks.append((label, ph["t_start"], ph["t_end"]))
        m = {"label": label, "prompt_tokens": len(ids), "new_tokens": len(stamps), "salt_token": ids[0],
             "ttft_s": round(stamps[0] - t_enq, 4) if stamps else None}
        if len(stamps) > 1:
            m["decode_tps"] = round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 3)
        if len(stamps) > 16:
            m["steady_tps"] = round((len(stamps) - 9) / (stamps[-1] - stamps[8]), 3)
        lat = list(np.diff(stamps)) if len(stamps) > 1 else []
        if lat:
            m.update({"tpot_ms_p50": round(1000 * pct(lat, 50), 3), "tpot_ms_p90": round(1000 * pct(lat, 90), 3),
                      "tpot_ms_max": round(1000 * max(lat), 3)})
            m.update(latency_fields(lat))
        if last is not None:
            for k in ("time_prefill", "time_generate", "cached_tokens"):
                if k in last:
                    v = last[k]
                    m[f"job_{k}"] = round(v, 4) if isinstance(v, float) else v
        m["phase"] = ph
        return m

    w = run(short, a.legacy_warmup, "legacy_warmup")
    out["warmup"] = w
    print(f"-- legacy warm-up: 24 tokens, {w['new_tokens']} new, ttft {w['ttft_s']} s", flush = True)
    for n in a.legacy_prefill:
        m = run(long_ids[:n], 2, f"legacy_prefill_{n}")
        m["prefill_tps"] = round(n / m["ttft_s"], 2) if m.get("ttft_s") else None
        out["prefill"].append(m)
        print(f"== legacy PREFILL n={n}: ttft {m['ttft_s']} s, {m['prefill_tps']} tok/s, foreign "
              f"{m['phase']['foreign_cores_0_15']}{' PROVISIONAL' if m['phase']['provisional'] else ''}", flush = True)
    for n in a.legacy_decode_ctx:
        ids = short if n == 24 else long_ids[:n]
        m = run(ids, a.gen_tokens, f"legacy_decode_{n}")
        m["ctx"] = n
        out["decode"].append(m)
        print(f"== legacy DECODE ctx={n}: {m.get('decode_tps')} tok/s (steady {m.get('steady_tps')}), ttft "
              f"{m['ttft_s']} s, p50 {m.get('tpot_ms_p50')} ms, stalls {m.get('stall_count')}, foreign "
              f"{m['phase']['foreign_cores_0_15']}, MoE-core foreign {m['phase'].get('moe_core_foreign_max')}"
              f"{' PROVISIONAL' if m['phase']['provisional'] else ''}", flush = True)
    del gen


# ----------------------------------------------------------------------------------------------------------------
# quality
# ----------------------------------------------------------------------------------------------------------------

def regions_for(n):
    r = {"all": slice(0, n - 1), "1-127": slice(0, min(127, n - 1)), "128-16383": slice(127, min(16383, n - 1)),
         "16384-65535": slice(16383, min(65535, n - 1))}
    if n - 1 > 65535:
        r["65536-end"] = slice(65535, n - 1)
    return {k: s for k, s in r.items() if s.stop > s.start}


def compare_anchor(path, actual, top1, n):
    z = np.load(path)
    if "actual_lp" not in z.files:
        return {"path": path, "error": f"no actual_lp in {z.files}"}
    alp = z["actual_lp"]
    at1 = z["top1"] if "top1" in z.files else z["top_ids"][:, 0]
    m = min(len(alp), len(actual))
    alp32 = alp[:m].astype(np.float32)
    exact = bool(np.array_equal(alp32.astype(np.float64), alp[:m].astype(np.float64)))
    mine = actual[:m].astype(np.float32)
    eq = mine.view(np.int32) == alp32.view(np.int32)
    t1eq = top1[:m].astype(np.int64) == at1[:m].astype(np.int64)
    both = eq & t1eq
    nd = np.nonzero(~both)[0]
    d = np.abs(mine.astype(np.float64) - alp32.astype(np.float64))
    return {"path": path, "anchor_positions": int(len(alp)), "compared": int(m), "anchor_f32_exact": exact,
            "lp_bitwise_equal": int(eq.sum()), "lp_differing": int(m - eq.sum()),
            "bitwise_equal": int(both.sum()), "differing": int(m - both.sum()),
            "first_diff_pos": int(nd[0]) + 1 if len(nd) else None,
            "top1_agree": int(t1eq.sum()), "top1_agree_frac": round(float(t1eq.mean()), 6) if m else None,
            "nll_anchor_same_range": float(np.mean(-alp[:m].astype(np.float64))),
            "nll_same_range": float(np.mean(-mine.astype(np.float64))),
            "dnll": float(np.mean(-mine.astype(np.float64)) - np.mean(-alp[:m].astype(np.float64))),
            "abs_dlp_mean": float(d.mean()), "abs_dlp_p99": float(np.quantile(d, 0.99)), "abs_dlp_max": float(d.max()),
            "abs_dlp_argmax_pos": int(d.argmax()) + 1}


def resolve_anchor_label(a, label):
    """The quality npz of the latest ok line with this label in --results ("label:<label>" when there is none)."""
    best = None
    try:
        for line in open(a.results):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if j.get("label") == label and j.get("status") == "ok" and (j.get("quality") or {}).get("npz"):
                best = j["quality"]["npz"]
    except OSError:
        pass
    return best or f"label:{label}"


def mode_quality(ctx, rec, ids, gpulog_marks):
    torch = ctx.torch
    a = ctx.args
    n = a.quality_tokens
    chunk = a.quality_chunk
    x = torch.as_tensor(ids[:n], dtype = torch.long)[None]
    actual = np.full(n - 1, np.nan, dtype = np.float32)
    top1 = np.full(n - 1, -1, dtype = np.int32)
    top1_lp = np.full(n - 1, np.nan, dtype = np.float32)
    ctx.cache.reset_states()
    drain(ctx)
    state = ctx.cache.get_new_state()
    pr = Probe(ctx).start()
    try:
        for s0 in range(0, n - 1, chunk):
            s1 = min(s0 + chunk, n - 1)
            y = ctx.model.forward(x[:, s0:s1], {"attn_mode": "flash_attn", "recurrent_states": [state]})[0]
            for s in range(0, y.shape[0], 512):
                lp = torch.log_softmax(y[s:s + 512].float(), dim = -1)
                tl, ti = torch.topk(lp, 5, dim = -1)
                p0 = s0 + s + 1
                m = lp.shape[0]
                r = slice(p0 - 1, p0 - 1 + m)
                actual[r] = lp.gather(1, x[0, p0:p0 + m].to(lp.device)[:, None])[:, 0].cpu().numpy()
                top1[r] = ti[:, 0].to(torch.int32).cpu().numpy()
                top1_lp[r] = tl[:, 0].cpu().numpy()
            del y
    finally:
        ctx.cache.release_state(state)
    ph = pr.stop()
    gpulog_marks.append(("quality", ph["t_start"], ph["t_end"]))
    npz = os.path.join(a.outdir, f"{a.run_id}_quality.npz")
    np.savez(npz, actual_lp = actual, top1 = top1, top1_lp = top1_lp)
    nll = {k: float(np.mean(-actual[s].astype(np.float64))) for k, s in regions_for(n).items()}
    q = {"tokens": n, "positions": n - 1, "chunk": chunk, "numerics": rec["config"].get("numerics"),
         "nll": nll["all"], "nll_regions": nll, "nan_positions": int(np.isnan(actual).sum()),
         "seconds": ph["wall_s"], "score_tps": round((n - 1) / ph["wall_s"], 2), "npz": npz,
         "ids_sha256_12": hashlib.sha256(np.ascontiguousarray(ids[:n]).tobytes()).hexdigest()[:12],
         "phase": ph, "anchors": []}
    for ap in list(a.anchor or []) + [resolve_anchor_label(a, lab) for lab in a.anchor_label or []]:
        try:
            if ap.startswith("label:"):
                raise FileNotFoundError(f"no ok quality line with {ap} in {a.results}")
            q["anchors"].append(compare_anchor(ap, actual, top1, n))
        except Exception as e:
            q["anchors"].append({"path": ap, "error": repr(e)})
    rec["quality"] = q
    print(f"== quality N={n}: NLL {nll['all']:.9f} ({q['score_tps']} tok/s scored)", flush = True)
    for c in q["anchors"]:
        if "error" in c:
            print(f"   anchor {c['path']}: {c['error']}", flush = True)
        else:
            print(f"   anchor {os.path.basename(c['path'])}: {c['bitwise_equal']}/{c['compared']} bitwise equal, first diff "
                  f"{c['first_diff_pos']}, top1 {c['top1_agree_frac']}, dNLL {c['dnll']:+.6f}", flush = True)


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required = True, help = "config label (free text, recorded)")
    ap.add_argument("--run-id", default = None, help = "unique id for this process (default: time + label)")
    ap.add_argument("--results", default = os.path.join(BENCH_DIR, "results", "results.jsonl"))
    ap.add_argument("--outdir", default = None, help = "npz / nvidia-smi log directory (default: dir of --results)")
    ap.add_argument("--modes", default = "speed,quality")
    ap.add_argument("--model-dir", default = DEFAULT_MODEL)
    ap.add_argument("--ids", default = DEFAULT_IDS, help = "canonical token ids (.npy int32)")
    ap.add_argument("--chunk", type = int, default = 4096, help = "load max_chunk_size and raw prefill slice")
    ap.add_argument("--gen-chunk", type = int, default = None, help = "Generator max_chunk_size (default: --chunk)")
    ap.add_argument("--cache-tokens", type = int, default = None,
                    help = "Cache max_num_tokens (default: 1048576 with speed, else N rounded up to 4096)")
    ap.add_argument("--cache-mode", default = "fp16", help = "fp16 | q4 | q6 | q8 | k,v bits e.g. 8,8")
    ap.add_argument("--reserve-mb", type = float, default = 96,
                    help = "autosplit reserve per device (serving: 96); negative: pass no reserve (model.load's default, "
                           "as bench_speed.py loaded)")
    ap.add_argument("--gpu-split", default = None, help = "use_per_device GiB list, e.g. 62,90 (replaces reserve)")
    ap.add_argument("--max-output-size", type = int, default = None,
                    help = "load max_output_size (default: 32 = serving for speed / legacy lines; --quality-chunk when "
                           "quality runs, which needs whole-chunk logits: speed measured in such a process is not the "
                           "serving configuration)")
    ap.add_argument("--numerics", default = None, help = "config.dsv41_numerics (default: env / engine default)")
    ap.add_argument("--prompt-tokens", default = "32768", help = "P, comma list allowed")
    ap.add_argument("--gen-tokens", type = int, default = 256)
    ap.add_argument("--repeats", type = int, default = 1)
    ap.add_argument("--warmup-tokens", type = int, default = None, help = "default min(max P, 2*chunk)")
    ap.add_argument("--warmup-gen", type = int, default = 16)
    ap.add_argument("--warmup-offset", type = int, default = 700000, help = "canonical-id offset of warmup prompts")
    ap.add_argument("--recurrent-cache-mb", type = int, default = None,
                    help = "Generator recurrent_cache_size in MiB (default 4096 = serving; 512 in legacy mode, as "
                           "bench_speed.py)")
    ap.add_argument("--vary", action = "append", default = None, metavar = "KEY=V1/V2/...",
                    help = f"per-repeat setting in speed mode, value = repeat index modulo the list; keys: "
                           f"{', '.join(VARY_KEYS)} (repeatable, one key per option)")
    ap.add_argument("--moe-cores", default = "auto",
                    help = "CPU-MoE-pinned cores for the per-core foreign-load check: auto (the cores the process "
                           "tree's threads are pinned to alone, found after load; none without a CPU-MoE worker), "
                           "none, or a list such as 0-11")
    ap.add_argument("--legacy-ref", default = DEFAULT_LEGACY_REF, help = "legacy mode: the vLLM reference prompts")
    ap.add_argument("--legacy-prefill", default = "8192,33600", help = "legacy mode: prefill job lengths")
    ap.add_argument("--legacy-decode-ctx", default = "24,20000", help = "legacy mode: decode job context lengths")
    ap.add_argument("--legacy-warmup", type = int, default = 8, help = "legacy mode: new tokens of the warm-up job")
    ap.add_argument("--raw", dest = "raw", action = "store_true", default = True)
    ap.add_argument("--no-raw", dest = "raw", action = "store_false")
    ap.add_argument("--prewarm", dest = "prewarm", action = "store_true", default = True,
                    help = "untimed raw prefill of each timed prompt first (engram rows into the page cache; default)")
    ap.add_argument("--no-prewarm", dest = "prewarm", action = "store_false")
    ap.add_argument("--quality-tokens", type = int, default = 65536)
    ap.add_argument("--quality-chunk", type = int, default = 4096)
    ap.add_argument("--anchor", action = "append", default = None, help = "anchor npz (repeatable)")
    ap.add_argument("--anchor-label", action = "append", default = None,
                    help = "use the quality npz of the latest ok line with this label in --results as an anchor "
                           "(repeatable; e.g. a same-config run earlier in the batch, for a run-to-run floor)")
    ap.add_argument("--expect-native-sha", default = None, help = "refuse to run unless the .so sha256 starts with this")
    ap.add_argument("--dry-run", action = "store_true", help = "no model: exercise probes and the JSON line only")
    # recipe stage 0 (h2h_modes.py): head-to-head scenarios, long context, engram page drop, PCIe log, device timing
    ap.add_argument("--scenarios", default = "full",
                    help = "h2h mode: scenario list or set (full | short | e.g. S0,S1:100000:32768,S2:100000:32768,S3,S5)")
    ap.add_argument("--prose-ids", default = None, help = "prose token ids (.npy int32; default recipe data/prose_ids.npy)")
    ap.add_argument("--session-first", default = "16384@400000", help = "S3 turn 1: [src:]N@OFF")
    ap.add_argument("--session-deltas", default = "1024,4096,512,8192,2048@420000",
                    help = "S3 turns 2..: consecutive delta sizes from one canonical offset, or [src:]N@OFF[:G],...")
    ap.add_argument("--session-switch", default = "prose:4096@0:512,canonical:2048@460000",
                    help = "S5 turns appended to the S3 session: [src:]N@OFF[:G],...")
    ap.add_argument("--session-fresh", default = "prose:8192@100000", help = "S5b: a fresh conversation [src:]N@OFF")
    ap.add_argument("--timeline", type = int, default = 32, help = "h2h / longctx: tier + time point every N new tokens (0 off)")
    ap.add_argument("--devtime", type = int, default = 32, help = "h2h DT scenario: decode steps timed per module (0 off)")
    ap.add_argument("--long-prompt", default = "524288", help = "longctx: prompt lengths, ascending, comma list")
    ap.add_argument("--long-decode-repeats", type = int, default = 3)
    ap.add_argument("--long-suffix", type = int, default = 64)
    ap.add_argument("--engram-drop", action = "store_true",
                    help = "after the load: posix_fadvise(DONTNEED) over the engram tables (cold engram page cache)")
    ap.add_argument("--pcie-log", type = int, default = 0, metavar = "MS",
                    help = "NVML PCIe TX / RX sampling period in ms in a separate process (0 off)")
    ap.add_argument("--ngram-draft", default = None, metavar = "MIN:K",
                    help = "h2h / longctx: n-gram drafting in every Generator (ngram_match_min MIN, num_draft_tokens K)")
    a = ap.parse_args()
    if a.ngram_draft is not None:
        try:
            _m, _k = (int(v) for v in a.ngram_draft.split(":"))
            assert _m >= 1 and _k >= 1
        except (ValueError, AssertionError):
            ap.error(f"--ngram-draft {a.ngram_draft!r}: expected MIN:K, two integers >= 1")
        a.ngram_draft = (_m, _k)
    a.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    for m in a.modes:
        if m not in ("speed", "quality", "legacy", "h2h", "longctx"):
            ap.error(f"unknown mode {m}")
    a.prompt_tokens = [int(v) for v in a.prompt_tokens.split(",") if v.strip()]
    a.legacy_prefill = [int(v) for v in a.legacy_prefill.split(",") if v.strip()]
    a.legacy_decode_ctx = [int(v) for v in a.legacy_decode_ctx.split(",") if v.strip()]
    a.gen_chunk = a.gen_chunk or a.chunk
    if a.warmup_tokens is None:
        a.warmup_tokens = max(256, min(max(a.prompt_tokens), 2 * a.chunk))
    if a.cache_tokens is None:
        a.cache_tokens = 1048576 if ("speed" in a.modes or "legacy" in a.modes or "h2h" in a.modes or
                                     "longctx" in a.modes) else \
            ((a.quality_tokens + 4095) // 4096) * 4096
    if a.recurrent_cache_mb is None:
        a.recurrent_cache_mb = 512 if a.modes == ["legacy"] else 4096
    a.max_output_size_source = "explicit"
    if a.max_output_size is None:
        a.max_output_size = a.quality_chunk if "quality" in a.modes else 32
        a.max_output_size_source = "quality-chunk" if "quality" in a.modes else "serving-default"
    if "quality" in a.modes and a.max_output_size < min(a.quality_chunk, a.quality_tokens - 1):
        ap.error(f"--max-output-size {a.max_output_size} with quality mode: scoring takes the logits of every row of "
                 f"a --quality-chunk ({a.quality_chunk}) slice; run speed on its own line for the serving size")
    vary = {}
    for spec in a.vary or []:
        k, sep, vals = spec.partition("=")
        k = k.strip()
        if not sep or k not in VARY_KEYS or not vals.strip():
            ap.error(f"--vary {spec!r}: expected KEY=V1/V2/... with KEY one of {', '.join(VARY_KEYS)}")
        if k in vary:
            ap.error(f"--vary {k} given twice")
        vs = [v.strip() for v in vals.split("/")]
        if k in ("xdev_bf16", "pipeline"):
            if any(v not in ("0", "1") for v in vs):
                ap.error(f"--vary {k}: values are 0 or 1")
            vs = [v == "1" for v in vs]
        elif k == "recurrent_cache_mb":
            try:
                vs = [int(v) for v in vs]
            except ValueError:
                ap.error(f"--vary {k}: integer MiB values")
        vary[k] = vs
    import h2h_modes
    try:
        a.scenarios = h2h_modes.parse_scenarios(a.scenarios)
        a.session_first = h2h_modes.parse_span(a.session_first)
        a.session_deltas = h2h_modes.parse_turns(a.session_deltas)
        a.session_switch = h2h_modes.parse_turns(a.session_switch)
        a.session_fresh = h2h_modes.parse_span(a.session_fresh)
    except ValueError as e:
        ap.error(str(e))
    a.long_prompt = sorted(int(v) for v in a.long_prompt.split(",") if v.strip())
    if a.prose_ids is None:
        a.prose_ids = h2h_modes.DEFAULT_PROSE
    if ("h2h" in a.modes or "longctx" in a.modes) and a.cache_tokens is not None:
        big = 0
        if "h2h" in a.modes:
            big = max([P + a.gen_tokens for k, _, P in a.scenarios if P] +
                      [h2h_modes.session_tokens(a, a.gen_tokens) if any(k == "S3" for k, _, _ in a.scenarios) else 0])
        if "longctx" in a.modes:
            big = max(big, max(a.long_prompt) + a.long_suffix + a.gen_tokens)
        if big > a.cache_tokens:
            ap.error(f"the h2h / longctx contexts reach {big} tokens, the cache holds {a.cache_tokens}")
    if vary and "speed" not in a.modes:
        ap.error("--vary applies to speed mode")
    a.vary = vary
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in a.label)
    if a.run_id is None:
        a.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + safe
    if a.outdir is None:
        a.outdir = os.path.dirname(os.path.abspath(a.results))
    return a


def append_line(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
    line = (json.dumps(rec, default = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok = True)
    t_start = time.time()
    rec = {"schema": 1, "harness_version": HARNESS_VERSION, "run_id": a.run_id, "label": a.label, "status": "running",
           "error": None, "t_start": datetime.datetime.fromtimestamp(t_start).isoformat(timespec = "seconds"),
           "host": {"hostname": socket.gethostname(), "kernel": platform.release(), "python": platform.python_version(),
                    "python_executable": sys.executable,
                    "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)), "nproc": os.cpu_count()},
           "env": env_snapshot(), "args": {k: v for k, v in vars(a).items()}, "model_dir": a.model_dir,
           "model_realpath": os.path.realpath(a.model_dir), "config": {}}
    ctx = Ctx(a)
    gpulog = GpuLog(os.path.join(a.outdir, f"{a.run_id}_nvsmi.csv"))
    pcie = None
    if a.pcie_log > 0 and not a.dry_run:
        import h2h_modes
        pcie = h2h_modes.PcieLog(os.path.join(a.outdir, f"{a.run_id}_pcie.csv"), a.pcie_log)
    marks = []
    interrupted = {"sig": None}

    def on_term(signum, frame):
        interrupted["sig"] = signum
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, on_term)

    try:
        ids = np.load(a.ids, mmap_mode = "r")
        rec["ids"] = {"path": a.ids, "n": int(len(ids)), "sha256_12": sha256_file(a.ids, 12)}
        need = max([a.quality_tokens if "quality" in a.modes else 1] +
                   ([max(a.prompt_tokens), a.warmup_offset + a.warmup_tokens + 4096 * (len(a.prompt_tokens) * a.repeats + 1)]
                    if "speed" in a.modes else []))
        if need > len(ids):
            raise ValueError(f"canonical ids hold {len(ids)} tokens, this run needs {need}")
        if "speed" in a.modes and max(a.prompt_tokens) + a.gen_tokens > a.cache_tokens:
            raise ValueError(f"P + G = {max(a.prompt_tokens) + a.gen_tokens} exceeds the cache ({a.cache_tokens})")
        import h2h_modes
        prose = None
        if "h2h" in a.modes or "longctx" in a.modes:
            nc, npr = h2h_modes.scenario_need(a)
            need = max(need, nc)
            if npr:
                prose = np.load(a.prose_ids, mmap_mode = "r")
                rec["prose_ids"] = {"path": a.prose_ids, "n": int(len(prose)), "sha256_12": sha256_file(a.prose_ids, 12)}
                if npr > len(prose):
                    raise ValueError(f"prose ids hold {len(prose)} tokens, this run needs {npr}")
                prose = np.ascontiguousarray(prose[:npr]).astype(np.int64)
            if need > len(ids):
                raise ValueError(f"canonical ids hold {len(ids)} tokens, this run needs {need}")
        ids = np.ascontiguousarray(ids[:need]).astype(np.int64)
        if "legacy" in a.modes:
            short, long_ids = legacy_prompts(a)
            rec["legacy_prompts"] = {"ref": a.legacy_ref, "short_tokens": len(short), "long_tokens": len(long_ids),
                                     "short_sha256_12": hashlib.sha256(np.asarray(short, np.int64).tobytes()).hexdigest()[:12]}
            if max(a.legacy_prefill + a.legacy_decode_ctx) + a.gen_tokens > a.cache_tokens:
                raise ValueError(f"legacy lengths + G exceed the cache ({a.cache_tokens})")
        for lab in a.anchor_label or []:
            rec.setdefault("anchor_labels", {})[lab] = resolve_anchor_label(a, lab)

        if a.dry_run:
            rec["build"] = {"bench_sha256_12": sha256_file(__file__, 12), "dry_run": True}
            # what can be checked without importing the tree (no torch, no extension): the interpreter, the tree and
            # extension paths, the modules a --vary needs, the anchors
            pp = [x for x in os.environ.get("PYTHONPATH", "").split(":") if x]
            tree = next((x for x in pp if os.path.isdir(os.path.join(x, "exllamav3"))), None)
            chk = {"python": sys.executable, "pythonpath_exists": {x: os.path.exists(x) for x in pp}, "tree": tree,
                   "extension_so": [os.path.join(x, f) for x in pp if os.path.isdir(x) for f in os.listdir(x)
                                    if f.startswith("exllamav3_ext") and f.endswith(".so")]}
            if tree:
                d41 = os.path.join(tree, "exllamav3", "architecture", "dsv41")
                chk["tree_has"] = {"numerics": os.path.exists(os.path.join(d41, "numerics.py")),
                                   "pipeline": os.path.exists(os.path.join(d41, "pipeline.py")),
                                   "placement_generic": os.path.exists(os.path.join(tree, "exllamav3", "model", "placement.py"))}
                if "numerics" in a.vary and not chk["tree_has"]["numerics"]:
                    raise ValueError("--vary numerics: the tree on PYTHONPATH has no attention numerics")
            chk["anchors_exist"] = {x: os.path.exists(x) for x in a.anchor or []}
            rec["dry_checks"] = chk
            if a.moe_cores not in ("auto", "none"):
                ctx.moe_cores = parse_cpu_list(a.moe_cores)
            elif a.moe_cores == "auto":
                ctx.moe_cores = list(range((os.cpu_count() or 2) // 2))      # the worker's default, to exercise the check
            pr = Probe(ctx).start()
            time.sleep(1.0)
            ph = pr.stop()
            marks.append(("dry", ph["t_start"], ph["t_end"]))
            rec["dry_phase"] = ph
            rec["status"] = "ok"
            return 0

        import torch
        import exllamav3_ext
        ctx.torch, ctx.ext = torch, exllamav3_ext
        rec["build"] = build_identity(a)
        if a.expect_native_sha and not rec["build"].get("native_sha256", "").startswith(a.expect_native_sha):
            raise RuntimeError(f"native .so sha256 {rec['build'].get('native_sha256')} != expected {a.expect_native_sha}")
        from exllamav3 import Config, Model, Cache, Tokenizer, CacheLayer_quant
        rec["host"].update({"torch": torch.__version__, "cuda": torch.version.cuda,
                            "torch_threads": torch.get_num_threads()})

        t0 = time.time()
        monitor = HostMonitor().start()     # load phase only; stopped before any measurement
        config = Config.from_directory(a.model_dir)
        if a.numerics is not None:
            config.dsv41_numerics = a.numerics
        ctx.config = config
        model = Model.from_config(config)
        ctx.model = model
        cm = a.cache_mode.lower()
        cm = {"q4": "4,4", "q6": "6,6", "q8": "8,8"}.get(cm, cm)
        if cm == "fp16":
            cache = Cache(model, max_num_tokens = a.cache_tokens, max_batch_size = 1, max_history = 0)
        else:
            kb, vb = (int(v) for v in cm.split(","))
            cache = Cache(model, max_num_tokens = a.cache_tokens, max_batch_size = 1, max_history = 0,
                          layer_type = CacheLayer_quant, k_bits = kb, v_bits = vb)
        ctx.cache = cache
        load_kw = dict(max_chunk_size = a.chunk, max_output_size = a.max_output_size, max_batch_size = 1,
                       progressbar = False)
        if a.gpu_split:
            load_kw["use_per_device"] = [float(v) for v in a.gpu_split.split(",")]
        elif a.reserve_mb >= 0:
            load_kw["reserve_per_device"] = [a.reserve_mb / 1024]
        load_kw_rec = json.loads(json.dumps(load_kw))      # model.load extends the reserve list in place
        mark_load0 = time.time()
        try:
            model.load(**load_kw)
            ctx.tokenizer = Tokenizer.from_config(config)
        finally:
            rec["load_monitor"] = monitor.stop()
        t_loaded = time.time()
        marks.append(("load", mark_load0, t_loaded))
        if a.moe_cores == "auto":
            ctx.moe_cores = pinned_cores(own_tree())
        elif a.moe_cores != "none":
            ctx.moe_cores = parse_cpu_list(a.moe_cores)
        vram_alloc = [round(torch.cuda.memory_allocated(i) / GIB, 3) for i in range(torch.cuda.device_count())]
        vram_res = [round(torch.cuda.memory_reserved(i) / GIB, 3) for i in range(torch.cuda.device_count())]
        cfg = rec["config"]
        cfg.update({"numerics": str(getattr(config, "dsv41_numerics", None)),
                    "placement": str(getattr(config.infer_params, "placement", None)),
                    "dsv41_placement": str(getattr(config, "dsv41_placement", None)),
                    "cache_mode": cm, "cache_tokens": a.cache_tokens, "chunk": a.chunk, "gen_chunk": a.gen_chunk,
                    "max_output_size": a.max_output_size, "load_kwargs": load_kw_rec,
                    "load_kwargs_after": {k: v for k, v in load_kw.items()},
                    "max_output_size_source": a.max_output_size_source, "moe_cores": ctx.moe_cores,
                    # the older serving tree's extension has neither query (None there)
                    "stable_arithmetic": bool(exllamav3_ext.stable_arithmetic())
                    if hasattr(exllamav3_ext, "stable_arithmetic") else None,
                    "hgemm_fixed_rows": int(exllamav3_ext.hgemm_fixed_rows())
                    if hasattr(exllamav3_ext, "hgemm_fixed_rows") else None})
        try:
            cfg["settings"] = current_settings(ctx)
        except Exception:
            pass
        try:
            cfg["disk_ngram_route"] = exllamav3_ext.disk_ngram_route()
            if cfg["disk_ngram_route"] == "engine":
                inf = exllamav3_ext.disk_engine_info()
                cfg["disk_engine"] = {k: inf[k] for k in ("backend", "auto_backend", "keepalive_ms", "direct_rows",
                                                          "direct_extents", "qd", "reserve0", "engram_qd", "window_hi",
                                                          "chunk_bytes", "sqpoll", "reg_files", "reg_buffers") if k in inf}
        except Exception as e:
            cfg["disk_engine_error"] = repr(e)
        try:
            from exllamav3.architecture.dsv41 import pipeline as _pl
            cfg["dsv41_pipeline"] = {"on": bool(getattr(_pl, "PIPELINE", False)), "sub_chunk": getattr(_pl, "SUB_CHUNK", None)}
        except Exception:
            pass
        rec["load"] = {"seconds": round(t_loaded - t0, 1), "load_call_s": round(t_loaded - mark_load0, 1),
                       "vram_alloc_gib": vram_alloc, "vram_reserved_gib": vram_res, "host": host_memory_snapshot(),
                       "cgroup": cgroup_mem(), "gpu_free_mib": gpu_free_mib(torch)}
        rec["tier_load"] = tier_snapshot(ctx)
        rec["tier_sizing"] = tier_sizing(ctx)
        if a.engram_drop:
            import engram_drop
            t_d = time.time()
            try:
                rec["engram_drop"] = engram_drop.drop(a.model_dir)
            except Exception as e:
                rec["engram_drop"] = {"error": repr(e)}
            rec["engram_drop"]["seconds"] = round(time.time() - t_d, 2)
            ed = rec["engram_drop"]
            print(f"-- engram page drop: {ed.get('bytes', 0) / GIB:.1f} GiB advised, resident "
                  f"{ed.get('resident_before', 0) / GIB:.2f} -> {ed.get('resident_after', 0) / GIB:.2f} GiB"
                  f"{' ERROR ' + ed['error'] if 'error' in ed else ''}", flush = True)
        if rec["tier_sizing"]:
            print(f"-- tier sizing: {json.dumps({k: v for k, v in rec['tier_sizing'].items() if k != 'text'})}",
                  flush = True)
        print(f"-- loaded in {t_loaded - t0:.1f} s; VRAM alloc {vram_alloc} GiB; numerics {cfg['numerics']}; "
              f"placement {cfg['placement']}; stable {cfg['stable_arithmetic']}", flush = True)

        with torch.inference_mode():
            if "speed" in a.modes:
                mode_speed(ctx, rec, ids, marks)
            if "legacy" in a.modes:
                mode_legacy(ctx, rec, ids, marks)
            if "h2h" in a.modes:
                h2h_modes.mode_h2h(sys.modules[__name__], ctx, rec, ids, prose, marks, pcie)
            if "longctx" in a.modes:
                h2h_modes.mode_longctx(sys.modules[__name__], ctx, rec, ids, marks, pcie)
            if "quality" in a.modes:
                mode_quality(ctx, rec, ids, marks)
        rec["status"] = "ok"
        return 0
    except KeyboardInterrupt as e:
        rec["status"] = "interrupted"
        rec["error"] = f"interrupted ({e})"
        return 130
    except BaseException as e:
        rec["status"] = "error"
        tb = traceback.format_exc()
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        rec["traceback_tail"] = tb[-3000:]
        print(tb, file = sys.stderr, flush = True)
        return 1
    finally:
        t_end = time.time()
        gpulog.stop()
        if pcie is not None:
            pcie.stop()
            rec["pcie_log"] = pcie.path
            try:
                rec["pcie_load"] = pcie.summarize(*next(((s_, e_) for n_, s_, e_ in marks if n_ == "load"), (t_start, t_start)))
            except Exception as e:
                rec["pcie_load"] = {"error": repr(e)}
        rows = gpulog.rows()
        try:
            if ctx.torch is not None:
                rec["devices"] = device_table(ctx.torch, rows)
                rec.setdefault("peaks", {})["vram_max_alloc_gib_process"] = None
        except Exception as e:
            rec["devices_error"] = repr(e)
        gpu = {"whole_run": GpuLog.summarize(rows, t_start, t_end)}
        for name, s, e in marks:
            gpu[name] = GpuLog.summarize(rows, s, e)
        rec["gpu"] = gpu
        rec["nvsmi_log"] = gpulog.path
        pk = rec.setdefault("peaks", {})
        wr = gpu["whole_run"]
        pk["gpu_temp_max_c"] = {k: v["temp_max_c"] for k, v in wr.items()}
        pk["gpu_power_max_w"] = {k: v["power_max_w"] for k, v in wr.items()}
        pk["gpu_mem_max_mib"] = {k: v["mem_max_mib"] for k, v in wr.items()}
        vr = []
        for sec in ("speed", "quality", "legacy", "h2h", "longctx"):
            s = rec.get(sec)
            if not s:
                continue
            phs = [s["phase"]] if "phase" in s else []
            phs += [r["phase"] for r in s.get("runs", []) + s.get("raw", []) if "phase" in r]
            phs += [r["warmup"]["phase"] for r in s.get("runs", []) if "warmup" in r]
            phs += [r["phase"] for r in s.get("prefill", []) + s.get("decode", []) if "phase" in r]
            phs += [r["phase"] for r in s.get("jobs", []) if "phase" in r]
            if isinstance(s.get("devtime"), dict) and "phase" in s["devtime"]:
                phs.append(s["devtime"]["phase"])
            vr += [p["vram_peak_alloc_gib"] for p in phs if p.get("vram_peak_alloc_gib")]
        pk["vram_peak_alloc_gib"] = [max((v[i] for v in vr if v[i] is not None), default = None)
                                     for i in range(len(vr[0]))] if vr else None
        pk.pop("vram_max_alloc_gib_process", None)
        try:
            pk["host_end"] = host_memory_snapshot()
        except Exception:
            pass
        pk["cgroup_end"] = cgroup_mem()
        pk["gpu_mem_min_free_mib"] = {k: (v["mem_total_mib"] - v["mem_max_mib"])
                                      if v.get("mem_total_mib") is not None and v.get("mem_max_mib") is not None
                                      else None for k, v in wr.items()}
        try:
            if rec["status"] == "ok" and ctx.model is not None and getattr(ctx.config, "expert_tiers", None) is not None:
                for td in ctx.config.expert_tiers.devices.values():
                    if td.handle is not None:
                        td.handle.drain()
                rec["tier_end"] = tier_snapshot(ctx)
                tl, te = rec.get("tier_load"), rec["tier_end"]
                if tl and te and "error" not in tl and "error" not in te:
                    rec["tier_ssd_reads_after_load"] = te.get("ssd_reads", 0) - tl.get("ssd_reads", 0)
                    rec["tier_ssd_bytes_after_load"] = te.get("ssd_bytes", 0) - tl.get("ssd_bytes", 0)
                    print(f"-- tier: SSD reads after the load {rec['tier_ssd_reads_after_load']} "
                          f"({rec['tier_ssd_bytes_after_load']} bytes)", flush = True)
        except Exception as e:
            rec["tier_end_error"] = repr(e)
        print(f"-- cgroup at the end: {pk['cgroup_end']}; nvidia-smi min free MiB {pk['gpu_mem_min_free_mib']}",
              flush = True)
        lm = rec.get("load_monitor")
        if lm is not None and "load" in gpu:
            lm["gpu_mem_max_mib"] = {k: v["mem_max_mib"] for k, v in gpu["load"].items()}
            lm["gpu_mem_min_free_mib"] = {k: (v["mem_total_mib"] - v["mem_max_mib"])
                                          if v.get("mem_total_mib") is not None and v.get("mem_max_mib") is not None
                                          else None for k, v in gpu["load"].items()}
        timed = []
        s = rec.get("speed") or {}
        timed += [r["phase"] for r in s.get("runs", []) + s.get("raw", []) if "phase" in r]
        s = rec.get("legacy") or {}
        timed += [r["phase"] for r in s.get("prefill", []) + s.get("decode", []) if "phase" in r]
        for sec in ("h2h", "longctx"):
            timed += [r["phase"] for r in (rec.get(sec) or {}).get("jobs", []) if "phase" in r]
        if rec.get("dry_phase"):
            timed.append(rec["dry_phase"])          # a dry run's stand-in for the timed phases
        if rec.get("quality"):
            timed.append(rec["quality"]["phase"])
        rec["provisional"] = any(p.get("provisional") for p in timed)
        rec["moe_cores_contended"] = any(p.get("moe_cores_contended") for p in timed)
        rec["stall_flag"] = any(r.get("stall_flag") for sec in ("h2h", "longctx") for r in (rec.get(sec) or {}).get("jobs", [])) or \
            any(r.get("stall_flag") for sec in ("speed",) for r in (rec.get(sec) or {}).get("runs", []) +
                                (rec.get(sec) or {}).get("raw", [])) or \
            any(r.get("stall_flag") for r in (rec.get("legacy") or {}).get("decode", []))
        rec["cores_16_23_contended"] = any(p.get("cores_16_23_contended") for p in timed)
        rec["t_end"] = datetime.datetime.fromtimestamp(t_end).isoformat(timespec = "seconds")
        rec["wall_s"] = round(t_end - t_start, 1)
        try:
            append_line(a.results, rec)
            print(f"-- result line appended to {a.results} (status {rec['status']}, provisional {rec['provisional']})",
                  flush = True)
        except Exception as e:
            print(f" !! could not write result line: {e!r}", flush = True)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # A CUDA error or a wedged worker can hang interpreter teardown; the result line is already written
    os._exit(rc if isinstance(rc, int) else 1)
