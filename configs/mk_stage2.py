"""Writes stage2.batch (recipe stage 2, PLAN.md v2 section 6): one-factor sweeps on cB = C12 (plain) and K4 = C11
(pipelined 4096 / 16384), round 1 (one process per arm, references first and last); round 2 / optional lines go to the
requeue file (mk_stage2.py --round2 ... prints them)."""
import sys
MEM = ("BENCH_PYTHON=/root/rnd/release/all/aux/py_memwatch.sh MEMWATCH_PY=/root/rnd/release/recipe/bench3/py_mem.sh "
       "PYTHONDONTWRITEBYTECODE=1 MEMMAX=119G")
ALLOC = "PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
COMMON = "--cache-tokens 1048576 --max-output-size 32 --engram-drop --timeline 32 --expect-native-sha 823174af46fb"
HEAT = "/root/rnd/release/recipe/stage2/heat"

def c12(extra_place = "", ram = "ram experts=all", disk = "disk experts=off"):
    p = "0-11=cuda:0; 12-39=cuda:1 experts=cache"
    if extra_place:
        p += "; " + extra_place
    return f"EXL3_PLACEMENT=\x27{p}; {ram}; {disk}\x27"

C11P = "EXL3_PLACEMENT=\x270-10=cuda:0; 11-39=cuda:1 experts=cache; ram experts=all; disk experts=off\x27"
A1MP = "CUDA_VISIBLE_DEVICES=0 EXL3_PLACEMENT=\x27*=cuda:0 experts=cache{x}; ram experts=102GiB\x27"

def line(label, env, args, est, minav, tmo = 1500, alloc = True, optional = False):
    d = f"#@ est={est} timeout={tmo} minavail={minav}" + (" optional=1" if optional else "")
    e = " ".join(x for x in (MEM, ALLOC if alloc else "", env) if x)
    return f"{d}\n{label} | {e} | {args} {COMMON}\n"

H_FULL = "--modes h2h --scenarios full"
H_SHORT = "--modes h2h --scenarios short"
PF = "--modes h2h --scenarios S0,S1:100000:32768,S2:100000:32768,S1:200000:131072"
PIPE = lambda p: f"EXL3_DSV41_PIPELINE=1 EXL3_DSV41_PIPELINE_CHUNK={p}"

R1 = [
 ("# ---- round 1: one process per arm; C12 reference first (it also writes the heat file for C12-HR) and last", None),
 line("C12-R1", f"{c12()} EXL3_MOE_HEAT_FILE={HEAT}/c12-r1.exl3moe", f"{H_FULL} --chunk 4096", 12, 87),
 ("# 2.5 tier policy (N): the T1 replay\x27s largest lever (admit=always) and evict=lfu (S3 / S5 prose)", None),
 line("C12-AA", c12("cuda:1 admit=always"), f"{H_FULL} --chunk 4096", 12, 87),
 line("C12-LFU", c12("cuda:1 evict=lfu"), f"{H_FULL} --chunk 4096", 12, 87),
 ("# 2.4 pool (N): single staging (+384 slots), headroom 2048 MiB (+~160 slots; 512K safety run if it wins)", None),
 line("C12-SS", f"{c12()} EXL3_MOE_TIER_STAGING=single", f"{H_FULL} --chunk 4096", 12, 87),
 line("C12-H2", f"{c12()} EXL3_MOE_TIER_HEADROOM_MB=2048", f"{H_FULL} --chunk 4096", 12, 87),
 ("# 2.3 engram rows (N): E2 ram ngram=auto (~35 GiB, rows buffered); E3 ngram=40GiB + disk io=direct (the row cache is", None),
 ("# the only row cache; 40 GiB, not auto with pagecache=8GiB, keeps pinned + resident <= 120 GB)", None),
 line("C12-E2", c12(ram = "ram experts=all ngram=auto"), f"{H_FULL} --chunk 4096", 12, 120),
 line("C12-E3", c12(ram = "ram experts=all ngram=40GiB", disk = "disk experts=off io=direct"), f"{H_FULL} --chunk 4096", 12, 125),
 ("# config a with the policy lever (fairness: the replay predicts admit=always helps a as well)", None),
 line("A1M-AA", A1MP.format(x = "; cuda:0 admit=always"), H_FULL, 9, 124),
 ("# 2.1 chunking / pipeline (A): C11 reference, C11 sub-chunk 8192, C12 sub-chunk 3072 (between 2048 = fits, 4096 = OOM)", None),
 line("C11-R1", f"{C11P} {PIPE(4096)}", f"{H_FULL} --chunk 4096 --gen-chunk 16384", 11, 92),
 line("C11-P8", f"{C11P} {PIPE(8192)}", f"{H_FULL} --chunk 8192 --gen-chunk 16384", 11, 92),
 line("C12-P3", f"{c12()} {PIPE(3072)}", f"{H_FULL} --chunk 4096 --gen-chunk 12288", 12, 87),
 ("# 2.2 BF16 residual crossing (L): C12 plain and C11 pipelined", None),
 line("C12-X", f"{c12()} EXL3_DSV41_XDEV_BF16=1", f"{H_FULL} --chunk 4096", 12, 87),
 line("C11-X", f"{C11P} {PIPE(4096)} EXL3_DSV41_XDEV_BF16=1", f"{H_FULL} --chunk 4096 --gen-chunk 16384", 11, 92),
 ("# 2.1 plain chunk 8192 on C12 (CMP fit probe) and 2.9 config a at chunk 8192 (a is staging-bound)", None),
 line("C12-K8", c12(), f"{H_FULL} --chunk 8192", 12, 87),
 line("A1M-K8", A1MP.format(x = ""), f"{H_FULL} --chunk 8192", 9, 124),
 ("# 2.7 PREFILL_ROWS (N) and 2.8 numerics speed (L, report only), H2H-short", None),
 line("C12-PR1K", f"{c12()} EXL3_MOE_TIER_PREFILL_ROWS=1024", f"{H_SHORT} --chunk 4096", 9, 87),
 line("C12-NP", f"{c12()} EXL3_DSV41_NUMERICS=precise", f"{H_SHORT} --chunk 4096", 9, 87),
 ("# 2.4 allocator: expandable segments instead of cudaMallocAsync (no PYTORCH_*ALLOC_CONF: the library sets them)", None),
 line("C12-ES", c12(), f"{H_FULL} --chunk 4096", 12, 87, alloc = False),
 ("# 2.1 C11 generator chunk 32768 (8 sub-chunks per call; prefill scenarios only) and the tabby-as-is pipeline 2048 / 4096", None),
 line("C11-G32", f"{C11P} {PIPE(4096)}", f"{PF} --chunk 4096 --gen-chunk 32768", 9, 92),
 line("C11-T", f"{C11P} {PIPE(2048)}", f"{H_FULL} --chunk 4096 --gen-chunk 4096", 11, 92),
 line("C12-R2", c12(), f"{H_FULL} --chunk 4096", 12, 87),
]

OPT = [
 ("# ---- optional (the requeue file is drained before this point: round 2 runs first)", None),
 line("C12-NG", c12(), f"{H_SHORT} --chunk 4096 --ngram-draft 2:4", 9, 87, optional = True),
 line("C12-HP", f"{c12()} EXL3_MOE_TIER_HUGEPAGE=1", f"{H_SHORT} --chunk 4096", 10, 87, optional = True),
 line("C12-SP0", f"{c12()} EXL3_MOE_TIER_SPIN_US=0", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
 line("C12-CW1", f"{c12()} EXL3_MOE_COOP_WIDE=1", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
 line("C12-CW0", f"{c12()} EXL3_MOE_COOP_WIDE=0", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
 line("C12-KS2", f"{c12()} EXL3_MOE_COOP_KSPLIT=2", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
 line("C12-FR0", f"{c12()} EXL3_HGEMM_FP32_REDUCTION=0", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
 line("C12-FC", f"{c12()} EXL3_DSV41_FUSED_COMPRESS=1", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
 line("C12-PR4K", f"{c12()} EXL3_MOE_TIER_PREFILL_ROWS=4096", f"{H_SHORT} --chunk 4096", 9, 87, optional = True),
]

hdr = """# Recipe stage 2 (PLAN.md v2 section 6): one-factor sweeps at a 1M FP16 cache on cB = C12 (plain, chunk 4096: the
# decode-best split of stage 1) and K4 = C11 (pipelined 4096 / 16384: the prefill-best split). Code: dsv41-all tip 5e55c062
# (source /root/rnd/release/all/src-5e55c0627ece via WS_SRC, native sha256 823174af46fb). Harness recipe/bench3 (version 5 =
# bench2 + --ngram-draft). ONE window: round 1 here (one process per arm, C12 references first and last), round 2 (second
# processes of the arms that beat the reference, mirrored, plus conditional lines) appended to the requeue file, which the
# driver drains before the first optional line. Memory: py_memwatch (8 GiB floor, kills only its own PID) in a swap-0
# cgroup of 119G; pinned + resident <= 120 GB on every line (E3 sized for it); minavail = expected RSS + ~18-20 GiB.
# N-class arms are gated by greedy-token equality with the C12 references (tokens_sha256_12 per job).
"""
def render(items):
    return "".join((i[0] + "\n") if isinstance(i, tuple) else i for i in items)

out = hdr + "\n" + render(R1 + OPT)
open(sys.argv[1], "w").write(out)
print(out)
