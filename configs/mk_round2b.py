"""Round 2 of stage 2 (appended to the requeue file): second processes of the arms that beat the reference, new
arms that round 1 motivated (written reasons in STAGE_2.md), quality lines, long-context safety runs."""
import subprocess, sys
B = "/root/rnd/release/recipe/bench3/batches/"

def gen(*items):
    return subprocess.run([sys.executable, B + "mk_round2.py", B + "stage2.batch", *items], check = True,
                          capture_output = True, text = True).stdout

def derive(src, new, subs, note):
    t = gen(f"{src}={new}")
    for a, b in subs:
        assert t.count(a) == 1, (src, a, t.count(a))
        t = t.replace(a, b)
    return f"# {note}\n" + t

C10 = "EXL3_PLACEMENT=\x270-9=cuda:0; 10-39=cuda:1 experts=cache; ram experts=all; disk experts=off\x27"
C11 = "EXL3_PLACEMENT=\x270-10=cuda:0; 11-39=cuda:1 experts=cache; ram experts=all; disk experts=off\x27"
out = []
out.append("# ---- round 2 (appended after round 1 was read)\n")
out.append(gen("C12-ES"))
out.append(derive("A1M-K8", "A1M-ES", [("PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync ", ""),
                                        ("--chunk 8192 ", "")],
                  "config a with expandable segments (fairness: C12-ES decoded 3-6% faster than cudaMallocAsync, same tokens)"))
out.append(derive("C12-P3", "C12-ES-P4", [("PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync ", ""),
                                           ("EXL3_DSV41_PIPELINE_CHUNK=3072", "EXL3_DSV41_PIPELINE_CHUNK=4096"),
                                           ("--gen-chunk 12288", "--gen-chunk 16384")],
                  "C12 pipelined 4096 / 16384 under expandable segments: C12-ES left 1.3 GB free on the CMP (cudaMallocAsync 0.8 GB)"))
out.append(derive("C12-P3", "C12-ES-P3", [("PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync ", "")],
                  "C12 pipelined 3072 / 12288 under expandable segments (with cudaMallocAsync 1 job of 15 hit a CMP OOM)"))
out.append(gen("C12-PR1K"))
out.append(derive("C12-PR1K", "C12-PR2K", [("EXL3_MOE_TIER_PREFILL_ROWS=1024", "EXL3_MOE_TIER_PREFILL_ROWS=2048")],
                  "PREFILL_ROWS 2048: the 1,280-row S3 turn 2 and 2,048-row remainders routed too"))
out.append(derive("A1M-K8", "A1M-K16", [("--chunk 8192", "--chunk 16384")],
                  "config a at chunk 16384: its prefill ceiling (a is staging-bound; chunk 8192 gave +52-65%)"))
out.append(derive("C11-P8", "C10-P8", [(C11, C10), ("--gen-chunk 16384", "--gen-chunk 32768"), ("minavail=92", "minavail=97")],
                  "C10 with 8192-row sub-chunks, 4 per call (the plan\x27s 2.1 arm for split <= 10): does c\x27s prefill follow a\x27s at 8192?"))
out.append(derive("C11-P8", "C10-P8X", [(C11, C10), ("--gen-chunk 16384", "--gen-chunk 32768"), ("minavail=92", "minavail=97"),
                                        ("EXL3_DSV41_PIPELINE_CHUNK=8192", "EXL3_DSV41_PIPELINE_CHUNK=8192 EXL3_DSV41_XDEV_BF16=1")],
                  "C10-P8 with the BF16 crossing (L): the family-c prefill ceiling at the x1 link"))
out.append(derive("C12-P3", "C12-P25", [("EXL3_DSV41_PIPELINE_CHUNK=3072", "EXL3_DSV41_PIPELINE_CHUNK=2560"),
                                         ("--gen-chunk 12288", "--gen-chunk 10240")],
                  "C12 pipelined with 2560-row sub-chunks: 3072 lost 1 job of 15 to a CMP OOM, 2048 fit but gained nothing (stage 0)"))
out.append(derive("C12-H2", "C12-H1", [("EXL3_MOE_TIER_HEADROOM_MB=2048", "EXL3_MOE_TIER_HEADROOM_MB=1024")],
                  "headroom 1024 MiB: at 2048 the PRO never dipped into it (min free 1,983 MiB, as 4,027 at 4096, also at 512K in stage 1)"))
out.append(gen("C12-H2", "C12-SS", "C12-AA", "heat"))
out.append(gen("q:C12-X", "q:C11-X"))
out.append(derive("C12-H2", "C12-H1-S4", [("EXL3_MOE_TIER_HEADROOM_MB=2048", "EXL3_MOE_TIER_HEADROOM_MB=1024"),
                                           ("--modes h2h --scenarios full", "--modes longctx --long-prompt 524288 --long-decode-repeats 3"),
                                           ("est=12 timeout=1500", "est=15 timeout=1800")],
                  "S4-512K safety run of the headroom cut (1024 MiB)"))
out.append(gen("ref"))
sys.stdout.write("".join(out))
