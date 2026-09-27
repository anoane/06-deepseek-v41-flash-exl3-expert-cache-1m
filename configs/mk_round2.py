#!/usr/bin/env python3
"""Stage-2 round 2 lines for the requeue file (append-only; the driver drains it before the first optional line).

usage: mk_round2.py <stage2.batch> <item> [...] > more.batch
items:
  <label>          a second process of a round-1 line: the same directive and line, labelled <label>-2
  <label>=<new>    the same line labelled <new>
  heat             C12-HR: C12 with the heat file C12-R1 wrote (seeded cold fill; S0 is the target)
  ref              C12-R3: the C12 reference once more (the end of the mirrored round)
  q:<label>        a 64K teacher-forced quality line in <label>'s configuration (chunk = its sub-chunk or load chunk)
  s4:<label>       an S4-512K line in <label>'s configuration (safety run for VRAM headroom cuts)
Every emitted line keeps the round-1 memory settings (py_memwatch, 119G swap-0 cgroup, minavail).
"""
import re, shlex, sys

ANCHORS = ["/root/rnd/proof/n64k_serving_deepseek-index.npz", "/root/rnd/numerics/n64k2_deepseek-index.npz",
           "/root/rnd/release/all/val/w1/runs-a/20260926-144016-09-W1-c-quality_quality.npz",
           "/root/rnd/release/all/val/w1/runs-b/20260926-145856-01-W1-a-quality_quality.npz",
           "/root/rnd/release/recipe/stage0/w0b/20260926-210510-01-RQ-4096_quality.npz",
           "/root/rnd/release/recipe/stage0/w0c/20260926-223752-01-A1M-Q_quality.npz",
           "/root/rnd/release/recipe/stage1/w1/20260927-012643-01-Q-C11_quality.npz"]
HEAT = "/root/rnd/release/recipe/stage2/heat/c12-r1.exl3moe"


def read(batch):
    lines, d = {}, None
    for l in open(batch):
        s = l.rstrip("\n")
        if s.startswith("#@"):
            d = s.replace(" optional=1", "")
            continue
        if not s or s.startswith("#"):
            continue
        label, env, args = [x.strip() for x in s.split("|")]
        lines[label] = (d, env, args)
        d = None
    return lines


def kv(d, k):
    m = re.search(rf"\b{k}=(\d+)", d)
    return int(m.group(1)) if m else None


def emit(label, d, env, args, note = None):
    out = []
    if note:
        out.append(f"# {note}")
    out.append(d)
    out.append(f"{label} | {env} | {args}")
    return "\n".join(out) + "\n"


def main():
    L = read(sys.argv[1])
    out = []
    for item in sys.argv[2:]:
        if item == "heat":
            d, env, args = L["C12-R1"]
            out.append(emit("C12-HR", d, env, args, f"C12 seeded from the heat file C12-R1 wrote ({HEAT})"))
        elif item == "ref":
            d, env, args = L["C12-R2"]
            out.append(emit("C12-R3", d, env, args, "C12 reference, end of the mirrored round 2"))
        elif item.startswith("q:"):
            src = item[2:]
            d, env, args = L[src]
            a = shlex.split(args)
            chunk = int(a[a.index("--chunk") + 1]) if "--chunk" in a else 4096
            m = re.search(r"EXL3_DSV41_PIPELINE_CHUNK=(\d+)", env)
            qc = int(m.group(1)) if m else chunk
            keep = []
            i = 0
            while i < len(a):          # drop the h2h / generator arguments; keep chunk and the common ones
                if a[i] in ("--modes", "--scenarios", "--gen-chunk", "--max-output-size", "--timeline", "--engram-drop",
                            "--cache-tokens", "--ngram-draft"):
                    i += 1 if a[i] == "--engram-drop" else 2
                    continue
                keep.append(a[i]); i += 1
            qa = " ".join(shlex.quote(x) for x in keep) + f" --modes quality --quality-tokens 65536 --quality-chunk {qc} " + \
                " ".join(f"--anchor {x}" for x in ANCHORS)
            dd = f"#@ est=7 timeout=900 minavail={kv(d, 'minavail') or 92}"
            out.append(emit(f"Q-{src}", dd, env, qa, f"64K teacher-forced quality of {src} (quality chunk {qc})"))
        elif item.startswith("s4:"):
            src = item[3:]
            d, env, args = L[src]
            a = shlex.split(args)
            keep = []
            i = 0
            while i < len(a):
                if a[i] in ("--modes", "--scenarios"):
                    i += 2
                    continue
                keep.append(a[i]); i += 1
            sa = "--modes longctx --long-prompt 524288 --long-decode-repeats 3 " + " ".join(shlex.quote(x) for x in keep)
            dd = f"#@ est=15 timeout=1800 minavail={kv(d, 'minavail') or 87}"
            out.append(emit(f"{src}-S4", dd, env, sa, f"S4-512K safety / long-context run of {src}"))
        else:
            src, _, new = item.partition("=")
            d, env, args = L[src]
            out.append(emit(new or f"{src}-2", d, env, args))
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
