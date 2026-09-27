#!/usr/bin/env python3
"""H4: drop the page cache of the DeepSeek-V4.1 engram tables (layers.*.engram.embed.weight / .scale) with
posix_fadvise(DONTNEED) over their byte ranges, read from the safetensors headers. Every process of a head-to-head
then starts with the same cold engram page cache, without a global drop_caches.
usage: engram_drop.py [model_dir]   (as a module: drop(model_dir) -> dict)"""
import json, os, struct, sys

DEFAULT_MODEL = "/root/serve/models/DeepSeek-V4.1-Flash-EXL3-3.0bpw"


def engram_ranges(model_dir, pattern = (".engram.embed.weight", ".engram.embed.scale")):
    """{file: [(name, start, end)]} absolute byte ranges of the engram tables"""
    ix = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    files = {}
    for k, f in ix.items():
        if k.endswith(pattern):
            files.setdefault(f, []).append(k)
    out = {}
    for f, keys in sorted(files.items()):
        p = os.path.join(model_dir, f)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        out[p] = [(k, base + hdr[k]["data_offsets"][0], base + hdr[k]["data_offsets"][1]) for k in sorted(keys)]
    return out


def resident_bytes(path, start, end):
    """Page-cache resident bytes of [start, end) of a file (mincore over a read-only mapping)"""
    import ctypes, mmap
    libc = ctypes.CDLL(None, use_errno = True)
    pg = mmap.PAGESIZE
    a0 = start - start % pg
    ln = end - a0
    if ln <= 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        addr = libc.mmap(None, ln, mmap.PROT_READ, mmap.MAP_SHARED, fd, a0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            npg = (ln + pg - 1) // pg
            vec = (ctypes.c_ubyte * npg)()
            if libc.mincore(ctypes.c_void_p(addr), ln, vec) != 0:
                return None
            return sum(v & 1 for v in vec) * pg
        finally:
            libc.munmap(ctypes.c_void_p(addr), ln)
    finally:
        os.close(fd)


def drop(model_dir = DEFAULT_MODEL, measure = True):
    rng = engram_ranges(model_dir)
    out = {"files": {}, "bytes": 0, "resident_before": 0, "resident_after": 0}
    for p, rs in rng.items():
        fd = os.open(p, os.O_RDONLY)
        try:
            for k, s, e in rs:
                rb = resident_bytes(p, s, e) if measure else None
                os.posix_fadvise(fd, s, e - s, os.POSIX_FADV_DONTNEED)
                ra = resident_bytes(p, s, e) if measure else None
                out["files"].setdefault(os.path.basename(p), []).append(
                    {"tensor": k, "bytes": e - s, "resident_before": rb, "resident_after": ra})
                out["bytes"] += e - s
                if measure and rb is not None and ra is not None:
                    out["resident_before"] += rb
                    out["resident_after"] += ra
        finally:
            os.close(fd)
    return out


if __name__ == "__main__":
    print(json.dumps(drop(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL), indent = 1))
