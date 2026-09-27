#!/usr/bin/env python3
"""H6: the prose token source for S5 (domain switch): wikitext-2 train (raw) tokenized with the model's tokenizer,
saved as int32 .npy, plus a check that no 64-token window of it occurs in the canonical (code) ids.
usage: make_prose_ids.py <wiki.train.raw> <tokenizer.json> <canonical_ids.npy> <out.npy>"""
import hashlib, json, sys, time
import numpy as np
from tokenizers import Tokenizer

raw, tokf, canf, out = sys.argv[1:5]
t0 = time.time()
tok = Tokenizer.from_file(tokf)
text = open(raw, encoding = "utf-8").read()
# one encode per article-sized block keeps memory flat; blocks split on blank lines, re-joined exactly
blocks, cur, n = [], [], 0
for line in text.split("\n"):
    cur.append(line)
    n += len(line) + 1
    if n > 200000 and line.strip() == "":
        blocks.append("\n".join(cur) + "\n")
        cur, n = [], 0
if cur:
    blocks.append("\n".join(cur))
assert "".join(blocks) == text or "".join(blocks) == text + "\n"
ids = []
for b in tok.encode_batch(blocks, add_special_tokens = False):
    ids.extend(b.ids)
ids = np.asarray(ids, dtype = np.int32)
canon = np.load(canf).astype(np.int32)
W = 64


def window_hashes(a):
    b = a.tobytes()
    return {hash(b[4 * i:4 * i + 4 * W]) for i in range(len(a) - W + 1)}


hc = window_hashes(canon)
b = ids.tobytes()
hits = [i for i in range(0, len(ids) - W + 1) if hash(b[4 * i:4 * i + 4 * W]) in hc]
# a hash hit is confirmed against the canonical ids byte for byte (hash() collisions are possible)
cb = canon.tobytes()
conf = [i for i in hits if b[4 * i:4 * i + 4 * W] in cb]
np.save(out, ids)
rep = {"source": raw, "tokenizer": tokf, "tokens": int(len(ids)), "blocks": len(blocks),
       "sha256_12": hashlib.sha256(ids.tobytes()).hexdigest()[:12],
       "windows_64_in_canonical": len(conf), "hash_hits": len(hits), "vocab_max": int(ids.max()),
       "seconds": round(time.time() - t0, 1)}
json.dump(rep, open(out + ".json", "w"), indent = 1)
print(json.dumps(rep))
sys.exit(0 if not conf else 1)
