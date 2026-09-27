# Recurrent-cache fix (2026-09-26 ~23:00 CEST)
The D-rc run crashed with KeyError('dictionary is empty') for recurrent_cache_mb=0: an upstream bug in
RecurrentCache.put() (exllamav3/cache/recurrent.py) whenever the cache is smaller than one checkpoint.
Fixed on dsv41-all as commit Generator: Drop a recurrent checkpoint larger than the whole recurrent cache
(pushed tip 5e55c062, base upstream dev 988fa662 v1.5.2; test tests/test_recurrent_cache_size_.py).
Python only: the native build is unchanged. To re-run D-rc (recurrent_cache_mb=0/4096), use
SRC=/root/rnd/release/all/src-5e55c0627ece with NATIVE=/root/rnd/release/all/native-5e55c0627ece (a symlink to native-b8105f2d6a05).
Every other recipe measurement is unaffected (it only changes behaviour when a checkpoint does not fit).
