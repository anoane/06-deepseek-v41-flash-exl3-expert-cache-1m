#!/usr/bin/env python3
"""H8: NVML PCIe throughput logger (a separate process, started and stopped by dsv41_bench.py).
usage: pcie_log.py <csv> [period_ms=100]
Writes `t,nvml_index,tx_kbs,rx_kbs` rows (time.time(), KB/s over NVML's 20 ms sample window; TX = GPU -> host,
RX = host -> GPU) every period until SIGTERM. Uses libnvidia-ml through ctypes (no pynvml dependency)."""
import ctypes, signal, sys, time

path = sys.argv[1]
period = (int(sys.argv[2]) if len(sys.argv) > 2 else 100) / 1000.0
stop = False


def on_term(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)
nv = ctypes.CDLL("libnvidia-ml.so.1")
assert nv.nvmlInit_v2() == 0, "nvmlInit failed"
n = ctypes.c_uint()
assert nv.nvmlDeviceGetCount_v2(ctypes.byref(n)) == 0
hs = []
for i in range(n.value):
    h = ctypes.c_void_p()
    assert nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0
    hs.append(h)
v = ctypes.c_uint()
with open(path, "w", buffering = 1) as f:
    f.write("t,idx,tx_kbs,rx_kbs\n")
    while not stop:
        t = time.time()
        for i, h in enumerate(hs):
            tx = rx = -1
            if nv.nvmlDeviceGetPcieThroughput(h, 0, ctypes.byref(v)) == 0:
                tx = v.value
            if nv.nvmlDeviceGetPcieThroughput(h, 1, ctypes.byref(v)) == 0:
                rx = v.value
            f.write(f"{t:.3f},{i},{tx},{rx}\n")
        dt = period - (time.time() - t)
        if dt > 0:
            time.sleep(dt)
nv.nvmlShutdown()
