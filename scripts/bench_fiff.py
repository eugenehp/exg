#!/usr/bin/env python3
"""Python/MNE baseline benchmark for FIF reading.

Run:  python3 scripts/bench_fiff.py
"""
import timeit, mne, numpy as np
from pathlib import Path

mne.set_log_level("ERROR")
FIF = Path(__file__).parent.parent / "data" / "sample1_raw.fif"

N = 100

# ── Benchmark 1: open header only ─────────────────────────────────────────
def open_raw_header():
    mne.io.read_raw_fif(FIF, preload=False, verbose=False)

t_header = timeit.timeit(open_raw_header, number=N) / N * 1e3

# ── Benchmark 2: open + preload ────────────────────────────────────────────
def open_raw_preload():
    mne.io.read_raw_fif(FIF, preload=True, verbose=False)

t_preload = timeit.timeit(open_raw_preload, number=N) / N * 1e3

# ── Benchmark 3: get_data() on already-loaded raw ─────────────────────────
raw = mne.io.read_raw_fif(FIF, preload=True, verbose=False)

def get_data():
    return raw.get_data()

t_getdata = timeit.timeit(get_data, number=N) / N * 1e3

# ── Benchmark 4: read 1 second slice ─────────────────────────────────────
raw2 = mne.io.read_raw_fif(FIF, preload=False, verbose=False)

def read_slice():
    return raw2[:, :256][0]

t_slice = timeit.timeit(read_slice, number=N) / N * 1e3

print("=== MNE Python FIF read benchmarks (mean over 100 runs) ===")
print(f"  open header only     : {t_header:8.3f} ms")
print(f"  open + preload       : {t_preload:8.3f} ms")
print(f"  get_data() preloaded : {t_getdata:8.3f} ms")
print(f"  read_slice [256 samp]: {t_slice:8.3f} ms")
print()
print("=== Rust exg FIF reader (from cargo bench, Alpine x86-64) ===")
print("  open_raw (header+tree)  :   ~0.176 ms")
print("  read_all_data [12×3840] :   ~0.298 ms")
print("  read_slice [256 samp]   :   ~0.084 ms")
print()
print(f"Speedup open_raw   : {t_header / 0.176:.1f}×")
print(f"Speedup read_all   : {t_preload / 0.298:.1f}×")
print(f"Speedup read_slice : {t_slice / 0.084:.1f}×")