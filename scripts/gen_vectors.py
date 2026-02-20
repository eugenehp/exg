#!/usr/bin/env python3
"""
gen_vectors.py — generate all test vectors for eeg-preproc Rust tests.

Each vector is a safetensors file containing the inputs and expected outputs
produced by the reference Python/MNE/SciPy implementation.

Run from the repo root:
    python3 eeg-preproc/scripts/gen_vectors.py
"""
import json, struct, sys, warnings
from pathlib import Path

import numpy as np
import scipy.signal as sig
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

OUT_DIR = Path(__file__).parent.parent / "tests" / "vectors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(seed=42)

# ── Safetensors writer ────────────────────────────────────────────────────────

def write_st(path: Path, tensors: dict[str, np.ndarray]) -> None:
    DTYPE = {
        np.dtype("float32"): "F32",
        np.dtype("int32"):   "I32",
        np.dtype("float64"): "F64",
    }
    header, parts, offset = {}, [], 0
    for name, arr in tensors.items():
        if arr.dtype not in DTYPE:
            arr = arr.astype(np.float32)
        raw = arr.tobytes()
        header[name] = {
            "dtype": DTYPE[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        parts.append(raw)
        offset += len(raw)
    hdr = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - len(hdr) % 8) % 8
    hdr += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for p in parts:
            f.write(p)
    print(f"  Wrote {path.name}  ({offset // 1024} KB)")


# ── Synthetic EEG generator ───────────────────────────────────────────────────

def make_eeg(n_ch: int, n_samples: int, sfreq: float) -> np.ndarray:
    """Synthetic EEG: physiological bands + noise."""
    t = np.arange(n_samples) / sfreq
    x = np.zeros((n_ch, n_samples), dtype=np.float64)
    bands = [(0.5, 0.3), (1.0, 0.5), (4.0, 1.2), (8.0, 0.8),
             (13.0, 0.6), (30.0, 0.3), (50.0, 0.1)]
    for freq, amp in bands:
        phase = RNG.uniform(0, 2 * np.pi, (n_ch, 1))
        x += amp * np.sin(2 * np.pi * freq * t + phase)
    x += 0.2 * RNG.standard_normal((n_ch, n_samples))
    return x.astype(np.float32)


def mne_raw(data: np.ndarray, sfreq: float) -> mne.io.RawArray:
    info = mne.create_info(data.shape[0], sfreq, "eeg")
    return mne.io.RawArray(data.astype(np.float64), info, verbose=False)


# ── 1. Resample vectors ───────────────────────────────────────────────────────

def gen_resample(src_sfreq: int, n_sec: float = 30.0) -> None:
    n_in = int(src_sfreq * n_sec)
    x = make_eeg(8, n_in, float(src_sfreq))

    raw = mne_raw(x, float(src_sfreq))
    raw.resample(256.0, verbose=False)
    y = raw.get_data().astype(np.float32)

    name = f"resample_{src_sfreq}_to_256"
    write_st(OUT_DIR / f"{name}.safetensors", {
        "input":     x,
        "output":    y,
        "src_sfreq": np.array([src_sfreq], dtype=np.float32),
        "dst_sfreq": np.array([256.0],     dtype=np.float32),
    })


# ── 2. FIR filter coefficients ────────────────────────────────────────────────

def gen_filter_coeffs() -> None:
    h = mne.filter.create_filter(
        None, 256.0, l_freq=0.5, h_freq=None,
        filter_length="auto", method="fir", fir_window="hamming",
        fir_design="firwin", phase="zero", verbose=False,
    )
    write_st(OUT_DIR / "filter_coeffs_hp05_256hz.safetensors", {
        "h": h.astype(np.float32),
    })
    print(f"    N={len(h)} taps,  h[N//2]={h[len(h)//2]:.8f}")


# ── 3. FIR filter application ─────────────────────────────────────────────────

def gen_filter_apply() -> None:
    sfreq = 256.0
    x = make_eeg(12, int(60 * sfreq), sfreq)       # 60 s, 12 channels

    raw = mne_raw(x, sfreq)
    raw.filter(0.5, None, verbose=False)
    y = raw.get_data().astype(np.float32)

    write_st(OUT_DIR / "filter_hp05_256hz.safetensors", {
        "input":  x,
        "output": y,
    })


# ── 4. Average reference ─────────────────────────────────────────────────────

def gen_average_reference() -> None:
    sfreq = 256.0
    x = make_eeg(12, int(60 * sfreq), sfreq)

    raw = mne_raw(x, sfreq)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    y = raw.get_data().astype(np.float32)

    write_st(OUT_DIR / "average_reference.safetensors", {
        "input":  x,
        "output": y,
    })


# ── 5. Global z-score ─────────────────────────────────────────────────────────

def gen_zscore() -> None:
    x = make_eeg(12, int(60 * 256), 256.0)
    mean = float(x.mean())
    std  = float(x.std())
    y = ((x - mean) / std).astype(np.float32)

    write_st(OUT_DIR / "zscore_global.safetensors", {
        "input":  x,
        "output": y,
        "mean":   np.array([mean], dtype=np.float32),
        "std":    np.array([std],  dtype=np.float32),
    })


# ── 6. Baseline correction ───────────────────────────────────────────────────

def gen_baseline() -> None:
    sfreq, n_ch, n_ep = 256.0, 12, 10
    epoch_samples = int(5.0 * sfreq)
    x = make_eeg(n_ch, epoch_samples * n_ep, sfreq)
    # Build MNE epochs object.
    raw = mne_raw(x, sfreq)
    events = mne.make_fixed_length_events(raw, duration=5.0)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=5.0 - 1/sfreq,
                        baseline=(None, None), preload=True, verbose=False)
    y = epochs.get_data().astype(np.float32)   # [E, C, T]

    # Corresponding raw input (reshape to [E, C, T]).
    n_e = y.shape[0]
    x_3d = np.stack(
        [x[:, e * epoch_samples:(e + 1) * epoch_samples] for e in range(n_e)],
        axis=0,
    ).astype(np.float32)

    write_st(OUT_DIR / "baseline_correction.safetensors", {
        "input":  x_3d,    # [E, C, T]
        "output": y,        # [E, C, T]
    })


# ── 7. Epoching ───────────────────────────────────────────────────────────────

def gen_epoch() -> None:
    sfreq = 256.0
    x = make_eeg(12, int(200 * sfreq), sfreq)    # 200 s  → 40 epochs of 5 s

    raw = mne_raw(x, sfreq)
    epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)
    epochs.apply_baseline((None, None))
    y = epochs.get_data().astype(np.float32)   # [E, C, T]

    write_st(OUT_DIR / "epoch_1280.safetensors", {
        "input":        x,
        "epochs":       y,
        "epoch_samples": np.array([1280], dtype=np.int32),
    })


# ── 8. Full pipeline ─────────────────────────────────────────────────────────

def gen_full_pipeline(src_sfreq: float = 512.0) -> None:
    n_sec = 60.0
    n_in = int(src_sfreq * n_sec)
    x_raw = make_eeg(12, n_in, src_sfreq)

    # Step-by-step with snapshots.
    raw = mne_raw(x_raw, src_sfreq)

    # 1. Resample.
    raw.resample(256.0, verbose=False)
    after_resample = raw.get_data().astype(np.float32)

    # 2. Highpass.
    raw.filter(0.5, None, verbose=False)
    after_hp = raw.get_data().astype(np.float32)

    # 3. Average reference.
    raw.set_eeg_reference("average", projection=False, verbose=False)
    after_ref = raw.get_data().astype(np.float32)

    # 4. Global z-score.
    d = raw.get_data()
    mean, std = d.mean(), d.std()
    d = (d - mean) / std
    raw._data[:] = d
    after_zscore = raw.get_data().astype(np.float32)

    # 5. Epoch + baseline.
    epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True, verbose=False)
    epochs.apply_baseline((None, None))
    ep_data = epochs.get_data().astype(np.float32)   # [E, C, T]

    # 6. Divide by data_norm=10.
    ep_out = (ep_data / 10.0).astype(np.float32)

    write_st(OUT_DIR / "full_pipeline.safetensors", {
        "input_raw":      x_raw,
        "after_resample": after_resample,
        "after_hp":       after_hp,
        "after_ref":      after_ref,
        "after_zscore":   after_zscore,
        "epochs":         ep_data,          # [E, C, T]  before /data_norm
        "output":         ep_out,           # [E, C, T]  after  /data_norm
        "src_sfreq":      np.array([src_sfreq], dtype=np.float32),
    })


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating test vectors...")

    print("\n[1/8] Resample vectors")
    for rate in [512, 1024, 250, 2000]:
        gen_resample(rate)

    print("\n[2/8] FIR coefficients")
    gen_filter_coeffs()

    print("\n[3/8] FIR application")
    gen_filter_apply()

    print("\n[4/8] Average reference")
    gen_average_reference()

    print("\n[5/8] Z-score")
    gen_zscore()

    print("\n[6/8] Baseline correction")
    gen_baseline()

    print("\n[7/8] Epoching")
    gen_epoch()

    print("\n[8/8] Full pipeline")
    gen_full_pipeline()

    print(f"\nAll vectors written to {OUT_DIR}/")
    total = sum(f.stat().st_size for f in OUT_DIR.glob("*.safetensors"))
    print(f"Total size: {total // 1024} KB")
