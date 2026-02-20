#!/usr/bin/env python3
"""Generate FIF reader test vectors from MNE.

Run from repo root:
    python3 eeg-preproc/scripts/gen_fiff_vectors.py

Requires: mne, safetensors, numpy
"""
import json, struct, numpy as np, mne
from pathlib import Path
mne.set_log_level("WARNING")

OUT = Path(__file__).parent.parent / "tests" / "vectors"
OUT.mkdir(parents=True, exist_ok=True)
FIF = Path(__file__).parent.parent / "data" / "sample1_raw.fif"


def write_st(path: Path, tensors: dict) -> None:
    """Write a safetensors file."""
    DTYPE = {np.dtype("float32"): "F32", np.dtype("float64"): "F64",
             np.dtype("int32"): "I32", np.dtype("int64"): "I64",
             np.dtype("uint8"): "U8"}
    header, parts, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": DTYPE[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        parts.append(raw); offset += len(raw)
    hdr = json.dumps(header, separators=(",",":")).encode()
    pad = (8 - len(hdr) % 8) % 8; hdr += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr))); f.write(hdr)
        for p in parts: f.write(p)
    print(f"  {path.name}  ({offset // 1024} KB)")


def main():
    raw = mne.io.read_raw_fif(FIF, preload=True, verbose=False)
    data = raw.get_data()           # [C, T] float64
    chs  = raw.info["chs"]

    # ── Channel name bytes [C, 16] uint8 ─────────────────────────────────
    ch_names_bytes = np.zeros((len(chs), 16), dtype=np.uint8)
    for i, ch in enumerate(chs):
        b = ch["ch_name"].encode("latin-1")[:16]
        ch_names_bytes[i, : len(b)] = list(b)

    # ── Buffer boundaries from _raw_extras ───────────────────────────────
    extra      = raw._raw_extras[0]
    buf_bounds = extra["bounds"].astype(np.int64)

    # ── Write info tensor ─────────────────────────────────────────────────
    print("[1/3] fiff_sample_info.safetensors")
    write_st(OUT / "fiff_sample_info.safetensors", {
        "nchan":      np.array([raw.info["nchan"]],   dtype=np.int32),
        "sfreq":      np.array([raw.info["sfreq"]],   dtype=np.float64),
        "ntimes":     np.array([raw.n_times],          dtype=np.int32),
        "first_samp": np.array([raw.first_samp],       dtype=np.int32),
        "ch_names":   ch_names_bytes,
        "ch_cal":     np.array([ch["cal"]       for ch in chs], dtype=np.float32),
        "ch_range":   np.array([ch["range"]     for ch in chs], dtype=np.float32),
        "ch_kind":    np.array([int(ch["kind"]) for ch in chs], dtype=np.int32),
        "ch_locs":    np.array([ch["loc"]       for ch in chs], dtype=np.float32),
        "ch_unit":    np.array([int(ch["unit"]) for ch in chs], dtype=np.int32),
        "ch_coil":    np.array([int(ch["coil_type"]) for ch in chs], dtype=np.int32),
        "buf_bounds": buf_bounds,
    })

    # ── Write full data tensor ────────────────────────────────────────────
    print("[2/3] fiff_sample_data.safetensors")
    write_st(OUT / "fiff_sample_data.safetensors", {
        "data": data.astype(np.float64),
    })

    # ── Write short data (first 256 samples) ─────────────────────────────
    print("[3/3] fiff_sample_data_short.safetensors")
    write_st(OUT / "fiff_sample_data_short.safetensors", {
        "data":     data[:, :256].astype(np.float64),
        "data_f32": data[:, :256].astype(np.float32),
    })

    print("\nAll FIF vectors written to", OUT)


if __name__ == "__main__":
    main()
