#!/usr/bin/env python3
"""
compare.py — EEG preprocessing: Rust vs Python/MNE comparison.

What it does
────────────
1. Builds the `pipeline_steps` Rust binary (release).
2. Runs it on sample1_raw.fif → /tmp/rust_steps.safetensors.
3. Runs the identical Python/MNE pipeline step-by-step.
4. Measures wall-clock timing for every step in both languages.
5. Produces a printed report + 5 matplotlib figures.

Usage
─────
    python3 exg/scripts/compare.py [--fif PATH] [--out-dir DIR]

Figures saved
─────────────
    01_raw_signal.png          – Raw EEG (12 channels, 15 s)
    02_pipeline_overlay.png    – Python vs Rust overlay at each step
    03_error_per_step.png      – Max/mean/std absolute error per step
    04_performance.png         – Timing bar chart (log scale)
    05_epoch_comparison.png    – Final epoch comparison (3 epochs × 12 ch)
"""

import argparse
import os
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

mne.set_log_level("ERROR")

# ── Paths ──────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parent.parent
CARGO  = REPO / "Cargo.toml"
BIN    = Path(os.environ.get("EXG_TARGET_DIR", "/tmp/exg-target")) / "release" / "pipeline_steps"
DEFAULT_FIF = REPO / "data/sample1_raw.fif"

# ── Safetensors loader ─────────────────────────────────────────────────────
def load_st(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    n   = int.from_bytes(raw[:8], "little")
    hdr = json.loads(raw[8:8 + n])
    ds  = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        buf   = raw[ds + s : ds + e]
        dtype = {"F32": "<f4", "F64": "<f8", "I32": "<i4", "I64": "<i8"}.get(v["dtype"])
        if dtype is None:
            continue
        arr = np.frombuffer(buf, dtype=dtype).reshape(v["shape"])
        out[k] = arr.astype(np.float64)
    return out

# ── Build ──────────────────────────────────────────────────────────────────
def build_binary() -> bool:
    print("Building pipeline_steps (release) …")
    r = subprocess.run(
        ["cargo", "build", "--release",
         "--bin", "pipeline_steps",
         "--manifest-path", str(CARGO)],
        env={**__import__("os").environ,
             "CARGO_TARGET_DIR": str(BIN.parent.parent)},
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("Build failed:\n" + r.stderr[-2000:])
        return False
    print("  OK →", BIN)
    return True

# ── Run Rust binary ────────────────────────────────────────────────────────
def run_rust(fif: Path, out: Path, n_runs: int = 5) -> tuple[dict, list[float], dict]:
    """
    Returns (arrays, wall_times, internal_step_times).
    internal_step_times: dict of step→ms, best over n_runs.
    """
    wall_times = []
    internal_best: dict[str, float] = {}

    for _ in range(n_runs):
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(BIN), "--fif", str(fif), "--output", str(out)],
            capture_output=True,
        )
        wall_times.append(time.perf_counter() - t0)
        if r.returncode != 0:
            sys.exit(f"pipeline_steps failed:\n{r.stderr.decode()}")

        # Parse "TIMING fif=Xms resample=Xms ..." from stderr.
        for line in r.stderr.decode().splitlines():
            if not line.startswith("TIMING "):
                continue
            for token in line.split()[1:]:
                key, val = token.split("=")
                ms = float(val.rstrip("ms"))
                if key not in internal_best or ms < internal_best[key]:
                    internal_best[key] = ms

    data = load_st(out)
    return data, wall_times, internal_best

# ── Python / MNE pipeline (step-by-step with timing) ──────────────────────
def run_python(fif: Path, n_runs: int = 5) -> tuple[dict, dict[str, float]]:
    """
    Returns (arrays_dict, timing_dict).
    arrays_dict keys match the Rust output keys.
    """
    def _pipeline_once():
        steps = {}

        # ── Read FIF ──────────────────────────────────────────────────────
        t = time.perf_counter()
        raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
        t_fif = time.perf_counter() - t
        data_raw = raw.get_data().astype(np.float32)   # [C, T]
        sfreq = raw.info["sfreq"]
        steps["raw"] = data_raw
        steps["_t_fif"] = t_fif

        # ── Resample ───────────────────────────────────────────────────────
        t = time.perf_counter()
        raw.resample(256.0, verbose=False)
        steps["_t_resample"] = time.perf_counter() - t
        steps["resample"] = raw.get_data().astype(np.float32)

        # ── Highpass filter ────────────────────────────────────────────────
        t = time.perf_counter()
        raw.filter(0.5, None, verbose=False)
        steps["_t_hp"] = time.perf_counter() - t
        steps["hp"] = raw.get_data().astype(np.float32)

        # ── Average reference ──────────────────────────────────────────────
        t = time.perf_counter()
        raw.set_eeg_reference("average", projection=False, verbose=False)
        steps["_t_ref"] = time.perf_counter() - t
        steps["ref"] = raw.get_data().astype(np.float32)

        # ── Z-score ────────────────────────────────────────────────────────
        t = time.perf_counter()
        d = raw.get_data()
        mean, std = d.mean(), d.std()
        d = ((d - mean) / std).astype(np.float32)
        steps["_t_zscore"] = time.perf_counter() - t
        steps["zscore"] = d

        # ── Epoch + baseline ───────────────────────────────────────────────
        # Overwrite raw data with z-scored values first
        raw_z = mne.io.RawArray(d.astype(np.float64), raw.info.copy(), verbose=False)
        t = time.perf_counter()
        epochs = mne.make_fixed_length_epochs(
            raw_z, duration=5.0, preload=True, verbose=False)
        epochs.apply_baseline((None, None))
        steps["_t_epoch"] = time.perf_counter() - t
        ep_data = epochs.get_data().astype(np.float32)   # [E, C, 1280]
        for i in range(ep_data.shape[0]):
            steps[f"epoch_{i}"] = ep_data[i]
            steps[f"final_{i}"] = (ep_data[i] / 10.0)
        steps["n_epochs"] = ep_data.shape[0]
        steps["zscore_mean"] = float(mean)
        steps["zscore_std"]  = float(std)
        return steps

    # Warm-up run to get the arrays.
    result = _pipeline_once()

    # Multiple runs for timing (just the pipeline, not array extraction).
    timing_keys = ["_t_fif", "_t_resample", "_t_hp", "_t_ref", "_t_zscore", "_t_epoch"]
    timing_acc  = {k: [] for k in timing_keys}
    for _ in range(n_runs):
        r = _pipeline_once()
        for k in timing_keys:
            timing_acc[k].append(r[k])

    timing = {k.lstrip("_t_"): min(v) for k, v in timing_acc.items()}
    return result, timing

# ── Precision analysis ─────────────────────────────────────────────────────
def precision_report(py: dict, rs: dict, steps: list[str]) -> dict:
    stats = {}
    for key in steps:
        if key not in py or key not in rs:
            continue
        a = np.asarray(py[key], dtype=np.float64)
        b = np.asarray(rs[key], dtype=np.float64)
        if a.shape != b.shape:
            continue
        err = np.abs(a - b)
        stats[key] = {
            "max":    float(err.max()),
            "mean":   float(err.mean()),
            "std":    float(err.std()),
            "rel_pct": float(err.max() / (np.abs(a).mean() + 1e-30) * 100),
        }
    return stats

# ── Figures ────────────────────────────────────────────────────────────────

COLORS = {
    "python": "#2196F3",   # blue
    "rust":   "#F44336",   # red
    "diff":   "#4CAF50",   # green
    "bg":     "#F5F5F5",
}

CH_NAMES = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8"]

def fig_raw_signal(py: dict, out_dir: Path, sfreq: float = 256.0) -> None:
    """Figure 1 – Raw 12-channel EEG time series."""
    data = py["raw"]                  # [12, T]
    n_ch, n_t = data.shape
    t = np.arange(n_t) / sfreq

    fig, axes = plt.subplots(n_ch, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("Raw EEG Signal (from sample1_raw.fif)", fontsize=14, y=0.99)

    cmap = plt.cm.tab20
    for ch, ax in enumerate(axes):
        color = cmap(ch / n_ch)
        ax.plot(t, data[ch] * 1e6, color=color, lw=0.6, rasterized=True)
        ax.set_ylabel(CH_NAMES[ch], rotation=0, labelpad=28, va="center", fontsize=8)
        ax.yaxis.set_label_position("right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        if ch < n_ch - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Time (s)")
    fig.text(0.01, 0.5, "Amplitude (µV)", va="center", rotation="vertical", fontsize=9)
    plt.tight_layout(rect=[0.03, 0, 1, 0.99])
    path = out_dir / "01_raw_signal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path.name}")


def fig_pipeline_overlay(py: dict, rs: dict, out_dir: Path, sfreq: float = 256.0) -> None:
    """Figure 2 – Python vs Rust overlay for each processing step (channel Fp1)."""
    steps     = ["raw", "resample", "hp", "ref", "zscore"]
    step_labels = {
        "raw":      "Raw FIF data",
        "resample": "After resample (256 Hz)",
        "hp":       "After highpass (0.5 Hz FIR)",
        "ref":      "After average reference",
        "zscore":   "After global z-score",
    }
    ch = 0   # Fp1

    fig, axes = plt.subplots(len(steps), 3, figsize=(22, 14),
                             gridspec_kw={"width_ratios": [3, 2, 1]})
    fig.suptitle(f"Python vs Rust — channel {CH_NAMES[ch]} (Fp1)", fontsize=13)

    for row, step in enumerate(steps):
        ax_ts   = axes[row, 0]
        ax_pct  = axes[row, 1]
        ax_hist = axes[row, 2]

        if step not in py or step not in rs:
            continue

        py_sig = np.asarray(py[step][ch], dtype=np.float64)
        rs_sig = np.asarray(rs[step][ch], dtype=np.float64)
        diff   = py_sig - rs_sig
        t      = np.arange(len(py_sig)) / sfreq

        # Normalise by the mean absolute value of the Python signal.
        # For near-zero signals (e.g. after re-referencing a flat channel)
        # fall back to the global std so the percentage stays meaningful.
        scale  = np.abs(py_sig).mean()
        if scale < 1e-30:
            scale = py_sig.std() + 1e-30
        pct    = diff / scale * 100.0          # signed normalised % error
        avg_pct = np.abs(pct).mean()

        # --- Col 0: Overlay time series ---
        ax_ts.plot(t, py_sig, color=COLORS["python"], lw=1.0, label="Python", alpha=0.9)
        ax_ts.plot(t, rs_sig, color=COLORS["rust"],   lw=0.6, label="Rust",
                   linestyle="--", alpha=0.85)
        ax_ts.set_ylabel(step_labels[step], fontsize=8)
        ax_ts.spines[["top", "right"]].set_visible(False)
        ax_ts.tick_params(labelsize=7)
        if row == 0:
            ax_ts.legend(fontsize=8, loc="upper right")
        if row < len(steps) - 1:
            ax_ts.tick_params(labelbottom=False)
        else:
            ax_ts.set_xlabel("Time (s)")

        # --- Col 1: Signed normalised % difference over time ---
        ax_pct.axhline(0, color="gray", lw=0.6, linestyle=":")
        ax_pct.fill_between(t,  pct, 0,
                            where=pct >= 0, color=COLORS["rust"],   alpha=0.45)
        ax_pct.fill_between(t,  pct, 0,
                            where=pct <  0, color=COLORS["python"], alpha=0.45)
        ax_pct.plot(t, pct, color=COLORS["diff"], lw=0.5, alpha=0.7)
        ax_pct.set_ylabel("err / |mean| (%)", fontsize=7)
        ax_pct.set_title(f"avg |err| = {avg_pct:.4f}%\nmax |err| = {np.abs(pct).max():.4f}%",
                         fontsize=7)
        ax_pct.spines[["top", "right"]].set_visible(False)
        ax_pct.tick_params(labelsize=7)
        if row < len(steps) - 1:
            ax_pct.tick_params(labelbottom=False)
        else:
            ax_pct.set_xlabel("Time (s)")

        # --- Col 2: Absolute error histogram ---
        ax_hist.hist(np.abs(diff), bins=35, color=COLORS["diff"],
                     edgecolor="none", alpha=0.85)
        ax_hist.set_xlabel("|err|", fontsize=8)
        ax_hist.set_title(f"max={np.abs(diff).max():.1e}\nmean={np.abs(diff).mean():.1e}",
                          fontsize=7)
        ax_hist.spines[["top", "right"]].set_visible(False)
        ax_hist.tick_params(labelsize=7)
        if row < len(steps) - 1:
            ax_hist.tick_params(labelbottom=False)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.06,
                        hspace=0.12, wspace=0.32)
    path = out_dir / "02_pipeline_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path.name}")


def fig_error_per_step(stats: dict, out_dir: Path) -> None:
    """Figure 3 – Max / mean absolute error at each pipeline step."""
    steps = [k for k in ["raw","resample","hp","ref","zscore"] if k in stats]
    # Add epoch steps
    for k in sorted(stats):
        if k.startswith("epoch_") or k.startswith("final_"):
            steps.append(k)

    maxes = [stats[s]["max"]  for s in steps]
    means = [stats[s]["mean"] for s in steps]

    x = np.arange(len(steps))
    w = 0.35

    # fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    fig, ax = plt.subplots(figsize=(14, 5))
    bars_max  = ax.bar(x - w/2, maxes, w, label="Max |err|",  color=COLORS["rust"],   alpha=0.85)
    bars_mean = ax.bar(x + w/2, means, w, label="Mean |err|", color=COLORS["python"], alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in steps], fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Absolute error (log scale)")
    ax.set_title("Rust vs Python: Numerical Precision per Pipeline Step")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate max values
    for bar, v in zip(bars_max, maxes):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.3, f"{v:.1e}",
                ha="center", va="bottom", fontsize=7, rotation=45)

    # Float32 epsilon reference line
    eps32 = np.finfo(np.float32).eps
    ax.axhline(eps32, color="gray", lw=1, linestyle=":", label=f"f32 ε = {eps32:.1e}")
    ax.legend(fontsize=9)

    path = out_dir / "03_error_per_step.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path.name}")


def fig_performance(py_timing: dict, rust_wall: list[float],
                    rust_step_ms: dict, out_dir: Path) -> None:
    """Figure 4 – Wall-clock timing comparison (Python vs Rust)."""
    step_map = {
        "fif":      "Read\nFIF",
        "resample": "Resample",
        "hp":       "HP\nfilter",
        "ref":      "Avg\nref",
        "zscore":   "Z-score",
        "epoch":    "Epoch",
    }
    py_steps = list(step_map.keys())
    py_ms    = [py_timing.get(s, 0) * 1000 for s in py_steps]
    ru_ms    = [rust_step_ms.get(s, 0)     for s in py_steps]

    total_py = sum(py_ms)
    total_ru = sum(rust_step_ms.values())

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.5], wspace=0.38)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    fig.suptitle("Performance: Python/MNE vs Rust — best of 5 runs", fontsize=13)

    # ── Left: stacked bar per-step ─────────────────────────────────────────
    cmap_py = plt.cm.Blues(np.linspace(0.4, 0.9, len(py_steps)))
    cmap_ru = plt.cm.Reds (np.linspace(0.4, 0.9, len(py_steps)))
    bot_py = bot_ru = 0
    labels = [step_map[s] for s in py_steps]

    for label, py_t, ru_t, cp, cr in zip(labels, py_ms, ru_ms, cmap_py, cmap_ru):
        ax1.bar("Python", py_t, bottom=bot_py, color=cp)
        ax1.bar("Rust",   ru_t, bottom=bot_ru, color=cr)
        if py_t > 0.5:
            ax1.text(0, bot_py + py_t/2,
                     f"{label}\n{py_t:.2f}ms",
                     ha="center", va="center", fontsize=7, color="white")
        if ru_t > 0.03:
            ax1.text(1, bot_ru + ru_t/2,
                     f"{label}\n{ru_t:.2f}ms",
                     ha="center", va="center", fontsize=7, color="white")
        bot_py += py_t
        bot_ru += ru_t

    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"Stacked step time\nPy={total_py:.1f} ms  Ru={total_ru:.2f} ms")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Middle: grouped bar per step ───────────────────────────────────────
    x = np.arange(len(py_steps))
    w = 0.38
    ax2.bar(x - w/2, py_ms, w, color=COLORS["python"], alpha=0.85, label="Python")
    ax2.bar(x + w/2, ru_ms, w, color=COLORS["rust"],   alpha=0.85, label="Rust")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([step_map[s] for s in py_steps], fontsize=8)
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_title("Per-step timing (log scale)")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Right: speedup bars ────────────────────────────────────────────────
    su_steps  = [s for s in py_steps if rust_step_ms.get(s, 0) > 0]
    su_labels = [step_map[s] for s in su_steps]
    speedups  = [(py_timing.get(s, 0) * 1000) / rust_step_ms[s] for s in su_steps]
    speedups.append(total_py / total_ru)
    su_labels.append("Total")

    colors_su = [COLORS["python"]] * len(su_steps) + ["#9C27B0"]
    bars = ax3.barh(su_labels, speedups, color=colors_su, alpha=0.85)
    ax3.axvline(1, color="gray", lw=1, linestyle="--")
    ax3.set_xlabel("Speedup (Python time / Rust time)")
    ax3.set_title("Rust speedup")
    ax3.spines[["top", "right"]].set_visible(False)
    for bar, su in zip(bars, speedups):
        ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{su:.1f}×", va="center", fontsize=9, weight="bold")

    path = out_dir / "04_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def fig_epoch_comparison(py: dict, rs: dict, out_dir: Path, sfreq: float = 256.0) -> None:
    """Figure 5 – Final epochs: Python vs Rust, all channels."""
    n_epochs = int(py.get("n_epochs", 0))
    if n_epochs == 0:
        return

    t = np.arange(1280) / sfreq   # 5 s @ 256 Hz

    fig, axes = plt.subplots(n_epochs, 12, figsize=(22, 3 * n_epochs),
                             sharex=True, sharey="row")
    if n_epochs == 1:
        axes = axes[np.newaxis, :]   # ensure 2-D
    fig.suptitle("Final epochs (after z-score ÷ 10): Python (blue) vs Rust (red)",
                 fontsize=13)

    for e in range(n_epochs):
        key_py = f"final_{e}"
        key_rs = f"final_{e}"
        if key_py not in py or key_rs not in rs:
            continue
        py_ep = np.asarray(py[key_py], dtype=np.float64)   # [12, 1280]
        rs_ep = np.asarray(rs[key_rs], dtype=np.float64)
        max_err = float(np.abs(py_ep - rs_ep).max())

        for ch in range(12):
            ax = axes[e, ch]
            ax.plot(t, py_ep[ch], color=COLORS["python"], lw=0.8, alpha=0.9)
            ax.plot(t, rs_ep[ch], color=COLORS["rust"],   lw=0.5,
                    linestyle="--", alpha=0.8)
            if e == 0:
                ax.set_title(CH_NAMES[ch], fontsize=8)
            if ch == 0:
                ax.set_ylabel(f"Epoch {e}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)
            if e < n_epochs - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("s", fontsize=7)

        fig.text(0.01, (n_epochs - e - 0.5) / n_epochs,
                 f"max |err| = {max_err:.2e}", fontsize=7,
                 va="center", ha="left", color="gray")

    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.06,
                        hspace=0.25, wspace=0.18)
    path = out_dir / "05_epoch_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path.name}")


# ── Report printer ─────────────────────────────────────────────────────────

def print_precision_table(stats: dict) -> None:
    keys = [k for k in ["raw","resample","hp","ref","zscore"] if k in stats]
    keys += sorted(k for k in stats if k.startswith("epoch_") or k.startswith("final_"))
    
    W = 12
    hdr = f"{'Step':<14} {'Max |err|':>12} {'Mean |err|':>12} {'Std |err|':>12} {'Rel %':>8}"
    print(hdr)
    print("─" * len(hdr))
    for k in keys:
        s = stats[k]
        print(f"{k:<14} {s['max']:>12.3e} {s['mean']:>12.3e} {s['std']:>12.3e} {s['rel_pct']:>7.4f}%")
    print()

def print_timing_table(py_timing: dict, rust_wall: list[float],
                       rust_step_ms: dict) -> None:
    step_map = {
        "fif":      "Read FIF",
        "resample": "Resample",
        "hp":       "HP filter",
        "ref":      "Avg ref",
        "zscore":   "Z-score",
        "epoch":    "Epoch",
    }
    total_py_ms   = sum(v * 1000 for v in py_timing.values())
    rust_total_ms = sum(rust_step_ms.values())
    rust_wall_ms  = min(rust_wall) * 1000

    hdr = f"{'Step':<18} {'Python (ms)':>13} {'Rust (ms)':>11} {'Speedup':>9}"
    print(hdr)
    print("─" * 55)
    for k, label in step_map.items():
        py_ms = py_timing.get(k, 0) * 1000
        ru_ms = rust_step_ms.get(k, 0)
        su    = f"{py_ms/ru_ms:.1f}×" if ru_ms > 0 else "—"
        print(f"{label:<18} {py_ms:>13.3f} {ru_ms:>11.3f} {su:>9}")
    print("─" * 55)
    print(f"{'TOTAL (steps)':<18} {total_py_ms:>13.3f} {rust_total_ms:>11.3f} "
          f"{total_py_ms/rust_total_ms:>8.1f}×")
    print(f"{'Rust wall (proc)':<18} {'':>13} {rust_wall_ms:>11.3f}  ← incl. I/O + startup")
    print()

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fif",     default=str(DEFAULT_FIF),
                        help="Path to .fif file")
    parser.add_argument("--out-dir", default=str(REPO / "comparison"),
                        help="Directory for output figures")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip cargo build (use existing binary)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of timing runs")
    args = parser.parse_args()

    fif     = Path(args.fif)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rust_st = out_dir / "rust_steps.safetensors"

    if not fif.exists():
        sys.exit(f"FIF file not found: {fif}")

    # ── Build ──────────────────────────────────────────────────────────────
    if not args.no_build:
        if not build_binary():
            sys.exit(1)

    # ── Run pipelines ──────────────────────────────────────────────────────
    print(f"\nRunning Rust pipeline ({args.runs} runs) …")
    rs, rust_times, rust_step_ms = run_rust(fif, rust_st, n_runs=args.runs)
    print(f"  subprocess wall: best={min(rust_times)*1000:.2f} ms  "
          f"mean={sum(rust_times)/len(rust_times)*1000:.2f} ms")
    print(f"  internal steps:  " +
          "  ".join(f"{k}={v:.3f}ms" for k, v in rust_step_ms.items()))

    print(f"\nRunning Python/MNE pipeline ({args.runs} runs) …")
    py, py_timing = run_python(fif, n_runs=args.runs)

    sfreq = 256.0  # after resample

    # ── Precision ──────────────────────────────────────────────────────────
    cmp_keys = ["raw", "resample", "hp", "ref", "zscore"]
    n_ep = int(py.get("n_epochs", 0))
    cmp_keys += [f"epoch_{i}" for i in range(n_ep)]
    cmp_keys += [f"final_{i}" for i in range(n_ep)]
    stats = precision_report(py, rs, cmp_keys)

    # ── Print report ───────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  EEG Preprocessing — Rust vs Python Comparison Report")
    print("═" * 62)
    print(f"\n  FIF file : {fif}")
    print(f"  Channels : {py['raw'].shape[0]}  "
          f"  sfreq={sfreq} Hz  "
          f"  n_epochs={n_ep}")
    print()

    print("── Numerical Precision (Rust vs Python) ─────────────────────")
    print_precision_table(stats)

    print("── Timing ────────────────────────────────────────────────────")
    print_timing_table(py_timing, rust_times, rust_step_ms)

    print("── Z-score parameters ────────────────────────────────────────")
    print(f"  Python : mean={py['zscore_mean']:.6e}  std={py['zscore_std']:.6e}")
    if "zscore_mean" in rs:
        print(f"  Rust   : mean={float(rs['zscore_mean'].flat[0]):.6e}  "
              f"std={float(rs['zscore_std'].flat[0]):.6e}")
    print()

    # ── Figures ────────────────────────────────────────────────────────────
    print("── Generating figures ────────────────────────────────────────")
    fig_raw_signal(py, out_dir, sfreq=float(
        mne.io.read_raw_fif(fif, preload=False, verbose=False).info["sfreq"]))
    fig_pipeline_overlay(py, rs, out_dir, sfreq=sfreq)
    fig_error_per_step(stats, out_dir)
    fig_performance(py_timing, rust_times, rust_step_ms, out_dir)
    fig_epoch_comparison(py, rs, out_dir, sfreq=sfreq)

    print(f"\nAll figures written to {out_dir}/")
    print("═" * 62)


if __name__ == "__main__":
    main()
