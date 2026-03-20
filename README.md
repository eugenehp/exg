# exg

> Native Rust EEG/ECG/EMG preprocessing — 100% numerical parity with MNE-Python, no Python required.

`exg` is a pure-Rust crate for EEG signal processing. Every DSP operation is
ported from [MNE-Python](https://mne.tools) and verified at the coefficient
level against MNE ground truth (200+ tests, < 5 × 10⁻⁸ max error).

_No Python, no BLAS, no C libraries.  Pure Rust + RustFFT._

---

## Workspace

| Crate | Description |
|-------|-------------|
| **`exg`** | Core DSP primitives, file I/O, generic preprocessing pipeline |
| **[`exg-luna`](exg-luna/)** | LUNA seizure-detection pipeline (TCP bipolar montage, 0.1–75 Hz BP, 60 Hz notch) |
| **[`exg-source`](exg-source/)** | Source localisation (eLORETA, MNE/dSPM/sLORETA, forward models, resolution metrics) |

---

## Features

| Category | What's included |
|----------|----------------|
| **File I/O** | FIF (`.fif`), EDF/EDF+ (`.edf`), CSV; HDF5 (`--features hdf5`); safetensors export |
| **FIR Filters** | Highpass, lowpass, bandpass, notch — all via MNE's `_firwin_design` with per-transition sub-filter lengths |
| **DSP** | FFT polyphase resampling, overlap-add zero-phase convolution, average reference |
| **Normalisation** | Global z-score (ddof=0), channel-wise z-score, per-epoch baseline correction |
| **Montages** | TCP bipolar (22-ch TUH), Siena unipolar (29-ch), SEED-V unipolar (62-ch), custom |
| **Pipelines** | Generic `preprocess()` in `exg`; LUNA-specific `preprocess_luna()` in `exg-luna` |
| **Export** | Safetensors batch writer; LUNA-compatible epoch export (via `exg-luna`) |
| **Source localisation** | eLORETA, MNE/dSPM/sLORETA, forward models, resolution metrics (via `exg-source`) |

---

## Quick start

### Generic pipeline (FIF input)

```rust
use exg::{preprocess, PipelineConfig, fiff::open_raw};
use ndarray::Array2;

let raw    = open_raw("data/sample1_raw.fif")?;
let data   = raw.read_all_data()?;                    // [C, T] f64
let pos    = Array2::<f32>::zeros((raw.info.n_chan, 3));
let cfg    = PipelineConfig::default();               // 256 Hz · 0.5 Hz HP · 5 s epochs
let epochs = preprocess(data.mapv(|v| v as f32), pos, raw.info.sfreq as f32, &cfg)?;
// → Vec<([C, 1280], [C, 3])>
```

### LUNA seizure-detection pipeline (EDF input)

Add both crates to your `Cargo.toml`:

```toml
[dependencies]
exg      = "0.0.3"
exg-luna = "0.0.3"
```

```rust
use exg::edf::open_raw_edf;
use exg_luna::{preprocess_luna, LunaPipelineConfig};

let raw    = open_raw_edf("recording.edf")?;
let data   = raw.read_all_data()?;                    // [C, T] f32
let names  = raw.channel_names();
let cfg    = LunaPipelineConfig::default();           // 0.1–75 Hz BP · 60 Hz notch · TCP bipolar
let epochs = preprocess_luna(data, &names, raw.header.sample_rate, &cfg)?;
// → Vec<([22, 1280], channel_names)>
```

### Individual DSP steps

```rust
use exg::filter::{design_bandpass, design_notch, apply_fir_zero_phase};
use exg::normalize::zscore_channelwise_inplace;
use exg::montage::{make_bipolar, TCP_MONTAGE};
use ndarray::Array2;

let mut data: Array2<f32> = /* ... */;

// Bandpass 0.1–75 Hz + notch 60 Hz
let h = design_bandpass(0.1, 75.0, 256.0);
apply_fir_zero_phase(&mut data, &h)?;
let h = design_notch(60.0, 256.0, None, None);
apply_fir_zero_phase(&mut data, &h)?;

// Bipolar montage
let (bipolar, names) = make_bipolar(&data, &ch_names, TCP_MONTAGE);

// Channel-wise z-score
zscore_channelwise_inplace(&mut bipolar);
```

### Export for LUNA inference

```rust
use exg_luna::{LunaEpoch, export_luna_epochs};

let epoch = LunaEpoch {
    signal: bipolar_epoch,            // [22, 1280]
    channel_positions: positions,     // [22, 3]
    channel_names: names,             // ["FP1-F7", "F7-T3", ...]
};
export_luna_epochs(&[epoch], "batch.safetensors".as_ref())?;
```

---

## Pipelines

### Generic pipeline (`exg::preprocess`)

```text
.fif / .edf / .csv
  │
  ├─ open_raw() / open_raw_edf()    native file reader
  ├─ resample()                      FFT polyphase → 256 Hz
  ├─ highpass FIR                    firwin + overlap-add → 0.5 Hz
  ├─ average reference               per-timepoint channel mean removed
  ├─ global z-score                  (data − μ) / σ  (ddof=0)
  ├─ epoch                           non-overlapping 5 s windows
  ├─ baseline correct                per-epoch per-channel mean removed
  └─ ÷ data_norm                     ÷ 10 → std ≈ 0.1
```

### LUNA pipeline (`exg_luna::preprocess_luna`)

```text
.edf (TUH corpus)
  │
  ├─ channel rename                  strip "EEG ", "-REF", "-LE"
  ├─ pick 10-20 channels             21 standard electrodes
  ├─ bandpass FIR                    0.1–75 Hz (MNE _firwin_design)
  ├─ notch FIR                       60 Hz (configurable 50 Hz)
  ├─ resample                        → 256 Hz
  ├─ TCP bipolar montage             22 channels from 21 electrodes
  └─ epoch                           non-overlapping 5 s windows
       │
       └─→ Vec<([22, 1280], channel_names)>
```

---

## Numerical parity with MNE-Python

All filter coefficients are verified at the individual-coefficient level
against MNE-Python 1.11.0.

| Operation | Algorithm match | Max error |
|-----------|:-:|--:|
| FIR design (HP/LP/BP/Notch) | ✅ `_firwin_design` exact | < 5 × 10⁻⁸ (f32 output) |
| Filter application | ✅ `_overlap_add_filter` | < 4 × 10⁻⁶ |
| Resampling | ✅ `_fft_resample` | < 5 × 10⁻⁴ |
| Average reference | ✅ bit-exact | 0 |
| Z-score (global / channel-wise) | ✅ ddof=0 | < 1 × 10⁻⁶ |
| Baseline correction | ✅ bit-exact | 0 |
| EDF read | ✅ digital→physical + unit scaling | < 1 × 10⁻³ (16-bit quantisation) |

Internally, filter design runs in f64 (matching numpy/scipy). The final
coefficients are returned as f32 for the signal path, introducing the sole
precision gap of ~5 × 10⁻⁸.

---

## Crate API

### Preprocessing (`exg`)

```rust
// Full pipeline
pub fn preprocess(data, chan_pos, sfreq, cfg) -> Result<Vec<(Array2, Array2)>>

// File I/O
pub mod fiff { pub fn open_raw(path) -> Result<RawFif> }
pub mod edf  { pub fn open_raw_edf(path) -> Result<RawEdf> }
pub mod csv  { pub fn read_eeg(path) -> Result<(Array2, Vec<String>, f32)> }
#[cfg(feature = "hdf5")]
pub mod hdf5 { pub fn read_dataset(path) -> Result<Vec<HDF5Sample>> }

// Filter design (100% MNE _firwin_design parity)
pub mod filter {
    pub fn design_highpass(l_freq, sfreq) -> Vec<f32>
    pub fn design_lowpass(h_freq, sfreq) -> Vec<f32>
    pub fn design_bandpass(l_freq, h_freq, sfreq) -> Vec<f32>
    pub fn design_notch(freq, sfreq, notch_width, trans_bandwidth) -> Vec<f32>
    pub fn apply_fir_zero_phase(data, h) -> Result<()>
}

// DSP
pub mod resample  { pub fn resample(data, src, dst) -> Result<Array2<f32>> }
pub mod reference { pub fn average_reference_inplace(data) }
pub mod normalize { pub fn zscore_global_inplace(data) -> (f32, f32)
                    pub fn zscore_channelwise_inplace(data)
                    pub fn baseline_correct_inplace(epochs) }
pub mod epoch     { pub fn epoch(data, n) -> Array3<f32> }

// Montages
pub mod montage {
    pub fn make_bipolar(data, names, defs) -> (Array2, Vec<String>)
    pub fn pick_channels(data, names, targets) -> (Array2, Vec<String>)
    pub const TCP_MONTAGE: &[BipolarDef]       // 22-ch TUH bipolar
    pub const SIENA_CHANNELS: &[&str]          // 29-ch unipolar
    pub const SEED_V_CHANNELS: &[&str]         // 62-ch unipolar
}

// Safetensors export
pub mod io {
    pub fn write_batch(epochs, positions, path) -> Result<()>
}
```

### LUNA pipeline (`exg-luna`)

```rust
pub fn preprocess_luna(data, ch_names, sfreq, cfg) -> Result<Vec<(Array2, Vec<String>)>>
pub struct LunaPipelineConfig { /* bandpass, notch, sfreq, epoch_dur, montage */ }
pub const STANDARD_10_20: &[&str]    // 21 electrodes

// Safetensors I/O (luna-rs InputBatch compatible)
pub fn export_luna_epochs(epochs, path) -> Result<()>
pub fn load_luna_epochs(path) -> Result<Vec<LunaEpoch>>
pub struct LunaEpoch { signal, channel_positions, channel_names }
```

### Source localisation (`exg-source`)

Enabled by the default `source` feature on `exg`. Disable with `default-features = false`.

```rust
pub fn make_sphere_forward(elec, src, nn, sphere) -> ForwardOperator
pub fn make_inverse_operator(fwd, cov, depth) -> Result<InverseOperator>
pub fn apply_inverse(data, inv, λ², method) -> Result<SourceEstimate>
// Also: dSPM, sLORETA, eLORETA, SNR estimation, resolution metrics
```

---

## Feature flags

| Flag | Default | What it enables |
|------|:-------:|-----------------|
| `source` | ✅ | Source localisation (eLORETA/MNE/dSPM/sLORETA) via `exg-source` |
| `hdf5` | ❌ | HDF5 dataset reader (requires `libhdf5` system library) |

```toml
# Just preprocessing, no source localisation
exg = { version = "0.0.3", default-features = false }

# Everything including HDF5
exg = { version = "0.0.3", features = ["hdf5"] }
```

---

## Running

```bash
cargo test --workspace           # 200+ tests across exg, exg-luna, exg-source
cargo bench                      # Criterion: open_raw / read_all_data / read_slice
```

---

## Benchmarks

| Step | MNE (ms) | Rust (ms) | Speedup |
|---|---:|---:|---:|
| Read FIF | 1.83 | 0.63 | **2.9×** |
| Resample | 0.03 | 0.02 | **1.4×** |
| HP filter | 5.48 | 3.68 | **1.5×** |
| Avg reference | 0.49 | 0.03 | **16.6×** |
| Z-score | 0.29 | 0.09 | **3.3×** |
| Epoch | 1.98 | 0.06 | **33.3×** |
| **Total** | **10.11** | **4.51** | **2.2×** |

---

## Project layout

```
exg/
├── Cargo.toml                        workspace root; features: source, hdf5
├── src/
│   ├── lib.rs                        preprocess() + re-exports
│   ├── config.rs                     PipelineConfig
│   ├── edf/mod.rs                    EDF/EDF+ reader (header, data, annotations)
│   ├── csv.rs                        CSV reader (auto-detect delimiter/timestamps)
│   ├── hdf5.rs                       HDF5 dataset reader (feature-gated)
│   ├── montage.rs                    TCP bipolar, Siena, SEED-V montages
│   ├── resample.rs                   FFT polyphase resampler
│   ├── filter/
│   │   ├── design.rs                 _firwin_design: HP, LP, BP, notch (MNE parity)
│   │   └── apply.rs                  overlap-add zero-phase FIR
│   ├── reference.rs                  average reference
│   ├── normalize.rs                  global z-score, channel-wise z-score, baseline
│   ├── epoch.rs                      fixed-length epoching
│   ├── io.rs                         safetensors I/O, batch writer
│   └── fiff/                         FIFF file format reader
│       ├── constants.rs              FIFF constants
│       ├── tag.rs                    tag header I/O
│       ├── tree.rs                   block tree + directory reader
│       ├── info.rs                   MeasInfo + ChannelInfo
│       └── raw.rs                    open_raw / read_all_data / read_slice
├── exg-luna/                         LUNA seizure-detection pipeline
│   ├── src/
│   │   ├── lib.rs                    re-exports
│   │   ├── pipeline.rs              preprocess_luna + LunaPipelineConfig
│   │   └── io.rs                     LunaEpoch export/load (safetensors)
│   └── README.md
├── exg-source/                       source localisation crate
│   ├── src/
│   │   ├── lib.rs                    re-exports
│   │   ├── forward.rs               forward models (sphere)
│   │   ├── inverse.rs               MNE / dSPM / sLORETA
│   │   ├── eloreta.rs               eLORETA
│   │   ├── covariance.rs            noise covariance estimation
│   │   ├── resolution.rs            resolution metrics (PSF, CTF)
│   │   ├── snr.rs                   SNR estimation
│   │   ├── source_space.rs          ico / grid source spaces
│   │   └── linalg.rs                SVD / regularisation helpers
│   └── README.md
├── tests/                            integration tests
├── benches/                          Criterion benchmarks
└── scripts/                          Python ground-truth generators
```

---

## License

[AI100](https://www.humanscommons.org/license/ai100/1.0)
