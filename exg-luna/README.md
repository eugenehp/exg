# exg-luna

> LUNA seizure-detection preprocessing pipeline for EEG — built on [`exg`](https://crates.io/crates/exg).

Part of the [`exg`](https://crates.io/crates/exg) workspace.
See also [`exg-source`](https://crates.io/crates/exg-source) for EEG source
localisation.

This crate implements the full preprocessing chain used by the LUNA seizure
detection model, matching the Python training pipeline. It uses `exg` DSP
primitives (filter design, resampling, montage conversion) under the hood.

## Installation

```toml
[dependencies]
exg-luna = "0.0.3"   # pulls in exg automatically
```

If you also need file readers (EDF, FIF) or source localisation from `exg`:

```toml
[dependencies]
exg      = "0.0.3"   # includes exg-source by default
exg-luna = "0.0.3"
```

## Pipeline steps

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Channel rename | Strip `"EEG "` prefix, `"-REF"` / `"-LE"` suffix |
| 2 | Pick 10-20 channels | 21 standard electrodes (`STANDARD_10_20`) |
| 3 | Bandpass filter | 0.1–75 Hz, zero-phase FIR (MNE `_firwin_design` parity) |
| 4 | Notch filter | 60 Hz (configurable for 50 Hz), zero-phase FIR |
| 5 | Resample | → 256 Hz (FFT polyphase) |
| 6 | TCP bipolar montage | 22 bipolar channels from 21 reference electrodes |
| 7 | Epoch | Non-overlapping 5 s windows (1280 samples) |

**Note:** Channel-wise z-score is _not_ applied here — LUNA does that at
inference time inside the model. Use `exg::normalize::zscore_channelwise_inplace`
separately if needed.

## Quick start

```rust,no_run
use exg::edf::open_raw_edf;
use exg_luna::{preprocess_luna, LunaPipelineConfig};

let raw = open_raw_edf("recording.edf").unwrap();
let data = raw.read_all_data().unwrap();
let ch_names = raw.channel_names();
let cfg = LunaPipelineConfig::default();  // 0.1–75 Hz BP · 60 Hz notch · TCP montage
let epochs = preprocess_luna(data, &ch_names, raw.header.sample_rate, &cfg).unwrap();
// → Vec<(Array2<f32>[22, 1280], Vec<String>)>
```

### Custom configuration

```rust
use exg_luna::LunaPipelineConfig;

let cfg = LunaPipelineConfig {
    notch_freq: Some(50.0),   // EU mains frequency
    epoch_dur: 10.0,          // 10 s epochs instead of 5 s
    ..LunaPipelineConfig::default()
};
```

## Safetensors I/O

Export and load preprocessed epochs in a format compatible with `luna-rs`
`InputBatch`:

```rust,no_run
use exg_luna::{LunaEpoch, export_luna_epochs, load_luna_epochs};
use ndarray::Array2;
use std::path::Path;

let epoch = LunaEpoch {
    signal: Array2::zeros((22, 1280)),
    channel_positions: Array2::zeros((22, 3)),
    channel_names: vec!["FP1-F7".into(), /* ... */],
};

export_luna_epochs(&[epoch], Path::new("batch.safetensors")).unwrap();
let loaded = load_luna_epochs(Path::new("batch.safetensors")).unwrap();
```

### File layout

```text
n_epochs:    I32 [1]
signal_0:    F32 [C, T]
positions_0: F32 [C, 3]
ch_names_0:  U8  [len]     ← newline-separated channel names
signal_1:    ...
```

## API

```rust
// Pipeline
pub fn preprocess_luna(data, ch_names, sfreq, cfg) -> Result<Vec<(Array2, Vec<String>)>>
pub struct LunaPipelineConfig { bandpass_low, bandpass_high, notch_freq, target_sfreq, epoch_dur, montage }
pub const STANDARD_10_20: &[&str]    // 21 electrodes

// I/O
pub struct LunaEpoch { signal, channel_positions, channel_names }
pub fn export_luna_epochs(epochs, path) -> Result<()>
pub fn load_luna_epochs(path) -> Result<Vec<LunaEpoch>>
```

## License

[AI100](https://www.humanscommons.org/license/ai100/1.0)
