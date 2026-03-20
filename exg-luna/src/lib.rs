//! # exg-luna — LUNA seizure-detection EEG preprocessing pipeline
//!
//! [![Crate](https://img.shields.io/crates/v/exg-luna.svg)](https://crates.io/crates/exg-luna)
//! [![Docs](https://docs.rs/exg-luna/badge.svg)](https://docs.rs/exg-luna)
//!
//! Part of the [`exg`](https://crates.io/crates/exg) workspace.
//! Uses `exg` DSP primitives (filter design, resampling, montage conversion)
//! under the hood.
//!
//! ## Pipeline
//!
//! Implements the full preprocessing chain used by the LUNA seizure detection
//! model, matching the Python training pipeline:
//!
//! 1. Channel rename (strip `"EEG "` prefix, `"-REF"` / `"-LE"` suffix)
//! 2. Pick standard 10-20 channels (21 electrodes)
//! 3. Bandpass filter 0.1–75 Hz (zero-phase FIR, MNE `_firwin_design` parity)
//! 4. Notch filter at 60 Hz (configurable for 50 Hz)
//! 5. Resample to 256 Hz (FFT polyphase)
//! 6. TCP bipolar montage (22 channels from 21 reference electrodes)
//! 7. Epoch into 5 s non-overlapping windows (1280 samples)
//!
//! **Note:** Channel-wise z-score is _not_ applied here — LUNA does that at
//! inference time inside the model. Use
//! `exg::normalize::zscore_channelwise_inplace` separately if needed.
//!
//! ## I/O
//!
//! This crate also provides safetensors serialization of preprocessed epochs
//! in a format compatible with `luna-rs` `InputBatch`.
//!
//! ## Quick start
//!
//! ```ignore
//! use exg::edf::open_raw_edf;
//! use exg_luna::{preprocess_luna, LunaPipelineConfig};
//!
//! let raw = open_raw_edf("recording.edf").unwrap();
//! let data = raw.read_all_data().unwrap();
//! let ch_names = raw.channel_names();
//! let cfg = LunaPipelineConfig::default();
//! let epochs = preprocess_luna(data, &ch_names, raw.header.sample_rate, &cfg).unwrap();
//! ```

mod io;
mod pipeline;

pub use io::{LunaEpoch, export_luna_epochs, load_luna_epochs};
pub use pipeline::{preprocess_luna, LunaPipelineConfig, STANDARD_10_20};
