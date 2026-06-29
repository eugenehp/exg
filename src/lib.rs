//! # exg — EEG/ECG/EMG preprocessing in pure Rust
//!
//! [![Crate](https://img.shields.io/crates/v/exg.svg)](https://crates.io/crates/exg)
//! [![Docs](https://docs.rs/exg/badge.svg)](https://docs.rs/exg)
//!
//! `exg` is a pure-Rust library for EEG preprocessing with 100% numerical
//! parity against [MNE-Python](https://mne.tools). Every DSP step is ported
//! from MNE and verified against ground-truth test vectors (200+ tests).
//!
//! _No Python, no BLAS, no C libraries — pure Rust + [RustFFT](https://crates.io/crates/rustfft)._
//!
//! ## Workspace
//!
//! | Crate | Description |
//! |-------|-------------|
//! | **`exg`** | Core DSP, file I/O, generic preprocessing pipeline |
//! | **[`exg-luna`](https://crates.io/crates/exg-luna)** | LUNA seizure-detection pipeline |
//! | **[`exg-source`](https://crates.io/crates/exg-source)** | Source localisation (eLORETA, MNE/dSPM/sLORETA) |
//!
//! ## Features
//!
//! | Category | Capabilities |
//! |----------|-------------|
//! | **File I/O** | FIF, EDF/EDF+, CSV readers; HDF5 (feature-gated); safetensors export |
//! | **Filters** | Highpass, lowpass, bandpass, notch — all MNE-parity `_firwin_design` |
//! | **DSP** | FFT polyphase resampling, average reference, overlap-add convolution |
//! | **Normalisation** | Global z-score, channel-wise z-score, per-epoch baseline correction |
//! | **Montages** | TCP bipolar (22-ch), Siena unipolar (29-ch), SEED-V unipolar (62-ch) |
//! | **Pipelines** | Generic `preprocess()`, LUNA-specific via `exg-luna` crate |
//! | **Source localisation** | eLORETA, MNE/dSPM/sLORETA (via `exg-source` crate) |
//!
//! ## Quick start — generic pipeline
//!
//! ```no_run
//! use exg::{preprocess, PipelineConfig, fiff::open_raw};
//! use ndarray::Array2;
//!
//! let raw  = open_raw("data/sample1_raw.fif").unwrap();
//! let data = raw.read_all_data().unwrap();
//! let chan_pos: Array2<f32> = Array2::zeros((raw.info.n_chan, 3));
//! let cfg  = PipelineConfig::default();           // 256 Hz · 0.5 Hz HP · 5 s epochs
//! let epochs = preprocess(data.mapv(|v| v as f32), chan_pos, raw.info.sfreq as f32, &cfg).unwrap();
//! ```
//!
//! ## Quick start — LUNA seizure-detection pipeline
//!
//! The LUNA pipeline lives in the separate [`exg-luna`](https://crates.io/crates/exg-luna) crate:
//!
//! ```ignore
//! use exg::edf::open_raw_edf;
//! use exg_luna::{preprocess_luna, LunaPipelineConfig};
//!
//! let raw = open_raw_edf("recording.edf").unwrap();
//! let data = raw.read_all_data().unwrap();
//! let ch_names = raw.channel_names();
//! let cfg = LunaPipelineConfig::default();        // 0.1–75 Hz BP · 60 Hz notch · TCP montage
//! let epochs = preprocess_luna(data, &ch_names, raw.header.sample_rate, &cfg).unwrap();
//! ```
//!
//! ## Individual DSP steps
//!
//! ```no_run
//! use exg::{resample::resample, filter::*, reference::*, normalize::*, epoch::*};
//! use ndarray::Array2;
//!
//! let mut data: Array2<f32> = Array2::zeros((22, 7680));
//!
//! // Resample 512 → 256 Hz
//! let mut data = resample(&data, 512.0, 256.0).unwrap();
//!
//! // Bandpass 0.1–75 Hz
//! let h = design_bandpass(0.1, 75.0, 256.0);
//! apply_fir_zero_phase(&mut data, &h).unwrap();
//!
//! // Notch 60 Hz
//! let h = design_notch(60.0, 256.0, None, None);
//! apply_fir_zero_phase(&mut data, &h).unwrap();
//!
//! // Average reference → channel-wise z-score → epoch
//! average_reference_inplace(&mut data);
//! zscore_channelwise_inplace(&mut data);
//! let epochs = epoch(&data, 1280);                // [E, 22, 1280]
//! ```

pub mod config;
pub mod csv;
pub mod edf;
pub mod eeglab;
pub mod epoch;
pub mod fiff;
pub mod filter;
#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod io;
pub mod montage;
pub mod normalize;
pub mod reference;
pub mod resample;
pub mod xdf;

/// EEG source localization (MNE / dSPM / sLORETA / eLORETA).
///
/// This module re-exports the [`exg_source`] crate.  It is available when
/// the **`source`** feature is enabled (on by default).
///
/// Disable it with `default-features = false` if you only need preprocessing:
///
/// ```toml
/// [dependencies]
/// exg = { version = "0.0.3", default-features = false }
/// ```
#[cfg(feature = "source")]
pub use exg_source as source_localization;

use anyhow::Result;
use ndarray::Array2;

// ── Crate-root re-exports ─────────────────────────────────────────────────
//
// Everything a downstream user is likely to need is available directly as
// `exg::Foo` without having to know the internal module layout.

// config
pub use config::PipelineConfig;

// csv
pub use csv::read_eeg;

// edf
pub use edf::{open_raw_edf, EdfAnnotation, EdfHeader, RawEdf, SignalHeader};

// epoch
pub use epoch::{epoch, epoch_and_baseline};

// fiff  — measurement info, raw reader, tag I/O, tree, constants
pub use fiff::{
    build_tree,
    // high-level
    open_raw,
    read_directory,
    read_f32,
    read_f32_array,
    read_f64,
    read_f64_array,
    read_i32,
    read_i32_array,
    read_meas_info,
    read_raw_bytes,
    read_string,
    read_tag_header,
    read_tree,
    scan_directory,
    try_load_directory,
    BufferRecord,
    ChannelInfo,
    MeasInfo,
    // tree
    Node,
    RawFif,
    // tag I/O
    TagHeader,
};

// filter — design helpers + convolution
pub use filter::{
    apply_fir_zero_phase, auto_filter_length, auto_trans_bandwidth, auto_trans_bandwidth_lowpass,
    design_bandpass, design_highpass, design_lowpass, design_notch, filter_1d, firwin, hamming,
};

// hdf5 — HDF5 dataset reader (feature-gated)
#[cfg(feature = "hdf5")]
pub use hdf5::{read_dataset as read_hdf5, read_dataset_split as read_hdf5_split, HDF5Sample};

// io — safetensors helpers
pub use io::{write_batch, RawData, StWriter};

// montage
pub use montage::{
    make_bipolar, normalize_channel_name, pick_channels, BipolarDef, SEED_V_CHANNELS,
    SIENA_CHANNELS, TCP_MONTAGE,
};

// normalize
pub use normalize::{baseline_correct_inplace, zscore_channelwise_inplace, zscore_global_inplace};

// reference
pub use reference::average_reference_inplace;

// resample — resampler + supporting math
pub use resample::{auto_npad, final_length, rational_approx, resample, resample_1d};

// source_localization — inverse modeling (MNE / dSPM / sLORETA / eLORETA)
#[cfg(feature = "source")]
pub use source_localization::{
    apply_inverse,
    apply_inverse_epochs,
    apply_inverse_epochs_full,
    apply_inverse_full,
    // covariance
    compute_covariance,
    compute_covariance_epochs,
    // snr
    estimate_snr,
    get_cross_talk,
    get_point_spread,
    grid_source_space,
    ico_n_vertices,
    // source space
    ico_source_space,
    // inverse
    make_inverse_operator,
    // resolution
    make_resolution_matrix,
    // forward model
    make_sphere_forward,
    make_sphere_forward_free,
    peak_localisation_error,
    relative_amplitude,
    spatial_spread,
    EloretaOptions,
    // types
    ForwardOperator,
    InverseMethod,
    InverseOperator,
    NoiseCov,
    PickOri,
    Regularization,
    SourceEstimate,
    SourceOrientation,
    SphereModel,
};

/// Run the **full EEG preprocessing pipeline** on a single continuous recording.
///
/// This is the main entry point for the `exg` library.  It chains all
/// preprocessing steps in the exact order used to train the model and
/// matches the MNE-Python reference implementation to within floating-point
/// rounding error (< 4 × 10⁻⁶ on typical EEG data).
///
/// # Pipeline steps
///
/// 1. Zero-fill channels listed in [`PipelineConfig::bad_channels`].
/// 2. Resample from `src_sfreq` to [`PipelineConfig::target_sfreq`] (FFT polyphase).
/// 3. Apply a zero-phase highpass FIR filter at [`PipelineConfig::hp_freq`].
/// 4. Subtract the per-timepoint channel mean (average reference).
/// 5. Apply global z-score normalisation (`ddof = 0`).
/// 6. Split into non-overlapping epochs of [`PipelineConfig::epoch_samples()`] samples
///    and apply per-epoch per-channel baseline correction.
/// 7. Divide each epoch by [`PipelineConfig::data_norm`].
///
/// # Arguments
///
/// * `data`     – Raw EEG signal, shape `[C, T]`, in original units (volts).
///   Must have at least `cfg.epoch_samples()` columns; shorter recordings
///   produce zero epochs.
/// * `chan_pos`  – Channel positions in **metres**, shape `[C, 3]`.
///   Returned unchanged alongside each epoch so downstream code
///   has direct access to spatial layout.
/// * `src_sfreq` – Sampling rate of `data` in Hz.
/// * `cfg`       – Pipeline configuration (see [`PipelineConfig`]).
///
/// # Returns
///
/// A `Vec` of `(epoch_data, chan_pos)` tuples:
/// * `epoch_data` — shape `[C, cfg.epoch_samples()]`, `f32`.
/// * `chan_pos`   — the original `chan_pos` argument (cloned, `f32`).
///
/// The length of the `Vec` is `floor(T_resampled / cfg.epoch_samples())`.
/// Trailing samples that do not fill a complete epoch are discarded.
///
/// # Errors
///
/// Returns an error if:
/// * The resampler fails (e.g. zero-length input).
/// * The FIR convolution fails (internal FFT planner error, extremely rare).
///
/// # Examples
///
/// ```no_run
/// use exg::{preprocess, PipelineConfig};
/// use ndarray::Array2;
///
/// // 12-channel, 15-second recording at 256 Hz
/// let data: Array2<f32> = Array2::zeros((12, 3840));
/// let chan_pos: Array2<f32> = Array2::zeros((12, 3));
///
/// let cfg    = PipelineConfig::default();
/// let epochs = preprocess(data, chan_pos, 256.0, &cfg).unwrap();
/// assert_eq!(epochs.len(), 2); // floor(3840 / 1280) = 2 (baseline uses 1 epoch's worth)
/// ```
pub fn preprocess(
    mut data: Array2<f32>,
    chan_pos: Array2<f32>,
    src_sfreq: f32,
    cfg: &PipelineConfig,
) -> Result<Vec<(Array2<f32>, Array2<f32>)>> {
    // 1. Zero out specified bad channels.
    zero_bad_channels(&mut data, &cfg.bad_channels, &[]);

    // 2. Resample to target sfreq.
    if (src_sfreq - cfg.target_sfreq).abs() > 1e-3 {
        data = resample::resample(&data, src_sfreq, cfg.target_sfreq)?;
    }

    // 3. Highpass FIR filter.
    let h = filter::design_highpass(cfg.hp_freq, cfg.target_sfreq);
    filter::apply_fir_zero_phase(&mut data, &h)?;

    // 4. Average reference.
    reference::average_reference_inplace(&mut data);

    // 5. Global z-score.
    normalize::zscore_global_inplace(&mut data);

    // 6. Epoch + baseline correction.
    let epochs_3d = epoch::epoch(&data, cfg.epoch_samples());
    let (n_epochs, _n_ch, _n_t) = epochs_3d.dim();

    // 7. Divide by data_norm and convert to Vec of 2D arrays.
    let inv_norm = 1.0 / cfg.data_norm;
    let mut result = Vec::with_capacity(n_epochs);
    for e in 0..n_epochs {
        let epoch_data: Array2<f32> = epochs_3d
            .slice(ndarray::s![e, .., ..])
            .to_owned()
            .mapv(|v| v * inv_norm);
        result.push((epoch_data, chan_pos.clone()));
    }

    Ok(result)
}

/// Zero-fill channels whose normalised name appears in `bad`.
///
/// Name normalisation: lowercase + strip spaces.
/// Silently skips names not found in `ch_names`.
pub fn zero_bad_channels(data: &mut Array2<f32>, bad: &[String], ch_names: &[String]) {
    for bad_ch in bad {
        let norm = |s: &str| s.replace(' ', "").to_lowercase();
        if let Some(idx) = ch_names.iter().position(|n| norm(n) == norm(bad_ch)) {
            data.row_mut(idx).fill(0.0);
        }
    }
}
