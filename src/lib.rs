//! # exg — EEG/ECG/EMG preprocessing in pure Rust
//!
//! `exg` is a zero-dependency\* Rust library that implements the EEG
//! preprocessing pipeline. Every DSP step is ported from
//! [MNE-Python](https://mne.tools) and verified against MNE ground truth via
//! safetensors test vectors (run `cargo test`).
//!
//! _\* No Python, no BLAS, no C libraries — pure Rust + [RustFFT](https://crates.io/crates/rustfft)._
//!
//! ## Pipeline overview
//!
//! ```text
//! sample_raw.fif
//!   │
//!   ├─ fiff::open_raw()       native FIFF reader (no MNE)
//!   ├─ resample::resample()   FFT polyphase → target_sfreq (default 256 Hz)
//!   ├─ filter (FIR HP)        firwin + overlap-add → 0.5 Hz cutoff
//!   ├─ reference              per-timepoint channel mean removed
//!   ├─ normalize (z-score)    (data − μ) / σ  over all ch × t
//!   ├─ epoch                  non-overlapping 5 s windows
//!   ├─ baseline correct       per-epoch per-channel mean removed
//!   └─ ÷ data_norm            ÷ 10 → std ≈ 0.1
//!        │
//!        └─→ Vec<([C, 1280] f32, [C, 3] f32)>   (epochs, channel positions)
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use exg::{preprocess, PipelineConfig};
//! use exg::fiff::open_raw;
//! use ndarray::Array2;
//!
//! // 1. Read a .fif file — no Python required
//! let raw  = open_raw("data/sample1_raw.fif").unwrap();
//! let data = raw.read_all_data().unwrap();          // [C, T]  f64
//!
//! // 2. Channel positions from the FIF file (metres)
//! let chan_pos: Array2<f32> = Array2::zeros((raw.info.n_chan, 3));
//!
//! // 3. Run the full preprocessing pipeline
//! let cfg    = PipelineConfig::default();
//! let epochs = preprocess(
//!     data.mapv(|v| v as f32),
//!     chan_pos,
//!     raw.info.sfreq as f32,
//!     &cfg,
//! ).unwrap();
//!
//! for (i, (epoch_data, pos)) in epochs.iter().enumerate() {
//!     println!("Epoch {i}: shape {:?}", epoch_data.dim());
//! }
//! ```
//!
//! ## Running individual steps
//!
//! Each preprocessing step is also exposed as a standalone function:
//!
//! ```no_run
//! use exg::resample::resample;
//! use exg::filter::{design_highpass, apply_fir_zero_phase};
//! use exg::reference::average_reference_inplace;
//! use exg::normalize::zscore_global_inplace;
//! use exg::epoch::epoch;
//! use ndarray::Array2;
//!
//! let mut data: Array2<f32> = Array2::zeros((12, 3840)); // [C, T]
//!
//! // Resample from 1024 Hz → 256 Hz
//! let data = resample(&data, 1024.0, 256.0).unwrap();
//!
//! // Apply 0.5 Hz highpass FIR
//! let h = design_highpass(0.5, 256.0);
//! let mut data = data;
//! apply_fir_zero_phase(&mut data, &h).unwrap();
//!
//! // Average reference
//! average_reference_inplace(&mut data);
//!
//! // Global z-score
//! let (mean, std) = zscore_global_inplace(&mut data);
//!
//! // Epoch into 5 s windows
//! let epochs = epoch(&data, 1280); // [E, C, 1280]
//! ```
//!
//! ## Feature coverage
//!
//! See the [README](https://github.com/Zyphra/exg#mne-feature-coverage) for a
//! full table of which MNE features are implemented and which are not yet ported.

pub mod config;
pub mod epoch;
pub mod fiff;
pub mod filter;
pub mod io;
pub mod normalize;
pub mod reference;
pub mod resample;

use anyhow::Result;
use ndarray::Array2;

// ── Crate-root re-exports ─────────────────────────────────────────────────
//
// Everything a downstream user is likely to need is available directly as
// `exg::Foo` without having to know the internal module layout.

// config
pub use config::PipelineConfig;

// epoch
pub use epoch::{epoch, epoch_and_baseline};

// fiff  — measurement info, raw reader, tag I/O, tree, constants
pub use fiff::{
    // high-level
    open_raw, RawFif, BufferRecord,
    ChannelInfo, MeasInfo, read_meas_info,
    // tag I/O
    TagHeader, read_tag_header,
    read_i32, read_f32, read_f64,
    read_string, read_i32_array, read_f32_array, read_f64_array,
    read_raw_bytes, read_directory,
    // tree
    Node, build_tree, read_tree, scan_directory, try_load_directory,
};

// filter — design helpers + convolution
pub use filter::{
    auto_trans_bandwidth, auto_filter_length, design_highpass,
    firwin, hamming,
    apply_fir_zero_phase, filter_1d,
};

// io — safetensors helpers
pub use io::{RawData, StWriter, write_batch};

// normalize
pub use normalize::{zscore_global_inplace, baseline_correct_inplace};

// reference
pub use reference::average_reference_inplace;

// resample — resampler + supporting math
pub use resample::{resample, resample_1d, auto_npad, rational_approx, final_length};

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
