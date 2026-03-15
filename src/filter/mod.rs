//! FIR filter design and application.
//!
//! - [`design`]: Hamming-windowed sinc highpass FIR design, matching
//!   `mne.filter.create_filter(fir_window='hamming', phase='zero')`.
//! - [`apply`]: Overlap-add zero-phase convolution, matching MNE's
//!   `_overlap_add_filter` / `_1d_overlap_filter`.

pub mod apply;
pub mod design;

pub use design::{auto_trans_bandwidth, auto_filter_length, design_highpass, firwin, hamming};
pub use apply::{apply_fir_zero_phase, filter_1d};
