//! FIR filter design and application.
//!
//! - [`design`]: Hamming-windowed sinc FIR design for highpass, lowpass,
//!   bandpass, and notch filters, matching MNE's `create_filter` and
//!   `notch_filter` parameter selection.
//! - [`apply`]: Overlap-add zero-phase convolution, matching MNE's
//!   `_overlap_add_filter` / `_1d_overlap_filter`.

pub mod apply;
pub mod design;

pub use design::{
    auto_trans_bandwidth, auto_trans_bandwidth_lowpass, auto_filter_length,
    design_highpass, design_lowpass, design_bandpass, design_notch,
    firwin, hamming,
};
pub use apply::{apply_fir_zero_phase, filter_1d};
