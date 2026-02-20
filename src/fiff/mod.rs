//! FIFF file format reader.
//!
//! Implements reading of `.fif` EEG recordings compatible with
//! [MNE-Python](https://mne.tools).
//!
//! # Quick start
//! ```no_run
//! use exg::fiff::raw::open_raw;
//!
//! let raw = open_raw("data/sample1_raw.fif").unwrap();
//! println!("{} channels @ {} Hz", raw.info.n_chan, raw.info.sfreq);
//! let data = raw.read_all_data().unwrap();  // [n_chan, n_times] f64
//! ```
pub mod constants;
pub mod info;
pub mod raw;
pub mod tag;
pub mod tree;

// Re-export the most commonly used items.
pub use info::{ChannelInfo, MeasInfo, read_meas_info};
pub use raw::{open_raw, RawFif, BufferRecord};
pub use tag::{
    TagHeader, read_tag_header,
    read_i32, read_f32, read_f64,
    read_string, read_i32_array, read_f32_array, read_f64_array,
    read_raw_bytes, read_directory,
};
pub use tree::{Node, build_tree, read_tree, scan_directory, try_load_directory};
