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
mod write;

pub use write::write_raw as write_raw_fif;

// Re-export the most commonly used items.
pub use info::{read_meas_info, ChannelInfo, MeasInfo};
pub use raw::{open_raw, BufferRecord, RawFif};
pub use tag::{
    read_directory, read_f32, read_f32_array, read_f64, read_f64_array, read_i32, read_i32_array,
    read_raw_bytes, read_string, read_tag_header, TagHeader,
};
pub use tree::{build_tree, read_tree, scan_directory, try_load_directory, Node};
