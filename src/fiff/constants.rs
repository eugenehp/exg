//! FIFF format constants.
//!
//! All names mirror [`mne/_fiff/constants.py`][mne-src] exactly so that the
//! Rust code can be cross-referenced with MNE line-by-line.
//!
//! The FIFF format is a self-describing binary file format used by Elekta /
//! Neuromag MEG and EEG systems and adopted by MNE-Python as its primary I/O
//! format.  Every piece of data in a FIF file is wrapped in a **tag** — a
//! 16-byte header (`kind`, `type`, `size`, `next`) followed by a payload.
//! Tags are grouped into **blocks** by `FIFF_BLOCK_START` / `FIFF_BLOCK_END`
//! sentinel tags, forming a tree.
//!
//! [mne-src]: https://github.com/mne-tools/mne-python/blob/main/mne/_fiff/constants.py

#![allow(dead_code)]

// ── Block kinds ───────────────────────────────────────────────────────────
//
// A block is opened by a FIFF_BLOCK_START tag whose i32 payload is the block
// kind, and closed by a matching FIFF_BLOCK_END tag.

/// Root / top-level block (rarely used explicitly).
pub const FIFFB_ROOT:           i32 = 999;
/// Measurement block — top-level container for one recording.
pub const FIFFB_MEAS:           i32 = 100;
/// Measurement-info block — channel metadata, sfreq, bad channels, etc.
pub const FIFFB_MEAS_INFO:      i32 = 101;
/// Raw (continuous) data block.
pub const FIFFB_RAW_DATA:       i32 = 102;
/// Processed data block.
pub const FIFFB_PROCESSED_DATA: i32 = 103;
/// Evoked / averaged data block.
pub const FIFFB_EVOKED:         i32 = 104;
/// Continuous data block (alias used by some acquisition systems).
pub const FIFFB_CONTINUOUS_DATA: i32 = 112;
/// IAS (internal active shielding) raw data block.
pub const FIFFB_IAS_RAW_DATA:   i32 = 119;
/// SSP projector block.
pub const FIFFB_PROJ:           i32 = 313;
/// One SSP projector item block.
pub const FIFFB_PROJ_ITEM:      i32 = 314;
/// MNE-specific extension block.
pub const FIFFB_MNE:            i32 = 350;
/// MNE epochs block.
pub const FIFFB_MNE_EPOCHS:     i32 = 373;

// ── Tag kinds — structural ─────────────────────────────────────────────────

/// Unique file identifier (first tag in every FIF file).
pub const FIFF_FILE_ID:         i32 = 100;
/// Pointer to the embedded tag directory (second tag, payload = byte offset).
pub const FIFF_DIR_POINTER:     i32 = 101;
/// Block identifier payload (inside `FIFF_BLOCK_START`).
pub const FIFF_BLOCK_ID:        i32 = 103;
/// Opens a new block; payload = block kind (i32).
pub const FIFF_BLOCK_START:     i32 = 104;
/// Closes the most recently opened block.
pub const FIFF_BLOCK_END:       i32 = 105;
/// Parent block identifier.
pub const FIFF_PARENT_BLOCK_ID: i32 = 110;

// ── Tag kinds — measurement info ──────────────────────────────────────────

/// Number of channels (i32).
pub const FIFF_NCHAN:           i32 = 200;
/// Sampling frequency in Hz (f32).
pub const FIFF_SFREQ:           i32 = 201;
/// Channel info struct (one per channel; see [`super::info::ChannelInfo`]).
pub const FIFF_CH_INFO:         i32 = 203;
/// Measurement date (Julian day, i32).
pub const FIFF_MEAS_DATE:       i32 = 204;
/// Free-text comment / description (string).
pub const FIFF_COMMENT:         i32 = 206;
/// Index of the first sample in acquisition time (i32).
pub const FIFF_FIRST_SAMPLE:    i32 = 208;
/// Index of the last sample in acquisition time (i32).
pub const FIFF_LAST_SAMPLE:     i32 = 209;
/// Online lowpass cutoff in Hz (f32); may be NaN if not set.
pub const FIFF_LOWPASS:         i32 = 219;
/// Colon-separated list of bad channel names (string).
pub const FIFF_BAD_CHS:         i32 = 220;
/// Online highpass cutoff in Hz (f32); may be NaN if not set.
pub const FIFF_HIGHPASS:        i32 = 223;
/// Power-line frequency in Hz (f32).
pub const FIFF_LINE_FREQ:       i32 = 235;
/// Experimenter name (string).
pub const FIFF_EXPERIMENTER:    i32 = 212;
/// Recording description — alias for `FIFF_COMMENT`.
pub const FIFF_DESCRIPTION:     i32 = FIFF_COMMENT;

// ── Tag kinds — data buffers ───────────────────────────────────────────────

/// One buffer of raw signal samples (interleaved `[n_samp, n_chan]`,
/// big-endian, type = `FIFFT_FLOAT` or `FIFFT_DOUBLE` or `FIFFT_SHORT`).
pub const FIFF_DATA_BUFFER:     i32 = 300;
/// Skip `n` complete buffers (inter-buffer gap; payload = n as i32).
pub const FIFF_DATA_SKIP:       i32 = 301;
/// Skip `n` individual samples (rarely used).
pub const FIFF_DATA_SKIP_SAMP:  i32 = 303;

// ── Tag kinds — multi-file references ─────────────────────────────────────

/// File identifier of a referenced file.
pub const FIFF_REF_FILE_ID:     i32 = 116;
/// Sequence number of a referenced file.
pub const FIFF_REF_FILE_NUM:    i32 = 117;
/// Name of a referenced file.
pub const FIFF_REF_FILE_NAME:   i32 = 118;
/// Role of a referenced file.
pub const FIFF_REF_ROLE:        i32 = 115;
/// `FIFF_REF_ROLE` value indicating the next file in a split recording.
pub const FIFFV_ROLE_NEXT_FILE: i32 = 2;

// ── Tag payload types (the `type` field of a tag header) ──────────────────

/// Void / no payload.
pub const FIFFT_VOID:              u32 = 0;
/// Unsigned 8-bit byte.
pub const FIFFT_BYTE:              u32 = 1;
/// Big-endian signed 16-bit integer.
pub const FIFFT_SHORT:             u32 = 2;
/// Big-endian signed 32-bit integer.
pub const FIFFT_INT:               u32 = 3;
/// Big-endian IEEE 754 single-precision float (4 bytes).
pub const FIFFT_FLOAT:             u32 = 4;
/// Big-endian IEEE 754 double-precision float (8 bytes).
pub const FIFFT_DOUBLE:            u32 = 5;
/// Julian date (i32).
pub const FIFFT_JULIAN:            u32 = 6;
/// Big-endian unsigned 16-bit integer.
pub const FIFFT_USHORT:            u32 = 7;
/// Big-endian unsigned 32-bit integer.
pub const FIFFT_UINT:              u32 = 8;
/// Latin-1 (ISO 8859-1) string, **not** NUL-terminated.
pub const FIFFT_STRING:            u32 = 10;
/// 16-bit DAU packed sample (same wire width as `FIFFT_SHORT`).
pub const FIFFT_DAU_PACK16:        u32 = 16;
/// Complex single-precision (real f32 + imag f32, 8 bytes).
pub const FIFFT_COMPLEX_FLOAT:     u32 = 20;
/// Complex double-precision (real f64 + imag f64, 16 bytes).
pub const FIFFT_COMPLEX_DOUBLE:    u32 = 21;
/// Legacy packed format (obsolete).
pub const FIFFT_OLD_PACK:          u32 = 23;
/// 96-byte channel info struct (see [`super::info::ChannelInfo`]).
pub const FIFFT_CH_INFO_STRUCT:    u32 = 30;
/// File-ID struct.
pub const FIFFT_ID_STRUCT:         u32 = 31;
/// Tag-directory entry struct (16 bytes per entry).
pub const FIFFT_DIR_ENTRY_STRUCT:  u32 = 32;
/// Digitisation point struct.
pub const FIFFT_DIG_POINT_STRUCT:  u32 = 33;
/// Coordinate transform struct.
pub const FIFFT_COORD_TRANS_STRUCT:u32 = 35;
/// Matrix modifier — OR this with an element type to indicate a matrix payload.
pub const FIFFT_MATRIX:            u32 = 0x4000_0000;

// ── `next` field sentinels in a tag header ────────────────────────────────

/// The next tag follows immediately: `next_pos = pos + 16 + size`.
pub const FIFFV_NEXT_SEQ:  i32 = 0;
/// There is no next tag (end of sequence / block).
pub const FIFFV_NEXT_NONE: i32 = -1;

// ── Channel kind codes (`ChannelInfo::kind`) ──────────────────────────────

/// MEG magnetometer or gradiometer channel.
pub const FIFFV_MEG_CH:     i32 = 1;
/// EEG scalp-potential channel.
pub const FIFFV_EEG_CH:     i32 = 2;
/// Stimulus / trigger channel.
pub const FIFFV_STIM_CH:    i32 = 3;
/// Electro-oculogram channel.
pub const FIFFV_EOG_CH:     i32 = 202;
/// Electromyogram channel.
pub const FIFFV_EMG_CH:     i32 = 302;
/// Electrocardiogram channel.
pub const FIFFV_ECG_CH:     i32 = 402;
/// Miscellaneous auxiliary channel.
pub const FIFFV_MISC_CH:    i32 = 502;
/// MEG reference (compensation) channel.
pub const FIFFV_REF_MEG_CH: i32 = 301;
/// Stereo-EEG depth electrode channel.
pub const FIFFV_SEEG_CH:    i32 = 802;
/// Electrocorticography (ECoG) channel.
pub const FIFFV_ECOG_CH:    i32 = 902;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Return the number of bytes occupied by one sample of the given tag type.
///
/// Returns `None` for types that do not represent scalar numeric samples
/// (e.g. strings, structs).
///
/// # Examples
///
/// ```
/// use exg::fiff::constants::{bytes_per_sample, FIFFT_FLOAT, FIFFT_DOUBLE, FIFFT_SHORT};
/// assert_eq!(bytes_per_sample(FIFFT_FLOAT),  Some(4));
/// assert_eq!(bytes_per_sample(FIFFT_DOUBLE), Some(8));
/// assert_eq!(bytes_per_sample(FIFFT_SHORT),  Some(2));
/// assert_eq!(bytes_per_sample(99),            None);
/// ```
pub fn bytes_per_sample(tag_type: u32) -> Option<usize> {
    match tag_type {
        FIFFT_DAU_PACK16 | FIFFT_SHORT  => Some(2),
        FIFFT_FLOAT                     => Some(4),
        FIFFT_DOUBLE                    => Some(8),
        FIFFT_INT                       => Some(4),
        FIFFT_COMPLEX_FLOAT             => Some(8),
        FIFFT_COMPLEX_DOUBLE            => Some(16),
        _                               => None,
    }
}
