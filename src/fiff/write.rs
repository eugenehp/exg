//! FIF file writer.
//!
//! Inverse of [`super::raw::open_raw`]: serializes a [`super::raw::RawFif`]'s
//! [`super::info::MeasInfo`] + a `[n_chan, n_times]` data array back to a
//! valid FIF file. The emitted file passes our own [`super::raw::open_raw`]
//! reader, the [`super::info::read_meas_info`] info reader, and
//! `mne.io.read_raw_fif` (single-buffer continuous recording).
//!
//! ## On-disk layout produced
//!
//! ```text
//! FIFF_FILE_ID                   (id_struct, 20 bytes)
//! FIFF_DIR_POINTER               (i32 = -1, no directory)
//! FIFF_BLOCK_START(MEAS)
//!   FIFF_BLOCK_START(MEAS_INFO)
//!     FIFF_NCHAN                 (i32)
//!     FIFF_SFREQ                 (f32, Hz)
//!     FIFF_HIGHPASS              (f32, Hz; optional)
//!     FIFF_LOWPASS               (f32, Hz; optional)
//!     FIFF_LINE_FREQ             (f32, Hz; optional)
//!     FIFF_CH_INFO × n_chan      (96-byte struct each)
//!   FIFF_BLOCK_END(MEAS_INFO)
//!   FIFF_BLOCK_START(RAW_DATA)
//!     FIFF_FIRST_SAMPLE          (i32)
//!     FIFF_DATA_BUFFER           (f32 × n_chan × n_times, big-endian)
//!   FIFF_BLOCK_END(RAW_DATA)
//! FIFF_BLOCK_END(MEAS)
//! ```
//!
//! All multi-byte values are big-endian (FIF native).
//!
//! ## Limitations
//!
//! - Single contiguous `FIFF_DATA_BUFFER` (no split-buffer chunking). For
//!   ZUNA-scale recordings this is fine (≤ a few hundred MB); for very long
//!   recordings the conventional split is ~10 MB per buffer.
//! - Float32 sample storage only.
//! - Identity calibration: `cal = 1.0`, `range = 1.0` per channel. The
//!   reader applies `data × cal × range`, so the values in your input
//!   array are the values that come back out.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::constants::*;
use super::info::{ChannelInfo, MeasInfo};

// FIF constant we don't yet have a name for in constants.rs.
const FIFFT_ID_STRUCT: u32 = 31;
const FIFFT_CH_INFO_STRUCT_LOCAL: u32 = FIFFT_CH_INFO_STRUCT;

/// Write a complete FIF recording.
///
/// `info` describes the channels and sample rate; `data` is `[n_chan, n_times]`
/// of f64 values that will be downcast to f32 on disk (FIF's native sample
/// format for continuous recordings).
pub fn write_raw<P: AsRef<Path>>(
    info: &MeasInfo,
    data: &ndarray::Array2<f64>,
    path: P,
) -> Result<()> {
    let path = path.as_ref();
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);

    let (n_chan, n_times) = (data.nrows(), data.ncols());
    anyhow::ensure!(
        n_chan == info.n_chan,
        "write_raw: data has {n_chan} channels, info says {}",
        info.n_chan,
    );
    anyhow::ensure!(
        info.chs.len() == n_chan,
        "info.chs has {} entries, n_chan = {n_chan}",
        info.chs.len()
    );

    // ── 1. FIFF_FILE_ID (id struct, 20 bytes) ────────────────────────────────
    // Layout used by MNE: version(i32) + machid[2](i32×2) + secs(i32) + usecs(i32).
    let mut id = [0u8; 20];
    id[0..4].copy_from_slice(&0x10001_i32.to_be_bytes()); // version 1.1
                                                          // machid: leave zero (anonymous)
                                                          // timestamp: 0 is allowed; readers tolerate it
    write_tag(&mut w, FIFF_FILE_ID, FIFFT_ID_STRUCT, &id, FIFFV_NEXT_SEQ)?;

    // ── 2. FIFF_DIR_POINTER = -1 (no directory; reader will scan) ────────────
    write_tag(
        &mut w,
        FIFF_DIR_POINTER,
        FIFFT_INT,
        &(-1_i32).to_be_bytes(),
        FIFFV_NEXT_SEQ,
    )?;

    // ── 3. FIFFB_MEAS ────────────────────────────────────────────────────────
    write_block_start(&mut w, FIFFB_MEAS)?;

    // 3a. FIFFB_MEAS_INFO ─────────────────────────────────────────────────────
    write_block_start(&mut w, FIFFB_MEAS_INFO)?;
    write_tag(
        &mut w,
        FIFF_NCHAN,
        FIFFT_INT,
        &(n_chan as i32).to_be_bytes(),
        FIFFV_NEXT_SEQ,
    )?;
    write_tag(
        &mut w,
        FIFF_SFREQ,
        FIFFT_FLOAT,
        &(info.sfreq as f32).to_be_bytes(),
        FIFFV_NEXT_SEQ,
    )?;
    if let Some(hp) = info.highpass {
        write_tag(
            &mut w,
            FIFF_HIGHPASS,
            FIFFT_FLOAT,
            &(hp as f32).to_be_bytes(),
            FIFFV_NEXT_SEQ,
        )?;
    }
    if let Some(lp) = info.lowpass {
        write_tag(
            &mut w,
            FIFF_LOWPASS,
            FIFFT_FLOAT,
            &(lp as f32).to_be_bytes(),
            FIFFV_NEXT_SEQ,
        )?;
    }
    if let Some(lf) = info.line_freq {
        write_tag(
            &mut w,
            FIFF_LINE_FREQ,
            FIFFT_FLOAT,
            &(lf as f32).to_be_bytes(),
            FIFFV_NEXT_SEQ,
        )?;
    }
    // One FIFF_CH_INFO tag per channel
    for ch in &info.chs {
        let payload = ch_info_to_bytes(ch);
        write_tag(
            &mut w,
            FIFF_CH_INFO,
            FIFFT_CH_INFO_STRUCT_LOCAL,
            &payload,
            FIFFV_NEXT_SEQ,
        )?;
    }
    write_block_end(&mut w, FIFFB_MEAS_INFO)?;

    // 3b. FIFFB_RAW_DATA ──────────────────────────────────────────────────────
    write_block_start(&mut w, FIFFB_RAW_DATA)?;
    write_tag(
        &mut w,
        FIFF_FIRST_SAMPLE,
        FIFFT_INT,
        &0_i32.to_be_bytes(),
        FIFFV_NEXT_SEQ,
    )?;
    // FIFF_DATA_BUFFER: f32 BE, layout [t][c] (sample-major, channel-minor)
    // — matches what `RawFif::read_all_data` expects via `read_buffer_data`.
    let total = n_chan * n_times;
    let mut buf: Vec<u8> = Vec::with_capacity(total * 4);
    for t in 0..n_times {
        for c in 0..n_chan {
            buf.extend_from_slice(&(data[[c, t]] as f32).to_be_bytes());
        }
    }
    write_tag(&mut w, FIFF_DATA_BUFFER, FIFFT_FLOAT, &buf, FIFFV_NEXT_SEQ)?;
    write_block_end(&mut w, FIFFB_RAW_DATA)?;

    // ── 4. Close FIFFB_MEAS, terminate stream ────────────────────────────────
    write_block_end_last(&mut w, FIFFB_MEAS)?;

    w.flush()?;
    Ok(())
}

// ── Low-level tag writers ────────────────────────────────────────────────────

fn write_tag<W: Write>(
    w: &mut W,
    kind: i32,
    ftype: u32,
    payload: &[u8],
    next: i32,
) -> std::io::Result<()> {
    w.write_all(&kind.to_be_bytes())?;
    w.write_all(&ftype.to_be_bytes())?;
    w.write_all(&(payload.len() as i32).to_be_bytes())?;
    w.write_all(&next.to_be_bytes())?;
    if !payload.is_empty() {
        w.write_all(payload)?;
    }
    Ok(())
}

fn write_block_start<W: Write>(w: &mut W, block: i32) -> std::io::Result<()> {
    write_tag(
        w,
        FIFF_BLOCK_START,
        FIFFT_INT,
        &block.to_be_bytes(),
        FIFFV_NEXT_SEQ,
    )
}

fn write_block_end<W: Write>(w: &mut W, block: i32) -> std::io::Result<()> {
    write_tag(
        w,
        FIFF_BLOCK_END,
        FIFFT_INT,
        &block.to_be_bytes(),
        FIFFV_NEXT_SEQ,
    )
}

fn write_block_end_last<W: Write>(w: &mut W, block: i32) -> std::io::Result<()> {
    // Last tag in the file: terminate with next = FIFFV_NEXT_NONE so a reader
    // that walks `next` pointers stops here cleanly. (Our reader scans
    // sequentially anyway, but MNE's reader respects `next`.)
    write_tag(
        w,
        FIFF_BLOCK_END,
        FIFFT_INT,
        &block.to_be_bytes(),
        FIFFV_NEXT_NONE,
    )
}

// ── Channel-info struct (inverse of `ChannelInfo::from_bytes`) ───────────────

fn ch_info_to_bytes(ch: &ChannelInfo) -> [u8; 96] {
    let mut out = [0u8; 96];
    out[0..4].copy_from_slice(&ch.scan_no.to_be_bytes());
    out[4..8].copy_from_slice(&ch.log_no.to_be_bytes());
    out[8..12].copy_from_slice(&ch.kind.to_be_bytes());
    out[12..16].copy_from_slice(&ch.range.to_be_bytes());
    out[16..20].copy_from_slice(&ch.cal.to_be_bytes());
    out[20..24].copy_from_slice(&ch.coil_type.to_be_bytes());
    for (i, &v) in ch.loc.iter().enumerate() {
        out[24 + i * 4..28 + i * 4].copy_from_slice(&v.to_be_bytes());
    }
    out[72..76].copy_from_slice(&ch.unit.to_be_bytes());
    out[76..80].copy_from_slice(&ch.unit_mul.to_be_bytes());
    // Name: 16-byte null-padded Latin-1
    let name_bytes = ch.name.as_bytes();
    let n = name_bytes.len().min(16);
    out[80..80 + n].copy_from_slice(&name_bytes[..n]);
    out
}
