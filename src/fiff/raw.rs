//! Raw FIF data reader — the Rust equivalent of `mne/io/fiff/raw.py`.
//!
//! # Algorithm
//! 1. Open the file.
//! 2. Load the tag directory (fast path: embedded dir tag; slow path: scan).
//! 3. Build the block tree from the directory.
//! 4. Read `MeasInfo` from `FIFFB_MEAS_INFO`.
//! 5. Find the `FIFFB_RAW_DATA` (or `FIFFB_CONTINUOUS_DATA`) node.
//! 6. Walk its tag entries to collect data-buffer records.
//! 7. Optionally pre-load all buffers into a `[n_chan, n_times]` array.
//!
//! # Calibration
//! ```text
//! calibrated_f64[ch, t] = raw_value[t, ch] × info.chs[ch].cal × info.chs[ch].range
//! ```
//! This matches MNE's `_cals = np.array([ch['range'] * ch['cal'] for ch in chs])`.
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use anyhow::{bail, Context, Result};
use ndarray::Array2;

use super::constants::*;
use super::info::{read_meas_info, MeasInfo};
use super::tag::{read_i32, TagHeader};
use super::tree::{read_tree, scan_directory, try_load_directory};

// ── Buffer record ─────────────────────────────────────────────────────────

/// Metadata for one raw-data buffer block in the file.
#[derive(Debug, Clone)]
pub struct BufferRecord {
    /// Tag header for this buffer (use `.pos + 16` to seek to data).
    pub tag:        TagHeader,
    /// Absolute first sample index (in acquisition time; may include initial skip).
    pub first_samp: u64,
    /// Number of samples in this buffer.
    pub n_samp:     usize,
}

// ── RawFif ───────────────────────────────────────────────────────────────

/// A loaded raw FIF recording.
#[derive(Debug, Clone)]
pub struct RawFif {
    /// Measurement info (channels, sfreq, …).
    pub info:       MeasInfo,
    /// First sample index in acquisition time.
    pub first_samp: u64,
    /// Last sample index (inclusive) in acquisition time.
    pub last_samp:  u64,
    /// File this was read from (for lazy re-reads).
    pub path:       PathBuf,
    /// Buffer table: one record per contiguous data block in the file.
    pub buffers:    Vec<BufferRecord>,
}

impl RawFif {
    /// Total number of time points.
    #[inline]
    pub fn n_times(&self) -> usize {
        (self.last_samp - self.first_samp + 1) as usize
    }

    /// Total duration in seconds.
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        self.n_times() as f64 / self.info.sfreq
    }

    /// Read **all** data into a `[n_chan, n_times]` f64 array with calibration.
    ///
    /// This is equivalent to `raw.get_data()` in MNE with `preload=True`.
    pub fn read_all_data(&self) -> Result<Array2<f64>> {
        let n_ch  = self.info.n_chan;
        let n_t   = self.n_times();
        let cals  = self.info.cals();
        let mut out = Array2::<f64>::zeros((n_ch, n_t));

        let file = File::open(&self.path)
            .with_context(|| format!("open {}", self.path.display()))?;
        let mut reader = BufReader::new(file);
        let mut t_offset: usize = 0;

        for buf in &self.buffers {
            let n_samp = buf.n_samp;
            let data = read_buffer_data(&mut reader, &buf.tag, n_samp, n_ch, &cals)?;
            // data is [n_ch, n_samp]
            out.slice_mut(ndarray::s![.., t_offset..t_offset + n_samp])
               .assign(&data);
            t_offset += n_samp;
        }
        assert_eq!(t_offset, n_t, "buffer totals don't match n_times");
        Ok(out)
    }

    /// Read a half-open time slice `[start, end)` relative to `first_samp`.
    ///
    /// Mirrors `raw[start:end]` in MNE.
    pub fn read_slice(&self, start: usize, end: usize) -> Result<Array2<f64>> {
        let n_ch = self.info.n_chan;
        let cals = self.info.cals();
        let mut out = Array2::<f64>::zeros((n_ch, end - start));

        let file = File::open(&self.path)
            .with_context(|| format!("open {}", self.path.display()))?;
        let mut reader = BufReader::new(file);

        let mut samp_base: usize = 0;     // cumulative sample offset across buffers
        let mut out_offset: usize = 0;

        for buf in &self.buffers {
            let n_samp = buf.n_samp;
            let buf_end = samp_base + n_samp;
            // Does this buffer overlap with [start, end)?
            if samp_base < end && buf_end > start {
                let pick_l = start.saturating_sub(samp_base);
                let pick_r = n_samp.min(end - samp_base);
                let n_pick = pick_r - pick_l;

                let data = read_buffer_data(&mut reader, &buf.tag, n_samp, n_ch, &cals)?;
                out.slice_mut(ndarray::s![.., out_offset..out_offset + n_pick])
                   .assign(&data.slice(ndarray::s![.., pick_l..pick_r]));
                out_offset += n_pick;
            }
            samp_base += n_samp;
        }
        Ok(out)
    }
}

// ── Reader entry point ────────────────────────────────────────────────────

/// Open a FIF file and return a `RawFif` without preloading data.
///
/// Mirrors `mne.io.read_raw_fif(fname, preload=False)`.
pub fn open_raw<P: AsRef<Path>>(path: P) -> Result<RawFif> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(file);

    // 1. Load tag directory ------------------------------------------------
    let directory = match try_load_directory(&mut reader)? {
        Some(d) => d,
        None    => scan_directory(&mut reader)?,
    };

    // 2. Build block tree --------------------------------------------------
    let tree = read_tree(&mut reader, &directory)?;

    // 3. Read measurement info --------------------------------------------
    let info = read_meas_info(&mut reader, &tree)?;

    // 4. Find raw data block -----------------------------------------------
    let meas_node = tree
        .find_block(FIFFB_MEAS)
        .ok_or_else(|| anyhow::anyhow!("FIFFB_MEAS not found"))?;

    let raw_node = meas_node
        .find_block(FIFFB_RAW_DATA)
        .or_else(|| meas_node.find_block(FIFFB_CONTINUOUS_DATA))
        .ok_or_else(|| anyhow::anyhow!("no raw-data block in FIF file"))?;

    // 5. Walk data directory -----------------------------------------------
    let nchan = info.n_chan;
    let mut first_samp: u64 = 0;
    let mut first_skip: usize = 0;
    let mut nskip: usize = 0;
    let mut buffers: Vec<BufferRecord> = Vec::new();
    let mut first = true;

    // Pre-scan for FIFF_FIRST_SAMPLE / initial DATA_SKIP before any DATA_BUFFER
    for ent in &raw_node.entries {
        if ent.kind == FIFF_FIRST_SAMPLE {
            first_samp = read_i32(&mut reader, ent)? as u64;
        }
    }

    for ent in &raw_node.entries {
        match ent.kind {
            FIFF_FIRST_SAMPLE => {} // already consumed above
            FIFF_DATA_SKIP if first => {
                first_skip = read_i32(&mut reader, ent)? as usize;
                first = false;
            }
            FIFF_DATA_BUFFER => {
                first = false;
                let bps = bytes_per_sample(ent.ftype)
                    .ok_or_else(|| anyhow::anyhow!("unknown buffer type {}", ent.ftype))?;
                let n_samp = ent.size as usize / (bps * nchan);

                // Apply first_skip (only once, before the first real buffer).
                if first_skip > 0 {
                    first_samp += (n_samp * first_skip) as u64;
                    first_skip = 0;
                }

                // Pending inter-buffer skip → emit a null gap.
                if nskip > 0 {
                    let gap_samp = n_samp * nskip;
                    // We represent gaps by a tag with kind=-1 (no real data).
                    let gap_tag = TagHeader { kind: -1, ftype: 0, size: 0, next: -1, pos: 0 };
                    buffers.push(BufferRecord {
                        tag: gap_tag,
                        first_samp,
                        n_samp: gap_samp,
                    });
                    first_samp += gap_samp as u64;
                    nskip = 0;
                }

                buffers.push(BufferRecord { tag: *ent, first_samp, n_samp });
                first_samp += n_samp as u64;
            }
            FIFF_DATA_SKIP => {
                nskip += read_i32(&mut reader, ent)? as usize;
            }
            _ => {}
        }
    }

    if buffers.is_empty() {
        bail!("no FIFF_DATA_BUFFER tags found in raw-data block");
    }

    let last_samp = first_samp - 1;
    // Recompute first_samp from buffers (it was mutated above).
    let actual_first = buffers[0].first_samp;

    Ok(RawFif {
        info,
        first_samp: actual_first,
        last_samp,
        path: path.to_path_buf(),
        buffers,
    })
}

// ── Buffer data reader ───────────────────────────────────────────────────

/// Read one data buffer from file and return `[n_chan, n_samp]` f64 with calibration.
///
/// The on-disk layout is `[n_samp, n_chan]` (row-major, big-endian) — i.e.
/// interleaved channels — identical to MNE's `one.reshape(nsamp, nchan)`.
fn read_buffer_data<R: Read + Seek>(
    reader: &mut R,
    tag:    &TagHeader,
    n_samp: usize,
    n_chan: usize,
    cals:   &[f64],
) -> Result<Array2<f64>> {
    // Gap buffers (kind == -1) → return zeros.
    if tag.kind < 0 {
        return Ok(Array2::<f64>::zeros((n_chan, n_samp)));
    }
    reader
        .seek(std::io::SeekFrom::Start(tag.data_pos()))
        .with_context(|| format!("seek to buffer data @ {:#x}", tag.data_pos()))?;

    let mut out = Array2::<f64>::zeros((n_chan, n_samp));

    match tag.ftype {
        FIFFT_FLOAT => {
            let mut buf = [0u8; 4];
            for t in 0..n_samp {
                for c in 0..n_chan {
                    reader.read_exact(&mut buf)?;
                    let raw = f32::from_be_bytes(buf) as f64;
                    out[[c, t]] = raw * cals[c];
                }
            }
        }
        FIFFT_DOUBLE => {
            let mut buf = [0u8; 8];
            for t in 0..n_samp {
                for c in 0..n_chan {
                    reader.read_exact(&mut buf)?;
                    let raw = f64::from_be_bytes(buf);
                    out[[c, t]] = raw * cals[c];
                }
            }
        }
        FIFFT_INT => {
            let mut buf = [0u8; 4];
            for t in 0..n_samp {
                for c in 0..n_chan {
                    reader.read_exact(&mut buf)?;
                    let raw = i32::from_be_bytes(buf) as f64;
                    out[[c, t]] = raw * cals[c];
                }
            }
        }
        FIFFT_SHORT | FIFFT_DAU_PACK16 => {
            let mut buf = [0u8; 2];
            for t in 0..n_samp {
                for c in 0..n_chan {
                    reader.read_exact(&mut buf)?;
                    let raw = i16::from_be_bytes(buf) as f64;
                    out[[c, t]] = raw * cals[c];
                }
            }
        }
        other => bail!("unsupported buffer type {other}"),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Path to the real FIF test file (only available in CI / local dev with data).
    fn sample_fif() -> Option<PathBuf> {
        let p = PathBuf::from(
            concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif"),
        );
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn open_raw_basic_info() {
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        assert_eq!(raw.info.n_chan, 12);
        assert_abs_diff_eq!(raw.info.sfreq, 256.0, epsilon = 1e-6);
        assert_eq!(raw.n_times(), 3840);
        assert_eq!(raw.first_samp, 2560);
        assert_eq!(raw.last_samp, 6399);
    }

    #[test]
    fn channel_names_match_mne() {
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        let expected = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8"];
        for (got, exp) in raw.info.ch_names().iter().zip(expected.iter()) {
            assert_eq!(*got, *exp, "channel name mismatch");
        }
    }

    #[test]
    fn all_channels_eeg() {
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        for ch in &raw.info.chs {
            assert_eq!(ch.kind, FIFFV_EEG_CH, "channel {} is not EEG", ch.name);
        }
    }

    #[test]
    fn calibration_factors_are_one() {
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        for ch in &raw.info.chs {
            assert_abs_diff_eq!(ch.calibration(), 1.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn buffer_count_matches_mne() {
        // MNE reports 15 data buffers for sample1_raw.fif
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        let n_real = raw.buffers.iter().filter(|b| b.tag.kind >= 0).count();
        assert_eq!(n_real, 15);
    }

    #[test]
    fn data_matches_mne_first_sample() {
        // Reference: data[0, 0] from MNE = 2.0721382e-05
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        let data = raw.read_all_data().unwrap();
        assert_abs_diff_eq!(data[[0, 0]], 2.0721382e-05_f64, epsilon = 1e-10);
    }

    #[test]
    fn data_shape() {
        let Some(fif) = sample_fif() else { return; };
        let raw = open_raw(&fif).unwrap();
        let data = raw.read_all_data().unwrap();
        assert_eq!(data.shape(), &[12, 3840]);
    }
}
