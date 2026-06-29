//! EDF/EDF+ file reader.
//!
//! Implements reading of `.edf` EEG recordings compatible with
//! [MNE-Python](https://mne.tools) `mne.io.read_raw_edf()`.
//!
//! # Quick start
//! ```no_run
//! use exg::edf::open_raw_edf;
//!
//! let raw = open_raw_edf("data/recording.edf").unwrap();
//! println!("{} channels @ {} Hz", raw.header.num_signals, raw.header.sample_rate);
//! let data = raw.read_all_data().unwrap();  // [C, T] f32, physical units
//! ```
//!
//! # EDF Format Reference
//!
//! The European Data Format (EDF) stores continuous polygraphic recordings.
//! Each file has a fixed-size header followed by data records containing
//! interleaved 16-bit integer samples for all channels.
//!
//! EDF+ extends EDF with annotations stored in special "EDF Annotations"
//! channels. This reader parses EDF+ annotations and exposes them via
//! [`RawEdf::annotations`].

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ── Header structures ─────────────────────────────────────────────────────────

/// Parsed EDF/EDF+ file header.
#[derive(Debug, Clone)]
pub struct EdfHeader {
    /// Version of data format (usually "0").
    pub version: String,
    /// Local patient identification.
    pub patient_id: String,
    /// Local recording identification.
    pub recording_id: String,
    /// Start date as string (dd.mm.yy).
    pub start_date: String,
    /// Start time as string (hh.mm.ss).
    pub start_time: String,
    /// Number of bytes in the header.
    pub header_bytes: usize,
    /// Reserved field (contains "EDF+C" or "EDF+D" for EDF+).
    pub reserved: String,
    /// Number of data records in the file.
    pub num_records: usize,
    /// Duration of each data record in seconds.
    pub record_duration: f64,
    /// Number of signals (channels) in the file.
    pub num_signals: usize,
    /// Per-signal information.
    pub signals: Vec<SignalHeader>,
    /// Maximum sampling rate across all signals (the effective sfreq).
    pub sample_rate: f32,
    /// Whether this is an EDF+ file.
    pub is_edfplus: bool,
}

/// Per-signal (channel) header information.
#[derive(Debug, Clone)]
pub struct SignalHeader {
    /// Channel label (e.g. "EEG FP1-REF").
    pub label: String,
    /// Transducer type.
    pub transducer: String,
    /// Physical dimension (e.g. "uV").
    pub physical_dimension: String,
    /// Physical minimum.
    pub physical_min: f64,
    /// Physical maximum.
    pub physical_max: f64,
    /// Digital minimum.
    pub digital_min: f64,
    /// Digital maximum.
    pub digital_max: f64,
    /// Prefiltering string.
    pub prefiltering: String,
    /// Number of samples per data record for this signal.
    pub samples_per_record: usize,
    /// Reserved field.
    pub reserved: String,
}

impl SignalHeader {
    /// Compute the calibration factor: physical_range / digital_range.
    pub fn cal(&self) -> f64 {
        let phys_range = self.physical_max - self.physical_min;
        let dig_range = self.digital_max - self.digital_min;
        if dig_range == 0.0 {
            1.0
        } else {
            phys_range / dig_range
        }
    }

    /// Compute the offset for converting digital to physical values.
    pub fn offset(&self) -> f64 {
        self.physical_min - self.digital_min * self.cal()
    }

    /// Get the unit scaling factor (to convert to volts).
    pub fn unit_scale(&self) -> f64 {
        let dim = self.physical_dimension.trim().to_lowercase();
        // Allow μ (greek mu), µ (micro symbol), u prefix
        if dim == "uv" || dim == "µv" || dim == "\u{03bc}v" {
            1e-6
        } else if dim == "mv" {
            1e-3
        } else {
            1.0 // "v" or unknown unit — no scaling
        }
    }
}

/// An EDF+ annotation event.
#[derive(Debug, Clone)]
pub struct EdfAnnotation {
    /// Onset time in seconds from the start of the recording.
    pub onset: f64,
    /// Duration in seconds (0 if not specified).
    pub duration: f64,
    /// Annotation text.
    pub description: String,
}

// ── Raw EDF structure ─────────────────────────────────────────────────────────

/// A loaded EDF/EDF+ file.
pub struct RawEdf {
    /// Parsed header.
    pub header: EdfHeader,
    /// Path to the file (for lazy reading).
    pub path: std::path::PathBuf,
    /// Parsed EDF+ annotations (empty for plain EDF).
    pub annotations: Vec<EdfAnnotation>,
}

/// Open an EDF/EDF+ file and parse the header.
///
/// # Arguments
/// * `path` — Path to the `.edf` file.
///
/// # Returns
/// A [`RawEdf`] struct with the parsed header. Use [`RawEdf::read_all_data()`]
/// to read the signal data.
pub fn open_raw_edf<P: AsRef<Path>>(path: P) -> Result<RawEdf> {
    let path = path.as_ref().to_path_buf();
    let mut file = std::fs::File::open(&path)
        .with_context(|| format!("opening EDF file: {}", path.display()))?;

    let header = read_header(&mut file)?;

    // Read annotations if EDF+
    let annotations = if header.is_edfplus {
        read_annotations(&mut file, &header)?
    } else {
        vec![]
    };

    Ok(RawEdf {
        header,
        path,
        annotations,
    })
}

impl RawEdf {
    /// Read all signal data from the EDF file.
    ///
    /// Returns an `Array2<f32>` of shape `[C, T]` where:
    /// * `C` = number of non-annotation channels
    /// * `T` = total samples (num_records × samples_per_record for the max-rate channel)
    ///
    /// Data is converted to physical units (scaled by cal + offset).
    /// Annotation channels (labeled "EDF Annotations") are excluded.
    pub fn read_all_data(&self) -> Result<Array2<f32>> {
        let mut file = std::fs::File::open(&self.path)?;
        read_data(&mut file, &self.header)
    }

    /// Channel names (excluding annotation channels).
    pub fn channel_names(&self) -> Vec<String> {
        self.header
            .signals
            .iter()
            .filter(|s| !is_annotation_channel(&s.label))
            .map(|s| s.label.clone())
            .collect()
    }

    /// Number of non-annotation channels.
    pub fn num_channels(&self) -> usize {
        self.header
            .signals
            .iter()
            .filter(|s| !is_annotation_channel(&s.label))
            .count()
    }

    /// Total number of samples per channel at the maximum sample rate.
    pub fn num_samples(&self) -> usize {
        let max_spr = self
            .header
            .signals
            .iter()
            .filter(|s| !is_annotation_channel(&s.label))
            .map(|s| s.samples_per_record)
            .max()
            .unwrap_or(0);
        self.header.num_records * max_spr
    }
}

fn is_annotation_channel(label: &str) -> bool {
    let l = label.trim().to_lowercase();
    l.contains("edf annotation")
        || l.contains("edf+annotation")
        || l == "edf annotations"
        || l == "bdf annotations"
}

// ── Header parsing ────────────────────────────────────────────────────────────

fn read_fixed_str<R: Read>(reader: &mut R, n: usize) -> Result<String> {
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf)?;
    // EDF uses ASCII/Latin-1
    Ok(String::from_utf8_lossy(&buf).trim_end().to_string())
}

fn read_fixed_f64<R: Read>(reader: &mut R, n: usize) -> Result<f64> {
    let s = read_fixed_str(reader, n)?;
    let s = s.replace(',', ".");
    s.trim()
        .parse::<f64>()
        .with_context(|| format!("parsing float from EDF header: '{s}'"))
}

fn read_fixed_usize<R: Read>(reader: &mut R, n: usize) -> Result<usize> {
    let s = read_fixed_str(reader, n)?;
    s.trim()
        .parse::<usize>()
        .with_context(|| format!("parsing int from EDF header: '{s}'"))
}

fn read_header<R: Read>(reader: &mut R) -> Result<EdfHeader> {
    let version = read_fixed_str(reader, 8)?;
    let patient_id = read_fixed_str(reader, 80)?;
    let recording_id = read_fixed_str(reader, 80)?;
    let start_date = read_fixed_str(reader, 8)?;
    let start_time = read_fixed_str(reader, 8)?;
    let header_bytes = read_fixed_usize(reader, 8)?;
    let reserved = read_fixed_str(reader, 44)?;
    let num_records = read_fixed_usize(reader, 8)?;
    let record_duration = read_fixed_f64(reader, 8)?;
    let num_signals = read_fixed_usize(reader, 4)?;

    let is_edfplus = reserved.contains("EDF+");

    // Read per-signal headers (each field is num_signals × field_width)
    let mut labels = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        labels.push(read_fixed_str(reader, 16)?);
    }

    let mut transducers = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        transducers.push(read_fixed_str(reader, 80)?);
    }

    let mut phys_dims = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        phys_dims.push(read_fixed_str(reader, 8)?);
    }

    let mut phys_mins = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        phys_mins.push(read_fixed_f64(reader, 8)?);
    }

    let mut phys_maxs = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        phys_maxs.push(read_fixed_f64(reader, 8)?);
    }

    let mut dig_mins = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        dig_mins.push(read_fixed_f64(reader, 8)?);
    }

    let mut dig_maxs = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        dig_maxs.push(read_fixed_f64(reader, 8)?);
    }

    let mut prefilterings = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        prefilterings.push(read_fixed_str(reader, 80)?);
    }

    let mut samples_per_record = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        samples_per_record.push(read_fixed_usize(reader, 8)?);
    }

    // Reserved per signal (32 bytes each)
    let mut sig_reserved = Vec::with_capacity(num_signals);
    for _ in 0..num_signals {
        sig_reserved.push(read_fixed_str(reader, 32)?);
    }

    let mut signals = Vec::with_capacity(num_signals);
    for i in 0..num_signals {
        signals.push(SignalHeader {
            label: labels[i].clone(),
            transducer: transducers[i].clone(),
            physical_dimension: phys_dims[i].clone(),
            physical_min: phys_mins[i],
            physical_max: phys_maxs[i],
            digital_min: dig_mins[i],
            digital_max: dig_maxs[i],
            prefiltering: prefilterings[i].clone(),
            samples_per_record: samples_per_record[i],
            reserved: sig_reserved[i].clone(),
        });
    }

    // Compute the maximum sample rate
    let max_spr = signals
        .iter()
        .filter(|s| !is_annotation_channel(&s.label))
        .map(|s| s.samples_per_record)
        .max()
        .unwrap_or(1);
    let sample_rate = if record_duration > 0.0 {
        max_spr as f32 / record_duration as f32
    } else {
        max_spr as f32
    };

    Ok(EdfHeader {
        version,
        patient_id,
        recording_id,
        start_date,
        start_time,
        header_bytes,
        reserved,
        num_records,
        record_duration,
        num_signals,
        signals,
        sample_rate,
        is_edfplus,
    })
}

// ── Data reading ──────────────────────────────────────────────────────────────

fn read_data<R: Read + Seek>(reader: &mut R, header: &EdfHeader) -> Result<Array2<f32>> {
    reader.seek(SeekFrom::Start(header.header_bytes as u64))?;

    // Identify non-annotation signal indices
    let sig_indices: Vec<usize> = (0..header.num_signals)
        .filter(|&i| !is_annotation_channel(&header.signals[i].label))
        .collect();

    if sig_indices.is_empty() {
        bail!("No non-annotation channels found in EDF file");
    }

    let max_spr = sig_indices
        .iter()
        .map(|&i| header.signals[i].samples_per_record)
        .max()
        .unwrap_or(1);

    let n_ch = sig_indices.len();
    let n_total = header.num_records * max_spr;
    let mut data = Array2::<f32>::zeros((n_ch, n_total));

    // Total samples in one data record across all signals
    let record_samples: usize = header.signals.iter().map(|s| s.samples_per_record).sum();

    // Read record by record
    let mut record_buf = vec![0i16; record_samples];
    let mut byte_buf = vec![0u8; record_samples * 2];

    for rec in 0..header.num_records {
        reader.read_exact(&mut byte_buf)?;
        // Convert bytes to i16 (little-endian)
        for (i, chunk) in byte_buf.chunks_exact(2).enumerate() {
            record_buf[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        // Extract each non-annotation channel
        let mut offset = 0;
        let mut out_ch = 0;
        for sig_idx in 0..header.num_signals {
            let spr = header.signals[sig_idx].samples_per_record;

            if sig_indices.contains(&sig_idx) {
                let sig = &header.signals[sig_idx];
                let cal = sig.cal();
                let off = sig.offset();
                let unit = sig.unit_scale();

                let dst_start = rec * max_spr;

                if spr == max_spr {
                    // Same sample rate as max → direct copy
                    for j in 0..spr {
                        let digital = record_buf[offset + j] as f64;
                        let physical = (digital * cal + off) * unit;
                        data[[out_ch, dst_start + j]] = physical as f32;
                    }
                } else {
                    // Different sample rate → simple nearest-neighbor upsampling
                    // (MNE uses scipy resample for mixed rates, but for LUNA
                    // all channels typically have the same rate)
                    for j in 0..max_spr {
                        let src_j = (j * spr) / max_spr;
                        let digital = record_buf[offset + src_j] as f64;
                        let physical = (digital * cal + off) * unit;
                        data[[out_ch, dst_start + j]] = physical as f32;
                    }
                }
                out_ch += 1;
            }

            offset += spr;
        }
    }

    Ok(data)
}

// ── EDF+ Annotations parsing ─────────────────────────────────────────────────

fn read_annotations<R: Read + Seek>(
    reader: &mut R,
    header: &EdfHeader,
) -> Result<Vec<EdfAnnotation>> {
    reader.seek(SeekFrom::Start(header.header_bytes as u64))?;

    let tal_indices: Vec<usize> = (0..header.num_signals)
        .filter(|&i| is_annotation_channel(&header.signals[i].label))
        .collect();

    if tal_indices.is_empty() {
        return Ok(vec![]);
    }

    let record_samples: usize = header.signals.iter().map(|s| s.samples_per_record).sum();
    let record_bytes = record_samples * 2;

    let mut annotations = Vec::new();
    let mut record_buf = vec![0u8; record_bytes];

    for _rec in 0..header.num_records {
        reader.read_exact(&mut record_buf)?;

        // Extract TAL bytes for each annotation channel
        for &tal_idx in &tal_indices {
            let mut byte_offset = 0;
            for i in 0..tal_idx {
                byte_offset += header.signals[i].samples_per_record * 2;
            }
            let tal_bytes = header.signals[tal_idx].samples_per_record * 2;
            let tal_data = &record_buf[byte_offset..byte_offset + tal_bytes];

            // Parse TAL (Time-stamped Annotation List)
            // Format: +onset\x15duration\x14annotation\x14\x00
            let tal_str = String::from_utf8_lossy(tal_data);
            for entry in tal_str.split('\x00') {
                if entry.is_empty() {
                    continue;
                }
                parse_tal_entry(entry, &mut annotations);
            }
        }
    }

    Ok(annotations)
}

fn parse_tal_entry(entry: &str, annotations: &mut Vec<EdfAnnotation>) {
    // TAL entry format: +onset[\x15duration]\x14annotation[\x14annotation...]
    let parts: Vec<&str> = entry.split('\x14').collect();
    if parts.is_empty() {
        return;
    }

    let time_part = parts[0];
    if time_part.is_empty() {
        return;
    }

    // Parse onset and optional duration separated by \x15
    let (onset_str, dur_str) = if let Some(pos) = time_part.find('\x15') {
        (&time_part[..pos], &time_part[pos + 1..])
    } else {
        (time_part, "")
    };

    let onset = match onset_str.replace('+', "").trim().parse::<f64>() {
        Ok(v) => v,
        Err(_) => return,
    };

    let duration = if dur_str.is_empty() {
        0.0
    } else {
        dur_str.parse::<f64>().unwrap_or(0.0)
    };

    for &desc in &parts[1..] {
        if desc.is_empty() {
            continue;
        }
        annotations.push(EdfAnnotation {
            onset,
            duration,
            description: desc.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_header_cal_offset() {
        let sig = SignalHeader {
            label: "EEG FP1".into(),
            transducer: String::new(),
            physical_dimension: "uV".into(),
            physical_min: -3200.0,
            physical_max: 3200.0,
            digital_min: -32768.0,
            digital_max: 32767.0,
            prefiltering: String::new(),
            samples_per_record: 256,
            reserved: String::new(),
        };
        let cal = sig.cal();
        // physical_range = 6400, digital_range = 65535
        approx::assert_abs_diff_eq!(cal, 6400.0 / 65535.0, epsilon = 1e-6);
    }

    #[test]
    fn test_unit_scale() {
        let mut sig = SignalHeader {
            label: String::new(),
            transducer: String::new(),
            physical_dimension: "uV".into(),
            physical_min: 0.0,
            physical_max: 0.0,
            digital_min: 0.0,
            digital_max: 0.0,
            prefiltering: String::new(),
            samples_per_record: 0,
            reserved: String::new(),
        };
        assert_eq!(sig.unit_scale(), 1e-6);
        sig.physical_dimension = "mV".into();
        assert_eq!(sig.unit_scale(), 1e-3);
        sig.physical_dimension = "V".into();
        assert_eq!(sig.unit_scale(), 1.0);
    }

    #[test]
    fn test_annotation_channel_detection() {
        assert!(is_annotation_channel("EDF Annotations"));
        assert!(is_annotation_channel("EDF Annotations "));
        assert!(!is_annotation_channel("EEG FP1-REF"));
    }

    #[test]
    fn test_parse_tal_entry() {
        let mut anns = Vec::new();
        parse_tal_entry("+0.0\x15\x14", &mut anns);
        // The time-keeping annotation (no description) produces nothing
        assert_eq!(anns.len(), 0);

        parse_tal_entry("+1.5\x152.0\x14Seizure onset\x14", &mut anns);
        assert_eq!(anns.len(), 1);
        approx::assert_abs_diff_eq!(anns[0].onset, 1.5, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(anns[0].duration, 2.0, epsilon = 1e-9);
        assert_eq!(anns[0].description, "Seizure onset");
    }
}
