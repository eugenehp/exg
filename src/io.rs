//! Safetensors I/O for the preprocessing pipeline.
//!
//! Reader: parses `raw.safetensors` written by `scripts/read_raw.py`.
use anyhow::{bail, Context, Result};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

// ── Low-level safetensors parser (no dependency on the `safetensors` crate's
//    tensor types — we just need raw bytes → ndarray). ─────────────────────────

fn parse_header(bytes: &[u8]) -> Result<(HashMap<String, serde_json::Value>, usize)> {
    if bytes.len() < 8 {
        bail!("safetensors file too small");
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n])
            .context("failed to parse safetensors header")?;
    Ok((header, 8 + n))
}

fn read_f32_tensor(
    bytes: &[u8],
    data_start: usize,
    entry: &serde_json::Value,
) -> Result<Vec<f32>> {
    let offsets = entry["data_offsets"].as_array().unwrap();
    let s = offsets[0].as_u64().unwrap() as usize;
    let e = offsets[1].as_u64().unwrap() as usize;
    let raw = &bytes[data_start + s..data_start + e];
    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn shape_of(entry: &serde_json::Value) -> Vec<usize> {
    entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

// ── Public structs ────────────────────────────────────────────────────────────

/// Raw EEG data loaded from `raw.safetensors` (output of `scripts/read_raw.py`).
pub struct RawData {
    /// [C, T] in original units (volts or arbitrary), original sample rate.
    pub data: Array2<f32>,
    /// [C, 3] channel positions in metres.
    pub chan_pos: Array2<f32>,
    /// Original sampling rate (Hz).
    pub sfreq: f32,
    /// Channel names (may be empty if not saved).
    pub ch_names: Vec<String>,
}

impl RawData {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).context("reading raw.safetensors")?;
        let (header, data_start) = parse_header(&bytes)?;

        let data_entry = header.get("data").context("missing 'data' key")?;
        let data_shape = shape_of(data_entry);
        let data_vec = read_f32_tensor(&bytes, data_start, data_entry)?;
        let data = Array2::from_shape_vec((data_shape[0], data_shape[1]), data_vec)?;

        let pos_entry = header.get("chan_pos").context("missing 'chan_pos' key")?;
        let pos_shape = shape_of(pos_entry);
        let pos_vec = read_f32_tensor(&bytes, data_start, pos_entry)?;
        let chan_pos = Array2::from_shape_vec((pos_shape[0], pos_shape[1]), pos_vec)?;

        let sfreq_entry = header.get("sfreq").context("missing 'sfreq' key")?;
        let sfreq = read_f32_tensor(&bytes, data_start, sfreq_entry)?[0];

        // Channel names are optional.
        let ch_names = if let Some(e) = header.get("ch_names") {
            let offsets = e["data_offsets"].as_array().unwrap();
            let s = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let raw_str = std::str::from_utf8(&bytes[data_start + s..data_start + end])?;
            raw_str
                .split('\n')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        } else {
            vec![]
        };

        Ok(RawData { data, chan_pos, sfreq, ch_names })
    }
}

// ── Generic safetensors builder ───────────────────────────────────────────────

/// Simple safetensors file writer that handles F32, F64, and I32 tensors.
///
/// Usage:
/// ```rust,no_run
/// use exg::io::StWriter;
/// use std::path::Path;
/// let mut w = StWriter::new();
/// w.add_f32("signal", &[1.0f32, 2.0, 3.0], &[1, 3]);
/// w.add_f64("signal_d", &[1.0f64, 2.0, 3.0], &[1, 3]);
/// w.write(Path::new("/tmp/out.safetensors")).unwrap();
/// ```
pub struct StWriter {
    entries: Vec<(String, Vec<u8>, &'static str, Vec<usize>)>,
}

impl StWriter {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn add_f32(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push((name.to_string(), bytes, "F32", shape.to_vec()));
    }

    pub fn add_f32_arr2(&mut self, name: &str, arr: &ndarray::Array2<f32>) {
        let data: Vec<f32> = arr.iter().copied().collect();
        self.add_f32(name, &data, &[arr.nrows(), arr.ncols()]);
    }

    pub fn add_f64(&mut self, name: &str, data: &[f64], shape: &[usize]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push((name.to_string(), bytes, "F64", shape.to_vec()));
    }

    pub fn add_f64_arr2(&mut self, name: &str, arr: &ndarray::Array2<f64>) {
        let data: Vec<f64> = arr.iter().copied().collect();
        self.add_f64(name, &data, &[arr.nrows(), arr.ncols()]);
    }

    pub fn add_i32(&mut self, name: &str, data: &[i32], shape: &[usize]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push((name.to_string(), bytes, "I32", shape.to_vec()));
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        let mut header_map = serde_json::Map::new();
        let mut offset: usize = 0;
        for (name, data, dtype, shape) in &self.entries {
            header_map.insert(name.clone(), serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + data.len()],
            }));
            offset += data.len();
        }
        let hdr_bytes = serde_json::to_vec(&header_map)?;
        let pad = (8 - hdr_bytes.len() % 8) % 8;
        let padded: Vec<u8> = hdr_bytes.into_iter()
            .chain(std::iter::repeat(b' ').take(pad))
            .collect();
        let mut f = std::fs::File::create(path)?;
        f.write_all(&(padded.len() as u64).to_le_bytes())?;
        f.write_all(&padded)?;
        for (_, data, _, _) in &self.entries {
            f.write_all(data)?;
        }
        Ok(())
    }
}

// ── Batch writer ──────────────────────────────────────────────────────────────

/// Write preprocessed epochs to `batch.safetensors`.
///
/// `epochs[e]`: [C, 1280]   `positions[e]`: [C, 3]
pub fn write_batch(
    epochs: &[Array2<f32>],
    positions: &[Array2<f32>],
    path: &Path,
) -> Result<()> {
    use std::io::Write;

    let n = epochs.len();
    assert_eq!(n, positions.len());

    // Collect tensors: (name, f32 data, shape).
    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = vec![];

    for i in 0..n {
        let eeg  = &epochs[i];
        let pos  = &positions[i];

        let eeg_bytes: Vec<u8> = eeg.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        tensors.push((
            format!("eeg_{i}"),
            eeg_bytes,
            vec![eeg.nrows(), eeg.ncols()],
        ));

        let pos_bytes: Vec<u8> = pos.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        tensors.push((
            format!("chan_pos_{i}"),
            pos_bytes,
            vec![pos.nrows(), pos.ncols()],
        ));
    }

    // n_samples scalar.
    tensors.push((
        "n_samples".into(),
        (n as i32).to_le_bytes().to_vec(),
        vec![1],
    ));

    // Build header.
    let mut header_map = serde_json::Map::new();
    let mut offset: usize = 0;
    let mut dtype_map: Vec<(String, String)> = vec![];

    for (name, data, shape) in &tensors {
        let dtype = if name == "n_samples" { "I32" } else { "F32" };
        header_map.insert(name.clone(), serde_json::json!({
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + data.len()],
        }));
        dtype_map.push((name.clone(), dtype.to_string()));
        offset += data.len();
    }

    let header_bytes = serde_json::to_vec(&header_map)?;
    let pad = (8 - header_bytes.len() % 8) % 8;
    let header_padded: Vec<u8> = header_bytes
        .into_iter()
        .chain(std::iter::repeat(b' ').take(pad))
        .collect();

    let mut f = std::fs::File::create(path)?;
    f.write_all(&(header_padded.len() as u64).to_le_bytes())?;
    f.write_all(&header_padded)?;
    for (_, data, _) in &tensors {
        f.write_all(data)?;
    }

    Ok(())
}
