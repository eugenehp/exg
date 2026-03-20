//! LUNA-format safetensors I/O.
//!
//! Serializes preprocessed epochs in a format compatible with the
//! `luna-rs` `InputBatch`.
//!
//! See [`export_luna_epochs`] and [`load_luna_epochs`] for the main entry points.

use anyhow::{Context, Result};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

use exg::io::StWriter;

/// A single epoch ready for LUNA inference.
///
/// Compatible with luna-rs `InputBatch` format when loaded from safetensors.
#[derive(Debug, Clone)]
pub struct LunaEpoch {
    /// Signal data, shape `[C, T]` (e.g. `[22, 1280]` for TCP montage).
    pub signal: Array2<f32>,
    /// Channel 3D positions in metres, shape `[C, 3]`.
    pub channel_positions: Array2<f32>,
    /// Bipolar channel names (e.g. `["FP1-F7", "F7-T3", ...]`).
    pub channel_names: Vec<String>,
}

/// Export LUNA-format epochs to a safetensors file.
///
/// The output file is compatible with luna-rs `InputBatch` loading.
///
/// # Layout
///
/// ```text
/// n_epochs: I32 [1]
/// signal_0: F32 [C, T]
/// positions_0: F32 [C, 3]
/// ch_names_0: U8 [len]     — newline-separated channel names
/// signal_1: ...
/// ...
/// ```
///
/// # Arguments
/// * `epochs` — the preprocessed epochs to export
/// * `path` — output file path (`.safetensors`)
pub fn export_luna_epochs(epochs: &[LunaEpoch], path: &Path) -> Result<()> {
    let n = epochs.len();
    let mut w = StWriter::new();

    w.add_i32("n_epochs", &[n as i32], &[1]);

    for (i, epoch) in epochs.iter().enumerate() {
        // Signal
        w.add_f32_arr2(&format!("signal_{i}"), &epoch.signal);

        // Channel positions
        w.add_f32_arr2(&format!("positions_{i}"), &epoch.channel_positions);

        // Channel names as newline-separated UTF-8 string
        let names_str = epoch.channel_names.join("\n");
        let name_bytes = names_str.as_bytes();
        w.entries.push((
            format!("ch_names_{i}"),
            name_bytes.to_vec(),
            "U8",
            vec![name_bytes.len()],
        ));
    }

    w.write(path)
}

// ── Low-level safetensors helpers ─────────────────────────────────────────────

fn parse_header(bytes: &[u8]) -> Result<(HashMap<String, serde_json::Value>, usize)> {
    anyhow::ensure!(bytes.len() >= 8, "safetensors file too small");
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

/// Load LUNA-format epochs from a safetensors file.
///
/// Inverse of [`export_luna_epochs`].
pub fn load_luna_epochs(path: &Path) -> Result<Vec<LunaEpoch>> {
    let bytes = std::fs::read(path).context("reading LUNA safetensors")?;
    let (header, data_start) = parse_header(&bytes)?;

    let n_epochs_entry = header.get("n_epochs").context("missing n_epochs")?;
    let n_epochs_bytes = {
        let offsets = n_epochs_entry["data_offsets"].as_array().unwrap();
        let s = offsets[0].as_u64().unwrap() as usize;
        let e = offsets[1].as_u64().unwrap() as usize;
        &bytes[data_start + s..data_start + e]
    };
    let n_epochs = i32::from_le_bytes([
        n_epochs_bytes[0], n_epochs_bytes[1],
        n_epochs_bytes[2], n_epochs_bytes[3],
    ]) as usize;

    let mut epochs = Vec::with_capacity(n_epochs);

    for i in 0..n_epochs {
        // Signal
        let sig_key = format!("signal_{i}");
        let sig_entry = header.get(&sig_key)
            .with_context(|| format!("missing {sig_key}"))?;
        let sig_shape = shape_of(sig_entry);
        let sig_vec = read_f32_tensor(&bytes, data_start, sig_entry)?;
        let signal = Array2::from_shape_vec((sig_shape[0], sig_shape[1]), sig_vec)?;

        // Positions
        let pos_key = format!("positions_{i}");
        let pos_entry = header.get(&pos_key)
            .with_context(|| format!("missing {pos_key}"))?;
        let pos_shape = shape_of(pos_entry);
        let pos_vec = read_f32_tensor(&bytes, data_start, pos_entry)?;
        let channel_positions = Array2::from_shape_vec((pos_shape[0], pos_shape[1]), pos_vec)?;

        // Channel names
        let names_key = format!("ch_names_{i}");
        let channel_names = if let Some(names_entry) = header.get(&names_key) {
            let offsets = names_entry["data_offsets"].as_array().unwrap();
            let s = offsets[0].as_u64().unwrap() as usize;
            let e = offsets[1].as_u64().unwrap() as usize;
            let raw = &bytes[data_start + s..data_start + e];
            std::str::from_utf8(raw)?
                .split('\n')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        } else {
            vec![]
        };

        epochs.push(LunaEpoch { signal, channel_positions, channel_names });
    }

    Ok(epochs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luna_epoch_roundtrip() {
        let epoch = LunaEpoch {
            signal: Array2::from_shape_fn((22, 1280), |(c, t)| {
                (c as f32 * 0.1 + t as f32 * 0.001).sin()
            }),
            channel_positions: Array2::zeros((22, 3)),
            channel_names: vec![
                "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
                "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
                "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                "FZ-CZ", "CZ-PZ",
                "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
            ].into_iter().map(String::from).collect(),
        };

        let dir = std::env::temp_dir().join("exg_luna_export_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_epochs.safetensors");

        // Export
        export_luna_epochs(&[epoch.clone()], &path).unwrap();

        // Load back
        let loaded = load_luna_epochs(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].signal.dim(), (22, 1280));
        assert_eq!(loaded[0].channel_positions.dim(), (22, 3));
        assert_eq!(loaded[0].channel_names.len(), 22);
        assert_eq!(loaded[0].channel_names[0], "FP1-F7");

        // Check signal values match
        for c in 0..22 {
            for t in 0..1280 {
                approx::assert_abs_diff_eq!(
                    loaded[0].signal[[c, t]],
                    epoch.signal[[c, t]],
                    epsilon = 1e-6
                );
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
