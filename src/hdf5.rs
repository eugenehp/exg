//! HDF5 reader for preprocessed EEG datasets.
//!
//! Reads TUH/LUNA-style HDF5 files where data is stored as:
//! ```text
//! data_group_0/
//!   X: [N, C, T] float32  — signal epochs
//!   y: [N] int64           — labels (optional)
//! data_group_1/
//!   X: [N, C, T] float32
//!   y: [N] int64
//! ...
//! ```
//!
//! This module is feature-gated behind `--features hdf5`.
//!
//! # Example
//! ```no_run
//! # #[cfg(feature = "hdf5")]
//! # {
//! use exg::hdf5::read_dataset;
//!
//! let samples = read_dataset("train.h5").unwrap();
//! for (signal, label) in &samples {
//!     println!("signal: {:?}, label: {:?}", signal.dim(), label);
//! }
//! # }
//! ```

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use std::path::Path;

/// A single sample read from an HDF5 dataset (signal + optional label).
#[derive(Debug, Clone)]
pub struct HDF5Sample {
    /// Signal data, shape `[C, T]`.
    pub signal: Array2<f32>,
    /// Optional label (e.g. seizure type).
    pub label: Option<i64>,
}

/// Read all samples from an HDF5 dataset file.
///
/// Expects the LUNA/BioFoundation HDF5 structure:
/// - Groups named `data_group_0`, `data_group_1`, ...
/// - Each group contains `X` (float32 signal) and optionally `y` (int labels).
/// - `X` shape: `[N, C, T]` (N samples, C channels, T time points)
/// - `y` shape: `[N]` (N labels)
///
/// Returns a flat `Vec<HDF5Sample>` with all samples from all groups.
pub fn read_dataset<P: AsRef<Path>>(path: P) -> Result<Vec<HDF5Sample>> {
    let path = path.as_ref();
    let file = hdf5::File::open(path)
        .with_context(|| format!("opening HDF5 file: {}", path.display()))?;

    let mut samples = Vec::new();

    // Iterate over all groups in the file
    let mut group_names: Vec<String> = file.member_names()
        .with_context(|| "listing HDF5 groups")?;
    group_names.sort(); // Ensure deterministic order

    for group_name in &group_names {
        let group = match file.group(group_name) {
            Ok(g) => g,
            Err(_) => continue, // Skip non-group members
        };

        // Read X dataset
        let x_ds = group.dataset("X")
            .with_context(|| format!("reading X from group {group_name}"))?;

        let x_shape = x_ds.shape();
        if x_shape.len() < 2 {
            bail!("X in {group_name} has unexpected shape: {x_shape:?}");
        }

        let x_data: Vec<f32> = x_ds.read_raw()
            .with_context(|| format!("reading X data from {group_name}"))?;

        // Determine shape: could be [N, C, T] or [N, T] (single channel)
        let (n_samples, n_ch, n_t) = if x_shape.len() == 3 {
            (x_shape[0], x_shape[1], x_shape[2])
        } else if x_shape.len() == 2 {
            (x_shape[0], 1, x_shape[1])
        } else {
            bail!("X in {group_name} has unsupported dimensionality: {}", x_shape.len());
        };

        // Read optional y dataset
        let labels: Option<Vec<i64>> = match group.dataset("y") {
            Ok(y_ds) => {
                let y_data: Vec<i64> = y_ds.read_raw()
                    .unwrap_or_default();
                if y_data.len() == n_samples {
                    Some(y_data)
                } else {
                    // Try reading as i32 and converting
                    let y_i32: Vec<i32> = y_ds.read_raw().unwrap_or_default();
                    if y_i32.len() == n_samples {
                        Some(y_i32.iter().map(|&v| v as i64).collect())
                    } else {
                        None
                    }
                }
            }
            Err(_) => None,
        };

        // Extract individual samples
        for i in 0..n_samples {
            let offset = i * n_ch * n_t;
            let sample_data: Vec<f32> = x_data[offset..offset + n_ch * n_t].to_vec();
            let signal = Array2::from_shape_vec((n_ch, n_t), sample_data)
                .with_context(|| format!("reshaping sample {i} from {group_name}"))?;

            let label = labels.as_ref().map(|l| l[i]);
            samples.push(HDF5Sample { signal, label });
        }
    }

    Ok(samples)
}

/// Read a dataset and return `(signals, labels)` as separate vectors.
///
/// Convenience wrapper around [`read_dataset`].
pub fn read_dataset_split<P: AsRef<Path>>(path: P) -> Result<(Vec<Array2<f32>>, Vec<Option<i64>>)> {
    let samples = read_dataset(path)?;
    let signals = samples.iter().map(|s| s.signal.clone()).collect();
    let labels = samples.iter().map(|s| s.label).collect();
    Ok((signals, labels))
}

#[cfg(test)]
mod tests {
    // HDF5 tests require actual .h5 files — tested via integration tests
    // with generated test data.
}
