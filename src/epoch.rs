//! Fixed-length epoching.
//!
//! Splits continuous [C, T] data into non-overlapping windows of
//! `epoch_samples` samples, dropping any trailing incomplete window.
//! Then applies per-channel baseline correction to each epoch.
use ndarray::{Array2, Array3, s};
use crate::normalize::baseline_correct_inplace;

/// Epoch `data` ([C, T]) into non-overlapping windows, return as `Vec<Array2<f32>>`.
/// Each entry is shape [C, epoch_samples] with baseline correction applied.
pub fn epoch_and_baseline(data: &Array2<f32>, epoch_samples: usize) -> Vec<Array2<f32>> {
    let arr3 = epoch(data, epoch_samples);
    let n_e = arr3.shape()[0];
    (0..n_e).map(|e| arr3.slice(s![e, .., ..]).to_owned()).collect()
}

/// Epoch `data` ([C, T]) into a 3-D array [E, C, epoch_samples].
/// Trailing samples that don't fill a complete epoch are discarded.
/// Baseline correction (subtract per-channel epoch mean) is applied.
pub fn epoch(data: &Array2<f32>, epoch_samples: usize) -> Array3<f32> {
    let (n_ch, n_t) = data.dim();
    let n_epochs = n_t / epoch_samples;

    let mut out = Array3::<f32>::zeros((n_epochs, n_ch, epoch_samples));
    for e in 0..n_epochs {
        let start = e * epoch_samples;
        out.slice_mut(s![e, .., ..])
           .assign(&data.slice(s![.., start..start + epoch_samples]));
    }

    baseline_correct_inplace(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn epoch_count_and_shape() {
        let data = Array2::from_elem((12, 3840), 1.0_f32);
        let epochs = epoch(&data, 1280);
        assert_eq!(epochs.shape(), &[3, 12, 1280]);
    }

    #[test]
    fn trailing_samples_dropped() {
        // 1300 samples with epoch_size=1280 → 1 epoch (20 trailing samples dropped).
        let data = Array2::from_elem((4, 1300), 0.5_f32);
        let epochs = epoch(&data, 1280);
        assert_eq!(epochs.shape()[0], 1);
    }

    #[test]
    fn baseline_applied() {
        // Constant signal: after baseline → all zeros.
        let data = Array2::from_elem((4, 2560), 3.0_f32);
        let epochs = epoch(&data, 1280);
        for &v in epochs.iter() {
            approx::assert_abs_diff_eq!(v, 0.0, epsilon = 1e-5_f32);
        }
    }
}
