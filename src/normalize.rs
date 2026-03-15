//! Z-score normalisation and epoch baseline correction.
//!
//! `zscore_global_inplace`  — matches `Normalizer.normalize_raw`:
//!   μ = mean(all channels × all times),  σ = std (ddof=0)
//!   data = (data - μ) / σ
//!
//! `baseline_correct_inplace` — matches `epochs.apply_baseline((None, None))`:
//!   for each channel: epoch[ch, :] -= mean(epoch[ch, :])
use ndarray::{Array2, Array3};

/// Global z-score over all channels and times.
/// Returns (mean, std) used for normalisation.
pub fn zscore_global_inplace(data: &mut Array2<f32>) -> (f32, f32) {
    let n = data.len() as f64;
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var  = data.iter().map(|&v| {
        let d = v as f64 - mean; d * d
    }).sum::<f64>() / n;
    let std  = var.sqrt() as f32;
    let mean = mean as f32;

    if std > 0.0 {
        data.mapv_inplace(|v| (v - mean) / std);
    }
    (mean, std)
}

/// Per-channel, per-epoch baseline correction.
/// `epochs`: [E, C, T]  →  epoch[e, c, :] -= mean(epoch[e, c, :])
pub fn baseline_correct_inplace(epochs: &mut Array3<f32>) {
    let (n_e, n_c, _n_t) = epochs.dim();
    for e in 0..n_e {
        for c in 0..n_c {
            let m = epochs.slice(ndarray::s![e, c, ..])
                          .mean()
                          .unwrap_or(0.0);
            epochs.slice_mut(ndarray::s![e, c, ..])
                  .mapv_inplace(|v| v - m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn zscore_mean_zero_std_one() {
        let mut data = Array2::from_shape_fn((8, 512), |(c, t)| {
            (c as f32 * 3.7 + t as f32 * 0.1).sin() * 50.0
        });
        let (mean, std) = zscore_global_inplace(&mut data);

        let out_mean = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
        let out_std: f64 = {
            let v = data.iter().map(|&v| {
                let d = v as f64 - out_mean; d * d
            }).sum::<f64>() / data.len() as f64;
            v.sqrt()
        };

        approx::assert_abs_diff_eq!(out_mean as f32, 0.0,  epsilon = 1e-5_f32);
        approx::assert_abs_diff_eq!(out_std  as f32, 1.0,  epsilon = 1e-4_f32);
        // Returned params should be original mean/std, not post-normalization.
        assert!(std > 0.0);
        let _ = mean;
    }

    #[test]
    fn zscore_constant_signal_no_panic() {
        let mut data = Array2::from_elem((4, 128), 7.0_f32);
        let (_m, s) = zscore_global_inplace(&mut data);
        // std=0: data unchanged.
        assert_eq!(s, 0.0);
        for &v in data.iter() {
            approx::assert_abs_diff_eq!(v, 7.0, epsilon = 1e-6_f32);
        }
    }

    #[test]
    fn baseline_removes_per_channel_mean() {
        let mut epochs = Array3::from_shape_fn((3, 8, 1280), |(e, c, _)| {
            e as f32 * 10.0 + c as f32 * 5.0 + 1.0
        });
        baseline_correct_inplace(&mut epochs);
        for e in 0..3usize {
            for c in 0..8usize {
                let ch_mean = epochs.slice(ndarray::s![e, c, ..]).mean().unwrap();
                approx::assert_abs_diff_eq!(ch_mean, 0.0, epsilon = 1e-5_f32);
            }
        }
    }
}
