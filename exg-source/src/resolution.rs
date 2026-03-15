//! Resolution matrix computation.
//!
//! The resolution matrix `R = K @ G` describes how source activity is
//! mapped through the inverse and forward operators. It characterises
//! the spatial blurring (leakage) of the inverse solution:
//!
//! - For an ideal inverse, `R = I` (perfect localisation).
//! - In practice, each row of `R` shows how activity at one source
//!   "leaks" to all other sources.
//!
//! ## Metrics
//!
//! - **Peak localisation error**: distance between the true source and
//!   the peak of the corresponding PSF.
//! - **Spatial spread**: width of the PSF (e.g., half-max radius).
//! - **Relative amplitude**: ratio of peak to off-peak activity.
//!
//! ## References
//!
//! Hauk et al. (2011). "Comparison of noise-normalized minimum norm
//! estimates for MEG analysis using a visual paradigm." NeuroImage.
//!
//! Ported from MNE-Python's `mne.minimum_norm.resolution_matrix`.

use anyhow::Result;
use ndarray::{Array1, Array2};

use super::inverse::prepare_inverse;
use super::{ForwardOperator, InverseMethod, InverseOperator};

/// Compute the resolution matrix `R = K @ G`.
///
/// # Arguments
///
/// * `inv`     — Inverse operator.
/// * `fwd`     — Forward operator (must match the one used to build `inv`).
/// * `lambda2` — Regularisation parameter.
/// * `method`  — Inverse method.
///
/// # Returns
///
/// Resolution matrix of shape `[n_sources, n_sources]` for fixed orientation,
/// or `[n_sources, n_sources]` (XYZ combined) for free orientation.
pub fn make_resolution_matrix(
    inv: &InverseOperator,
    fwd: &ForwardOperator,
    lambda2: f64,
    method: InverseMethod,
) -> Result<Array2<f64>> {
    let prepared = prepare_inverse(inv, lambda2, method, None)?;
    let kernel = &prepared.kernel; // [n_src*n_orient, n_chan]
    let gain = &fwd.gain; // [n_chan, n_src*n_orient]

    // R_raw = K @ G  [n_src*n_orient, n_src*n_orient]
    let r_raw = kernel.dot(gain);

    let n_orient = fwd.n_orient();
    let n_src = fwd.n_sources;

    if n_orient == 1 {
        // Apply noise normalisation if present
        let mut r = r_raw;
        if let Some(ref nn) = prepared.noisenorm {
            for i in 0..r.nrows() {
                for j in 0..r.ncols() {
                    r[[i, j]] *= nn[i];
                }
            }
        }
        Ok(r)
    } else {
        // Free orientation: combine XYZ → [n_src, n_src]
        // R_combined[i,j] = ‖R_raw[3i..3i+3, 3j..3j+3]‖_F
        let mut r = Array2::zeros((n_src, n_src));
        for i in 0..n_src {
            for j in 0..n_src {
                let mut sum_sq = 0.0;
                for oi in 0..3 {
                    for oj in 0..3 {
                        let v = r_raw[[i * 3 + oi, j * 3 + oj]];
                        sum_sq += v * v;
                    }
                }
                r[[i, j]] = sum_sq.sqrt();
            }
        }
        // Apply noise normalisation
        if let Some(ref nn) = prepared.noisenorm {
            for i in 0..n_src {
                for j in 0..n_src {
                    r[[i, j]] *= nn[i];
                }
            }
        }
        Ok(r)
    }
}

/// Point-spread function (PSF) for a given source index.
///
/// Returns a column of the resolution matrix: how activity at source `idx`
/// appears across all sources after inverse modelling.
///
/// Shape: `[n_sources]`.
pub fn get_point_spread(resolution: &Array2<f64>, source_idx: usize) -> Array1<f64> {
    resolution.column(source_idx).to_owned()
}

/// Cross-talk function (CTF) for a given source index.
///
/// Returns a row of the resolution matrix: how activity at all other
/// sources leaks into the estimate at source `idx`.
///
/// Shape: `[n_sources]`.
pub fn get_cross_talk(resolution: &Array2<f64>, source_idx: usize) -> Array1<f64> {
    resolution.row(source_idx).to_owned()
}

/// Compute peak localisation error for each source.
///
/// For each source `i`, finds the index of the maximum in the PSF
/// and reports the index offset `|argmax(R[:, i]) − i|`.
///
/// Returns an array of length `n_sources`.
pub fn peak_localisation_error(resolution: &Array2<f64>) -> Array1<usize> {
    let n = resolution.ncols();
    let mut errors = Array1::zeros(n);
    for j in 0..n {
        let col = resolution.column(j);
        let peak = col
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(j);
        errors[j] = if peak > j { peak - j } else { j - peak };
    }
    errors
}

/// Compute spatial spread (half-max width) for each PSF.
///
/// For each source, counts how many sources in the PSF have amplitude
/// ≥ 50% of the peak amplitude. Smaller is better.
///
/// Returns an array of length `n_sources`.
pub fn spatial_spread(resolution: &Array2<f64>) -> Array1<usize> {
    let n = resolution.ncols();
    let mut widths = Array1::zeros(n);
    for j in 0..n {
        let col = resolution.column(j);
        let peak_abs = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if peak_abs > 0.0 {
            let threshold = peak_abs * 0.5;
            widths[j] = col.iter().filter(|v| v.abs() >= threshold).count();
        }
    }
    widths
}

/// Compute relative amplitude for each PSF.
///
/// Ratio of the peak value to the sum of absolute values in the PSF.
/// A value of 1.0 means perfect localisation (delta function).
///
/// Returns an array of length `n_sources`.
pub fn relative_amplitude(resolution: &Array2<f64>) -> Array1<f64> {
    let n = resolution.ncols();
    let mut rel_amp = Array1::zeros(n);
    for j in 0..n {
        let col = resolution.column(j);
        let peak_abs = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let total_abs: f64 = col.iter().map(|v| v.abs()).sum();
        rel_amp[j] = if total_abs > 0.0 {
            peak_abs / total_abs
        } else {
            0.0
        };
    }
    rel_amp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_inverse_operator, ForwardOperator, InverseMethod, NoiseCov};
    use ndarray::Array2;

    fn make_test_setup() -> (ForwardOperator, NoiseCov) {
        let n_chan = 16;
        let n_src = 20;
        let mut gain = Array2::zeros((n_chan, n_src));
        for i in 0..n_chan {
            for j in 0..n_src {
                let dist =
                    ((i as f64 - j as f64 * n_chan as f64 / n_src as f64).powi(2) + 1.0).sqrt();
                gain[[i, j]] = 1e-8 / dist;
            }
        }
        (
            ForwardOperator::new_fixed(gain),
            NoiseCov::diagonal(vec![1e-12; n_chan]),
        )
    }

    #[test]
    fn test_resolution_matrix_shape() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let r = make_resolution_matrix(&inv, &fwd, 1.0 / 9.0, InverseMethod::MNE).unwrap();
        assert_eq!(r.dim(), (20, 20));
    }

    #[test]
    fn test_resolution_matrix_diagonal_dominance() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let r = make_resolution_matrix(&inv, &fwd, 1.0 / 9.0, InverseMethod::MNE).unwrap();

        // For many sources the diagonal should be among the largest values in each column
        let mut diag_is_large = 0;
        for j in 0..20 {
            let col = r.column(j);
            let diag = col[j].abs();
            let max = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            if diag >= max * 0.5 {
                diag_is_large += 1;
            }
        }
        assert!(
            diag_is_large >= 10,
            "At least half the sources should have diag ≥ 50% of max, got {diag_is_large}"
        );
    }

    #[test]
    fn test_psf_ctf() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let r = make_resolution_matrix(&inv, &fwd, 1.0 / 9.0, InverseMethod::MNE).unwrap();

        let psf = get_point_spread(&r, 10);
        let ctf = get_cross_talk(&r, 10);
        assert_eq!(psf.len(), 20);
        assert_eq!(ctf.len(), 20);
    }

    #[test]
    fn test_peak_localisation_error() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let r = make_resolution_matrix(&inv, &fwd, 1.0 / 9.0, InverseMethod::MNE).unwrap();

        let errors = peak_localisation_error(&r);
        assert_eq!(errors.len(), 20);
        // Most sources should have small localisation error
        let mean_error: f64 = errors.iter().map(|&e| e as f64).sum::<f64>() / 20.0;
        assert!(
            mean_error < 5.0,
            "Mean peak error = {mean_error}, expected < 5"
        );
    }

    #[test]
    fn test_spatial_spread_and_relative_amplitude() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let r = make_resolution_matrix(&inv, &fwd, 1.0 / 9.0, InverseMethod::MNE).unwrap();

        let spread = spatial_spread(&r);
        let rel_amp = relative_amplitude(&r);
        assert_eq!(spread.len(), 20);
        assert_eq!(rel_amp.len(), 20);

        // Relative amplitude should be in (0, 1]
        for &v in rel_amp.iter() {
            assert!(v >= 0.0 && v <= 1.0, "rel_amp = {v} out of range");
        }
    }

    #[test]
    fn test_resolution_with_dspm() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let r = make_resolution_matrix(&inv, &fwd, 1.0 / 9.0, InverseMethod::DSPM).unwrap();
        assert_eq!(r.dim(), (20, 20));
        assert!(r.iter().all(|v| v.is_finite()));
    }
}
