//! Noise covariance estimation from sensor data.
//!
//! Ported from MNE-Python's `mne.compute_covariance` /
//! `mne.compute_raw_covariance`.
//!
//! ## Methods
//!
//! - **Empirical**: `C = (1/N) Σ xᵢ xᵢᵀ` (sample covariance)
//! - **Shrunk (Ledoit–Wolf)**: `C_shrunk = (1−α) C + α tr(C)/p · I`
//! - **Diagonal**: keep only diagonal entries
//!
//! ## Example
//!
//! ```
//! use exg_source::covariance::{compute_covariance, Regularization};
//! use ndarray::Array2;
//!
//! // 3 channels, 1000 samples
//! let data = Array2::<f64>::from_shape_fn((3, 1000), |(i, j)| {
//!     ((i * 1000 + j) as f64).sin() * 1e-6
//! });
//! let cov = compute_covariance(&data, Regularization::Empirical);
//! assert_eq!(cov.n_channels(), 3);
//! ```

use ndarray::Array2;

use super::NoiseCov;

/// Regularisation strategy for covariance estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regularization {
    /// Raw sample covariance, no regularisation.
    Empirical,
    /// Ledoit–Wolf shrinkage towards scaled identity.
    ///
    /// If `None`, the optimal shrinkage coefficient is estimated
    /// automatically. Otherwise, the given `alpha` ∈ [0, 1] is used.
    ShrunkIdentity(Option<f64>),
    /// Keep only the diagonal (channel variances).
    Diagonal,
}

/// Compute noise covariance from continuous data `[n_channels, n_times]`.
///
/// The data is assumed to be baseline or empty-room recording.
/// The mean is subtracted per channel before computing the covariance.
///
/// # Arguments
///
/// * `data` — Sensor data, shape `[n_channels, n_times]`.
/// * `reg`  — Regularisation method.
///
/// # Returns
///
/// A [`NoiseCov`] suitable for use with [`make_inverse_operator`](super::make_inverse_operator).
pub fn compute_covariance(data: &Array2<f64>, reg: Regularization) -> NoiseCov {
    let (n_ch, n_t) = data.dim();
    assert!(n_t > 1, "Need at least 2 time points for covariance");

    // Subtract per-channel mean
    let means = data.mean_axis(ndarray::Axis(1)).unwrap();
    let mut centered = data.clone();
    for i in 0..n_ch {
        for j in 0..n_t {
            centered[[i, j]] -= means[i];
        }
    }

    match reg {
        Regularization::Empirical => {
            let cov = centered.dot(&centered.t()) / (n_t - 1) as f64;
            NoiseCov::full(cov)
        }
        Regularization::ShrunkIdentity(alpha_opt) => {
            let cov = centered.dot(&centered.t()) / (n_t - 1) as f64;
            let alpha = alpha_opt.unwrap_or_else(|| ledoit_wolf_alpha(&centered, &cov));
            let alpha = alpha.clamp(0.0, 1.0);
            let trace = cov.diag().sum();
            let mu = trace / n_ch as f64;
            let shrunk = cov.mapv(|v| v * (1.0 - alpha)) + Array2::<f64>::eye(n_ch).mapv(|v: f64| v * alpha * mu);
            NoiseCov::full(shrunk)
        }
        Regularization::Diagonal => {
            let mut vars = Vec::with_capacity(n_ch);
            for i in 0..n_ch {
                let mut sum_sq = 0.0;
                for j in 0..n_t {
                    sum_sq += centered[[i, j]].powi(2);
                }
                vars.push(sum_sq / (n_t - 1) as f64);
            }
            NoiseCov::diagonal(vars)
        }
    }
}

/// Compute noise covariance from epoched data `[n_epochs, n_channels, n_times]`.
///
/// Concatenates all epochs before computing covariance, subtracting the
/// per-epoch, per-channel mean (i.e., each epoch is baseline-corrected).
pub fn compute_covariance_epochs(
    epochs: &ndarray::Array3<f64>,
    reg: Regularization,
) -> NoiseCov {
    let (n_epochs, n_ch, n_t) = epochs.dim();
    let total_t = n_epochs * n_t;

    // Concatenate all epochs into [n_ch, total_t], subtracting per-epoch mean
    let mut concat = Array2::zeros((n_ch, total_t));
    for e in 0..n_epochs {
        let epoch = epochs.slice(ndarray::s![e, .., ..]);
        let mean = epoch.mean_axis(ndarray::Axis(1)).unwrap();
        for i in 0..n_ch {
            for j in 0..n_t {
                concat[[i, e * n_t + j]] = epoch[[i, j]] - mean[i];
            }
        }
    }

    // Now compute covariance on the already-centered data
    let (_, total) = concat.dim();
    match reg {
        Regularization::Empirical => {
            let cov = concat.dot(&concat.t()) / (total - 1) as f64;
            NoiseCov::full(cov)
        }
        Regularization::ShrunkIdentity(alpha_opt) => {
            let cov = concat.dot(&concat.t()) / (total - 1) as f64;
            let alpha = alpha_opt.unwrap_or_else(|| ledoit_wolf_alpha(&concat, &cov));
            let alpha = alpha.clamp(0.0, 1.0);
            let trace = cov.diag().sum();
            let mu = trace / n_ch as f64;
            let shrunk = cov.mapv(|v| v * (1.0 - alpha)) + Array2::<f64>::eye(n_ch).mapv(|v: f64| v * alpha * mu);
            NoiseCov::full(shrunk)
        }
        Regularization::Diagonal => {
            let mut vars = Vec::with_capacity(n_ch);
            for i in 0..n_ch {
                let mut sum_sq = 0.0;
                for j in 0..total {
                    sum_sq += concat[[i, j]].powi(2);
                }
                vars.push(sum_sq / (total - 1) as f64);
            }
            NoiseCov::diagonal(vars)
        }
    }
}

/// Ledoit–Wolf optimal shrinkage coefficient towards scaled identity.
///
/// Implements the analytical formula from Ledoit & Wolf (2004),
/// "A well-conditioned estimator for large-dimensional covariance matrices."
fn ledoit_wolf_alpha(x: &Array2<f64>, sample_cov: &Array2<f64>) -> f64 {
    let (p, n) = x.dim(); // p = channels, n = samples

    if n < 2 {
        return 1.0;
    }

    let trace_s = sample_cov.diag().sum();
    let trace_s2 = sample_cov.iter().map(|v| v * v).sum::<f64>();
    let mu = trace_s / p as f64;

    // Compute sum of squared norms of x_i x_i^T - S
    // β̂² = (1/n²) Σ_i ‖x_i x_i^T − S‖²_F
    let mut beta_sum = 0.0;
    for t in 0..n {
        // x_t is column t of x
        // ‖x_t x_t^T - S‖²_F = (x_t^T x_t)² - 2 x_t^T S x_t + ‖S‖²_F
        let mut xtx = 0.0;
        for i in 0..p {
            xtx += x[[i, t]] * x[[i, t]];
        }
        let mut xt_s_xt = 0.0;
        for i in 0..p {
            let mut row_dot = 0.0;
            for j in 0..p {
                row_dot += sample_cov[[i, j]] * x[[j, t]];
            }
            xt_s_xt += x[[i, t]] * row_dot;
        }
        beta_sum += xtx * xtx - 2.0 * xt_s_xt + trace_s2;
    }
    let beta = beta_sum / (n * n) as f64;

    // δ² = ‖S − μI‖²_F = ‖S‖²_F − p·μ²
    let delta = trace_s2 - p as f64 * mu * mu;

    if delta <= 0.0 {
        return 1.0;
    }

    (beta / delta).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_empirical_covariance_shape() {
        let data = Array2::<f64>::from_shape_fn((4, 100), |(i, j)| {
            ((i * 100 + j) as f64 * 0.1).sin()
        });
        let cov = compute_covariance(&data, Regularization::Empirical);
        assert_eq!(cov.n_channels(), 4);
        let full = cov.to_full();
        assert_eq!(full.dim(), (4, 4));
    }

    #[test]
    fn test_empirical_covariance_symmetric() {
        let data = Array2::<f64>::from_shape_fn((5, 200), |(i, j)| {
            ((i * 200 + j) as f64 * 0.3).cos() * 1e-6
        });
        let cov = compute_covariance(&data, Regularization::Empirical);
        let full = cov.to_full();
        for i in 0..5 {
            for j in 0..5 {
                approx::assert_abs_diff_eq!(full[[i, j]], full[[j, i]], epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_empirical_covariance_positive_diagonal() {
        let data = Array2::<f64>::from_shape_fn((3, 500), |(i, j)| {
            ((i * 500 + j) as f64 * 0.7).sin() * 1e-6
        });
        let cov = compute_covariance(&data, Regularization::Empirical);
        let diag = cov.diag_elements();
        for &v in diag.iter() {
            assert!(v > 0.0, "Diagonal should be positive");
        }
    }

    #[test]
    fn test_diagonal_covariance() {
        let data = Array2::<f64>::from_shape_fn((3, 500), |(i, j)| {
            ((i * 500 + j) as f64 * 0.2).sin() * (i as f64 + 1.0) * 1e-6
        });
        let cov = compute_covariance(&data, Regularization::Diagonal);
        assert!(cov.diag);
        assert_eq!(cov.n_channels(), 3);
    }

    #[test]
    fn test_shrunk_covariance_between_empirical_and_identity() {
        let data = Array2::<f64>::from_shape_fn((4, 200), |(i, j)| {
            ((i * 200 + j) as f64 * 0.5).sin() * 1e-6
        });
        let emp = compute_covariance(&data, Regularization::Empirical).to_full();
        let shrunk = compute_covariance(&data, Regularization::ShrunkIdentity(None)).to_full();

        // Off-diagonal elements should be smaller in shrunk than empirical
        let mut emp_offdiag = 0.0;
        let mut shrunk_offdiag = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    emp_offdiag += emp[[i, j]].abs();
                    shrunk_offdiag += shrunk[[i, j]].abs();
                }
            }
        }
        assert!(
            shrunk_offdiag <= emp_offdiag + 1e-20,
            "Shrinkage should reduce off-diagonal: shrunk={shrunk_offdiag}, emp={emp_offdiag}"
        );
    }

    #[test]
    fn test_covariance_from_epochs() {
        let epochs = ndarray::Array3::<f64>::from_shape_fn((10, 3, 50), |(e, i, j)| {
            ((e * 150 + i * 50 + j) as f64 * 0.4).sin() * 1e-6
        });
        let cov = compute_covariance_epochs(&epochs, Regularization::Empirical);
        assert_eq!(cov.n_channels(), 3);
        let full = cov.to_full();
        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                approx::assert_abs_diff_eq!(full[[i, j]], full[[j, i]], epsilon = 1e-15);
            }
        }
        // Positive diagonal
        for i in 0..3 {
            assert!(full[[i, i]] > 0.0);
        }
    }
}
