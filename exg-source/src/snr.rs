//! SNR estimation from sensor data and an inverse operator.
//!
//! Ported from MNE-Python's `mne.minimum_norm.estimate_snr`.
//!
//! ## Overview
//!
//! Two SNR measures are provided:
//!
//! - **Whitened GFP** (`snr`): Global field power of the whitened data,
//!   normalised by the effective channel count. This is a data-driven SNR
//!   that does not depend on the regularisation parameter.
//!
//! - **Regularisation-based** (`snr_est`): Finds the smallest `λ²` for
//!   which the residual (unregularised − regularised prediction) stays
//!   within a χ² confidence bound. Returns `1 / √λ²`.
//!
//! ## Example
//!
//! ```no_run
//! use exg_source::snr::estimate_snr;
//! use exg_source::*;
//! use ndarray::Array2;
//!
//! # let n_chan = 16; let n_src = 50;
//! # let gain = Array2::<f64>::zeros((n_chan, n_src));
//! # let fwd = ForwardOperator::new_fixed(gain);
//! # let cov = NoiseCov::diagonal(vec![1e-12; n_chan]);
//! # let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
//! let data = Array2::<f64>::zeros((n_chan, 100));
//! let (snr, snr_est) = estimate_snr(&data, &inv);
//! println!("SNR (whitened GFP): {:?}", &snr.as_slice().unwrap()[..5]);
//! println!("SNR (estimated):    {:?}", &snr_est.as_slice().unwrap()[..5]);
//! ```

use ndarray::Array1;
use ndarray::Array2;

use super::InverseOperator;

/// Estimate SNR as a function of time.
///
/// # Arguments
///
/// * `data` — Sensor data, shape `[n_channels, n_times]`.
/// * `inv`  — Inverse operator.
///
/// # Returns
///
/// `(snr, snr_est)` — both `Array1<f64>` of length `n_times`.
///
/// - `snr`: whitened GFP — `√(‖W x(t)‖² / n_eff)`
/// - `snr_est`: regularisation-based estimate — `1 / √λ²_opt(t)`
pub fn estimate_snr(data: &Array2<f64>, inv: &InverseOperator) -> (Array1<f64>, Array1<f64>) {
    let n_times = data.ncols();
    let n_eff = inv.n_nzero;

    // Whiten the data: w(t) = W @ x(t)
    let data_white = inv.whitener.dot(data);

    // Project onto eigen-field basis: w_ef(t) = U^T @ w(t)
    let data_white_ef = inv.eigen_fields.dot(&data_white);

    // ── SNR from whitened GFP ──────────────────────────────────────────
    let mut snr = Array1::zeros(n_times);
    for t in 0..n_times {
        let mut sum_sq = 0.0;
        for i in 0..data_white.nrows() {
            sum_sq += data_white[[i, t]].powi(2);
        }
        snr[t] = (sum_sq / n_eff as f64).sqrt();
    }

    // ── SNR from regularisation mismatch ───────────────────────────────
    //
    // For each time point, find the largest λ² for which the residual
    // between unregularised and regularised solutions exceeds a χ² threshold.
    //
    // Π_k(λ²) = s_k² / (s_k² + λ²)
    // error(t) = Σ_k w_ef_k(t)² × (1 − Π_k(λ²))²
    //
    // We sweep λ² downward until error < χ²_{n_eff}(0.001).

    let sing2: Vec<f64> = inv.sing.iter().map(|s| s * s).collect();
    let n_k = sing2.len();

    // χ² critical value approximation for p=0.001 (Wilson–Hilferty)
    let chi2_val = chi2_isf(1e-3, n_eff);

    let mut snr_est = Array1::zeros(n_times);
    let lambda_mult = 0.99_f64;

    for t in 0..n_times {
        // Check if signal is too weak
        let sig: f64 = (0..data_white.nrows())
            .map(|i| data_white[[i, t]].powi(2))
            .sum();
        if sig / n_eff as f64 <= 1.0 {
            snr_est[t] = 0.0;
            continue;
        }

        let mut lambda2 = 10.0_f64;
        let mut converged = false;
        for _ in 0..1000 {
            let mut err = 0.0;
            for k in 0..n_k {
                if sing2[k] > 0.0 {
                    let pi_k = sing2[k] / (sing2[k] + lambda2);
                    let residual = data_white_ef[[k, t]] * (1.0 - pi_k);
                    err += residual * residual;
                }
            }
            if err < chi2_val {
                converged = true;
                break;
            }
            lambda2 *= lambda_mult;
        }

        snr_est[t] = if converged {
            1.0 / lambda2.sqrt()
        } else {
            1.0 / lambda2.sqrt() // best estimate even if not converged
        };
    }

    (snr, snr_est)
}

/// Approximate the inverse survival function of χ²(k) at probability p.
///
/// Uses the Wilson–Hilferty normal approximation:
/// `χ²_p ≈ k × (1 − 2/(9k) + z_p × √(2/(9k)))³`
fn chi2_isf(p: f64, k: usize) -> f64 {
    // z_p for the standard normal (approximate for small p)
    // For p = 0.001, z ≈ 3.09
    let z = normal_quantile(1.0 - p);
    let kf = k as f64;
    let term = 1.0 - 2.0 / (9.0 * kf) + z * (2.0 / (9.0 * kf)).sqrt();
    kf * term.powi(3)
}

/// Approximate quantile of the standard normal distribution.
///
/// Uses the rational approximation from Abramowitz & Stegun (26.2.23).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let sign;
    let pp;
    if p < 0.5 {
        sign = -1.0;
        pp = p;
    } else {
        sign = 1.0;
        pp = 1.0 - p;
    };

    let t = (-2.0 * pp.ln()).sqrt();

    // Rational approximation coefficients
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    sign * z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_inverse_operator, ForwardOperator, NoiseCov};
    use ndarray::Array2;

    fn make_test_setup() -> (ForwardOperator, NoiseCov) {
        let n_chan = 16;
        let n_src = 30;
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
    fn test_estimate_snr_shapes() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let data = Array2::from_elem((16, 20), 1e-6);
        let (snr, snr_est) = estimate_snr(&data, &inv);
        assert_eq!(snr.len(), 20);
        assert_eq!(snr_est.len(), 20);
        assert!(snr.iter().all(|v| v.is_finite()));
        assert!(snr_est.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_snr_signal_vs_noise() {
        let (fwd, cov) = make_test_setup();
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        // High SNR: strong signal
        let strong = Array2::from_elem((16, 10), 1e-3);
        let (snr_strong, _) = estimate_snr(&strong, &inv);

        // Low SNR: weak signal
        let weak = Array2::from_elem((16, 10), 1e-15);
        let (snr_weak, _) = estimate_snr(&weak, &inv);

        // Strong signal should have higher whitened GFP
        assert!(
            snr_strong[0] > snr_weak[0],
            "Strong signal SNR ({}) should exceed weak ({})",
            snr_strong[0],
            snr_weak[0]
        );
    }

    #[test]
    fn test_chi2_isf_sanity() {
        // For large k, chi2_isf(0.5, k) ≈ k
        let val = chi2_isf(0.5, 100);
        assert!(
            (val - 100.0).abs() < 5.0,
            "chi2_isf(0.5, 100) = {val}, expected ≈ 100"
        );
    }

    #[test]
    fn test_normal_quantile() {
        let z = normal_quantile(0.975);
        assert!(
            (z - 1.96).abs() < 0.01,
            "normal_quantile(0.975) = {z}, expected ≈ 1.96"
        );
    }
}
