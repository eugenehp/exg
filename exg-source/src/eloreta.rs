//! eLORETA (exact Low Resolution Electromagnetic Tomography) solver.
//!
//! Ported from MNE-Python's `mne.minimum_norm._eloreta._compute_eloreta`.
//!
//! eLORETA iteratively computes optimal source weights that yield exact
//! localisation (zero localisation bias) for single dipole sources.
//!
//! ## References
//!
//! Pascual-Marqui, R. D. (2011). Discrete, 3D distributed, linear imaging
//! methods of electric neuronal activity. Part 1: exact, zero error
//! localization. *arXiv:0710.3341*.

use anyhow::{bail, Result};
use ndarray::{Array1, Array2};

use super::linalg;
use super::{EloretaOptions, InverseOperator, SourceOrientation};

/// Compute the eLORETA inverse kernel.
///
/// Returns `(kernel, reginv)` where:
/// - `kernel` has shape `[n_sources × n_orient, n_channels]`
/// - `reginv` has length `n_nzero`
pub fn compute_eloreta(
    inv: &InverseOperator,
    lambda2: f64,
    opts: &EloretaOptions,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if inv.eigen_leads_weighted {
        bail!("eLORETA cannot be computed with weighted eigen leads");
    }

    let n_orient = match inv.orientation {
        SourceOrientation::Fixed => 1,
        SourceOrientation::Free => 3,
    };
    let n_src = inv.n_sources;
    let n_nzero = inv.n_nzero;

    // Reassemble the gain matrix: G = U @ diag(s) @ V^T (in whitened space)
    // eigen_fields = U^T [k, n_chan_w], sing = s [k], eigen_leads = V [n_cols, k]
    // G_whitened = (eigen_fields^T @ diag(s) @ eigen_leads^T) ... but this is
    // whitened gain. Let's reconstruct: G_w = U @ S @ V^T
    let n_k = inv.sing.len();
    let n_cols = n_src * n_orient;

    // G = eigen_fields^T * diag(sing) * eigen_leads^T  [n_chan_w, n_cols]
    // But eigen_fields = U^T [k, n_chan_w], so U = eigen_fields^T [n_chan_w, k]
    let mut g = Array2::zeros((inv.eigen_fields.ncols(), n_cols));
    for c in 0..n_cols {
        for ch in 0..g.nrows() {
            let mut val = 0.0;
            for k in 0..n_k {
                val += inv.eigen_fields[[k, ch]] * inv.sing[k] * inv.eigen_leads[[c, k]];
            }
            g[[ch, c]] = val;
        }
    }

    // Remove source_cov weighting to get the "raw" whitened gain
    for c in 0..n_cols {
        let w = inv.source_cov[c].sqrt();
        if w > 0.0 {
            for r in 0..g.nrows() {
                g[[r, c]] /= w;
            }
        }
    }

    let force_equal = opts.force_equal.unwrap_or(n_orient == 1);

    // Initialise weights R
    let mut r_diag = Array1::ones(n_cols); // used when force_equal or n_orient==1

    // Main iteration
    for _iter in 0..opts.max_iter {
        // Apply R to G: G_R = G @ diag(R) for diagonal R
        let mut g_r = g.clone();
        for c in 0..n_cols {
            let w = r_diag[c];
            for r in 0..g_r.nrows() {
                g_r[[r, c]] *= w;
            }
        }

        // G_R_Gt = G_R @ G^T = G @ diag(R) @ G^T
        let g_r_gt = g_r.dot(&g.t());

        // Normalise so trace = n_nzero
        let trace = g_r_gt.diag().sum();
        let norm = trace / n_nzero as f64;
        let g_r_gt_normed = g_r_gt.mapv(|v| v / norm);
        let _r_norm = norm;

        // Eigendecompose G_R_Gt
        let (evals, evecs) = linalg::eigh_sorted(&g_r_gt_normed)?;

        // Compute N = (G_R_Gt + lambda2 I)^{-1} using eigendecomposition
        let mut n_mat = Array2::zeros((g.nrows(), g.nrows()));
        for k in 0..n_nzero {
            if evals[k].abs() > 0.0 {
                let inv_val = 1.0 / (evals[k] + lambda2);
                for i in 0..n_mat.nrows() {
                    for j in 0..n_mat.ncols() {
                        n_mat[[i, j]] += inv_val * evecs[[i, k]] * evecs[[j, k]];
                    }
                }
            }
        }

        // Update weights
        let r_diag_old = r_diag.clone();

        if n_orient == 1 || force_equal {
            // R_k = 1 / sqrt(G_k^T @ N @ G_k)
            for s in 0..n_src {
                let mut val = 0.0;
                for o in 0..n_orient {
                    let c = s * n_orient + o;
                    // G_k^T @ N @ G_k for this column
                    let ng = n_mat.dot(&g.column(c).to_owned());
                    val += g.column(c).dot(&ng);
                }
                let w = if val > 0.0 {
                    1.0 / (val / n_orient as f64).sqrt()
                } else {
                    1.0
                };
                for o in 0..n_orient {
                    r_diag[s * n_orient + o] = w;
                }
            }
        } else {
            // Free orientation, not force_equal: use per-component weights
            for c in 0..n_cols {
                let ng = n_mat.dot(&g.column(c).to_owned());
                let val = g.column(c).dot(&ng);
                r_diag[c] = if val > 0.0 { 1.0 / val.sqrt() } else { 1.0 };
            }
        }

        // Normalise R to keep things stable
        let r_trace: f64 = {
            let mut gr = g.clone();
            for c in 0..n_cols {
                for r in 0..gr.nrows() {
                    gr[[r, c]] *= r_diag[c];
                }
            }
            let grgt = gr.dot(&g.t());
            grgt.diag().sum() / n_nzero as f64
        };
        if r_trace > 0.0 {
            r_diag.mapv_inplace(|v| v / r_trace.sqrt());
        }

        // Check convergence
        let delta_num: f64 = r_diag
            .iter()
            .zip(r_diag_old.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let delta_den: f64 = r_diag_old.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let delta = if delta_den > 0.0 {
            delta_num / delta_den
        } else {
            0.0
        };

        if delta < opts.eps {
            break;
        }
    }

    // Build final kernel with eLORETA weights
    // G_weighted = G @ diag(R)
    let mut g_weighted = g.clone();
    for c in 0..n_cols {
        for r in 0..g_weighted.nrows() {
            g_weighted[[r, c]] *= r_diag[c];
        }
    }

    // SVD of weighted gain
    let (u, sing, vt) = linalg::svd_thin(&g_weighted)?;

    // Compute reginv
    let mut reginv = Array1::zeros(sing.len());
    for k in 0..sing.len().min(n_nzero) {
        let s = sing[k];
        if s > 0.0 {
            reginv[k] = s / (s * s + lambda2);
        }
    }

    // trans = diag(reginv) @ U^T @ whitener  [k, n_chan]
    let ut = u.t().to_owned();
    let trans = {
        let ut_w = ut.dot(&inv.whitener);
        let mut t = Array2::zeros(ut_w.dim());
        for i in 0..t.nrows() {
            for j in 0..t.ncols() {
                t[[i, j]] = ut_w[[i, j]] * reginv[i];
            }
        }
        t
    };

    // kernel = diag(R) @ V @ trans  =  diag(R) @ Vt^T @ trans
    let v = vt.t().to_owned();
    let mut kernel = Array2::zeros((n_cols, trans.ncols()));
    let v_trans = v.dot(&trans);
    for c in 0..n_cols {
        let w = r_diag[c];
        for j in 0..kernel.ncols() {
            kernel[[c, j]] = w * v_trans[[c, j]];
        }
    }

    Ok((kernel, reginv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inverse::{apply_inverse_with_options, make_inverse_operator},
        ForwardOperator, InverseMethod, NoiseCov,
    };
    use ndarray::Array2;

    #[test]
    fn test_eloreta_basic() {
        let n_chan = 16;
        let n_src = 30;
        let mut gain = Array2::zeros((n_chan, n_src));
        for i in 0..n_chan {
            for j in 0..n_src {
                let dist = ((i as f64 - j as f64 * n_chan as f64 / n_src as f64).powi(2)
                    + 1.0)
                    .sqrt();
                gain[[i, j]] = 1e-8 / dist;
            }
        }
        let fwd = ForwardOperator::new_fixed(gain);
        let cov = NoiseCov::diagonal(vec![1e-12; n_chan]);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        let data = Array2::from_elem((n_chan, 5), 1e-6);
        let opts = EloretaOptions {
            max_iter: 10,
            eps: 1e-4,
            force_equal: Some(true),
        };
        let stc = apply_inverse_with_options(
            &data, &inv, 1.0 / 9.0, InverseMethod::ELORETA, Some(&opts),
        )
        .unwrap();
        assert_eq!(stc.data.nrows(), n_src);
        assert!(stc.data.iter().all(|v: &f64| v.is_finite()));
    }
}
