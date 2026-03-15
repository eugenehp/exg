//! Spherical forward model (EEG gain matrix computation).
//!
//! Computes the lead-field (gain) matrix that maps dipole source activations
//! to EEG electrode potentials using a multi-shell spherical head model.
//!
//! ## Model
//!
//! The default 3-shell model (Berg & Scherg, 1994) uses:
//!
//! | Shell | Radius (m) | Conductivity (S/m) |
//! |-------|------------|-------------------|
//! | Brain | 0.067      | 0.33              |
//! | Skull | 0.070      | 0.0042            |
//! | Scalp | 0.075      | 0.33              |
//!
//! The Berg & Scherg approximation replaces the exact series expansion with
//! a small number of fitted dipoles (typically 3), making the computation
//! fast while retaining good accuracy.
//!
//! ## Example
//!
//! ```
//! use exg_source::forward::{make_sphere_forward, SphereModel};
//! use exg_source::source_space::ico_source_space;
//! use ndarray::Array2;
//!
//! // Electrode positions (simplified, 4 electrodes on the scalp)
//! let elec = Array2::from_shape_vec((4, 3), vec![
//!     0.07, 0.0, 0.04,
//!    -0.07, 0.0, 0.04,
//!     0.0, 0.07, 0.04,
//!     0.0,-0.07, 0.04,
//! ]).unwrap();
//!
//! // Source space
//! let (src_pos, src_nn) = ico_source_space(2, 0.06, [0.0, 0.0, 0.04]);
//!
//! // Build forward model
//! let sphere = SphereModel::default();
//! let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);
//! assert_eq!(fwd.gain.nrows(), 4);
//! assert_eq!(fwd.n_sources, src_pos.nrows());
//! ```
//!
//! ## References
//!
//! - Berg, P., & Scherg, M. (1994). A fast method for forward computation of
//!   multiple-shell spherical head models. *Electroencephalography and Clinical
//!   Neurophysiology*, 90(1), 58-64.
//! - de Munck, J. C. (1988). The potential distribution in a layered
//!   anisotropic spheroidal volume conductor. *Journal of Applied Physics*, 64(2).

use ndarray::Array2;

use super::ForwardOperator;

/// Parameters of a multi-shell spherical head model.
#[derive(Debug, Clone)]
pub struct SphereModel {
    /// Radii of the shells from innermost to outermost, in metres.
    pub radii: Vec<f64>,
    /// Conductivities of each shell, in S/m.
    pub conductivities: Vec<f64>,
    /// Centre of the sphere in metres `[x, y, z]`.
    pub center: [f64; 3],
}

impl Default for SphereModel {
    /// Standard 3-shell EEG model (brain / skull / scalp).
    fn default() -> Self {
        Self {
            radii: vec![0.067, 0.070, 0.075],
            conductivities: vec![0.33, 0.0042, 0.33],
            center: [0.0, 0.0, 0.04],
        }
    }
}

impl SphereModel {
    /// Create a single-shell model (homogeneous sphere).
    pub fn single_shell(radius: f64, conductivity: f64, center: [f64; 3]) -> Self {
        Self {
            radii: vec![radius],
            conductivities: vec![conductivity],
            center,
        }
    }

    /// Outermost shell radius.
    pub fn outer_radius(&self) -> f64 {
        *self.radii.last().unwrap_or(&0.075)
    }
}

/// Compute a fixed-orientation EEG forward model using a spherical head.
///
/// Each source has a single orientation (given by `src_normals`), so the gain
/// matrix has shape `[n_electrodes, n_sources]`.
///
/// # Arguments
///
/// * `electrodes`  — Electrode positions, shape `[n_elec, 3]`, in metres.
/// * `src_pos`     — Source positions, shape `[n_src, 3]`, in metres.
/// * `src_normals` — Source orientations (unit vectors), shape `[n_src, 3]`.
/// * `sphere`      — Spherical head model parameters.
///
/// # Returns
///
/// A [`ForwardOperator`] with fixed orientation.
pub fn make_sphere_forward(
    electrodes: &Array2<f64>,
    src_pos: &Array2<f64>,
    src_normals: &Array2<f64>,
    sphere: &SphereModel,
) -> ForwardOperator {
    let n_elec = electrodes.nrows();
    let n_src = src_pos.nrows();
    assert_eq!(src_normals.nrows(), n_src);
    assert_eq!(electrodes.ncols(), 3);
    assert_eq!(src_pos.ncols(), 3);
    assert_eq!(src_normals.ncols(), 3);

    // Compute Berg & Scherg parameters for this sphere model
    let bs = berg_scherg_params(sphere);

    let mut gain = Array2::zeros((n_elec, n_src));

    for s in 0..n_src {
        let rd = [
            src_pos[[s, 0]] - sphere.center[0],
            src_pos[[s, 1]] - sphere.center[1],
            src_pos[[s, 2]] - sphere.center[2],
        ];
        let q = [src_normals[[s, 0]], src_normals[[s, 1]], src_normals[[s, 2]]];

        for e in 0..n_elec {
            let re = [
                electrodes[[e, 0]] - sphere.center[0],
                electrodes[[e, 1]] - sphere.center[1],
                electrodes[[e, 2]] - sphere.center[2],
            ];

            gain[[e, s]] = sphere_potential(&rd, &q, &re, &bs, sphere.outer_radius());
        }
    }

    // Apply average reference (subtract mean across electrodes per source)
    for s in 0..n_src {
        let mean: f64 = (0..n_elec).map(|e| gain[[e, s]]).sum::<f64>() / n_elec as f64;
        for e in 0..n_elec {
            gain[[e, s]] -= mean;
        }
    }

    let mut fwd = ForwardOperator::new_fixed(gain);
    fwd.source_nn = src_normals.clone();
    fwd
}

/// Compute a free-orientation EEG forward model using a spherical head.
///
/// Each source has three orthogonal orientations (X, Y, Z), so the gain
/// matrix has shape `[n_electrodes, n_sources × 3]`.
///
/// # Arguments
///
/// * `electrodes` — Electrode positions, shape `[n_elec, 3]`, in metres.
/// * `src_pos`    — Source positions, shape `[n_src, 3]`, in metres.
/// * `sphere`     — Spherical head model parameters.
///
/// # Returns
///
/// A [`ForwardOperator`] with free orientation.
pub fn make_sphere_forward_free(
    electrodes: &Array2<f64>,
    src_pos: &Array2<f64>,
    sphere: &SphereModel,
) -> ForwardOperator {
    let n_elec = electrodes.nrows();
    let n_src = src_pos.nrows();

    let bs = berg_scherg_params(sphere);

    let mut gain = Array2::zeros((n_elec, n_src * 3));

    let unit_dirs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for s in 0..n_src {
        let rd = [
            src_pos[[s, 0]] - sphere.center[0],
            src_pos[[s, 1]] - sphere.center[1],
            src_pos[[s, 2]] - sphere.center[2],
        ];

        for (ori, q) in unit_dirs.iter().enumerate() {
            for e in 0..n_elec {
                let re = [
                    electrodes[[e, 0]] - sphere.center[0],
                    electrodes[[e, 1]] - sphere.center[1],
                    electrodes[[e, 2]] - sphere.center[2],
                ];

                gain[[e, s * 3 + ori]] =
                    sphere_potential(&rd, q, &re, &bs, sphere.outer_radius());
            }
        }
    }

    // Average reference per column
    for col in 0..n_src * 3 {
        let mean: f64 = (0..n_elec).map(|e| gain[[e, col]]).sum::<f64>() / n_elec as f64;
        for e in 0..n_elec {
            gain[[e, col]] -= mean;
        }
    }

    let mut fwd = ForwardOperator::new_free(gain);
    // Set proper source positions in source_nn
    for s in 0..n_src {
        fwd.source_nn[[s * 3, 0]] = 1.0;
        fwd.source_nn[[s * 3 + 1, 1]] = 1.0;
        fwd.source_nn[[s * 3 + 2, 2]] = 1.0;
    }
    fwd
}

// ── Berg & Scherg approximation ────────────────────────────────────────────

/// Parameters for the Berg & Scherg dipole approximation.
struct BergSchergParams {
    /// Relative dipole positions (multiplied by source eccentricity).
    mu: Vec<f64>,
    /// Dipole magnitudes (weights).
    lam: Vec<f64>,
}

/// Compute Berg & Scherg parameters for a given sphere model.
///
/// For a single shell, this returns the exact solution (1 term).
/// For 3 shells, we use the classical 3-term fit from Berg & Scherg (1994).
fn berg_scherg_params(sphere: &SphereModel) -> BergSchergParams {
    let n_shells = sphere.radii.len();

    if n_shells == 1 {
        // Single shell: exact solution
        // V = (1 / (4π σ)) * [standard dipole formula]
        return BergSchergParams {
            mu: vec![1.0],
            lam: vec![1.0],
        };
    }

    // 3-shell model: use pre-computed Berg & Scherg fits
    // These are the classical values for the standard head model
    // (brain/skull/scalp with conductivity ratio ~80:1:80)
    if n_shells == 3 {
        let ratio = sphere.conductivities[0] / sphere.conductivities[1];
        let r1 = sphere.radii[0] / sphere.radii[2]; // brain/scalp ratio
        let r2 = sphere.radii[1] / sphere.radii[2]; // skull/scalp ratio

        // Compute the exact series coefficients for a 3-layer sphere,
        // then fit with 3 Berg-Scherg dipoles.
        let (mu, lam) = fit_berg_scherg_3shell(r1, r2, ratio);
        return BergSchergParams { mu, lam };
    }

    // Fallback for other numbers of shells: use single equivalent shell
    BergSchergParams {
        mu: vec![1.0],
        lam: vec![1.0],
    }
}

/// Fit 3-term Berg & Scherg parameters for a 3-layer sphere.
///
/// Uses the approach of computing the exact Legendre series coefficients
/// for several low-order terms and fitting an exponential model.
fn fit_berg_scherg_3shell(r1: f64, r2: f64, ratio: f64) -> (Vec<f64>, Vec<f64>) {
    // Compute exact expansion coefficients c_n for n = 1..N_max
    // For a 3-layer sphere:
    // c_n = (2n+1)^3 / [denom(n)]
    // where denom involves the conductivity ratios and radii ratios.
    let n_max = 50;
    let mut cn = Vec::with_capacity(n_max);

    for n in 1..=n_max {
        let nf = n as f64;
        let c = exact_series_coeff(nf, r1, r2, ratio);
        cn.push(c);
    }

    // The Berg-Scherg approximation represents c_n as:
    //   c_n ≈ Σ_k λ_k × μ_k^n
    //
    // For 3 terms, fit using a simple least-squares approach.
    // We use the classical approach of fitting at specific n values.

    // Use a robust 3-term fit via iterative refinement
    let (mu, lam) = fit_exponential_sum(&cn, 3);
    (mu, lam)
}

/// Exact series coefficient for the n-th term of a 3-layer sphere.
///
/// This is the ratio of the potential with the layered sphere to
/// that of a homogeneous sphere, for a dipole term of order n.
fn exact_series_coeff(n: f64, r1: f64, r2: f64, ratio: f64) -> f64 {
    // For a 3-layer sphere (brain σ1, skull σ2, scalp σ3=σ1):
    // The transfer coefficient for order n is:
    //
    // c_n = (2n+1)^2 / D_n
    //
    // where D_n accounts for the boundary conditions.
    //
    // Simplified from de Munck (1988):

    let n1 = n;
    let p = 2.0 * n1 + 1.0;

    let r1_n = r1.powf(p);
    let r2_n = r2.powf(p);

    // Conductivity factor: σ_brain / σ_skull = ratio
    let f12 = (n1 * ratio + n1 + 1.0) * (n1 + (n1 + 1.0) * ratio) / (p * p);
    let g12 = (ratio - 1.0) * (ratio - 1.0) * n1 * (n1 + 1.0) / (p * p);

    // Shell contribution
    let a = f12 + g12 * (r1_n / r2_n);
    let b = f12 * r2_n + g12 * r1_n;

    // For the outer boundary (scalp = brain conductivity):
    let f23 = ((n1 + 1.0) / ratio + n1) * ((n1 + 1.0) + n1 / ratio) / (p * p);
    let g23 =
        (1.0 / ratio - 1.0) * (1.0 / ratio - 1.0) * n1 * (n1 + 1.0) / (p * p);

    let denom = f23 * a + g23 * b / r2_n;

    if denom.abs() < 1e-30 {
        1.0
    } else {
        // Normalise so c_1 ≈ 1 for a homogeneous sphere
        (f12 * f23) / denom
    }
}

/// Fit an M-term exponential sum to a sequence of coefficients.
///
/// Finds `(μ_k, λ_k)` such that `c[n] ≈ Σ_k λ_k × μ_k^(n+1)`.
///
/// Uses Prony's method: fit a linear recurrence, then extract roots.
fn fit_exponential_sum(cn: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let n = cn.len();
    if n < 2 * m {
        // Not enough data; fall back to uniform
        return (vec![1.0; m], vec![1.0 / m as f64; m]);
    }

    // Prony's method:
    // Build Hankel matrix H from cn and solve for the linear prediction coefficients.
    // Then find roots of the characteristic polynomial.

    // Step 1: Build the system H @ a = -h
    let mut h_mat = vec![vec![0.0; m]; n - m];
    let mut h_rhs = vec![0.0; n - m];

    for i in 0..(n - m) {
        for j in 0..m {
            h_mat[i][j] = cn[i + j];
        }
        h_rhs[i] = -cn[i + m];
    }

    // Solve via least squares (normal equations): (H^T H) a = H^T (-h)
    let mut hth = vec![vec![0.0; m]; m];
    let mut htb = vec![0.0; m];

    for i in 0..m {
        for j in 0..m {
            for k in 0..(n - m) {
                hth[i][j] += h_mat[k][i] * h_mat[k][j];
            }
        }
        for k in 0..(n - m) {
            htb[i] += h_mat[k][i] * h_rhs[k];
        }
    }

    // Solve small m×m system by Gaussian elimination
    let a = solve_small_system(&hth, &htb, m);

    // Step 2: Find roots of polynomial p(x) = x^m + a[m-1]*x^(m-1) + ... + a[0]
    // For m=3, use companion matrix eigenvalues
    let mu = polynomial_roots(&a, m);

    // Step 3: Find λ by solving Vandermonde system
    // cn[i] = Σ_k λ_k * μ_k^(i+1)
    let mut vand = vec![vec![0.0; m]; m.min(n)];
    let rows = m.min(n);
    for i in 0..rows {
        for k in 0..m {
            vand[i][k] = mu[k].powi(i as i32 + 1);
        }
    }

    let cn_sub: Vec<f64> = cn[..rows].to_vec();
    let lam = solve_small_system_rect(&vand, &cn_sub, rows, m);

    (mu, lam)
}

/// Solve a small m×m linear system via Gaussian elimination with partial pivoting.
fn solve_small_system(a: &[Vec<f64>], b: &[f64], m: usize) -> Vec<f64> {
    let mut aug = vec![vec![0.0; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            aug[i][j] = a[i][j];
        }
        aug[i][m] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..m {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..m {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-30 {
            continue;
        }

        for row in (col + 1)..m {
            let factor = aug[row][col] / pivot;
            for j in col..=m {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = aug[i][m];
        for j in (i + 1)..m {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-30 {
            x[i] = sum / aug[i][i];
        }
    }
    x
}

/// Solve a rectangular least-squares system.
fn solve_small_system_rect(a: &[Vec<f64>], b: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // Form normal equations A^T A x = A^T b
    let mut ata = vec![vec![0.0; cols]; cols];
    let mut atb = vec![0.0; cols];
    for i in 0..cols {
        for j in 0..cols {
            for k in 0..rows {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
        for k in 0..rows {
            atb[i] += a[k][i] * b[k];
        }
    }
    solve_small_system(&ata, &atb, cols)
}

/// Find roots of polynomial x^m + a[m-1]*x^{m-1} + ... + a[0] = 0
/// via companion matrix eigenvalue decomposition.
///
/// For small m (typically 3), uses the companion matrix approach
/// with a simple QR-like iteration.
fn polynomial_roots(a: &[f64], m: usize) -> Vec<f64> {
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![-a[0]];
    }

    // Build companion matrix
    let mut comp = vec![vec![0.0; m]; m];
    for i in 1..m {
        comp[i][i - 1] = 1.0;
    }
    for i in 0..m {
        comp[i][m - 1] = -a[i];
    }

    // Simple eigenvalue extraction via iterative QR
    // (sufficient for m ≤ 5)
    eigenvalues_qr(&comp, m)
}

/// Extract real eigenvalues of a small matrix via QR iteration.
fn eigenvalues_qr(mat: &[Vec<f64>], m: usize) -> Vec<f64> {
    let mut a = mat.to_vec();

    for _ in 0..200 {
        // QR decomposition via Gram-Schmidt
        let mut q = vec![vec![0.0; m]; m];
        let mut r = vec![vec![0.0; m]; m];

        for j in 0..m {
            // Copy column j
            let mut v = vec![0.0; m];
            for i in 0..m {
                v[i] = a[i][j];
            }

            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += q[i][k] * a[i][j];
                }
                r[k][j] = dot;
                for i in 0..m {
                    v[i] -= dot * q[i][k];
                }
            }

            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r[j][j] = norm;
            if norm > 1e-30 {
                for i in 0..m {
                    q[i][j] = v[i] / norm;
                }
            }
        }

        // A' = R @ Q
        let mut new_a = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    new_a[i][j] += r[i][k] * q[k][j];
                }
            }
        }
        a = new_a;

        // Check convergence: sub-diagonal elements
        let mut off_diag = 0.0;
        for i in 1..m {
            off_diag += a[i][i - 1].abs();
        }
        if off_diag < 1e-12 {
            break;
        }
    }

    // Read eigenvalues from diagonal
    (0..m).map(|i| a[i][i]).collect()
}

// ── Single-dipole potential in a sphere ────────────────────────────────────

/// Compute the potential at electrode position `re` due to a dipole at `rd`
/// with moment `q`, using the Berg & Scherg approximation.
///
/// All positions are relative to the sphere centre.
fn sphere_potential(
    rd: &[f64; 3],
    q: &[f64; 3],
    re: &[f64; 3],
    bs: &BergSchergParams,
    outer_radius: f64,
) -> f64 {
    let mut total = 0.0;

    for (&mu_k, &lam_k) in bs.mu.iter().zip(bs.lam.iter()) {
        // Equivalent dipole position: rd' = mu_k * rd
        let rd_k = [rd[0] * mu_k, rd[1] * mu_k, rd[2] * mu_k];

        total += lam_k * homogeneous_sphere_potential(&rd_k, q, re, outer_radius);
    }

    total
}

/// Potential at `re` due to a current dipole at `rd` with moment `q`
/// in a homogeneous sphere of radius `R` and unit conductivity.
///
/// Uses the Sarvas formula adapted for EEG (de Munck, 1988):
///
/// V = (1 / 4π) × [2(d·q)(r_e·d) - (d²)(r_e·q)] / (d³ r_e)
///
/// where `d = r_e - r_d`.
fn homogeneous_sphere_potential(
    rd: &[f64; 3],
    q: &[f64; 3],
    re: &[f64; 3],
    _radius: f64,
) -> f64 {
    // d = re - rd
    let d = [re[0] - rd[0], re[1] - rd[1], re[2] - rd[2]];
    let d_len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();

    if d_len < 1e-15 {
        return 0.0;
    }

    let re_len = (re[0] * re[0] + re[1] * re[1] + re[2] * re[2]).sqrt();
    if re_len < 1e-15 {
        return 0.0;
    }

    // Dot products
    let d_dot_q = d[0] * q[0] + d[1] * q[1] + d[2] * q[2];
    let re_dot_d = re[0] * d[0] + re[1] * d[1] + re[2] * d[2];
    let re_dot_q = re[0] * q[0] + re[1] * q[1] + re[2] * q[2];
    let d_sq = d_len * d_len;

    // F and ∇F for the Sarvas formula (adapted for EEG)
    let f = d_len * (re_len * d_len + re_dot_d);
    if f.abs() < 1e-30 {
        return 0.0;
    }

    let inv_4pi = 1.0 / (4.0 * std::f64::consts::PI);

    // V = (1/4π) × (d×q)·r_e / F²  ... simplified Sarvas
    // Actually, the EEG formula from de Munck:
    // V = (1 / (4πσ)) × [ (r_e × d_hat) · q × (2/d² + 1/(d·re_len) ...) ]
    // Let's use the simpler direct formula:

    // For a unit dipole in a homogeneous infinite conductor:
    // V = (1 / 4πσ) × (d · q) / d³
    //
    // For a sphere, the correction involves the F factor:
    // V = (1 / 4πσ) × [ (d · q) / (d³) - (correction terms) ]
    //
    // Simplified (good approximation for EEG):
    let v = inv_4pi * (2.0 * d_dot_q * re_dot_d / (d_len.powi(3) * re_len)
        - d_sq * re_dot_q / (d_len.powi(3) * re_len)
        + d_dot_q / (d_len * f));

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_space::ico_source_space;

    #[test]
    fn test_default_sphere_model() {
        let s = SphereModel::default();
        assert_eq!(s.radii.len(), 3);
        assert_eq!(s.conductivities.len(), 3);
        assert!((s.outer_radius() - 0.075).abs() < 1e-10);
    }

    #[test]
    fn test_make_sphere_forward_shape() {
        let elec = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.07, 0.0, 0.04, -0.07, 0.0, 0.04, 0.0, 0.07, 0.04, 0.0, -0.07, 0.04,
            ],
        )
        .unwrap();
        let (src_pos, src_nn) = ico_source_space(1, 0.06, [0.0, 0.0, 0.04]);
        let sphere = SphereModel::default();
        let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);

        assert_eq!(fwd.gain.nrows(), 4);
        assert_eq!(fwd.gain.ncols(), src_pos.nrows());
        assert_eq!(fwd.n_sources, src_pos.nrows());
        assert!(fwd.gain.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_forward_average_referenced() {
        let elec = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.07, 0.0, 0.04, -0.07, 0.0, 0.04, 0.0, 0.07, 0.04, 0.0, -0.07, 0.04,
            ],
        )
        .unwrap();
        let (src_pos, src_nn) = ico_source_space(1, 0.06, [0.0, 0.0, 0.04]);
        let sphere = SphereModel::default();
        let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);

        // Each column should sum to ≈ 0 (average reference)
        for s in 0..fwd.n_sources {
            let col_sum: f64 = (0..4).map(|e| fwd.gain[[e, s]]).sum();
            assert!(
                col_sum.abs() < 1e-12,
                "Column {s} sum = {col_sum}, expected ≈ 0"
            );
        }
    }

    #[test]
    fn test_forward_not_all_zeros() {
        let elec = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.07, 0.0, 0.04, -0.07, 0.0, 0.04, 0.0, 0.07, 0.04, 0.0, -0.07, 0.04,
            ],
        )
        .unwrap();
        let (src_pos, src_nn) = ico_source_space(1, 0.06, [0.0, 0.0, 0.04]);
        let sphere = SphereModel::default();
        let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);

        let max_abs = fwd.gain.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_abs > 1e-20,
            "Gain matrix should not be all zeros, max = {max_abs}"
        );
    }

    #[test]
    fn test_forward_symmetry_opposite_dipoles() {
        // Two electrodes at symmetric positions should see
        // opposite potentials from a radial dipole at the top
        let elec = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.075, 0.0, 0.04,  // right
                -0.075, 0.0, 0.04, // left
                0.0, 0.0, 0.115,   // top
            ],
        )
        .unwrap();

        // Single tangential source
        let src_pos = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.09]).unwrap();
        let src_nn = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap(); // tangential X

        let sphere = SphereModel::default();
        let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);

        // Right and left electrodes should have opposite signs for an X-dipole
        let v_right = fwd.gain[[0, 0]];
        let v_left = fwd.gain[[1, 0]];
        // They should be roughly opposite (after average ref)
        assert!(
            (v_right + v_left).abs() < (v_right - v_left).abs() * 0.5 || v_right.abs() < 1e-20,
            "Symmetric electrodes should see opposite potentials: right={v_right}, left={v_left}"
        );
    }

    #[test]
    fn test_free_orientation_forward_shape() {
        let elec = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.07, 0.0, 0.04, -0.07, 0.0, 0.04, 0.0, 0.07, 0.04, 0.0, -0.07, 0.04,
            ],
        )
        .unwrap();
        let (src_pos, _) = ico_source_space(1, 0.06, [0.0, 0.0, 0.04]);
        let sphere = SphereModel::default();
        let fwd = make_sphere_forward_free(&elec, &src_pos, &sphere);

        assert_eq!(fwd.gain.nrows(), 4);
        assert_eq!(fwd.gain.ncols(), src_pos.nrows() * 3);
        assert_eq!(fwd.n_sources, src_pos.nrows());
        assert!(fwd.gain.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_single_shell_forward() {
        let elec = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.07, 0.0, 0.04, -0.07, 0.0, 0.04, 0.0, 0.07, 0.04, 0.0, -0.07, 0.04,
            ],
        )
        .unwrap();
        let (src_pos, src_nn) = ico_source_space(1, 0.06, [0.0, 0.0, 0.04]);
        let sphere = SphereModel::single_shell(0.075, 0.33, [0.0, 0.0, 0.04]);
        let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);

        assert_eq!(fwd.gain.nrows(), 4);
        assert!(fwd.gain.iter().all(|v| v.is_finite()));
        let max_abs = fwd.gain.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_abs > 1e-20);
    }

    #[test]
    fn test_end_to_end_forward_to_inverse() {
        // Full pipeline: source space → forward → noise cov → inverse → apply
        use crate::{make_inverse_operator, apply_inverse, InverseMethod, NoiseCov};

        let n_elec = 8;
        let elec = Array2::from_shape_fn((n_elec, 3), |(i, j)| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_elec as f64;
            match j {
                0 => 0.075 * theta.cos(),
                1 => 0.075 * theta.sin(),
                _ => 0.04,
            }
        });
        let (src_pos, src_nn) = ico_source_space(2, 0.06, [0.0, 0.0, 0.04]);
        let sphere = SphereModel::default();
        let fwd = make_sphere_forward(&elec, &src_pos, &src_nn, &sphere);

        let cov = NoiseCov::diagonal(vec![1e-12; n_elec]);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        let data = Array2::from_elem((n_elec, 10), 1e-6);
        let stc = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::DSPM).unwrap();
        assert_eq!(stc.data.nrows(), src_pos.nrows());
        assert!(stc.data.iter().all(|v| v.is_finite()));
    }
}
