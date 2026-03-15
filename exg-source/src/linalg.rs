//! Pure-Rust linear algebra helpers (SVD, eigendecomposition, whitener).
//!
//! Uses [`faer`](https://crates.io/crates/faer) under the hood and converts
//! to/from `ndarray`.

use anyhow::{bail, Result};
use faer::Mat;
use ndarray::{Array1, Array2};

/// Convert `ndarray::Array2<f64>` → `faer::Mat<f64>`.
pub fn ndarray_to_faer(a: &Array2<f64>) -> Mat<f64> {
    let (r, c) = a.dim();
    Mat::<f64>::from_fn(r, c, |i, j| a[[i, j]])
}

/// Convert `faer::Mat<f64>` → `ndarray::Array2<f64>`.
pub fn faer_to_ndarray(m: &Mat<f64>) -> Array2<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    Array2::from_shape_fn((r, c), |(i, j)| m[(i, j)])
}

/// Thin SVD: `A = U @ diag(s) @ V^T`.
///
/// Returns `(U, s, Vt)` where:
/// - `U`  has shape `[m, k]`
/// - `s`  has length `k`
/// - `Vt` has shape `[k, n]`
///
/// with `k = min(m, n)`.
pub fn svd_thin(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let m = ndarray_to_faer(a);
    let svd = m
        .thin_svd()
        .map_err(|e| anyhow::anyhow!("SVD failed: {e:?}"))?;

    let u = svd.U();
    let s_diag = svd.S();
    let v = svd.V();
    let k = u.ncols();

    let s = Array1::from_iter((0..k).map(|i| s_diag[i]));
    let u_nd = faer_to_ndarray(&u.to_owned());
    // faer returns V, we need V^T
    let v_nd = faer_to_ndarray(&v.to_owned());
    let vt_nd = v_nd.t().to_owned();

    Ok((u_nd, s, vt_nd))
}

/// Symmetric eigendecomposition of a real symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` sorted in **descending** order of
/// eigenvalue. `eigenvectors` has shape `[n, n]` with eigenvectors as columns.
pub fn eigh_sorted(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let (n, nc) = a.dim();
    if n != nc {
        bail!("eigh_sorted: matrix must be square, got [{n}, {nc}]");
    }
    let m = ndarray_to_faer(a);
    let evd = m
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| anyhow::anyhow!("Eigendecomposition failed: {e:?}"))?;

    let s_diag = evd.S();
    let u = evd.U();

    // faer returns eigenvalues in ascending order — build index for descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| s_diag[b].partial_cmp(&s_diag[a]).unwrap());

    let vals = Array1::from_iter(indices.iter().map(|&i| s_diag[i]));
    let mut vecs = Array2::zeros((n, n));
    for (col_out, &col_in) in indices.iter().enumerate() {
        for row in 0..n {
            vecs[[row, col_out]] = u[(row, col_in)];
        }
    }

    Ok((vals, vecs))
}

/// Compute a whitening matrix from a noise covariance.
///
/// Returns `(whitener, n_nzero)` where:
/// - `whitener` has shape `[n_nzero, n_channels]`
/// - `n_nzero` is the number of positive eigenvalues
///
/// Whitener satisfies `W @ C @ W^T ≈ I` (restricted to the non-zero subspace).
pub fn compute_whitener(noise_cov: &Array2<f64>) -> Result<(Array2<f64>, usize)> {
    let (evals, evecs) = eigh_sorted(noise_cov)?;
    let n = evals.len();

    // Count positive eigenvalues (numerical rank)
    let tol = evals[0].abs() * 1e-12;
    let n_nzero = evals.iter().filter(|&&v| v > tol).count();
    if n_nzero == 0 {
        bail!("Noise covariance has no positive eigenvalues");
    }

    // Whitener: W = diag(1/√λ) @ V^T  for the non-zero subspace
    let mut whitener = Array2::zeros((n_nzero, n));
    for k in 0..n_nzero {
        let inv_sqrt = 1.0 / evals[k].sqrt();
        for j in 0..n {
            whitener[[k, j]] = inv_sqrt * evecs[[j, k]];
        }
    }

    Ok((whitener, n_nzero))
}

/// Matrix square root of a symmetric positive (semi-)definite matrix.
///
/// Returns `M^{1/2}` such that `M^{1/2} @ M^{1/2} ≈ M`.
pub fn sqrtm_sym(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (evals, evecs) = eigh_sorted(a)?;
    let n = evals.len();
    let mut result = Array2::zeros((n, n));

    for k in 0..n {
        let s = if evals[k] > 0.0 { evals[k].sqrt() } else { 0.0 };
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += s * evecs[[i, k]] * evecs[[j, k]];
            }
        }
    }

    Ok(result)
}

/// Inverse square root of a symmetric positive definite matrix.
///
/// Returns `M^{-1/2}` such that `M^{-1/2} @ M @ M^{-1/2} ≈ I`.
pub fn inv_sqrtm_sym(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (evals, evecs) = eigh_sorted(a)?;
    let n = evals.len();
    let tol = evals[0].abs() * 1e-12;
    let mut result = Array2::zeros((n, n));

    for k in 0..n {
        let s = if evals[k] > tol {
            1.0 / evals[k].sqrt()
        } else {
            0.0
        };
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += s * evecs[[i, k]] * evecs[[j, k]];
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_svd_identity() {
        let eye = Array2::<f64>::eye(4);
        let (u, s, vt) = svd_thin(&eye).unwrap();
        for &sv in s.iter() {
            approx::assert_abs_diff_eq!(sv, 1.0, epsilon = 1e-10);
        }
        // U @ diag(s) @ Vt ≈ I
        let reconstructed = u.dot(&Array2::from_diag(&s)).dot(&vt);
        for ((i, j), &v) in reconstructed.indexed_iter() {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx::assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_whitener_diagonal() {
        let mut cov = Array2::<f64>::zeros((3, 3));
        cov[[0, 0]] = 4.0;
        cov[[1, 1]] = 9.0;
        cov[[2, 2]] = 16.0;
        let (w, n_nz) = compute_whitener(&cov).unwrap();
        assert_eq!(n_nz, 3);

        // W @ C @ W^T should be ≈ I
        let result = w.dot(&cov).dot(&w.t());
        for ((i, j), &v) in result.indexed_iter() {
            let expected = if i == j { 1.0 } else { 0.0 };
            approx::assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_eigh_sorted_descending() {
        let mut m = Array2::<f64>::zeros((3, 3));
        m[[0, 0]] = 1.0;
        m[[1, 1]] = 3.0;
        m[[2, 2]] = 2.0;
        let (evals, _) = eigh_sorted(&m).unwrap();
        assert!(evals[0] >= evals[1] && evals[1] >= evals[2]);
        approx::assert_abs_diff_eq!(evals[0], 3.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(evals[1], 2.0, epsilon = 1e-10);
        approx::assert_abs_diff_eq!(evals[2], 1.0, epsilon = 1e-10);
    }
}
