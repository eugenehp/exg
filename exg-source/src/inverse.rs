//! Inverse operator construction and application.
//!
//! Implements `make_inverse_operator` and `apply_inverse`, ported from
//! MNE-Python's `mne.minimum_norm.inverse`.

use anyhow::{bail, Result};
use ndarray::{Array1, Array2};

use super::eloreta::compute_eloreta;
use super::linalg;
use super::{
    EloretaOptions, ForwardOperator, InverseMethod, InverseOperator, NoiseCov, PickOri,
    SourceEstimate, SourceOrientation,
};

/// Compute the depth-weighting prior from the gain matrix.
///
/// Following MNE-Python's `compute_depth_prior`, the depth weight for source
/// `k` is `(‖G_k‖ / max ‖G_k‖)^exp` where `G_k` is the column (or 3-column
/// block for free orientation) of the whitened gain matrix.
fn compute_depth_prior(
    whitened_gain: &Array2<f64>,
    n_sources: usize,
    n_orient: usize,
    exp: f64,
) -> Array1<f64> {
    let n_cols = n_sources * n_orient;
    let mut col_norms = Array1::zeros(n_sources);

    for s in 0..n_sources {
        let mut norm_sq = 0.0;
        for o in 0..n_orient {
            let col_idx = s * n_orient + o;
            if col_idx < n_cols {
                for r in 0..whitened_gain.nrows() {
                    norm_sq += whitened_gain[[r, col_idx]].powi(2);
                }
            }
        }
        col_norms[s] = norm_sq.sqrt();
    }

    let max_norm = col_norms.iter().copied().fold(0.0_f64, f64::max);
    if max_norm <= 0.0 {
        return Array1::ones(n_cols);
    }

    let mut prior = Array1::zeros(n_cols);
    for s in 0..n_sources {
        let w = (col_norms[s] / max_norm).powf(exp);
        for o in 0..n_orient {
            prior[s * n_orient + o] = w;
        }
    }
    prior
}

/// Build an inverse operator from a forward model and noise covariance.
///
/// This is the Rust equivalent of `mne.minimum_norm.make_inverse_operator`.
///
/// # Arguments
///
/// * `fwd` — Forward operator (gain matrix + source info).
/// * `noise_cov` — Sensor noise covariance.
/// * `depth_exp` — Optional depth-weighting exponent. If `None`, uses
///   `fwd.depth_exp` or falls back to no depth weighting.
///
/// # Returns
///
/// An [`InverseOperator`] containing the SVD decomposition of the whitened
/// and weighted gain matrix, ready for use with [`apply_inverse`].
pub fn make_inverse_operator(
    fwd: &ForwardOperator,
    noise_cov: &NoiseCov,
    depth_exp: Option<f64>,
) -> Result<InverseOperator> {
    let n_chan = fwd.gain.nrows();
    let n_orient = fwd.n_orient();
    let n_cols = fwd.n_sources * n_orient;

    if fwd.gain.ncols() != n_cols {
        bail!(
            "Gain matrix has {} columns but expected {} (n_sources={} × n_orient={})",
            fwd.gain.ncols(),
            n_cols,
            fwd.n_sources,
            n_orient,
        );
    }
    if noise_cov.n_channels() != n_chan {
        bail!(
            "Noise covariance has {} channels but gain has {} rows",
            noise_cov.n_channels(),
            n_chan,
        );
    }

    // 1. Compute whitener from noise covariance
    let cov_full = noise_cov.to_full();
    let (whitener, n_nzero) = linalg::compute_whitener(&cov_full)?;

    // 2. Whiten the gain matrix: G_w = W @ G
    let gain_w = whitener.dot(&fwd.gain);

    // 3. Compute depth prior (optional)
    let exp = depth_exp.or(fwd.depth_exp);
    let source_std = if let Some(e) = exp {
        let prior = compute_depth_prior(&gain_w, fwd.n_sources, n_orient, e);
        let mut std = Array1::zeros(n_cols);
        for i in 0..n_cols {
            std[i] = prior[i].sqrt();
        }
        std
    } else {
        Array1::ones(n_cols)
    };

    // 4. Apply source weighting to whitened gain: G_w *= source_std
    let mut gain_ws = gain_w;
    for j in 0..n_cols {
        for i in 0..gain_ws.nrows() {
            gain_ws[[i, j]] *= source_std[j];
        }
    }

    // 5. Scale so that trace(G_ws @ G_ws^T) = n_nzero
    let trace_grgt = gain_ws.iter().map(|v| v * v).sum::<f64>();
    let scale = (n_nzero as f64 / trace_grgt).sqrt();
    gain_ws.mapv_inplace(|v| v * scale);
    let source_std = source_std.mapv(|v| v * scale);

    // 6. SVD of whitened, weighted gain
    let (u, sing, vt) = linalg::svd_thin(&gain_ws)?;

    // eigen_fields = U^T  [k, n_chan]
    let eigen_fields = u.t().to_owned();
    // eigen_leads = V  [n_cols, k]  (from Vt -> V = Vt^T)
    let eigen_leads = vt.t().to_owned();

    let source_cov = source_std.mapv(|v| v * v);

    Ok(InverseOperator {
        eigen_fields,
        sing,
        eigen_leads,
        source_cov,
        eigen_leads_weighted: false,
        n_sources: fwd.n_sources,
        orientation: fwd.orientation,
        source_nn: fwd.source_nn.clone(),
        whitener,
        n_nzero,
        noise_cov: noise_cov.clone(),
    })
}

/// Intermediate prepared state for an inverse operator.
pub struct PreparedInverse {
    /// Regularised inverse of singular values: `s / (s² + λ²)`.
    pub reginv: Array1<f64>,
    /// Noise-normalisation factors (one per source), or `None` for MNE.
    pub noisenorm: Option<Array1<f64>>,
    /// Imaging kernel `K` [n_sources_out, n_channels].
    pub kernel: Array2<f64>,
}

/// Prepare an inverse operator for a specific method and regularisation.
///
/// Computes the imaging kernel and noise normalisation.
///
/// # Arguments
///
/// * `inv` — Inverse operator from [`make_inverse_operator`].
/// * `lambda2` — Regularisation parameter (recommended: 1/SNR²).
/// * `method` — Inverse method to use.
/// * `eloreta_opts` — Options for eLORETA (ignored for other methods).
pub fn prepare_inverse(
    inv: &InverseOperator,
    lambda2: f64,
    method: InverseMethod,
    eloreta_opts: Option<&EloretaOptions>,
) -> Result<PreparedInverse> {
    let n_orient = match inv.orientation {
        SourceOrientation::Fixed => 1,
        SourceOrientation::Free => 3,
    };

    if method == InverseMethod::ELORETA {
        return prepare_eloreta(inv, lambda2, eloreta_opts);
    }

    // Compute regularised inverse: reginv_k = s_k / (s_k² + λ²)
    let reginv = compute_reginv(&inv.sing, lambda2, inv.n_nzero);

    // Noise normalisation
    let noisenorm = match method {
        InverseMethod::MNE => None,
        InverseMethod::DSPM => {
            let noise_weight = reginv.clone();
            Some(compute_noise_norm(inv, &noise_weight, n_orient))
        }
        InverseMethod::SLORETA => {
            let noise_weight = Array1::from_iter(
                reginv
                    .iter()
                    .zip(inv.sing.iter())
                    .map(|(&ri, &si)| ri * (1.0 + si * si / lambda2).sqrt()),
            );
            Some(compute_noise_norm(inv, &noise_weight, n_orient))
        }
        InverseMethod::ELORETA => unreachable!(),
    };

    // Assemble kernel: K = sqrt(source_cov) @ V @ diag(reginv) @ U^T @ W
    // trans = U^T @ W  has shape [k, n_chan] (but eigen_fields is already U^T)
    // So trans = eigen_fields @ whitener... no, eigen_fields IS U^T [k, n_chan_whitened]
    // We need: trans = diag(reginv) @ eigen_fields @ whitener
    // Wait, let me re-read MNE:
    // trans = eigen_fields @ whitener @ proj
    // trans *= reginv[:, None]
    // K = eigen_leads @ trans
    // K *= sqrt(source_cov)[:, None]

    let n_k = inv.sing.len();
    let n_chan = inv.whitener.ncols();

    // trans = eigen_fields @ whitener  [k, n_chan]
    let trans = inv.eigen_fields.dot(&inv.whitener);
    // trans *= reginv (row-wise)
    let mut trans_scaled = Array2::zeros((n_k, n_chan));
    for i in 0..n_k {
        for j in 0..n_chan {
            trans_scaled[[i, j]] = trans[[i, j]] * reginv[i];
        }
    }

    // K = eigen_leads @ trans_scaled  [n_src*n_orient, n_chan]
    let mut kernel = inv.eigen_leads.dot(&trans_scaled);

    // K *= sqrt(source_cov)
    if !inv.eigen_leads_weighted {
        for i in 0..kernel.nrows() {
            let w = inv.source_cov[i].sqrt();
            for j in 0..kernel.ncols() {
                kernel[[i, j]] *= w;
            }
        }
    }

    Ok(PreparedInverse {
        reginv,
        noisenorm,
        kernel,
    })
}

/// Prepare the eLORETA inverse.
fn prepare_eloreta(
    inv: &InverseOperator,
    lambda2: f64,
    opts: Option<&EloretaOptions>,
) -> Result<PreparedInverse> {
    let default_opts = EloretaOptions::default();
    let opts = opts.unwrap_or(&default_opts);

    let (kernel, reginv) = compute_eloreta(inv, lambda2, opts)?;

    Ok(PreparedInverse {
        reginv,
        noisenorm: None, // eLORETA embeds normalisation in the kernel
        kernel,
    })
}

/// Compute `reginv[k] = s[k] / (s[k]² + λ²)` for the first `n_nzero` values.
fn compute_reginv(sing: &Array1<f64>, lambda2: f64, n_nzero: usize) -> Array1<f64> {
    let n = sing.len();
    let mut reginv = Array1::zeros(n);
    for k in 0..n.min(n_nzero) {
        let s = sing[k];
        if s > 0.0 {
            reginv[k] = s / (s * s + lambda2);
        }
    }
    reginv
}

/// Compute noise-normalisation factors (dSPM / sLORETA).
///
/// For each source, compute `1 / ‖row_k @ diag(noise_weight)‖₂`.
fn compute_noise_norm(
    inv: &InverseOperator,
    noise_weight: &Array1<f64>,
    n_orient: usize,
) -> Array1<f64> {
    let n_rows = inv.eigen_leads.nrows();
    let n_k = noise_weight.len();

    let mut raw_norm = Array1::zeros(n_rows);
    for k in 0..n_rows {
        let mut sq_sum = 0.0;
        for j in 0..n_k {
            let lead = if inv.eigen_leads_weighted {
                inv.eigen_leads[[k, j]]
            } else {
                inv.source_cov[k].sqrt() * inv.eigen_leads[[k, j]]
            };
            let val = lead * noise_weight[j];
            sq_sum += val * val;
        }
        raw_norm[k] = sq_sum.sqrt();
    }

    // For free orientation: combine XYZ triplets
    if n_orient == 3 {
        let n_src = n_rows / 3;
        let mut combined = Array1::zeros(n_src);
        for s in 0..n_src {
            let mut sum_sq = 0.0;
            for o in 0..3 {
                sum_sq += raw_norm[s * 3 + o].powi(2);
            }
            combined[s] = sum_sq.sqrt();
        }
        combined.mapv(|v| if v.abs() > 0.0 { 1.0 / v } else { 0.0 })
    } else {
        raw_norm.mapv(|v| if v.abs() > 0.0 { 1.0 / v } else { 0.0 })
    }
}

/// Combine free-orientation XYZ triplets: `√(x² + y² + z²)` per source.
fn combine_xyz(sol: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_times) = sol.dim();
    assert!(n_rows % 3 == 0, "combine_xyz: rows must be divisible by 3");
    let n_src = n_rows / 3;
    let mut out = Array2::zeros((n_src, n_times));
    for s in 0..n_src {
        for t in 0..n_times {
            let x = sol[[s * 3, t]];
            let y = sol[[s * 3 + 1, t]];
            let z = sol[[s * 3 + 2, t]];
            out[[s, t]] = (x * x + y * y + z * z).sqrt();
        }
    }
    out
}

/// Apply an inverse operator to sensor-space data.
///
/// This is the Rust equivalent of `mne.minimum_norm.apply_inverse`.
///
/// # Arguments
///
/// * `data` — Sensor data, shape `[n_channels, n_times]`.
/// * `inv` — Inverse operator from [`make_inverse_operator`].
/// * `lambda2` — Regularisation parameter (recommended: `1.0 / SNR.powi(2)`).
/// * `method` — Which method to use.
///
/// # Returns
///
/// A [`SourceEstimate`] with shape `[n_sources, n_times]` (magnitudes for
/// free orientation are combined across XYZ).
///
/// # Example
///
/// ```no_run
/// use exg_source::*;
/// use ndarray::Array2;
///
/// let n_chan = 32;
/// let n_src  = 500;
/// let gain = Array2::<f64>::from_elem((n_chan, n_src), 1e-8);
/// let fwd  = ForwardOperator::new_fixed(gain);
/// let cov  = NoiseCov::diagonal(vec![1e-12; n_chan]);
/// let inv  = make_inverse_operator(&fwd, &cov, None).unwrap();
///
/// let data = Array2::<f64>::zeros((n_chan, 100));
/// let stc  = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::SLORETA).unwrap();
/// assert_eq!(stc.data.nrows(), n_src);
/// assert_eq!(stc.data.ncols(), 100);
/// ```
pub fn apply_inverse(
    data: &Array2<f64>,
    inv: &InverseOperator,
    lambda2: f64,
    method: InverseMethod,
) -> Result<SourceEstimate> {
    apply_inverse_with_options(data, inv, lambda2, method, None)
}

/// Apply inverse with optional eLORETA parameters.
pub fn apply_inverse_with_options(
    data: &Array2<f64>,
    inv: &InverseOperator,
    lambda2: f64,
    method: InverseMethod,
    eloreta_opts: Option<&EloretaOptions>,
) -> Result<SourceEstimate> {
    apply_inverse_full(data, inv, lambda2, method, PickOri::None, eloreta_opts)
}

/// Apply inverse with full control over orientation picking and eLORETA options.
///
/// # Arguments
///
/// * `data`          — Sensor data, shape `[n_channels, n_times]`.
/// * `inv`           — Inverse operator.
/// * `lambda2`       — Regularisation parameter.
/// * `method`        — Inverse method.
/// * `pick_ori`      — How to handle source orientations (see [`PickOri`]).
/// * `eloreta_opts`  — Options for eLORETA (ignored for other methods).
pub fn apply_inverse_full(
    data: &Array2<f64>,
    inv: &InverseOperator,
    lambda2: f64,
    method: InverseMethod,
    pick_ori: PickOri,
    eloreta_opts: Option<&EloretaOptions>,
) -> Result<SourceEstimate> {
    let n_chan = data.nrows();
    if n_chan != inv.whitener.ncols() {
        bail!(
            "Data has {} channels but inverse expects {}",
            n_chan,
            inv.whitener.ncols()
        );
    }

    let n_orient = match inv.orientation {
        SourceOrientation::Fixed => 1,
        SourceOrientation::Free => 3,
    };

    if pick_ori == PickOri::Normal && n_orient != 3 {
        bail!("pick_ori=Normal requires free-orientation inverse");
    }

    let prepared = prepare_inverse(inv, lambda2, method, eloreta_opts)?;

    // Apply imaging kernel: sol = K @ data
    let mut sol = prepared.kernel.dot(data);

    let is_free = n_orient == 3;

    match pick_ori {
        PickOri::None => {
            // Default: combine XYZ for free orientation
            if is_free {
                sol = combine_xyz(&sol);
            }
            // Apply noise normalisation
            apply_noisenorm(&mut sol, &prepared.noisenorm);
        }
        PickOri::Normal => {
            // Pick only the Z (normal) component: every 3rd row starting at index 2
            let n_src = inv.n_sources;
            let n_times = sol.ncols();
            let mut normal_sol = Array2::zeros((n_src, n_times));
            for s in 0..n_src {
                for t in 0..n_times {
                    normal_sol[[s, t]] = sol[[s * 3 + 2, t]];
                }
            }
            sol = normal_sol;
            // Apply noise normalisation
            apply_noisenorm(&mut sol, &prepared.noisenorm);
        }
        PickOri::Vector => {
            // Return all 3 components — noise norm must be expanded
            if let Some(ref nn) = prepared.noisenorm {
                if is_free {
                    // noisenorm has n_src entries; repeat for each orientation
                    for s in 0..inv.n_sources {
                        let norm = nn[s];
                        for o in 0..3 {
                            for t in 0..sol.ncols() {
                                sol[[s * 3 + o, t]] *= norm;
                            }
                        }
                    }
                } else {
                    apply_noisenorm(&mut sol, &prepared.noisenorm);
                }
            }
        }
    }

    Ok(SourceEstimate {
        data: sol,
        n_sources: inv.n_sources,
        orientation: inv.orientation,
    })
}

/// Apply noise normalisation in-place.
fn apply_noisenorm(sol: &mut Array2<f64>, noisenorm: &Option<Array1<f64>>) {
    if let Some(ref nn) = noisenorm {
        let n_src_out = sol.nrows();
        for s in 0..n_src_out {
            let norm = nn[s];
            for t in 0..sol.ncols() {
                sol[[s, t]] *= norm;
            }
        }
    }
}

/// Apply inverse operator to each epoch in a batch.
///
/// This is the Rust equivalent of `mne.minimum_norm.apply_inverse_epochs`.
///
/// # Arguments
///
/// * `epochs`  — Epoched data, shape `[n_epochs, n_channels, n_times]`.
/// * `inv`     — Inverse operator.
/// * `lambda2` — Regularisation parameter.
/// * `method`  — Inverse method.
///
/// # Returns
///
/// A `Vec<SourceEstimate>`, one per epoch.
pub fn apply_inverse_epochs(
    epochs: &ndarray::Array3<f64>,
    inv: &InverseOperator,
    lambda2: f64,
    method: InverseMethod,
) -> Result<Vec<SourceEstimate>> {
    apply_inverse_epochs_full(epochs, inv, lambda2, method, PickOri::None, None)
}

/// Apply inverse to epochs with full options.
pub fn apply_inverse_epochs_full(
    epochs: &ndarray::Array3<f64>,
    inv: &InverseOperator,
    lambda2: f64,
    method: InverseMethod,
    pick_ori: PickOri,
    eloreta_opts: Option<&EloretaOptions>,
) -> Result<Vec<SourceEstimate>> {
    let (n_epochs, _n_ch, _n_t) = epochs.dim();
    let mut results = Vec::with_capacity(n_epochs);
    for e in 0..n_epochs {
        let epoch = epochs.slice(ndarray::s![e, .., ..]).to_owned();
        let stc = apply_inverse_full(&epoch, inv, lambda2, method, pick_ori, eloreta_opts)?;
        results.push(stc);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build a simple test forward model and noise cov.
    fn make_test_setup(n_chan: usize, n_src: usize) -> (ForwardOperator, NoiseCov) {
        // Simple forward: each source contributes to all channels
        let mut gain = Array2::zeros((n_chan, n_src));
        for i in 0..n_chan {
            for j in 0..n_src {
                // Distance-like falloff
                let dist =
                    ((i as f64 - j as f64 * n_chan as f64 / n_src as f64).powi(2) + 1.0).sqrt();
                gain[[i, j]] = 1e-8 / dist;
            }
        }
        let fwd = ForwardOperator::new_fixed(gain);
        let cov = NoiseCov::diagonal(vec![1e-12; n_chan]);
        (fwd, cov)
    }

    #[test]
    fn test_make_inverse_operator() {
        let (fwd, cov) = make_test_setup(16, 50);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        assert_eq!(inv.n_sources, 50);
        assert_eq!(inv.sing.len(), 16); // min(n_chan, n_src)
    }

    #[test]
    fn test_apply_inverse_mne() {
        let (fwd, cov) = make_test_setup(16, 50);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        // Simulate data from a single source
        let n_times = 10;
        let source_idx = 25;
        let mut source_signal = Array2::zeros((50, n_times));
        for t in 0..n_times {
            source_signal[[source_idx, t]] = 1e-9;
        }
        let data = fwd.gain.dot(&source_signal);

        let stc = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::MNE).unwrap();
        assert_eq!(stc.data.dim(), (50, n_times));

        // The peak source should be near the simulated source
        let mut peak_src = 0;
        let mut peak_val = 0.0_f64;
        for s in 0..50 {
            let val = stc.data[[s, 0]].abs();
            if val > peak_val {
                peak_val = val;
                peak_src = s;
            }
        }
        // Should be within a few sources of the true location
        assert!(
            (peak_src as i32 - source_idx as i32).unsigned_abs() <= 5,
            "Peak at {peak_src}, expected near {source_idx}"
        );
    }

    #[test]
    fn test_apply_inverse_dspm() {
        let (fwd, cov) = make_test_setup(16, 50);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let data = Array2::from_elem((16, 5), 1e-6);
        let stc = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::DSPM).unwrap();
        assert_eq!(stc.data.nrows(), 50);
        // dSPM values should be finite and non-NaN
        assert!(stc.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_apply_inverse_sloreta() {
        let (fwd, cov) = make_test_setup(16, 50);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();
        let data = Array2::from_elem((16, 5), 1e-6);
        let stc = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::SLORETA).unwrap();
        assert_eq!(stc.data.nrows(), 50);
        assert!(stc.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_free_orientation() {
        let n_chan = 16;
        let n_src = 20;
        let mut gain = Array2::zeros((n_chan, n_src * 3));
        for i in 0..n_chan {
            for j in 0..n_src * 3 {
                let dist = ((i as f64 - j as f64 / 3.0 * n_chan as f64 / n_src as f64).powi(2)
                    + 1.0)
                    .sqrt();
                gain[[i, j]] = 1e-8 / dist;
            }
        }
        let fwd = ForwardOperator::new_free(gain);
        let cov = NoiseCov::diagonal(vec![1e-12; n_chan]);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        let data = Array2::from_elem((n_chan, 5), 1e-6);
        let stc = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::DSPM).unwrap();
        // Free orientation: XYZ are combined → n_sources rows
        assert_eq!(stc.data.nrows(), n_src);
        assert!(stc.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_apply_inverse_epochs() {
        let (fwd, cov) = make_test_setup(16, 50);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        let epochs = ndarray::Array3::from_shape_fn((5, 16, 10), |(_, i, j)| {
            ((i * 10 + j) as f64).sin() * 1e-6
        });
        let stcs = apply_inverse_epochs(&epochs, &inv, 1.0 / 9.0, InverseMethod::DSPM).unwrap();
        assert_eq!(stcs.len(), 5);
        for stc in &stcs {
            assert_eq!(stc.data.dim(), (50, 10));
            assert!(stc.data.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_pick_ori_vector() {
        let n_chan = 16;
        let n_src = 20;
        let mut gain = Array2::zeros((n_chan, n_src * 3));
        for i in 0..n_chan {
            for j in 0..n_src * 3 {
                let dist = ((i as f64 - j as f64 / 3.0 * n_chan as f64 / n_src as f64).powi(2)
                    + 1.0)
                    .sqrt();
                gain[[i, j]] = 1e-8 / dist;
            }
        }
        let fwd = ForwardOperator::new_free(gain);
        let cov = NoiseCov::diagonal(vec![1e-12; n_chan]);
        let inv = make_inverse_operator(&fwd, &cov, None).unwrap();

        let data = Array2::from_elem((n_chan, 5), 1e-6);

        // Vector: should return n_src*3 rows
        let stc_vec = apply_inverse_full(
            &data,
            &inv,
            1.0 / 9.0,
            InverseMethod::MNE,
            PickOri::Vector,
            None,
        )
        .unwrap();
        assert_eq!(stc_vec.data.nrows(), n_src * 3);

        // Normal: should return n_src rows
        let stc_norm = apply_inverse_full(
            &data,
            &inv,
            1.0 / 9.0,
            InverseMethod::MNE,
            PickOri::Normal,
            None,
        )
        .unwrap();
        assert_eq!(stc_norm.data.nrows(), n_src);

        // Default (None): should return n_src rows (combined)
        let stc_comb = apply_inverse_full(
            &data,
            &inv,
            1.0 / 9.0,
            InverseMethod::MNE,
            PickOri::None,
            None,
        )
        .unwrap();
        assert_eq!(stc_comb.data.nrows(), n_src);
    }

    #[test]
    fn test_depth_weighting() {
        let (fwd, cov) = make_test_setup(16, 50);
        // With depth weighting
        let inv_depth = make_inverse_operator(&fwd, &cov, Some(0.8)).unwrap();
        // Without depth weighting
        let inv_nodepth = make_inverse_operator(&fwd, &cov, None).unwrap();

        // Source covariances should differ
        let diff: f64 = (&inv_depth.source_cov - &inv_nodepth.source_cov)
            .mapv(f64::abs)
            .sum();
        assert!(diff > 1e-10, "Depth weighting should change source_cov");
    }
}
