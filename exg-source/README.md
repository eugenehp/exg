# exg-source

> EEG source localization in pure Rust — MNE / dSPM / sLORETA / eLORETA

`exg-source` implements the complete minimum-norm source localization pipeline
ported from [MNE-Python](https://mne.tools), with no C, BLAS, or Python
dependencies. Linear algebra is handled by
[faer](https://crates.io/crates/faer) (pure Rust).

This crate can be used **standalone** or as part of
[exg](https://crates.io/crates/exg) via the default `source` feature.

---

## Quick start

```toml
[dependencies]
exg-source = "0.0.2"
```

```rust
use exg_source::*;
use exg_source::covariance::Regularization;
use ndarray::Array2;

// 1. Source space — 162 dipoles on a cortical sphere
let (src_pos, src_nn) = ico_source_space(2, 0.06, [0.0, 0.0, 0.04]);

// 2. Forward model — 3-shell spherical head (Berg & Scherg)
let electrodes: Array2<f64> = /* your electrode positions [n_elec, 3] */
# Array2::zeros((32, 3));
let fwd = make_sphere_forward(&electrodes, &src_pos, &src_nn, &SphereModel::default());

// 3. Noise covariance from baseline data
let baseline: Array2<f64> = /* [n_elec, n_samples] */
# Array2::zeros((32, 5000));
let noise_cov = compute_covariance(&baseline, Regularization::ShrunkIdentity(None));

// 4. Inverse operator
let inv = make_inverse_operator(&fwd, &noise_cov, Some(0.8)).unwrap();

// 5. Source estimate
let data: Array2<f64> = /* [n_elec, n_times] */
# Array2::zeros((32, 1000));
let stc = apply_inverse(&data, &inv, 1.0 / 9.0, InverseMethod::SLORETA).unwrap();
println!("{} sources × {} time points", stc.data.nrows(), stc.data.ncols());
```

---

## Pipeline overview

```text
Electrode positions          Source positions
       │                            │
       ▼                            ▼
┌──────────────┐           ┌────────────────┐
│ SphereModel  │           │ ico/grid       │
│ (3-shell)    │           │ source_space   │
└──────┬───────┘           └───────┬────────┘
       │                           │
       └─────────┬─────────────────┘
                 ▼
        ┌────────────────┐
        │ make_sphere_   │
        │ forward()      │──→ ForwardOperator (gain matrix)
        └────────┬───────┘
                 │
   Baseline ─→ compute_covariance() ──→ NoiseCov
                 │
                 ▼
        ┌────────────────┐
        │ make_inverse_  │
        │ operator()     │──→ InverseOperator (SVD + priors)
        └────────┬───────┘
                 │
    EEG data ──→ apply_inverse() ──→ SourceEstimate
                 │
                 ├──→ estimate_snr()
                 └──→ make_resolution_matrix()
```

---

## Modules

### Source space — `source_space`

Generate source positions where dipoles are placed.

| Function | Description |
|----------|-------------|
| `ico_source_space(n, r, center)` | Icosahedron subdivided `n` times, projected onto sphere of radius `r` |
| `grid_source_space(spacing, r, center)` | Regular 3-D grid inside a sphere |
| `ico_n_vertices(n)` | Expected vertex count: `10 × 4^n + 2` |

Subdivision levels: 0 → 12, 1 → 42, 2 → 162, 3 → 642, 4 → 2562, 5 → 10242 vertices.

### Forward model — `forward`

Compute the EEG lead-field (gain) matrix using a multi-shell spherical head.

| Function | Description |
|----------|-------------|
| `make_sphere_forward(elec, src, nn, sphere)` | Fixed-orientation forward `[n_elec, n_src]` |
| `make_sphere_forward_free(elec, src, sphere)` | Free-orientation forward `[n_elec, n_src × 3]` |
| `SphereModel::default()` | 3-shell: brain (0.067 m, 0.33 S/m), skull (0.070 m, 0.0042 S/m), scalp (0.075 m, 0.33 S/m) |
| `SphereModel::single_shell(r, σ, c)` | Homogeneous sphere |

Based on the Berg & Scherg (1994) approximation with Prony's method for parameter fitting.

### Noise covariance — `covariance`

| Function | Description |
|----------|-------------|
| `compute_covariance(data, reg)` | From continuous `[n_ch, n_t]` data |
| `compute_covariance_epochs(epochs, reg)` | From epoched `[n_ep, n_ch, n_t]` data |

Regularisation options:
- `Empirical` — raw sample covariance
- `ShrunkIdentity(None)` — Ledoit–Wolf automatic shrinkage
- `ShrunkIdentity(Some(α))` — manual shrinkage, α ∈ [0, 1]
- `Diagonal` — channel variances only

### Inverse operator — `inverse`

| Function | Description |
|----------|-------------|
| `make_inverse_operator(fwd, cov, depth)` | Build from forward + noise covariance |
| `apply_inverse(data, inv, λ², method)` | Apply to `[n_ch, n_t]` data |
| `apply_inverse_full(data, inv, λ², method, pick_ori, opts)` | With orientation picking |
| `apply_inverse_epochs(epochs, inv, λ², method)` | Batch over `[n_ep, n_ch, n_t]` |

Methods: `InverseMethod::MNE`, `DSPM`, `SLORETA`, `ELORETA`

Orientation picking (`PickOri`):
- `None` — combine XYZ as `√(x² + y² + z²)` (default)
- `Normal` — cortical normal component only
- `Vector` — all three components `[n_src × 3, n_t]`

### Resolution analysis — `resolution`

| Function | Returns |
|----------|---------|
| `make_resolution_matrix(inv, fwd, λ², method)` | `R = K @ G` `[n_src, n_src]` |
| `get_point_spread(R, idx)` | PSF for source `idx` |
| `get_cross_talk(R, idx)` | CTF for source `idx` |
| `peak_localisation_error(R)` | Index offset of PSF peak from true source |
| `spatial_spread(R)` | Number of sources within half-max of PSF |
| `relative_amplitude(R)` | Peak / total ratio (1.0 = perfect) |

### SNR estimation — `snr`

| Function | Returns |
|----------|---------|
| `estimate_snr(data, inv)` | `(snr, snr_est)` — whitened GFP and regularisation-based SNR per time point |

### eLORETA — `eloreta`

Iterative solver for exact zero-bias source localisation. Configure via `EloretaOptions`:

```rust
let opts = EloretaOptions {
    eps: 1e-6,       // convergence threshold
    max_iter: 20,    // maximum iterations
    force_equal: None, // automatic orientation handling
};
```

### Linear algebra — `linalg`

Pure-Rust helpers using [faer](https://crates.io/crates/faer):

| Function | Description |
|----------|-------------|
| `svd_thin(A)` | Thin SVD: `(U, s, Vt)` |
| `eigh_sorted(A)` | Symmetric eigendecomposition, descending order |
| `compute_whitener(C)` | Whitening matrix from noise covariance |
| `sqrtm_sym(A)` | Matrix square root of symmetric PSD matrix |
| `inv_sqrtm_sym(A)` | Inverse matrix square root |

---

## Choosing λ² and method

| Parameter | Typical value | Meaning |
|-----------|--------------|---------|
| `lambda2 = 1.0 / 9.0` | SNR = 3 | Good default for evoked responses |
| `lambda2 = 1.0 / 1.0` | SNR = 1 | More regularisation (noisy data) |
| `lambda2 = 1.0 / 100.0` | SNR = 10 | Less regularisation (clean data) |

| Method | When to use |
|--------|-------------|
| **MNE** | Raw current estimates, custom normalisation |
| **dSPM** | Statistical inference, group analysis |
| **sLORETA** | Best localisation accuracy for focal sources |
| **eLORETA** | Guaranteed zero localisation bias (slower) |

---

## References

- Hämäläinen, M. S. & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain. *Medical & Biological Engineering & Computing*, 32(1), 35-42.
- Dale, A. M. et al. (2000). Dynamic statistical parametric mapping. *Neuron*, 26(1), 55-67.
- Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA). *Methods and Findings in Experimental and Clinical Pharmacology*, 24D, 5-12.
- Pascual-Marqui, R. D. (2011). Discrete, 3D distributed, linear imaging methods of electric neuronal activity. *arXiv:0710.3341*.
- Berg, P. & Scherg, M. (1994). A fast method for forward computation of multiple-shell spherical head models. *Electroencephalography and Clinical Neurophysiology*, 90(1), 58-64.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

## License

[AI100](https://www.humanscommons.org/license/ai100/1.0)
