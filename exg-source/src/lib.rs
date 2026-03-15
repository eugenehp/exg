//! # exg-source — EEG Source Localization in Pure Rust
//!
//! [![Crate](https://img.shields.io/crates/v/exg-source.svg)](https://crates.io/crates/exg-source)
//! [![Docs](https://docs.rs/exg-source/badge.svg)](https://docs.rs/exg-source)
//!
//! Complete EEG source localization pipeline ported from
//! [MNE-Python](https://mne.tools), implemented in pure Rust with no C/BLAS
//! dependencies. Uses [`faer`](https://crates.io/crates/faer) for SVD and
//! eigendecomposition.
//!
//! This crate can be used **standalone** or as an optional dependency of
//! [`exg`](https://crates.io/crates/exg) (enabled by the default `source`
//! feature).
//!
//! ## End-to-end example
//!
//! ```no_run
//! use exg_source::*;
//! use exg_source::covariance::Regularization;
//! use ndarray::Array2;
//!
//! // 1. Define source space — 162 dipoles on a cortical sphere
//! let (src_pos, src_nn) = ico_source_space(2, 0.06, [0.0, 0.0, 0.04]);
//!
//! // 2. Define electrode positions (e.g., from a 10-20 montage)
//! let n_elec = 32;
//! let electrodes = Array2::<f64>::zeros((n_elec, 3)); // your positions here
//!
//! // 3. Compute forward model (3-shell spherical head)
//! let sphere = SphereModel::default(); // brain/skull/scalp
//! let fwd = make_sphere_forward(&electrodes, &src_pos, &src_nn, &sphere);
//!
//! // 4. Estimate noise covariance from a baseline recording
//! let baseline = Array2::<f64>::zeros((n_elec, 5000));
//! let noise_cov = compute_covariance(&baseline, Regularization::ShrunkIdentity(None));
//!
//! // 5. Build the inverse operator
//! let inv = make_inverse_operator(&fwd, &noise_cov, Some(0.8)).unwrap();
//!
//! // 6. Apply to EEG data
//! let data = Array2::<f64>::zeros((n_elec, 1000));
//! let lambda2 = 1.0 / 9.0; // SNR² = 9
//! let stc = apply_inverse(&data, &inv, lambda2, InverseMethod::SLORETA).unwrap();
//!
//! // 7. Assess solution quality
//! let (snr, snr_est) = estimate_snr(&data, &inv);
//! let r = make_resolution_matrix(&inv, &fwd, lambda2, InverseMethod::SLORETA).unwrap();
//! let errors = peak_localisation_error(&r);
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`source_space`] | Icosahedron and grid source space generation |
//! | [`forward`] | Spherical head model (Berg & Scherg) forward computation |
//! | [`covariance`] | Noise covariance estimation (empirical, Ledoit–Wolf, diagonal) |
//! | [`inverse`] | Inverse operator construction and application |
//! | [`eloreta`] | eLORETA iterative weight solver |
//! | [`resolution`] | Resolution matrix, PSF, CTF, and spatial metrics |
//! | [`snr`] | SNR estimation from whitened data |
//! | [`linalg`] | SVD, eigendecomposition, whitener (faer backend) |
//!
//! ## Inverse methods
//!
//! | Method | Noise-normalised? | Iterative? | Best for |
//! |--------|:-:|:-:|----------|
//! | **MNE** | ✗ | ✗ | Raw current density estimates |
//! | **dSPM** | ✓ | ✗ | Statistical maps, group studies |
//! | **sLORETA** | ✓ | ✗ | Localisation with low bias |
//! | **eLORETA** | ✓ | ✓ | Exact zero-bias localisation |
//!
//! ## Mathematical background
//!
//! The whitened gain matrix is decomposed via SVD:
//!
//! ```text
//! G_w = W @ G @ R^{1/2} = U @ S @ V^T
//! ```
//!
//! where **W** is the whitener (`C_noise^{-1/2}`), **R** is the source
//! covariance (depth / orientation priors), and **U**, **S**, **V** are the
//! SVD factors.
//!
//! The inverse kernel is:
//!
//! ```text
//! K = R^{1/2} @ V @ diag(s / (s² + λ²)) @ U^T @ W
//! ```
//!
//! Noise normalisation divides each source by its noise sensitivity:
//!
//! - **dSPM**: `norm_k = ‖V_k @ diag(reginv)‖`
//! - **sLORETA**: `norm_k = ‖V_k @ diag(reginv × √(1 + s²/λ²))‖`
//! - **eLORETA**: iterative reweighting to achieve `R = I` (see [`eloreta`])

pub mod covariance;
pub mod eloreta;
pub mod forward;
pub mod inverse;
pub mod linalg;
pub mod resolution;
pub mod snr;
pub mod source_space;

use ndarray::{Array1, Array2};

// ── Public types ───────────────────────────────────────────────────────────

/// Choice of inverse method.
///
/// See the [module-level docs](crate) for a comparison table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseMethod {
    /// Minimum-norm estimate (no noise normalisation).
    MNE,
    /// Dynamic statistical parametric mapping (noise-normalised).
    DSPM,
    /// Standardised low-resolution electromagnetic tomography.
    SLORETA,
    /// Exact low-resolution electromagnetic tomography (iterative).
    ELORETA,
}

/// Source orientation constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceOrientation {
    /// Fixed orientation: one value per source (`n_orient = 1`).
    Fixed,
    /// Free orientation: three Cartesian components per source (`n_orient = 3`).
    Free,
}

/// Forward operator (gain matrix + metadata).
///
/// The gain matrix maps source activations to sensor measurements:
/// `x(t) = G @ j(t) + noise`.
///
/// # Construction
///
/// - From a raw gain matrix: [`ForwardOperator::new_fixed`] or [`ForwardOperator::new_free`].
/// - From electrode positions + source space: [`forward::make_sphere_forward`].
#[derive(Debug, Clone)]
pub struct ForwardOperator {
    /// Gain matrix, shape `[n_channels, n_sources × n_orient]`.
    pub gain: Array2<f64>,
    /// Source normals/orientations, shape `[n_sources × n_orient, 3]`.
    pub source_nn: Array2<f64>,
    /// Number of source locations.
    pub n_sources: usize,
    /// Orientation mode.
    pub orientation: SourceOrientation,
    /// Depth-weighting exponent (0.8 for MEG, 2–5 for EEG). `None` = no depth weighting.
    pub depth_exp: Option<f64>,
}

impl ForwardOperator {
    /// Create a fixed-orientation forward operator from a gain matrix
    /// `[n_channels, n_sources]`.
    ///
    /// Source normals are initialised to zero; set [`ForwardOperator::source_nn`]
    /// manually if you need proper normals for `PickOri::Normal`.
    pub fn new_fixed(gain: Array2<f64>) -> Self {
        let n_src = gain.ncols();
        let source_nn = Array2::zeros((n_src, 3));
        Self {
            gain,
            source_nn,
            n_sources: n_src,
            orientation: SourceOrientation::Fixed,
            depth_exp: None,
        }
    }

    /// Create a free-orientation forward operator from a gain matrix
    /// `[n_channels, n_sources × 3]`.
    ///
    /// Panics if `gain.ncols()` is not divisible by 3.
    pub fn new_free(gain: Array2<f64>) -> Self {
        let n_cols = gain.ncols();
        assert!(
            n_cols % 3 == 0,
            "Free-orientation gain must have 3N columns, got {n_cols}"
        );
        let n_src = n_cols / 3;
        let mut source_nn = Array2::zeros((n_cols, 3));
        for i in 0..n_src {
            source_nn[[i * 3, 0]] = 1.0;
            source_nn[[i * 3 + 1, 1]] = 1.0;
            source_nn[[i * 3 + 2, 2]] = 1.0;
        }
        Self {
            gain,
            source_nn,
            n_sources: n_src,
            orientation: SourceOrientation::Free,
            depth_exp: None,
        }
    }

    /// Number of orientations per source (1 for fixed, 3 for free).
    pub fn n_orient(&self) -> usize {
        match self.orientation {
            SourceOrientation::Fixed => 1,
            SourceOrientation::Free => 3,
        }
    }
}

/// Noise covariance matrix.
///
/// Can be a full `[n_channels, n_channels]` matrix or a diagonal vector.
///
/// # Construction
///
/// - Directly: [`NoiseCov::full`] or [`NoiseCov::diagonal`].
/// - From data: [`covariance::compute_covariance`] or [`covariance::compute_covariance_epochs`].
#[derive(Debug, Clone)]
pub struct NoiseCov {
    /// Full covariance matrix `[n, n]`, or a diagonal stored as `[n, 1]`.
    data: Array2<f64>,
    /// If true, `data` is `[n, 1]` (diagonal elements only).
    pub diag: bool,
}

impl NoiseCov {
    /// Create from a full covariance matrix `[n, n]`.
    pub fn full(data: Array2<f64>) -> Self {
        assert_eq!(data.nrows(), data.ncols(), "Covariance must be square");
        Self { data, diag: false }
    }

    /// Create from diagonal variances (channel noise powers).
    pub fn diagonal(variances: Vec<f64>) -> Self {
        let n = variances.len();
        let data = Array2::from_shape_vec((n, 1), variances).unwrap();
        Self { data, diag: true }
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.data.nrows()
    }

    /// Return the full covariance matrix (expanding diagonal if needed).
    pub fn to_full(&self) -> Array2<f64> {
        if self.diag {
            let n = self.data.nrows();
            let mut out = Array2::zeros((n, n));
            for i in 0..n {
                out[[i, i]] = self.data[[i, 0]];
            }
            out
        } else {
            self.data.clone()
        }
    }

    /// Return the diagonal elements (channel variances).
    pub fn diag_elements(&self) -> Array1<f64> {
        if self.diag {
            self.data.column(0).to_owned()
        } else {
            self.data.diag().to_owned()
        }
    }
}

/// Prepared inverse operator, ready for application to data.
///
/// Created by [`make_inverse_operator`] and consumed by [`apply_inverse`].
/// Contains the SVD decomposition of the whitened gain matrix plus
/// all metadata needed to reconstruct source currents.
#[derive(Debug, Clone)]
pub struct InverseOperator {
    /// Left singular vectors transposed: `U^T`, shape `[n_nzero, n_channels]`.
    pub eigen_fields: Array2<f64>,
    /// Singular values, length `n_nzero`.
    pub sing: Array1<f64>,
    /// Right singular vectors: `V`, shape `[n_sources × n_orient, n_nzero]`.
    pub eigen_leads: Array2<f64>,
    /// Source covariance diagonal (depth + orient priors), length `n_sources × n_orient`.
    pub source_cov: Array1<f64>,
    /// Whether `eigen_leads` already includes `√source_cov` (set by eLORETA).
    pub eigen_leads_weighted: bool,
    /// Number of source locations.
    pub n_sources: usize,
    /// Orientation mode.
    pub orientation: SourceOrientation,
    /// Source normals, shape `[n_sources × n_orient, 3]`.
    pub source_nn: Array2<f64>,
    /// Whitener matrix, shape `[n_nzero, n_channels]`.
    pub whitener: Array2<f64>,
    /// Number of non-zero eigenvalues in the noise covariance.
    pub n_nzero: usize,
    /// Noise covariance (retained for eLORETA computation).
    pub noise_cov: NoiseCov,
}

/// Source-space estimate produced by [`apply_inverse`].
///
/// Contains source time courses and metadata about the source space.
#[derive(Debug, Clone)]
pub struct SourceEstimate {
    /// Source time courses.
    ///
    /// Shape depends on [`PickOri`]:
    /// - `PickOri::None` / `PickOri::Normal` → `[n_sources, n_times]`
    /// - `PickOri::Vector` → `[n_sources × 3, n_times]`
    pub data: Array2<f64>,
    /// Number of source locations.
    pub n_sources: usize,
    /// Orientation mode of the inverse operator.
    pub orientation: SourceOrientation,
}

/// How to handle source orientations in free-orientation inverse solutions.
///
/// Only relevant when the inverse operator has [`SourceOrientation::Free`].
/// For fixed-orientation operators, all variants behave identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickOri {
    /// Combine XYZ components as `√(x² + y² + z²)` per source (default).
    None,
    /// Extract only the component normal to the cortical surface.
    /// Requires a free-orientation inverse operator.
    Normal,
    /// Return all three orientation components without combining.
    /// Output has shape `[n_sources × 3, n_times]`.
    Vector,
}

/// Options for the eLORETA iterative solver.
///
/// See [`inverse::apply_inverse_full`] and [`eloreta`] for details.
#[derive(Debug, Clone)]
pub struct EloretaOptions {
    /// Convergence threshold (default: `1e-6`).
    pub eps: f64,
    /// Maximum number of iterations (default: `20`).
    pub max_iter: usize,
    /// Force equal weights across XYZ orientations at each source.
    ///
    /// - `None` — automatic: `true` for fixed, `false` for free orientation.
    /// - `Some(true)` — uniform weights (like dSPM/sLORETA), recommended for loose orientation.
    /// - `Some(false)` — independent 3×3 weights per source (reference eLORETA).
    pub force_equal: Option<bool>,
}

impl Default for EloretaOptions {
    fn default() -> Self {
        Self {
            eps: 1e-6,
            max_iter: 20,
            force_equal: None,
        }
    }
}

// ── Public API re-exports ──────────────────────────────────────────────────

// Inverse operator
pub use inverse::{
    apply_inverse, apply_inverse_epochs, apply_inverse_epochs_full, apply_inverse_full,
    apply_inverse_with_options, make_inverse_operator, prepare_inverse,
};

// Covariance estimation
pub use covariance::{compute_covariance, compute_covariance_epochs, Regularization};

// Resolution analysis
pub use resolution::{
    get_cross_talk, get_point_spread, make_resolution_matrix, peak_localisation_error,
    relative_amplitude, spatial_spread,
};

// SNR estimation
pub use snr::estimate_snr;

// Forward model
pub use forward::{make_sphere_forward, make_sphere_forward_free, SphereModel};

// Source space
pub use source_space::{grid_source_space, ico_n_vertices, ico_source_space};
