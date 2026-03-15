mod common;
use common::{load_vectors, max_abs_diff, array_std};
use exg::filter::{design_highpass, apply_fir_zero_phase};
use ndarray::Array2;

// ── Coefficient tests ─────────────────────────────────────────────────────────

#[test]
fn filter_coeffs_match_scipy() {
    let vecs = load_vectors("filter_coeffs_hp05_256hz");
    let h_ref = vecs.get("h").expect("missing 'h'");

    let h_rust = design_highpass(0.5, 256.0);

    assert_eq!(h_rust.len(), h_ref.len(),
        "tap count mismatch: Rust={} SciPy={}", h_rust.len(), h_ref.len());

    let h_rust_arr = ndarray::Array::from_vec(h_rust.clone())
        .into_dyn();
    let max_err = max_abs_diff(&h_rust_arr, h_ref);
    assert!(max_err < 1e-7,
        "max coeff error {max_err:.2e} >= 1e-7  \
         (hint: check firwin cutoff or Hamming window formula)");
}

#[test]
fn filter_coeffs_sum_near_zero() {
    // Highpass: sum of coefficients ≈ 0 (zero DC gain).
    let h = design_highpass(0.5, 256.0);
    let s: f32 = h.iter().sum();
    assert!(s.abs() < 1e-5, "sum(h) = {s:.2e}, expected ≈ 0 for highpass");
}

#[test]
fn filter_coeffs_symmetric() {
    let h = design_highpass(0.5, 256.0);
    let n = h.len();
    for i in 0..n / 2 {
        let diff = (h[i] - h[n - 1 - i]).abs();
        assert!(diff < 1e-7, "h[{i}]={} ≠ h[{}]={}", h[i], n-1-i, h[n-1-i]);
    }
}

// ── Application tests ─────────────────────────────────────────────────────────

#[test]
fn filter_application_matches_mne() {
    let vecs = load_vectors("filter_hp05_256hz");
    let x_arr = vecs.get("input").expect("missing 'input'");
    let y_ref  = vecs.get("output").expect("missing 'output'");

    let shape = x_arr.shape();
    let x = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    let h = design_highpass(0.5, 256.0);
    let mut data = x.clone();
    apply_fir_zero_phase(&mut data, &h).unwrap();

    let data_dyn = data.clone().into_dyn();
    let max_err = max_abs_diff(&data_dyn, y_ref);
    let sigma   = array_std(y_ref);

    assert!(max_err < 1e-4,
        "max filter error {max_err:.2e} >= 1e-4");
    assert!(max_err / sigma < 0.0001,
        "relative error {:.4}% >= 0.01%  (sigma={sigma:.4})",
        max_err / sigma * 100.0);
}

#[test]
fn filter_removes_sub_hz_content() {
    // After highpass at 0.5 Hz, a 0.1 Hz sine should be heavily attenuated.
    let sfreq = 256.0_f32;
    let n = 60 * 256;  // 60 seconds
    let t: Vec<f32> = (0..n).map(|i| i as f32 / sfreq).collect();

    // 0.1 Hz (stop band) + 5 Hz (pass band)
    let row: Vec<f32> = t.iter().map(|&ti| {
        (2.0 * std::f32::consts::PI * 0.1 * ti).sin()
        + (2.0 * std::f32::consts::PI * 5.0 * ti).sin()
    }).collect();
    let mut data = Array2::from_shape_vec((1, n), row.clone()).unwrap();
    let h = design_highpass(0.5, sfreq);
    apply_fir_zero_phase(&mut data, &h).unwrap();

    // RMS power in the 5 Hz component should dominate.
    // Rough check: variance of filtered signal should be close to 0.5 (5 Hz has amp 1).
    let filtered: Vec<f32> = data.row(0).to_vec();
    // Skip edges (transient region).
    let guard = h.len();
    let interior = &filtered[guard..filtered.len() - guard];
    let rms: f32 = (interior.iter().map(|v| v * v).sum::<f32>() / interior.len() as f32).sqrt();
    // Pure 5 Hz sine has RMS = 1/sqrt(2) ≈ 0.707; 0.1 Hz mostly removed.
    assert!(rms > 0.5, "RMS too low ({rms:.3}), pass-band signal attenuated?");
    assert!(rms < 0.85, "RMS too high ({rms:.3}), stop-band not attenuated?");
}
