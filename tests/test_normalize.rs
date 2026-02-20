mod common;
use common::{load_vectors, max_abs_diff};
use exg::normalize::{zscore_global_inplace, baseline_correct_inplace};
use ndarray::{Array2, Array3};

#[test]
fn zscore_matches_numpy() {
    let vecs = load_vectors("zscore_global");
    let x_arr   = vecs.get("input").unwrap();
    let y_ref    = vecs.get("output").unwrap();
    let mean_ref = vecs.get("mean").unwrap()[[0]];
    let std_ref  = vecs.get("std").unwrap()[[0]];

    let shape = x_arr.shape();
    let mut data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    let (mean, std) = zscore_global_inplace(&mut data);

    approx::assert_abs_diff_eq!(mean, mean_ref, epsilon = 1e-4_f32);
    approx::assert_abs_diff_eq!(std,  std_ref,  epsilon = 1e-4_f32);

    let got_dyn = data.into_dyn();
    let max_err = max_abs_diff(&got_dyn, y_ref);
    assert!(max_err < 1e-5, "max abs error {max_err:.2e} >= 1e-5");
}

#[test]
fn zscore_postconditions() {
    let vecs = load_vectors("zscore_global");
    let x_arr = vecs.get("input").unwrap();
    let shape = x_arr.shape();
    let mut data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();
    zscore_global_inplace(&mut data);

    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let std: f32 = (data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();

    assert!(mean.abs() < 1e-4, "post-zscore mean = {mean:.2e}");
    approx::assert_abs_diff_eq!(std, 1.0, epsilon = 1e-3_f32);
}

#[test]
fn baseline_correction_matches_mne() {
    let vecs = load_vectors("baseline_correction");
    let x_arr = vecs.get("input").expect("missing 'input'");
    let y_ref  = vecs.get("output").expect("missing 'output'");

    let s = x_arr.shape();
    let (n_e, n_c, n_t) = (s[0], s[1], s[2]);
    let mut epochs = Array3::from_shape_vec(
        (n_e, n_c, n_t),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    baseline_correct_inplace(&mut epochs);

    let got_dyn = epochs.into_dyn();
    let max_err = max_abs_diff(&got_dyn, y_ref);
    assert!(max_err < 1e-5, "max abs error {max_err:.2e} >= 1e-5");
}

#[test]
fn baseline_per_channel_mean_is_zero() {
    let vecs = load_vectors("baseline_correction");
    let x_arr = vecs.get("input").unwrap();
    let s = x_arr.shape();
    let mut epochs = Array3::from_shape_vec(
        (s[0], s[1], s[2]),
        x_arr.iter().cloned().collect(),
    ).unwrap();
    baseline_correct_inplace(&mut epochs);

    for e in 0..s[0] {
        for c in 0..s[1] {
            let m = epochs.slice(ndarray::s![e, c, ..]).mean().unwrap();
            assert!(m.abs() < 1e-4,
                "epoch={e} ch={c} mean={m:.2e} after baseline correction");
        }
    }
}
