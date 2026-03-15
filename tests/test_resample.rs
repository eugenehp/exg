mod common;
use common::{load_vectors, max_abs_diff, array_std};
use exg::resample::resample;
use ndarray::Array2;

fn run_resample_test(vec_name: &str, abs_tol: f32, rel_tol_pct: f32) {
    let vecs = load_vectors(vec_name);
    let x_arr = vecs.get("input").expect("missing 'input'");
    let y_ref  = vecs.get("output").expect("missing 'output'");
    let src    = vecs.get("src_sfreq").unwrap()[[0]] as f32;
    let dst    = vecs.get("dst_sfreq").unwrap()[[0]] as f32;

    let shape = x_arr.shape();
    let x = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    let got = resample(&x, src, dst).unwrap();

    // Shape check.
    let ref_shape = y_ref.shape();
    assert_eq!(got.nrows(), ref_shape[0], "channel count mismatch");
    assert_eq!(got.ncols(), ref_shape[1],
        "sample count: got={} expected={}", got.ncols(), ref_shape[1]);

    let got_dyn = got.into_dyn();
    let max_err = max_abs_diff(&got_dyn, y_ref);
    let sigma   = array_std(y_ref);

    assert!(max_err < abs_tol,
        "[{vec_name}] max abs error {max_err:.2e} >= {abs_tol:.2e}");
    assert!(max_err / sigma * 100.0 < rel_tol_pct,
        "[{vec_name}] relative error {:.4}% >= {rel_tol_pct}%",
        max_err / sigma * 100.0);
}

#[test]
fn resample_512_to_256() {
    run_resample_test("resample_512_to_256", 5e-4, 0.1);
}

#[test]
fn resample_1024_to_256() {
    run_resample_test("resample_1024_to_256", 5e-4, 0.1);
}

#[test]
fn resample_250_to_256() {
    // Fractional ratio — harder; MNE uses up=128, down=125.
    run_resample_test("resample_250_to_256", 2e-3, 0.2);
}

#[test]
fn resample_2000_to_256() {
    run_resample_test("resample_2000_to_256", 1e-3, 0.2);
}

#[test]
fn resample_noop_exact() {
    // Same-rate resample should be identity.
    let vecs = load_vectors("resample_512_to_256");
    let x_arr = vecs.get("input").unwrap();
    let shape = x_arr.shape();
    let x = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();
    let got = resample(&x, 512.0, 512.0).unwrap();
    assert_eq!(got.shape(), x.shape());
    for (a, b) in got.iter().zip(x.iter()) {
        approx::assert_abs_diff_eq!(a, b, epsilon = 1e-7_f32);
    }
}

#[test]
fn resample_output_length_correct() {
    // Verify length formula: round(n_in * dst / src).
    for (src, dst, n_in) in [
        (512.0_f32, 256.0, 15360_usize),
        (1024.0, 256.0, 30720),
        (250.0, 256.0, 7500),
        (2000.0, 256.0, 60000),
    ] {
        let expected = (n_in as f64 * dst as f64 / src as f64).round() as usize;
        let x = Array2::zeros((1, n_in));
        let got = resample(&x, src, dst).unwrap();
        assert_eq!(got.ncols(), expected,
            "src={src} dst={dst} n={n_in}: got={} expected={expected}", got.ncols());
    }
}
