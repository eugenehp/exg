mod common;
use common::{load_vectors, max_abs_diff};
use exg::reference::average_reference_inplace;
use ndarray::{Array2, Axis};

#[test]
fn average_reference_matches_mne() {
    let vecs = load_vectors("average_reference");
    let x_arr = vecs.get("input").expect("missing 'input'");
    let y_ref  = vecs.get("output").expect("missing 'output'");

    let shape = x_arr.shape();
    let mut data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    average_reference_inplace(&mut data);

    let got_dyn = data.clone().into_dyn();
    let max_err = max_abs_diff(&got_dyn, y_ref);
    assert!(max_err < 1e-5,
        "max abs error {max_err:.2e} >= 1e-5");
}

#[test]
fn average_reference_zero_column_sum() {
    let vecs = load_vectors("average_reference");
    let x_arr = vecs.get("input").unwrap();
    let shape = x_arr.shape();
    let mut data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();
    average_reference_inplace(&mut data);
    let col_sums = data.sum_axis(Axis(0));
    for (t, &s) in col_sums.iter().enumerate() {
        assert!(s.abs() < 1e-3,
            "column {t} sum = {s:.2e} after average reference");
    }
}
