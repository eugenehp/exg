mod common;
use common::load_vectors;
use exg::epoch::epoch_and_baseline;
use ndarray::Array2;

#[test]
fn epoch_count_matches_mne() {
    let vecs = load_vectors("epoch_1280");
    let x_arr = vecs.get("input").expect("missing 'input'");
    let y_ref  = vecs.get("epochs").expect("missing 'epochs'");
    let n_ep_expected = y_ref.shape()[0];

    let shape = x_arr.shape();
    let data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    let epochs = epoch_and_baseline(&data, 1280);
    assert_eq!(epochs.len(), n_ep_expected,
        "epoch count: Rust={} MNE={n_ep_expected}", epochs.len());
}

#[test]
fn epoch_shapes_are_correct() {
    let vecs = load_vectors("epoch_1280");
    let x_arr = vecs.get("input").unwrap();
    let y_ref  = vecs.get("epochs").unwrap();
    let (n_ep, n_ch, n_t) = (y_ref.shape()[0], y_ref.shape()[1], y_ref.shape()[2]);

    let shape = x_arr.shape();
    let data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    let epochs = epoch_and_baseline(&data, 1280);
    for (i, ep) in epochs.iter().enumerate() {
        assert_eq!(ep.nrows(), n_ch, "epoch {i}: nrows");
        assert_eq!(ep.ncols(), n_t,  "epoch {i}: ncols");
    }
    let _ = n_ep; // already checked above
}

#[test]
fn epoch_values_match_mne() {
    let vecs = load_vectors("epoch_1280");
    let x_arr = vecs.get("input").unwrap();
    let y_ref  = vecs.get("epochs").unwrap();   // [E, C, T] f32

    let shape = x_arr.shape();
    let data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();

    let epochs = epoch_and_baseline(&data, 1280);
    let n_ep = epochs.len();
    let n_ch = epochs[0].nrows();
    let n_t  = epochs[0].ncols();

    for e in 0..n_ep {
        for c in 0..n_ch {
            for t in 0..n_t {
                let got = epochs[e][[c, t]];
                let exp = y_ref[[e, c, t]];
                let err = (got - exp).abs();
                assert!(err < 1e-4,
                    "epoch={e} ch={c} t={t}: got={got:.6} expected={exp:.6} err={err:.2e}");
            }
        }
    }
}

#[test]
fn epoch_baseline_mean_is_zero() {
    let vecs = load_vectors("epoch_1280");
    let x_arr = vecs.get("input").unwrap();
    let shape = x_arr.shape();
    let data = Array2::from_shape_vec(
        (shape[0], shape[1]),
        x_arr.iter().cloned().collect(),
    ).unwrap();
    let epochs = epoch_and_baseline(&data, 1280);
    for (e, ep) in epochs.iter().enumerate() {
        for c in 0..ep.nrows() {
            let m = ep.row(c).mean().unwrap();
            assert!(m.abs() < 1e-3,
                "epoch={e} ch={c} mean={m:.2e} after baseline");
        }
    }
}
