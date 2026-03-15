//! Average reference: subtract the mean across channels at each time point.
//!
//! Matches `raw.set_eeg_reference('average', projection=False)`.
//!
//! `data`: [C, T]  →  `data[c, t] -= mean(data[:, t])`
use ndarray::{Array2, Axis};

pub fn average_reference_inplace(data: &mut Array2<f32>) {
    let means = data.mean_axis(Axis(0)).unwrap(); // shape [T]
    for mut row in data.rows_mut() {
        row -= &means;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn channel_sum_is_zero_after_reference() {
        let mut data = Array2::from_shape_fn((8, 512), |(c, t)| {
            ((c * 7 + t * 3) as f32).sin()
        });
        average_reference_inplace(&mut data);
        // Column sums should be ≈ 0.
        let col_sums = data.sum_axis(Axis(0));
        for &s in col_sums.iter() {
            approx::assert_abs_diff_eq!(s, 0.0, epsilon = 1e-4_f32);
        }
    }

    #[test]
    fn reference_of_constant_gives_zero() {
        let mut data = Array2::from_elem((4, 100), 5.0_f32);
        average_reference_inplace(&mut data);
        for &v in data.iter() {
            approx::assert_abs_diff_eq!(v, 0.0, epsilon = 1e-6_f32);
        }
    }

    #[test]
    fn reference_preserves_channel_differences() {
        // x[0] = 2, x[1] = 4 → mean = 3 → x[0]-x[1] is preserved.
        let mut data = Array2::from_shape_fn((2, 10), |(c, _)| if c == 0 { 2.0_f32 } else { 4.0 });
        average_reference_inplace(&mut data);
        for t in 0..10 {
            approx::assert_abs_diff_eq!(
                data[[0, t]] - data[[1, t]],
                -2.0_f32,
                epsilon = 1e-6
            );
        }
    }
}
