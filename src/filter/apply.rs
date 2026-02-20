//! Overlap-add zero-phase FIR convolution.
//!
//! Matches MNE's `_overlap_add_filter` + `_1d_overlap_filter`.
//!
//! Zero-phase is achieved by shifting the output left by `(N-1)/2` samples,
//! NOT by running filtfilt. The edge transient is suppressed by
//! reflect-limited padding of `N-1` samples on each side.
use anyhow::Result;
use ndarray::Array2;
use rustfft::{FftPlanner, num_complex::Complex};

/// Apply a zero-phase FIR filter to each channel of `data` ([C, T]) in-place.
///
/// `h` must have odd length (guaranteed by `design_highpass`).
pub fn apply_fir_zero_phase(data: &mut Array2<f32>, h: &[f32]) -> Result<()> {
    let n_ch = data.nrows();
    for ch in 0..n_ch {
        let row: Vec<f32> = data.row(ch).to_vec();
        let filtered = filter_1d(&row, h)?;
        data.row_mut(ch).assign(&ndarray::ArrayView1::from(&filtered));
    }
    Ok(())
}

/// Filter a single 1-D signal with the overlap-add algorithm.
///
/// Returns a vector of the same length as `x`.
pub fn filter_1d(x: &[f32], h: &[f32]) -> Result<Vec<f32>> {
    let n_x = x.len();
    let n_h = h.len();

    if n_x == 0 {
        return Ok(vec![]);
    }

    // Shift for zero-phase: (N-1)/2  (N must be odd).
    let shift = (n_h - 1) / 2;
    // Edge padding (reflect-limited).
    let n_edge = n_h - 1;

    // Build padded signal.
    let x_ext = reflect_limited_pad(x, n_edge, n_edge);
    let n_ext = x_ext.len();

    // Choose FFT block size.
    let n_fft = choose_fft_len(n_h, n_ext);

    // Precompute FFT of h (zero-padded to n_fft).
    let h_fft = fft_of_h(h, n_fft);

    // Overlap-add.
    let n_seg = n_fft - n_h + 1;
    let n_segments = n_ext.div_ceil(n_seg);
    let mut x_filtered = vec![0.0_f32; n_ext];

    let mut planner: FftPlanner<f32> = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    let fft_inv = planner.plan_fft_inverse(n_fft);
    let inv_scale = 1.0 / n_fft as f32;

    for seg_idx in 0..n_segments {
        let start = seg_idx * n_seg;
        let stop  = (start + n_seg).min(n_ext);

        // Zero-pad segment to n_fft.
        let mut buf: Vec<Complex<f32>> = x_ext[start..stop]
            .iter()
            .map(|&v| Complex { re: v, im: 0.0 })
            .chain(std::iter::repeat(Complex::default()))
            .take(n_fft)
            .collect();

        fft_fwd.process(&mut buf);

        // Multiply with H.
        for (b, &hf) in buf.iter_mut().zip(h_fft.iter()) {
            *b *= hf;
        }

        fft_inv.process(&mut buf);

        // Accumulate with overlap-add (accounting for zero-phase shift).
        let out_start = start.saturating_sub(shift);
        let out_end   = (out_start + n_fft).min(n_ext);
        let prod_start = if start < shift { shift - start } else { 0 };

        for (o, p) in (out_start..out_end).zip(prod_start..) {
            if p < buf.len() {
                x_filtered[o] += buf[p].re * inv_scale;
            }
        }
    }

    // Strip edge padding.
    let result: Vec<f32> = x_filtered[n_edge..n_edge + n_x].to_vec();
    Ok(result)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Reflect-limited padding (matches MNE's `_smart_pad`).
///
/// Left:  `pad[i] = 2*x[0] - x[n_l-i]`  for i in 1..=n_l
/// Right: `pad[i] = 2*x[-1] - x[-(i+1)]` for i in 1..=n_r
fn reflect_limited_pad(x: &[f32], n_l: usize, n_r: usize) -> Vec<f32> {
    let n = x.len();
    let actual_l = n_l.min(n - 1);
    let actual_r = n_r.min(n - 1);

    let mut out = Vec::with_capacity(actual_l + n + actual_r);

    // Left padding (reversed, odd reflection around x[0]).
    for i in (1..=actual_l).rev() {
        out.push(2.0 * x[0] - x[i]);
    }
    // If requested padding exceeds signal, prepend zeros.
    for _ in actual_l..n_l {
        out.insert(0, 0.0);
    }

    out.extend_from_slice(x);

    // Right padding (odd reflection around x[-1]).
    let last = x[n - 1];
    for i in 1..=actual_r {
        let idx = (n - 1).saturating_sub(i);
        out.push(2.0 * last - x[idx]);
    }
    // If requested padding exceeds signal, append zeros.
    for _ in actual_r..n_r {
        out.push(0.0);
    }

    out
}

/// Choose the optimal FFT block size (power of 2 minimising operation count).
///
/// Matches MNE's cost function:
///   `cost = ceil(n_x / (N - n_h + 1)) * N * (log2(N) + 1) + 4e-5 * N * n_x`
fn choose_fft_len(n_h: usize, n_x: usize) -> usize {
    let min_fft = 2 * n_h - 1;

    // Upper bound: next power of 2 above n_x.
    let max_pow = (n_x as f64).log2().ceil() as u32 + 1;
    let min_pow = (min_fft as f64).log2().ceil() as u32;

    let mut best_n = 1_usize << max_pow;
    let mut best_cost = f64::INFINITY;

    for pow in min_pow..=max_pow {
        let n = 1_usize << pow;
        if n < min_fft { continue; }
        let n_seg = (n - n_h + 1) as f64;
        let cost = (n_x as f64 / n_seg).ceil() * n as f64 * (pow as f64 + 1.0)
            + 4e-5 * n as f64 * n_x as f64;
        if cost < best_cost {
            best_cost = cost;
            best_n = n;
        }
    }
    best_n
}

/// Compute the FFT of `h` zero-padded to `n_fft`.
fn fft_of_h(h: &[f32], n_fft: usize) -> Vec<Complex<f32>> {
    let mut buf: Vec<Complex<f32>> = h
        .iter()
        .map(|&v| Complex { re: v, im: 0.0 })
        .chain(std::iter::repeat(Complex::default()))
        .take(n_fft)
        .collect();
    let mut planner: FftPlanner<f32> = FftPlanner::new();
    planner.plan_fft_forward(n_fft).process(&mut buf);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::design::design_highpass;

    #[test]
    fn filter_preserves_length() {
        let x: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0).sin()).collect();
        let h = design_highpass(0.5, 256.0);
        let y = filter_1d(&x, &h).unwrap();
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn filter_removes_dc() {
        // A constant signal should become zero after highpass filtering.
        let x = vec![1.0_f32; 4096];
        let h = design_highpass(0.5, 256.0);
        let y = filter_1d(&x, &h).unwrap();
        // Skip edges (transient region = filter length).
        let n_h = h.len();
        let interior = &y[n_h..y.len() - n_h];
        let max_val: f32 = interior.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(max_val < 1e-3, "DC not removed: max={max_val}");
    }

    #[test]
    fn reflect_limited_left_pad() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_limited_pad(&x, 3, 0);
        // left pad: 2*1 - x[3]=4 → -2, 2*1 - x[2]=3 → -1, 2*1 - x[1]=2 → 0
        assert_eq!(&padded[..3], &[-2.0_f32, -1.0, 0.0]);
        assert_eq!(&padded[3..], &x[..]);
    }
}
