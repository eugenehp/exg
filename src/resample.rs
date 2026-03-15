//! FFT-based rational resampler exactly matching MNE's `resample(..., method='fft')`.
//!
//! Algorithm (from `mne/cuda.py _fft_resample`):
//!   1. Pad with npad=100 reflect-limited samples on each side.
//!   2. rfft(padded)  →  complex half-spectrum.
//!   3. If downsampling: double the Nyquist bin (use_len = new_len).
//!      If upsampling:   halve  the Nyquist bin (use_len = old_len).
//!   4. Scale all bins by `new_len_padded / old_len_padded`.
//!   5. irfft(spectrum, n=new_len_padded)  — irfft handles zero-padding or
//!      truncation of the spectrum automatically.
//!   6. Strip the resampled padding edges.
use anyhow::Result;
use ndarray::Array2;
use rustfft::FftPlanner;

/// Compute the auto npad as MNE does: pad to the next power of 2.
///
/// ```text
/// min_add = min(n // 8, 100) * 2
/// total   = 2^ceil(log2(n + min_add)) - n
/// npads   = [total // 2, total - total // 2]
/// ```
pub fn auto_npad(n: usize) -> (usize, usize) {
    let min_add = (n / 8).min(100) * 2;
    let sum = n + min_add;
    let next_pow2 = 1usize << ((sum as f64).log2().ceil() as u32);
    let total = next_pow2 - n;
    (total / 2, total - total / 2)
}

/// Resample `data` ([C, T]) from `src_sfreq` to `dst_sfreq`.
pub fn resample(data: &Array2<f32>, src_sfreq: f32, dst_sfreq: f32) -> Result<Array2<f32>> {
    if (src_sfreq - dst_sfreq).abs() < 1e-6 {
        return Ok(data.clone());
    }
    let ratio = dst_sfreq as f64 / src_sfreq as f64;
    let n_in = data.ncols();
    let final_len = (ratio * n_in as f64).round() as usize;
    let n_ch = data.nrows();

    let (npad_l, npad_r) = auto_npad(n_in);
    let mut out = Array2::<f32>::zeros((n_ch, final_len));
    for ch in 0..n_ch {
        let row: Vec<f32> = data.row(ch).to_vec();
        let resampled = resample_1d(&row, ratio, npad_l, npad_r)?;
        out.row_mut(ch).assign(&ndarray::ArrayView1::from(&resampled));
    }
    Ok(out)
}

/// Resample a single 1-D f32 signal with explicit (possibly asymmetric) padding.
pub fn resample_1d(x: &[f32], ratio: f64, npad_l: usize, npad_r: usize) -> Result<Vec<f32>> {
    let n_in = x.len();
    if n_in == 0 {
        return Ok(vec![]);
    }
    let final_len = (ratio * n_in as f64).round() as usize;

    // --- 1. Reflect-limited padding (matches MNE's _smart_pad) ----------
    let pad_l = npad_l.min(n_in - 1);
    let pad_r = npad_r.min(n_in - 1);
    let old_len = n_in + pad_l + pad_r;
    // Note: if npad > n_in-1, MNE zero-pads the extra. We clamp to n_in-1.

    let mut x_ext = Vec::with_capacity(old_len);
    for i in (1..=pad_l).rev() {
        x_ext.push(2.0 * x[0] - x[i]);
    }
    x_ext.extend_from_slice(x);
    let last = x[n_in - 1];
    for i in 1..=pad_r {
        let idx = (n_in - 1).saturating_sub(i);
        x_ext.push(2.0 * last - x[idx]);
    }

    // --- 2. Compute padded output length ---------------------------------
    let new_len_padded = (ratio * old_len as f64).round() as usize;
    let shorter = new_len_padded < old_len;
    let use_len = if shorter { new_len_padded } else { old_len };

    // --- 3. rfft of padded signal ----------------------------------------
    // MNE uses scipy.fft.rfft which returns (n//2 + 1) complex coefficients.
    // We simulate rfft with a full FFT and take the first half.
    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_forward(old_len);
    let mut buf: Vec<rustfft::num_complex::Complex<f64>> = x_ext
        .iter()
        .map(|&v| rustfft::num_complex::Complex { re: v as f64, im: 0.0 })
        .collect();
    fft.process(&mut buf);

    let rfft_len = old_len / 2 + 1;
    let mut x_fft: Vec<rustfft::num_complex::Complex<f64>> = buf[..rfft_len].to_vec();

    // --- 4. Handle Nyquist bin -------------------------------------------
    // MNE: if use_len % 2 == 0:
    //          nyq = use_len // 2
    //          x_fft[nyq] *= 2 if shorter else 0.5
    if use_len % 2 == 0 {
        let nyq = use_len / 2;
        if nyq < x_fft.len() {
            let factor = if shorter { 2.0 } else { 0.5 };
            x_fft[nyq] *= factor;
        }
    }

    // --- 5. Scale by new_len_padded / old_len_padded ---------------------
    // (This is what MNE's boxcar window does: W = scale * ones)
    let scale = new_len_padded as f64 / old_len as f64;
    for v in &mut x_fft {
        *v *= scale;
    }

    // --- 6. irfft(x_fft, n=new_len_padded) --------------------------------
    // irfft with n=new_len_padded:
    //   - if new_len_padded < old_len (downsampling): takes only x_fft[0..new_rfft_len],
    //     truncating high frequencies.
    //   - if new_len_padded > old_len (upsampling): zero-pads the spectrum.
    let new_rfft_len = new_len_padded / 2 + 1;
    let mut irfft_in = vec![rustfft::num_complex::Complex::<f64>::default(); new_len_padded];

    // Copy available spectrum (truncate or zero-pad).
    let n_copy = x_fft.len().min(new_rfft_len);
    irfft_in[..n_copy].copy_from_slice(&x_fft[..n_copy]);

    // Reconstruct full spectrum from half-spectrum (Hermitian symmetry).
    // irfft_in[0..new_rfft_len] is already set.
    // irfft_in[new_rfft_len..] = conj(irfft_in[new_len_padded - i]) for i in 1..
    for i in 1..new_rfft_len {
        let idx = new_len_padded - i;
        if idx < new_len_padded && idx >= new_rfft_len {
            irfft_in[idx] = irfft_in[i].conj();
        }
    }

    let ifft = planner.plan_fft_inverse(new_len_padded);
    ifft.process(&mut irfft_in);
    let inv_scale = 1.0 / new_len_padded as f64;

    // --- 7. Strip padding ------------------------------------------------
    let to_remove_l = (ratio * npad_l as f64).round() as usize;
    let to_remove_r = new_len_padded - final_len - to_remove_l;
    let strip_end = new_len_padded.saturating_sub(to_remove_r);

    let mut result: Vec<f32> = irfft_in[to_remove_l..strip_end]
        .iter()
        .map(|c| (c.re * inv_scale) as f32)
        .collect();
    result.resize(final_len, 0.0);
    Ok(result)
}

/// Compute `(up, down)` from dst/src via GCD reduction (for length checks).
pub fn rational_approx(dst: f32, src: f32) -> (usize, usize) {
    let scale = 1000usize;
    let up0 = (dst * scale as f32).round() as usize;
    let down0 = (src * scale as f32).round() as usize;
    let g = gcd(up0, down0);
    (up0 / g, down0 / g)
}

/// Exact output length: round(n * dst / src).
pub fn final_length(n: usize, up: usize, down: usize) -> usize {
    (n as f64 * up as f64 / down as f64).round() as usize
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_noop_passthrough() {
        let data = Array2::from_shape_fn((2, 512), |(_, t)| t as f32 / 512.0);
        let out = resample(&data, 256.0, 256.0).unwrap();
        assert_eq!(out.shape(), data.shape());
    }

    #[test]
    fn resample_half_rate_length() {
        let data = Array2::zeros((1, 1024));
        let out = resample(&data, 512.0, 256.0).unwrap();
        assert_eq!(out.ncols(), 512);
    }

    #[test]
    fn resample_preserves_dc() {
        let data = Array2::from_elem((1, 1024), 3.14_f32);
        let out = resample(&data, 512.0, 256.0).unwrap();
        for &v in out.iter() {
            approx::assert_abs_diff_eq!(v, 3.14, epsilon = 1e-2);
        }
    }

    #[test]
    fn rational_approx_integer_ratio() {
        assert_eq!(rational_approx(256.0, 512.0),  (1, 2));
        assert_eq!(rational_approx(256.0, 1024.0), (1, 4));
        assert_eq!(rational_approx(256.0, 2048.0), (1, 8));
    }

    #[test]
    fn auto_npad_correct() {
        // 512 Hz, 30s = 15360 samples → npads = [512, 512]
        assert_eq!(auto_npad(15360), (512, 512));
        // 1024 Hz, 30s = 30720 → npads = [1024, 1024]
        assert_eq!(auto_npad(30720), (1024, 1024));
        // 512 Hz, 60s = 30720 → npads = [1024, 1024]
        assert_eq!(auto_npad(30720), (1024, 1024));
    }
}
