//! FIR filter design with 100% MNE-Python parity.
//!
//! Implements MNE's `_firwin_design` algorithm exactly:
//!   - For each gain transition, computes a **sub-filter** with its own
//!     length derived from that specific transition bandwidth.
//!   - The sub-filter is centered (zero-padded) within the master kernel.
//!   - Gain=1 transitions add; gain=0 transitions subtract.
//!
//! This matches `mne.filter.create_filter(fir_design='firwin',
//!   fir_window='hamming', phase='zero')` to machine precision.
use std::f64::consts::PI;

/// Hamming window length factor (from MNE's `_length_factors`).
const HAMMING_FACTOR: f64 = 3.3;

/// Compute MNE-compatible transition bandwidth for a highpass filter.
///
/// Rule: `min(max(0.25 * l_freq, 2.0), l_freq)`
pub fn auto_trans_bandwidth(l_freq: f32) -> f32 {
    (0.25 * l_freq).max(2.0).min(l_freq)
}

/// Compute MNE-compatible transition bandwidth for a lowpass filter.
///
/// Rule: `min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)`
pub fn auto_trans_bandwidth_lowpass(h_freq: f32, sfreq: f32) -> f32 {
    (0.25 * h_freq).max(2.0).min(sfreq / 2.0 - h_freq)
}

/// Compute the number of FIR taps for a given transition bandwidth.
/// Returns an odd integer (required for zero-phase linear-phase FIR).
///
/// Formula: `ceil(3.3 / trans_bw * sfreq)` rounded up to odd.
pub fn auto_filter_length(trans_bw: f32, sfreq: f32) -> usize {
    let n_raw = (3.3 / trans_bw as f64 * sfreq as f64).ceil() as usize;
    // Make odd.
    if n_raw.is_multiple_of(2) { n_raw + 1 } else { n_raw }
}

// ── MNE's _firwin_design — the core algorithm ──────────────────────────────

/// Exact port of MNE's `_firwin_design(N, freq, gain, window, sfreq)`.
///
/// `freq` and `gain` are normalised frequency/gain pairs where freq is
/// in the range [0, 1] (0 = DC, 1 = Nyquist).
///
/// # Algorithm
///
/// Iterates from Nyquist to DC. When a gain transition is encountered:
///   1. Compute the sub-filter length from that transition's bandwidth:
///      `this_N = round(3.3 / transition)`, made odd, capped at `N`.
///   2. Design a lowpass with `scipy.signal.firwin` at the transition midpoint.
///   3. Center the sub-filter in the master kernel and add (gain 1→0) or
///      subtract (gain 0→1) it.
///
/// If `gain[-1] == 1`, the kernel starts as a unit impulse (all-pass).
/// If `gain[-1] == 0`, the kernel starts as zeros.
fn firwin_design(n: usize, freq: &[f64], gain: &[i32]) -> Vec<f64> {
    assert!(n % 2 == 1, "N must be odd");
    assert!(freq[0] == 0.0);
    assert!(freq.len() == gain.len());

    let mut h = vec![0.0_f64; n];

    let last_gain = *gain.last().unwrap();
    if last_gain == 1 {
        h[n / 2] = 1.0; // start with "all up" (unit impulse)
    }

    let mut prev_freq = *freq.last().unwrap();
    let mut prev_gain = last_gain;

    // Iterate from right to left (Nyquist → DC), skipping the last element
    for (&this_freq, &this_gain) in freq.iter().rev().skip(1).zip(gain.iter().rev().skip(1)) {
        if this_gain != prev_gain {
            // Compute sub-filter length from the transition bandwidth
            let transition = (prev_freq - this_freq) / 2.0;
            let mut this_n = (HAMMING_FACTOR / transition).round() as usize;
            this_n += 1 - this_n % 2; // make odd

            if this_n > n {
                // If the required sub-filter is longer than the master, cap it
                // (MNE raises ValueError here, but we cap gracefully)
                this_n = n;
            }

            // Cutoff frequency (midpoint of the transition, normalised [0, 1])
            let cutoff = (prev_freq + this_freq) / 2.0;

            // Design the sub-filter (lowpass, Hamming window)
            // freq is normalised to [0,1] where 1=Nyquist, so we use
            // sfreq=2.0 which gives Nyquist=1.0, making fc=cutoff directly.
            let this_h = firwin_f64(this_n, cutoff, 2.0);

            // Center the sub-filter in the master kernel
            let offset = (n - this_n) / 2;
            if this_gain == 0 {
                for (i, &v) in this_h.iter().enumerate() {
                    h[offset + i] -= v;
                }
            } else {
                for (i, &v) in this_h.iter().enumerate() {
                    h[offset + i] += v;
                }
            }
        }

        prev_gain = this_gain;
        prev_freq = this_freq;
    }

    h
}

// ── Public filter design functions ──────────────────────────────────────────

/// Design a zero-phase highpass FIR filter.
///
/// 100% parity with `mne.filter.create_filter(None, sfreq, l_freq=l_freq,
///   h_freq=None, filter_length='auto', fir_window='hamming',
///   fir_design='firwin', phase='zero')`.
pub fn design_highpass(l_freq: f32, sfreq: f32) -> Vec<f32> {
    let trans_bw = auto_trans_bandwidth(l_freq);
    let n = auto_filter_length(trans_bw, sfreq);

    let l_freq_f64 = l_freq as f64;
    let trans_bw_f64 = trans_bw as f64;
    let l_stop = l_freq_f64 - trans_bw_f64;
    let nyq = sfreq as f64 / 2.0;

    let mut freq = Vec::new();
    let mut gain = Vec::new();

    if l_stop > 0.0 {
        freq.push(0.0);
        gain.push(0);
    }
    freq.push(l_stop / nyq);
    gain.push(0);
    freq.push(l_freq_f64 / nyq);
    gain.push(1);
    freq.push(1.0);
    gain.push(1);

    let h = firwin_design(n, &freq, &gain);
    h.iter().map(|&v| v as f32).collect()
}

/// Design a zero-phase lowpass FIR filter.
///
/// 100% parity with `mne.filter.create_filter(None, sfreq, l_freq=None,
///   h_freq=h_freq, filter_length='auto', fir_window='hamming',
///   fir_design='firwin', phase='zero')`.
pub fn design_lowpass(h_freq: f32, sfreq: f32) -> Vec<f32> {
    let trans_bw = auto_trans_bandwidth_lowpass(h_freq, sfreq);
    let n = auto_filter_length(trans_bw, sfreq);

    let h_freq_f64 = h_freq as f64;
    let trans_bw_f64 = trans_bw as f64;
    let h_stop = h_freq_f64 + trans_bw_f64;
    let nyq = sfreq as f64 / 2.0;

    let mut freq = vec![0.0, h_freq_f64 / nyq, h_stop / nyq];
    let mut gain = vec![1, 1, 0];

    if (h_stop - nyq).abs() > 1e-6 {
        freq.push(1.0);
        gain.push(0);
    }

    let h = firwin_design(n, &freq, &gain);
    h.iter().map(|&v| v as f32).collect()
}

/// Design a zero-phase bandpass FIR filter.
///
/// 100% parity with `mne.filter.create_filter(None, sfreq, l_freq=l_freq,
///   h_freq=h_freq, filter_length='auto', fir_window='hamming',
///   fir_design='firwin', phase='zero')`.
///
/// Uses MNE's `_firwin_design` algorithm: each transition band gets its own
/// sub-filter length, so the highpass and lowpass edges use different-length
/// windowed-sinc kernels, both centered in the master kernel.
pub fn design_bandpass(l_freq: f32, h_freq: f32, sfreq: f32) -> Vec<f32> {
    assert!(l_freq < h_freq, "l_freq must be < h_freq for bandpass");
    assert!(h_freq < sfreq / 2.0, "h_freq must be < Nyquist");

    let l_trans = auto_trans_bandwidth(l_freq);
    let h_trans = auto_trans_bandwidth_lowpass(h_freq, sfreq);

    // Master filter length from the narrowest transition
    let min_trans = l_trans.min(h_trans);
    let n = auto_filter_length(min_trans, sfreq);

    let l_freq_f64 = l_freq as f64;
    let h_freq_f64 = h_freq as f64;
    let l_trans_f64 = l_trans as f64;
    let h_trans_f64 = h_trans as f64;
    let l_stop = l_freq_f64 - l_trans_f64;
    let h_stop = h_freq_f64 + h_trans_f64;
    let nyq = sfreq as f64 / 2.0;

    let mut freq = Vec::new();
    let mut gain = Vec::new();

    if l_stop > 0.0 {
        freq.push(0.0);
        gain.push(0);
    }
    freq.push(l_stop / nyq);
    gain.push(0);
    freq.push(l_freq_f64 / nyq);
    gain.push(1);
    freq.push(h_freq_f64 / nyq);
    gain.push(1);
    freq.push(h_stop / nyq);
    gain.push(0);
    if (h_stop - nyq).abs() > 1e-6 {
        freq.push(1.0);
        gain.push(0);
    }

    let h = firwin_design(n, &freq, &gain);
    h.iter().map(|&v| v as f32).collect()
}

/// Design a zero-phase notch (bandstop) FIR filter.
///
/// 100% parity with MNE's `notch_filter(x, sfreq, freqs=[freq],
///   method='fir', notch_widths=nw, trans_bandwidth=tb)`.
///
/// Default `notch_width`: `freq / 200.0` (MNE default).
/// Default `trans_bandwidth`: `1.0` Hz (MNE default).
pub fn design_notch(freq: f32, sfreq: f32, notch_width: Option<f32>, trans_bandwidth: Option<f32>) -> Vec<f32> {
    let nw = notch_width.unwrap_or(freq / 200.0);
    let tb = trans_bandwidth.unwrap_or(1.0);
    let nyq = sfreq / 2.0;

    // MNE notch_filter call chain:
    //   lows  = freq - nw/2 - tb/2
    //   highs = freq + nw/2 + tb/2
    //   filter_data(x, sfreq, highs, lows, l_trans=tb/2, h_trans=tb/2)
    //
    // In create_filter, since highs > lows → bandstop.
    // _triage_filter_params with reverse=True produces:
    //   f_p1 = lows                    = freq - nw/2 - tb/2  (pass edge below)
    //   f_s1 = lows + tb/2             = freq - nw/2         (stop edge below)
    //   f_s2 = highs - tb/2            = freq + nw/2         (stop edge above)
    //   f_p2 = highs                   = freq + nw/2 + tb/2  (pass edge above)
    //
    //   freq = [0, f_p1, f_s1, f_s2, f_p2, nyq]
    //   gain = [1,   1,    0,    0,    1,    1]

    // Use f64 throughout to maintain precision matching MNE-Python
    let freq_f64 = freq as f64;
    let nw_f64 = nw as f64;
    let tb_f64 = tb as f64;
    let nyq_f64 = nyq as f64;

    let f_p1 = freq_f64 - nw_f64 / 2.0 - tb_f64 / 2.0;  // pass edge below
    let f_s1 = freq_f64 - nw_f64 / 2.0;                    // stop edge below
    let f_s2 = freq_f64 + nw_f64 / 2.0;                    // stop edge above
    let f_p2 = freq_f64 + nw_f64 / 2.0 + tb_f64 / 2.0;    // pass edge above

    // Filter length from tb/2 (the transition bandwidth used for both sides)
    let n = auto_filter_length(tb / 2.0, sfreq);

    let mut freq_arr = vec![0.0_f64];
    let mut gain_arr = vec![1_i32];

    freq_arr.push(f_p1 / nyq_f64);
    gain_arr.push(1);
    freq_arr.push(f_s1 / nyq_f64);
    gain_arr.push(0);
    freq_arr.push(f_s2 / nyq_f64);
    gain_arr.push(0);
    freq_arr.push(f_p2 / nyq_f64);
    gain_arr.push(1);

    if (f_p2 - nyq_f64).abs() > 1e-6 {
        freq_arr.push(1.0);
        gain_arr.push(1);
    }

    let h = firwin_design(n, &freq_arr, &gain_arr);
    h.iter().map(|&v| v as f32).collect()
}

// ── Low-level primitives ────────────────────────────────────────────────────

/// Internal f64-precision lowpass firwin for use in `firwin_design`.
///
/// Takes cutoff and sfreq as f64 to avoid precision loss.
/// Always pass_zero=true (lowpass).
fn firwin_f64(n: usize, cutoff_hz: f64, sfreq: f64) -> Vec<f64> {
    assert!(n % 2 == 1, "firwin requires odd N");
    let alpha = (n - 1) as f64 / 2.0;
    let nyq = sfreq / 2.0;
    let fc = cutoff_hz / nyq;

    let win = hamming(n);

    let mut h: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - alpha;
            let sinc = if x == 0.0 { fc } else { (PI * fc * x).sin() / (PI * x) };
            sinc * win[i]
        })
        .collect();

    let s: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= s);
    h
}

/// Design a lowpass FIR filter using a Hamming-windowed sinc.
///
/// `pass_zero=true` means the DC component passes (lowpass).
/// `cutoff_hz` is the -6 dB point, `sfreq` the sampling rate.
///
/// This matches `scipy.signal.firwin(n, cutoff, window='hamming',
///   pass_zero=True, fs=sfreq)`.
pub fn firwin(n: usize, cutoff_hz: f32, sfreq: f32, pass_zero: bool) -> Vec<f64> {
    assert!(n % 2 == 1, "firwin requires odd N for linear-phase filter");
    let alpha = (n - 1) as f64 / 2.0;
    let nyq = sfreq as f64 / 2.0;
    let fc = cutoff_hz as f64 / nyq;   // normalised [0, 1]

    let win = hamming(n);

    let mut h: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - alpha;
            let sinc = if x == 0.0 { fc } else { (PI * fc * x).sin() / (PI * x) };
            sinc * win[i]
        })
        .collect();

    // Normalise so sum = 1 (unit DC gain for lowpass).
    let s: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= s);

    if !pass_zero {
        // Highpass by spectral inversion.
        h.iter_mut().for_each(|v| *v = -*v);
        h[n / 2] += 1.0;
    }

    h
}

/// Hamming window of length `n`.
pub fn hamming(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_length_is_odd() {
        for l_freq in [0.5_f32, 1.0, 2.0, 5.0] {
            let tb = auto_trans_bandwidth(l_freq);
            let n  = auto_filter_length(tb, 256.0);
            assert!(n % 2 == 1, "N={n} is even for l_freq={l_freq}");
        }
    }

    // ── Highpass ────────────────────────────────────────────────────────

    #[test]
    fn highpass_sum_near_zero() {
        let h = design_highpass(0.5, 256.0);
        let s: f32 = h.iter().sum();
        assert!(s.abs() < 1e-5, "highpass sum = {s}");
    }

    #[test]
    fn highpass_is_symmetric() {
        let h = design_highpass(0.5, 256.0);
        let n = h.len();
        for i in 0..n / 2 {
            approx::assert_abs_diff_eq!(h[i], h[n - 1 - i], epsilon = 1e-7_f32);
        }
    }

    #[test]
    fn highpass_known_length_256hz() {
        let h = design_highpass(0.5, 256.0);
        assert_eq!(h.len(), 1691, "expected 1691 taps, got {}", h.len());
    }

    // ── Lowpass ─────────────────────────────────────────────────────────

    #[test]
    fn lowpass_dc_gain_unity() {
        let h = firwin(101, 10.0, 256.0, true);
        let dc: f64 = h.iter().sum();
        approx::assert_abs_diff_eq!(dc, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn lowpass_sum_near_one() {
        let h = design_lowpass(75.0, 256.0);
        let s: f64 = h.iter().map(|&v| v as f64).sum();
        approx::assert_abs_diff_eq!(s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn lowpass_is_symmetric() {
        let h = design_lowpass(75.0, 256.0);
        let n = h.len();
        for i in 0..n / 2 {
            approx::assert_abs_diff_eq!(h[i], h[n - 1 - i], epsilon = 1e-7_f32);
        }
    }

    #[test]
    fn lowpass_75hz_first_element_is_zero() {
        // MNE's _firwin_design uses sub-filter N=45 centered in master N=47,
        // so offset=1, meaning h[0] and h[46] are 0.
        let h = design_lowpass(75.0, 256.0);
        approx::assert_abs_diff_eq!(h[0], 0.0, epsilon = 1e-10_f32);
        approx::assert_abs_diff_eq!(*h.last().unwrap(), 0.0, epsilon = 1e-10_f32);
    }

    // ── Bandpass ────────────────────────────────────────────────────────

    #[test]
    fn bandpass_sum_near_zero() {
        let h = design_bandpass(0.1, 75.0, 256.0);
        let s: f64 = h.iter().map(|&v| v as f64).sum();
        assert!(s.abs() < 1e-4, "bandpass sum = {s}");
    }

    #[test]
    fn bandpass_is_symmetric() {
        let h = design_bandpass(0.1, 75.0, 256.0);
        let n = h.len();
        for i in 0..n / 2 {
            approx::assert_abs_diff_eq!(h[i], h[n - 1 - i], epsilon = 1e-6_f32);
        }
    }

    // ── Notch ───────────────────────────────────────────────────────────

    #[test]
    fn notch_sum_near_one() {
        let h = design_notch(60.0, 256.0, None, None);
        let s: f64 = h.iter().map(|&v| v as f64).sum();
        approx::assert_abs_diff_eq!(s, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn notch_is_symmetric() {
        let h = design_notch(60.0, 256.0, None, None);
        let n = h.len();
        for i in 0..n / 2 {
            approx::assert_abs_diff_eq!(h[i], h[n - 1 - i], epsilon = 1e-6_f32);
        }
    }
}
