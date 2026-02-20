//! FIR filter design matching MNE / `scipy.signal.firwin`.
//!
//! For a highpass filter at `l_freq` Hz with sampling rate `sfreq`:
//!   • transition bandwidth = min(max(0.25 * l_freq, 2.0), l_freq)
//!   • filter length N      = ceil(3.3 / trans_bw * sfreq), rounded to odd
//!   • windowed-sinc design (Hamming window) via spectral inversion
use std::f64::consts::PI;

/// Compute MNE-compatible transition bandwidth for a highpass filter.
///
/// Rule: `min(max(0.25 * l_freq, 2.0), l_freq)`
pub fn auto_trans_bandwidth(l_freq: f32) -> f32 {
    let tb = (0.25 * l_freq).max(2.0).min(l_freq);
    tb
}

/// Compute the number of FIR taps for a given transition bandwidth.
/// Returns an odd integer (required for zero-phase linear-phase FIR).
///
/// Formula: `ceil(3.3 / trans_bw * sfreq)` rounded up to odd.
pub fn auto_filter_length(trans_bw: f32, sfreq: f32) -> usize {
    let n_raw = (3.3 / trans_bw * sfreq).ceil() as usize;
    // Make odd.
    if n_raw % 2 == 0 { n_raw + 1 } else { n_raw }
}

/// Design a zero-phase highpass FIR filter using a Hamming-windowed sinc.
///
/// Matches `mne.filter.create_filter(None, sfreq, l_freq=l_freq, h_freq=None,
///   filter_length='auto', fir_window='hamming', fir_design='firwin', phase='zero')`.
///
/// Returns the impulse response `h[N]` as `Vec<f32>`.
pub fn design_highpass(l_freq: f32, sfreq: f32) -> Vec<f32> {
    let trans_bw = auto_trans_bandwidth(l_freq);
    let n = auto_filter_length(trans_bw, sfreq);
    let l_stop = l_freq - trans_bw;  // lower stop frequency (Hz)

    // Midpoint of transition band → firwin cutoff.
    let cutoff_hz = (l_stop + l_freq) / 2.0;

    // Build lowpass at cutoff_hz, then spectrally invert to highpass.
    let h_lp = firwin(n, cutoff_hz, sfreq, true);

    // Spectral inversion: highpass = delta[n=N/2] - lowpass
    let mut h: Vec<f64> = h_lp.iter().map(|&v| -(v as f64)).collect();
    h[n / 2] += 1.0;

    h.iter().map(|&v| v as f32).collect()
}

/// Design a lowpass FIR filter using a Hamming-windowed sinc.
///
/// `pass_zero=true` means the DC component passes (lowpass).
/// `cutoff_hz` is the -6 dB point.
pub fn firwin(n: usize, cutoff_hz: f32, sfreq: f32, pass_zero: bool) -> Vec<f64> {
    assert!(n % 2 == 1, "firwin requires odd N for linear-phase filter");
    let alpha = (n - 1) as f64 / 2.0;
    let nyq = sfreq as f64 / 2.0;
    let fc = cutoff_hz as f64 / nyq;   // normalised [0, 1]

    let win = hamming(n);

    let mut h: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - alpha;
            // f(x) = sin(π·fc·x) / (π·x);  lim_{x→0} f(x) = fc  (L'Hôpital)
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

    #[test]
    fn highpass_sum_near_zero() {
        // A highpass filter should sum to ≈ 0 (no DC component passes).
        let h = design_highpass(0.5, 256.0);
        let s: f32 = h.iter().sum();
        assert!(s.abs() < 1e-5, "highpass sum = {s}");
    }

    #[test]
    fn highpass_is_symmetric() {
        // Linear-phase FIR must be symmetric.
        let h = design_highpass(0.5, 256.0);
        let n = h.len();
        for i in 0..n / 2 {
            approx::assert_abs_diff_eq!(h[i], h[n - 1 - i], epsilon = 1e-7_f32);
        }
    }

    #[test]
    fn highpass_known_length_256hz() {
        // At 256 Hz / 0.5 Hz: MNE produces 1691 taps.
        let h = design_highpass(0.5, 256.0);
        assert_eq!(h.len(), 1691, "expected 1691 taps, got {}", h.len());
    }

    #[test]
    fn lowpass_dc_gain_unity() {
        let h = firwin(101, 10.0, 256.0, true);
        let dc: f64 = h.iter().sum();
        approx::assert_abs_diff_eq!(dc, 1.0, epsilon = 1e-9);
    }
}
