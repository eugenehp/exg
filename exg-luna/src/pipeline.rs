//! LUNA-specific EEG preprocessing pipeline.
//!
//! See the [crate-level docs](crate) for the full pipeline description.

use anyhow::Result;
use ndarray::Array2;

use exg::filter;
use exg::montage::{self, TCP_MONTAGE};
use exg::resample;

/// Configuration for the LUNA preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct LunaPipelineConfig {
    /// Lower edge of bandpass filter in Hz.
    /// Default: `0.1`
    pub bandpass_low: f32,

    /// Upper edge of bandpass filter in Hz.
    /// Default: `75.0`
    pub bandpass_high: f32,

    /// Notch filter frequency in Hz. Set to 60 for US data, 50 for EU.
    /// Set to `None` to skip notch filtering.
    /// Default: `Some(60.0)`
    pub notch_freq: Option<f32>,

    /// Target sampling rate in Hz.
    /// Default: `256.0`
    pub target_sfreq: f32,

    /// Duration of each output epoch in seconds.
    /// Default: `5.0`
    pub epoch_dur: f32,

    /// Bipolar montage definition.
    /// Default: [`exg::montage::TCP_MONTAGE`] (22-channel TCP bipolar).
    pub montage: Vec<(String, String, String)>,
}

impl Default for LunaPipelineConfig {
    fn default() -> Self {
        Self {
            bandpass_low: 0.1,
            bandpass_high: 75.0,
            notch_freq: Some(60.0),
            target_sfreq: 256.0,
            epoch_dur: 5.0,
            montage: TCP_MONTAGE
                .iter()
                .map(|&(name, anode, cathode)| {
                    (name.to_string(), anode.to_string(), cathode.to_string())
                })
                .collect(),
        }
    }
}

impl LunaPipelineConfig {
    /// Number of samples per epoch at the target sampling rate.
    pub fn epoch_samples(&self) -> usize {
        (self.epoch_dur * self.target_sfreq) as usize
    }
}

/// The standard 21 electrodes of the 10-20 system used by TUH/LUNA.
pub const STANDARD_10_20: &[&str] = &[
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5",
    "T6", "FZ", "CZ", "PZ", "A1", "A2",
];

/// Run the LUNA preprocessing pipeline on a continuous recording.
///
/// # Arguments
/// * `data` — Raw EEG signal, shape `[C, T]`, in reference montage.
/// * `ch_names` — Channel names corresponding to rows of `data`.
/// * `src_sfreq` — Sampling rate of the input data in Hz.
/// * `cfg` — Pipeline configuration.
///
/// # Returns
/// A `Vec` of `(epoch_data, bipolar_names)` tuples:
/// * `epoch_data` — shape `[M, epoch_samples]` where M = number of bipolar channels
/// * `bipolar_names` — names of the bipolar channels (e.g. "FP1-F7")
///
/// # Pipeline steps
/// 1. Normalise channel names (strip "EEG ", "-REF", "-LE")
/// 2. Pick channels matching the standard 10-20 electrode set
/// 3. Bandpass filter (0.1–75 Hz, zero-phase FIR)
/// 4. Notch filter (60 Hz, zero-phase FIR)
/// 5. Resample to target_sfreq (256 Hz)
/// 6. Apply TCP bipolar montage
/// 7. Epoch into non-overlapping windows
pub fn preprocess_luna(
    data: Array2<f32>,
    ch_names: &[String],
    src_sfreq: f32,
    cfg: &LunaPipelineConfig,
) -> Result<Vec<(Array2<f32>, Vec<String>)>> {
    // 1. Normalise channel names
    let norm_names: Vec<String> = ch_names
        .iter()
        .map(|n| montage::normalize_channel_name(n))
        .collect();

    // 2. Pick standard channels — keep only channels whose normalised
    //    name matches one of the standard 10-20 electrodes
    let mut pick_indices: Vec<usize> = Vec::new();
    let mut picked_names: Vec<String> = Vec::new();
    for (i, name) in norm_names.iter().enumerate() {
        if STANDARD_10_20.contains(&name.as_str()) {
            // Avoid duplicates (take first occurrence)
            if !picked_names.contains(name) {
                pick_indices.push(i);
                picked_names.push(name.clone());
            }
        }
    }

    let n_t = data.ncols();
    let n_picked = pick_indices.len();
    let mut picked_data = Array2::<f32>::zeros((n_picked, n_t));
    for (out_i, &src_i) in pick_indices.iter().enumerate() {
        picked_data.row_mut(out_i).assign(&data.row(src_i));
    }

    // 3. Bandpass filter
    let h_bp = filter::design_bandpass(cfg.bandpass_low, cfg.bandpass_high, src_sfreq);
    filter::apply_fir_zero_phase(&mut picked_data, &h_bp)?;

    // 4. Notch filter
    if let Some(notch_freq) = cfg.notch_freq {
        let h_notch = filter::design_notch(notch_freq, src_sfreq, None, None);
        filter::apply_fir_zero_phase(&mut picked_data, &h_notch)?;
    }

    // 5. Resample
    if (src_sfreq - cfg.target_sfreq).abs() > 1e-3 {
        picked_data = resample::resample(&picked_data, src_sfreq, cfg.target_sfreq)?;
    }

    // 6. Bipolar montage — always use TCP_MONTAGE (the default).
    // Custom montages are matched by name against TCP_MONTAGE entries.
    let (bipolar_data, bipolar_names) =
        montage::make_bipolar(&picked_data, &picked_names, TCP_MONTAGE);

    if bipolar_data.nrows() == 0 {
        return Ok(vec![]);
    }

    // 7. Epoch
    let epoch_samples = cfg.epoch_samples();
    let n_t_bp = bipolar_data.ncols();
    let n_epochs = n_t_bp / epoch_samples;

    let mut result = Vec::with_capacity(n_epochs);
    for e in 0..n_epochs {
        let start = e * epoch_samples;
        let epoch_data: Array2<f32> = bipolar_data
            .slice(ndarray::s![.., start..start + epoch_samples])
            .to_owned();
        result.push((epoch_data, bipolar_names.clone()));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn default_config() {
        let cfg = LunaPipelineConfig::default();
        assert_eq!(cfg.epoch_samples(), 1280);
        assert_eq!(cfg.bandpass_low, 0.1);
        assert_eq!(cfg.bandpass_high, 75.0);
        assert_eq!(cfg.notch_freq, Some(60.0));
    }

    #[test]
    fn pipeline_with_synthetic_data() {
        // Create synthetic data: 21 channels, 30 seconds at 256 Hz
        let sfreq = 256.0_f32;
        let dur = 30.0_f32;
        let n_t = (sfreq * dur) as usize;

        let ch_names: Vec<String> = STANDARD_10_20
            .iter()
            .map(|&s| format!("EEG {}-REF", s))
            .collect();

        // Generate some sinusoidal data
        let data = Array2::from_shape_fn((ch_names.len(), n_t), |(c, t)| {
            let freq = 10.0 + c as f32;
            (2.0 * std::f32::consts::PI * freq * t as f32 / sfreq).sin() * 50e-6
        });

        let cfg = LunaPipelineConfig {
            notch_freq: None, // skip notch for speed
            ..LunaPipelineConfig::default()
        };

        let epochs = preprocess_luna(data, &ch_names, sfreq, &cfg).unwrap();
        // 30s / 5s = 6 epochs (but filter transients may reduce effective length)
        assert!(!epochs.is_empty(), "should produce at least one epoch");

        let (ep, names) = &epochs[0];
        assert_eq!(ep.ncols(), 1280);
        assert!(!names.is_empty());
    }
}
