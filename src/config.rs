//! Pipeline configuration.
//!
//! [`PipelineConfig`] holds every tunable parameter for the full preprocessing
//! pipeline.  All fields have sensible defaults that match the values used to
//! train the EEG model.

/// Configuration for the full EEG preprocessing pipeline.
///
/// All fields are `pub` so you can construct one with struct-update syntax:
///
/// ```
/// use exg::PipelineConfig;
///
/// let cfg = PipelineConfig {
///     target_sfreq: 128.0,   // resample to 128 Hz instead of 256
///     hp_freq:      1.0,     // stronger highpass
///     ..PipelineConfig::default()
/// };
/// ```
///
/// Or just call [`PipelineConfig::default()`] for the training settings.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Target sampling rate in Hz after resampling.
    ///
    /// The resampler is skipped entirely when the source rate already equals
    /// this value (within 1 mHz).
    ///
    /// Default: `256.0` Hz.
    pub target_sfreq: f32,

    /// Cutoff frequency of the zero-phase highpass FIR filter in Hz.
    ///
    /// The filter is designed with a Hamming window and an automatically
    /// computed transition bandwidth (`min(max(0.25 · hp_freq, 2.0), hp_freq)`)
    /// and filter length (`⌈3.3 / trans_bw · sfreq⌉`, rounded to odd).
    ///
    /// At the default 0.5 Hz / 256 Hz this produces a 1 691-tap kernel,
    /// matching `mne.filter.create_filter(l_freq=0.5, fir_window='hamming')`.
    ///
    /// Default: `0.5` Hz.
    pub hp_freq: f32,

    /// Duration of each output epoch in seconds.
    ///
    /// Non-overlapping windows of this length are cut from the continuous
    /// recording after all per-sample preprocessing steps (resample, filter,
    /// reference, z-score).  Trailing samples that do not fill a complete
    /// window are discarded.
    ///
    /// At the default 5.0 s / 256 Hz each epoch has **1 280 samples**.
    ///
    /// Default: `5.0` s.
    pub epoch_dur: f32,

    /// Divisor applied to every epoch element after z-scoring.
    ///
    /// After global z-score the signal has `std ≈ 1`.  Dividing by 10 brings
    /// typical values into the range `[−0.3, +0.3]`, which improves numerical
    /// stability in the diffusion model.
    ///
    /// Set to `1.0` to disable.
    ///
    /// Default: `10.0`.
    pub data_norm: f32,

    /// Channel names to zero-fill before any processing.
    ///
    /// Useful for simulating channel dropout during inference or for excluding
    /// known-bad channels from contributing to the average reference and
    /// z-score statistics.
    ///
    /// Name matching is case-insensitive and ignores spaces
    /// (e.g. `"fp 1"` matches `"Fp1"`).
    ///
    /// Default: `[]` (no channels zeroed).
    pub bad_channels: Vec<String>,
}

impl Default for PipelineConfig {
    /// Returns the training configuration:
    /// 256 Hz · 0.5 Hz HP · 5 s epochs · data_norm = 10.
    fn default() -> Self {
        Self {
            target_sfreq: 256.0,
            hp_freq: 0.5,
            epoch_dur: 5.0,
            data_norm: 10.0,
            bad_channels: vec![],
        }
    }
}

impl PipelineConfig {
    /// Number of samples per epoch at the target sampling rate.
    ///
    /// Computed as `floor(epoch_dur × target_sfreq)`.  At the defaults this
    /// returns **1 280** (= 5 s × 256 Hz).
    ///
    /// # Examples
    ///
    /// ```
    /// use exg::PipelineConfig;
    /// let cfg = PipelineConfig::default();
    /// assert_eq!(cfg.epoch_samples(), 1280);
    /// ```
    pub fn epoch_samples(&self) -> usize {
        (self.epoch_dur * self.target_sfreq) as usize
    }
}
