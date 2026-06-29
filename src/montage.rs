//! Bipolar montage conversion for EEG data.
//!
//! Implements the TCP (Temporal Central Parasagittal) bipolar montage
//! used by LUNA and the TUH EEG corpus.
//!
//! Each bipolar channel is computed as `electrode1 - electrode2`.

use ndarray::Array2;

/// A single bipolar channel definition: `(output_name, anode_name, cathode_name)`.
///
/// The output signal is `anode - cathode`.
pub type BipolarDef = (&'static str, &'static str, &'static str);

/// TCP bipolar montage: 22 channels derived from 21 reference electrodes.
///
/// This is the standard montage used by the TUH EEG Corpus and LUNA.
/// Each entry is `(bipolar_name, anode_electrode, cathode_electrode)`.
pub const TCP_MONTAGE: &[BipolarDef] = &[
    // Left temporal chain
    ("FP1-F7", "FP1", "F7"),
    ("F7-T3", "F7", "T3"),
    ("T3-T5", "T3", "T5"),
    ("T5-O1", "T5", "O1"),
    // Right temporal chain
    ("FP2-F8", "FP2", "F8"),
    ("F8-T4", "F8", "T4"),
    ("T4-T6", "T4", "T6"),
    ("T6-O2", "T6", "O2"),
    // Left parasagittal chain
    ("FP1-F3", "FP1", "F3"),
    ("F3-C3", "F3", "C3"),
    ("C3-P3", "C3", "P3"),
    ("P3-O1", "P3", "O1"),
    // Right parasagittal chain
    ("FP2-F4", "FP2", "F4"),
    ("F4-C4", "F4", "C4"),
    ("C4-P4", "C4", "P4"),
    ("P4-O2", "P4", "O2"),
    // Central chain
    ("FZ-CZ", "FZ", "CZ"),
    ("CZ-PZ", "CZ", "PZ"),
    // Additional temporal (alt naming)
    ("T3-C3", "T3", "C3"),
    ("C3-CZ", "C3", "CZ"),
    ("CZ-C4", "CZ", "C4"),
    ("C4-T4", "C4", "T4"),
];

/// Normalise a TUH-style channel name to a standard electrode label.
///
/// Strips "EEG " prefix, "-REF" / "-LE" suffix, and converts to uppercase.
/// Examples:
///   "EEG FP1-REF" → "FP1"
///   "EEG FP1-LE"  → "FP1"
///   "FP1"         → "FP1"
pub fn normalize_channel_name(name: &str) -> String {
    let mut s = name.trim().to_uppercase();
    // Strip "EEG " prefix
    if let Some(rest) = s.strip_prefix("EEG ") {
        s = rest.to_string();
    }
    // Strip "-REF" or "-LE" suffix
    if let Some(rest) = s.strip_suffix("-REF") {
        s = rest.to_string();
    } else if let Some(rest) = s.strip_suffix("-LE") {
        s = rest.to_string();
    }
    s
}

/// Apply a bipolar montage to EEG data.
///
/// # Arguments
/// * `data` — Raw EEG data, shape `[C, T]`, in reference montage.
/// * `ch_names` — Channel names corresponding to rows of `data`.
/// * `montage` — Bipolar channel definitions. Use [`TCP_MONTAGE`] for LUNA.
///
/// # Returns
/// `(bipolar_data, bipolar_names)`:
/// * `bipolar_data` — shape `[M, T]` where M is the number of montage channels
///   for which both electrodes were found.
/// * `bipolar_names` — the names of the output bipolar channels.
///
/// Channels whose electrodes are not found in `ch_names` are silently skipped.
pub fn make_bipolar(
    data: &Array2<f32>,
    ch_names: &[String],
    montage: &[BipolarDef],
) -> (Array2<f32>, Vec<String>) {
    // Build lookup: normalised name → row index
    let norm_names: Vec<String> = ch_names.iter().map(|n| normalize_channel_name(n)).collect();

    let n_t = data.ncols();
    let mut out_rows: Vec<Vec<f32>> = Vec::new();
    let mut out_names: Vec<String> = Vec::new();

    for &(bp_name, anode, cathode) in montage {
        let a_idx = norm_names.iter().position(|n| n == anode);
        let c_idx = norm_names.iter().position(|n| n == cathode);
        if let (Some(ai), Some(ci)) = (a_idx, c_idx) {
            let row: Vec<f32> = data
                .row(ai)
                .iter()
                .zip(data.row(ci).iter())
                .map(|(&a, &c)| a - c)
                .collect();
            out_rows.push(row);
            out_names.push(bp_name.to_string());
        }
    }

    let n_out = out_rows.len();
    if n_out == 0 {
        return (Array2::zeros((0, n_t)), vec![]);
    }

    let flat: Vec<f32> = out_rows.into_iter().flatten().collect();
    let bipolar_data =
        Array2::from_shape_vec((n_out, n_t), flat).expect("shape mismatch in bipolar montage");

    (bipolar_data, out_names)
}

// ── Siena Scalp EEG montage (29 unipolar channels) ──────────────────────────

/// Siena Scalp EEG dataset: 29 unipolar channels.
///
/// No bipolar conversion needed — the data is used directly in unipolar montage.
/// Channel order matches the Siena dataset from the BioFoundation project.
pub const SIENA_CHANNELS: &[&str] = &[
    "FP1", "FP2", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "FC1", "FC5", "CP1", "CP5", "F9", "FZ",
    "CZ", "PZ", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "FC2", "FC6", "CP2", "CP6", "F10",
];

// ── SEED-V dataset montage (62 unipolar channels) ───────────────────────────

/// SEED / SEED-V dataset: 62-channel ESI NeuroScan system.
///
/// Unipolar montage — no bipolar conversion needed.
/// Channel order matches the SEED_CHANNEL_LIST from the torcheeg library.
pub const SEED_V_CHANNELS: &[&str] = &[
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2",
    "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5",
    "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
];

/// Pick channels from data that match a given unipolar channel set.
///
/// Useful for Siena and SEED-V datasets where no bipolar conversion is needed.
///
/// # Arguments
/// * `data` — shape `[C, T]`
/// * `ch_names` — channel names for each row of `data`
/// * `target_channels` — the desired channel set (e.g. [`SIENA_CHANNELS`])
///
/// # Returns
/// `(picked_data, picked_names)` — the subset of channels found, in the
/// order specified by `target_channels`.
pub fn pick_channels(
    data: &Array2<f32>,
    ch_names: &[String],
    target_channels: &[&str],
) -> (Array2<f32>, Vec<String>) {
    let norm_names: Vec<String> = ch_names.iter().map(|n| normalize_channel_name(n)).collect();

    let n_t = data.ncols();
    let mut out_rows: Vec<Vec<f32>> = Vec::new();
    let mut out_names: Vec<String> = Vec::new();

    for &target in target_channels {
        let target_upper = target.to_uppercase();
        if let Some(idx) = norm_names.iter().position(|n| n == &target_upper) {
            out_rows.push(data.row(idx).to_vec());
            out_names.push(target_upper);
        }
    }

    let n_out = out_rows.len();
    if n_out == 0 {
        return (Array2::zeros((0, n_t)), vec![]);
    }

    let flat: Vec<f32> = out_rows.into_iter().flatten().collect();
    let picked =
        Array2::from_shape_vec((n_out, n_t), flat).expect("shape mismatch in pick_channels");

    (picked, out_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn normalize_tuh_names() {
        assert_eq!(normalize_channel_name("EEG FP1-REF"), "FP1");
        assert_eq!(normalize_channel_name("EEG FP1-LE"), "FP1");
        assert_eq!(normalize_channel_name("FP1"), "FP1");
        assert_eq!(normalize_channel_name("eeg fp1-ref"), "FP1");
    }

    #[test]
    fn bipolar_subtraction() {
        // 3 channels: FP1=1.0, F7=0.5, F3=0.2 (constant signals)
        let names: Vec<String> = vec!["EEG FP1-REF", "EEG F7-REF", "EEG F3-REF"]
            .into_iter()
            .map(String::from)
            .collect();
        let data = Array2::from_shape_fn((3, 100), |(c, _)| [1.0_f32, 0.5, 0.2][c]);

        let montage: &[BipolarDef] = &[("FP1-F7", "FP1", "F7"), ("FP1-F3", "FP1", "F3")];

        let (bp_data, bp_names) = make_bipolar(&data, &names, montage);
        assert_eq!(bp_names, vec!["FP1-F7", "FP1-F3"]);
        assert_eq!(bp_data.nrows(), 2);

        // FP1-F7 = 1.0 - 0.5 = 0.5
        for &v in bp_data.row(0).iter() {
            approx::assert_abs_diff_eq!(v, 0.5, epsilon = 1e-6);
        }
        // FP1-F3 = 1.0 - 0.2 = 0.8
        for &v in bp_data.row(1).iter() {
            approx::assert_abs_diff_eq!(v, 0.8, epsilon = 1e-6);
        }
    }

    #[test]
    fn pick_channels_works() {
        let names: Vec<String> = vec!["EEG FP1-REF", "EEG F7-REF", "EEG CZ-REF", "EEG O1-REF"]
            .into_iter()
            .map(String::from)
            .collect();
        let data = Array2::from_shape_fn((4, 50), |(c, _)| c as f32);

        let (picked, picked_names) = pick_channels(&data, &names, &["FP1", "CZ", "PZ"]);
        // PZ not found → only 2 channels
        assert_eq!(picked_names, vec!["FP1", "CZ"]);
        assert_eq!(picked.nrows(), 2);
        // FP1 was row 0 → values = 0.0
        approx::assert_abs_diff_eq!(picked[[0, 0]], 0.0, epsilon = 1e-6);
        // CZ was row 2 → values = 2.0
        approx::assert_abs_diff_eq!(picked[[1, 0]], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn siena_channel_count() {
        assert_eq!(SIENA_CHANNELS.len(), 29);
    }

    #[test]
    fn seed_v_channel_count() {
        assert_eq!(SEED_V_CHANNELS.len(), 62);
    }

    #[test]
    fn missing_channels_skipped() {
        let names: Vec<String> = vec!["FP1", "F7"].into_iter().map(String::from).collect();
        let data = Array2::zeros((2, 50));

        let montage: &[BipolarDef] = &[
            ("FP1-F7", "FP1", "F7"),
            ("FP1-F3", "FP1", "F3"), // F3 missing
        ];

        let (bp_data, bp_names) = make_bipolar(&data, &names, montage);
        assert_eq!(bp_names.len(), 1);
        assert_eq!(bp_names[0], "FP1-F7");
        assert_eq!(bp_data.nrows(), 1);
    }
}
