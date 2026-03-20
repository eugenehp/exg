//! CSV reader for EEG data.
//!
//! Reads EEG data from CSV files with optional timestamp column and
//! channel columns.
//!
//! # Format
//!
//! The CSV file should have:
//! - An optional header row with channel names
//! - An optional timestamp column (auto-detected as monotonically increasing)
//! - Numeric data columns (one per channel)
//! - Comment lines starting with `#` are skipped
//!
//! # Example
//! ```no_run
//! use exg::csv::read_eeg;
//!
//! let (data, ch_names, sfreq) = read_eeg("recording.csv").unwrap();
//! println!("{} channels @ {} Hz, {} samples", data.nrows(), sfreq, data.ncols());
//! ```

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use std::path::Path;

/// Read EEG data from a CSV file.
///
/// # Returns
/// `(data, channel_names, sample_rate)`:
/// * `data` — shape `[C, T]` in original units
/// * `channel_names` — names from the header row (or generated Ch0, Ch1, ...)
/// * `sample_rate` — inferred from timestamps (or 0.0 if no timestamp column)
pub fn read_eeg<P: AsRef<Path>>(path: P) -> Result<(Array2<f32>, Vec<String>, f32)> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("reading CSV: {}", path.as_ref().display()))?;

    let lines: Vec<&str> = content.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();

    if lines.is_empty() {
        bail!("CSV file is empty");
    }

    // Detect delimiter
    let delim = detect_delimiter(lines[0]);

    // Try to parse the first line as data; if it fails, it's a header
    let first_fields: Vec<&str> = lines[0].split(delim).map(|s| s.trim()).collect();
    let has_header = first_fields.iter().any(|f| f.parse::<f64>().is_err());

    let (header, data_lines) = if has_header {
        let hdr: Vec<String> = first_fields.iter().map(|s| s.to_string()).collect();
        (Some(hdr), &lines[1..])
    } else {
        (None, &lines[..])
    };

    if data_lines.is_empty() {
        bail!("CSV file has no data rows");
    }

    // Parse all data rows
    let n_cols = first_fields.len();
    let n_rows = data_lines.len();
    let mut raw_data = Vec::with_capacity(n_rows * n_cols);

    for (line_idx, line) in data_lines.iter().enumerate() {
        let fields: Vec<&str> = line.split(delim).map(|s| s.trim()).collect();
        if fields.len() != n_cols {
            bail!("Row {} has {} columns, expected {}", line_idx + 1, fields.len(), n_cols);
        }
        for field in &fields {
            let val = field.parse::<f64>()
                .with_context(|| format!("parsing value '{}' at row {}", field, line_idx + 1))?;
            raw_data.push(val as f32);
        }
    }

    // Detect timestamp column (first column that is monotonically increasing with > 0 diff)
    let mut timestamp_col: Option<usize> = None;
    'col_loop: for col in 0..n_cols {
        if n_rows < 2 { break; }
        let mut prev = raw_data[col];
        for row in 1..n_rows {
            let val = raw_data[row * n_cols + col];
            if val <= prev {
                continue 'col_loop;
            }
            prev = val;
        }
        timestamp_col = Some(col);
        break;
    }

    // Compute sample rate from timestamps
    let sfreq = if let Some(ts_col) = timestamp_col {
        if n_rows >= 2 {
            let dt = raw_data[n_cols + ts_col] - raw_data[ts_col];
            if dt > 0.0 { 1.0 / dt } else { 0.0 }
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Build data array excluding timestamp column
    let data_cols: Vec<usize> = (0..n_cols)
        .filter(|&c| Some(c) != timestamp_col)
        .collect();
    let n_ch = data_cols.len();

    let mut data = Array2::<f32>::zeros((n_ch, n_rows));
    for (ch_out, &col) in data_cols.iter().enumerate() {
        for row in 0..n_rows {
            data[[ch_out, row]] = raw_data[row * n_cols + col];
        }
    }

    // Channel names
    let ch_names: Vec<String> = if let Some(ref hdr) = header {
        data_cols.iter().map(|&c| hdr[c].clone()).collect()
    } else {
        (0..n_ch).map(|i| format!("Ch{}", i)).collect()
    };

    Ok((data, ch_names, sfreq))
}

fn detect_delimiter(line: &str) -> char {
    // Count occurrences of common delimiters
    let tab_count = line.matches('\t').count();
    let comma_count = line.matches(',').count();
    let semi_count = line.matches(';').count();

    if tab_count >= comma_count && tab_count >= semi_count && tab_count > 0 {
        '\t'
    } else if semi_count > comma_count && semi_count > 0 {
        ';'
    } else {
        ','
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_csv_with_header_and_timestamp() {
        let dir = std::env::temp_dir().join("exg_csv_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");

        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "time,ch1,ch2").unwrap();
        writeln!(f, "0.000,1.0,2.0").unwrap();
        writeln!(f, "0.004,3.0,4.0").unwrap();
        writeln!(f, "0.008,5.0,6.0").unwrap();

        let (data, names, sfreq) = read_eeg(&path).unwrap();
        assert_eq!(names, vec!["ch1", "ch2"]);
        assert_eq!(data.dim(), (2, 3));
        approx::assert_abs_diff_eq!(sfreq, 250.0, epsilon = 1.0);
        approx::assert_abs_diff_eq!(data[[0, 0]], 1.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(data[[1, 2]], 6.0, epsilon = 1e-6);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_csv_no_header() {
        let dir = std::env::temp_dir().join("exg_csv_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test2.csv");

        let mut f = std::fs::File::create(&path).unwrap();
        // Use non-monotonic columns to avoid timestamp detection
        writeln!(f, "5.0,2.0,9.0").unwrap();
        writeln!(f, "1.0,1.0,3.0").unwrap();

        let (data, names, _sfreq) = read_eeg(&path).unwrap();
        assert_eq!(names.len(), 3);
        assert_eq!(data.dim(), (3, 2));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detect_delimiter_works() {
        assert_eq!(detect_delimiter("a,b,c"), ',');
        assert_eq!(detect_delimiter("a\tb\tc"), '\t');
        assert_eq!(detect_delimiter("a;b;c"), ';');
    }
}
