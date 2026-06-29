//! EEGLAB `.set + .fdt` reader and writer.
//!
//! EEGLAB stores a recording as a pair of files:
//!
//! - `*.set` — MATLAB v5 `.mat` file holding the `EEG` struct (metadata:
//!   channel count, sample rate, channel locations, events, …).
//! - `*.fdt` — raw `float32` little-endian binary, layout
//!   `[nbchan, pnts, trials]` written in MATLAB column-major order
//!   (channel index varies fastest).
//!
//! For epoched data, `pnts` is the per-trial length and `trials > 1`.
//! For continuous data, `trials = 1` and `pnts` is the total length.
//!
//! Only the subset of the EEGLAB schema needed for downstream RLX / ZUNA
//! consumers is parsed: `nbchan`, `pnts`, `srate`, `trials`, `data` (the
//! .fdt filename), and `chanlocs.{labels, X, Y, Z}` (best-effort — falls
//! back to zeros if absent).
//!
//! Compressed `.set` files (`miCOMPRESSED` top-level elements) are not
//! supported — re-save without compression in EEGLAB first.

pub mod mat5;
mod write;
pub use write::write_raw;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::Array2;
use std::fs::File;
use std::path::{Path, PathBuf};

use mat5::MatValue;

/// One EEGLAB channel's metadata: label + Cartesian position (metres).
#[derive(Debug, Clone)]
pub struct EeglabChannel {
    pub label: String,
    /// `[x, y, z]` in metres. EEGLAB stores in metres natively; if a file
    /// has positions in millimetres we leave them as-is and let downstream
    /// code rescale.
    pub xyz: [f32; 3],
}

/// An EEGLAB recording.
#[derive(Debug, Clone)]
pub struct RawSet {
    /// Path to the `.set` file this was opened from.
    pub set_path: PathBuf,
    /// Path to the `.fdt` companion (or in-`.set` external name).
    pub fdt_path: PathBuf,
    pub n_chan: usize,
    pub n_samples: usize, // per trial if `trials > 1`, else total length
    pub n_trials: usize,
    pub sfreq: f64,
    pub channels: Vec<EeglabChannel>,
}

impl RawSet {
    /// `true` iff the recording is epoched (`trials > 1`).
    pub fn is_epoched(&self) -> bool {
        self.n_trials > 1
    }

    /// Total samples across all trials (i.e. continuous flatten).
    pub fn n_times_total(&self) -> usize {
        self.n_samples * self.n_trials
    }

    /// Channel names in order.
    pub fn ch_names(&self) -> Vec<&str> {
        self.channels.iter().map(|c| c.label.as_str()).collect()
    }

    /// Channel positions as `[n_chan, 3]` (metres). Zeros if not in the file.
    pub fn chan_pos_meters(&self) -> Array2<f32> {
        let mut a = Array2::<f32>::zeros((self.n_chan, 3));
        for (i, ch) in self.channels.iter().enumerate() {
            a[[i, 0]] = ch.xyz[0];
            a[[i, 1]] = ch.xyz[1];
            a[[i, 2]] = ch.xyz[2];
        }
        a
    }

    /// Read the `.fdt` companion as `[n_chan, n_times_total]` f64.
    ///
    /// For epoched data, trials are concatenated in MATLAB column-major
    /// flatten order: `trial 0 sample 0`, `trial 0 sample 1`, …, `trial 0
    /// sample (n_samples − 1)`, `trial 1 sample 0`, …
    ///
    /// Mirrors `exg::fiff::RawFif::read_all_data`'s return shape so the
    /// generic `preprocess()` pipeline can consume it the same way.
    pub fn read_all_data(&self) -> Result<Array2<f64>> {
        let n_ch = self.n_chan;
        let n_t = self.n_times_total();
        let total = n_ch * n_t;
        let need_bytes = total * 4;

        let f = File::open(&self.fdt_path)
            .with_context(|| format!("open {}", self.fdt_path.display()))?;
        let on_disk = f.metadata()?.len() as usize;
        if on_disk != need_bytes {
            bail!(
                ".fdt size mismatch: file {} has {} bytes, struct claims {}×{}×{} = {} samples ({} bytes)",
                self.fdt_path.display(), on_disk,
                n_ch, self.n_samples, self.n_trials, total, need_bytes,
            );
        }

        // mmap the .fdt — it's flat little-endian f32, so we can view it
        // as `&[f32]` directly and avoid the per-byte decode loop.
        // SAFETY: we hold the mmap alive for the duration of this function;
        // the file is not modified by anyone else (process-private mapping).
        let mmap = unsafe { memmap2::Mmap::map(&f) }
            .with_context(|| format!("mmap {}", self.fdt_path.display()))?;
        debug_assert_eq!(mmap.len(), need_bytes);

        // Re-view the bytes as f32 if the alignment cooperates. On the OS X
        // / Linux page allocators mmap returns a page-aligned slice, so the
        // 4-byte alignment is always satisfied. We still guard with
        // `align_to` and fall back to a chunked path if it doesn't.
        let mut out = Array2::<f64>::zeros((n_ch, n_t));
        let out_slice = out.as_slice_mut().expect("contiguous");
        // .fdt layout: per-sample chunk of n_chan f32s, written for every
        // sample of trial 0, then trial 1, etc. (channel index fastest).
        // Target layout (`Array2::<f64>` row-major): channel-major
        // (`out[c, t]` is at offset `c * n_t + t`). So we transpose during
        // the f32→f64 widen.
        let (head, mid, tail) = unsafe { mmap.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() {
            // Fast path: native &[f32] view → vectorizable transpose+widen.
            for t in 0..n_t {
                let row = &mid[t * n_ch..(t + 1) * n_ch];
                for c in 0..n_ch {
                    out_slice[c * n_t + t] = row[c] as f64;
                }
            }
        } else {
            // Fallback: decode bytes 4 at a time.
            for t in 0..n_t {
                let off = t * n_ch * 4;
                for c in 0..n_ch {
                    let b = &mmap[off + c * 4..off + c * 4 + 4];
                    out_slice[c * n_t + t] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
                }
            }
        }
        Ok(out)
    }

    /// Read epochs as `[n_trials, n_chan, n_samples]` f32. Only valid when
    /// `is_epoched()`. Convenience wrapper around `read_all_data()` + reshape.
    pub fn read_epochs(&self) -> Result<ndarray::Array3<f32>> {
        if !self.is_epoched() {
            bail!("read_epochs called on non-epoched RawSet");
        }
        let flat = self.read_all_data()?;
        let mut out = ndarray::Array3::<f32>::zeros((self.n_trials, self.n_chan, self.n_samples));
        for tr in 0..self.n_trials {
            for c in 0..self.n_chan {
                for s in 0..self.n_samples {
                    out[[tr, c, s]] = flat[[c, tr * self.n_samples + s]] as f32;
                }
            }
        }
        Ok(out)
    }

    /// Per-trial event-type string read from `EEG.epoch[i].eventtype`.
    ///
    /// Returns `Vec<Option<String>>` of length `n_trials`. Each entry is the
    /// trial-defining event tag (e.g. `"B3(43)"` for level 3, block 4).
    /// Entries are `None` if the EEGLAB file omits the field or stores it as
    /// something other than a string (e.g. a cell of multiple events — we
    /// only pull the simple-string form here).
    ///
    /// Re-reads the `.set` file once. For batch use, prefer
    /// [`Self::epoch_event_types_from_mat`].
    pub fn epoch_event_types(&self) -> Result<Vec<Option<String>>> {
        let bytes = std::fs::read(&self.set_path)
            .with_context(|| format!("re-read {}", self.set_path.display()))?;
        let mat = mat5::read_mat_v5(std::io::Cursor::new(bytes))?;
        Self::epoch_event_types_from_mat(&mat, self.n_trials)
    }

    /// Same as [`Self::epoch_event_types`] but takes a pre-parsed `MatFile`
    /// so the caller can reuse it across other lookups.
    pub fn epoch_event_types_from_mat(
        mat: &mat5::MatFile,
        n_trials: usize,
    ) -> Result<Vec<Option<String>>> {
        let eeg = mat.get("EEG").ok_or_else(|| anyhow!("no `EEG` struct"))?;
        let fields = match eeg {
            MatValue::Struct(m) => m,
            _ => bail!("`EEG` is not a struct"),
        };
        let elems: &Vec<std::collections::HashMap<String, MatValue>> = match fields.get("epoch") {
            Some(MatValue::StructArray { elems, .. }) => elems,
            // Single-element EEG.epoch (rare, but tolerated).
            Some(MatValue::Struct(m)) => {
                let mut out = vec![None; n_trials.max(1)];
                if let Some(s) = m.get("eventtype").and_then(|v| v.as_str()) {
                    out[0] = Some(s.to_string());
                }
                return Ok(out);
            }
            _ => return Ok(vec![None; n_trials]),
        };
        let mut out = Vec::with_capacity(n_trials);
        for i in 0..n_trials {
            let tag = elems
                .get(i)
                .and_then(|m| m.get("eventtype"))
                .and_then(extract_event_string);
            out.push(tag);
        }
        Ok(out)
    }

    /// Per-trial integer "level" extracted from the event tags by the
    /// `B<level>(<digits>)` pattern. Returns `-1` for trials whose tag does
    /// not match (those are the ones `prepare_dataset.py` drops). The
    /// `level` value is `1..=7` for kept trials.
    ///
    /// This mirrors [`scripts/prepare_dataset.py::event_tag_to_level`] but
    /// runs in pure Rust against the raw EEGLAB file.
    pub fn epoch_levels(&self) -> Result<Vec<i32>> {
        let tags = self.epoch_event_types()?;
        Ok(tags
            .iter()
            .map(|t| t.as_deref().and_then(parse_b_level).unwrap_or(-1))
            .collect())
    }
}

/// EEGLAB epoch eventtype may be stored as either:
///   - a bare string (`MatValue::Str`)
///   - a 1-element cell of strings (one event per epoch — common in EEGLAB
///     `EEG.epoch[i].eventtype`)
/// Both reduce to a single string for our purposes.
fn extract_event_string(v: &MatValue) -> Option<String> {
    match v {
        MatValue::Str(s) => Some(s.clone()),
        MatValue::Cell { elems, .. } => elems.iter().find_map(extract_event_string),
        _ => None,
    }
}

/// Parse the level from a `"B<L>(<dd>)"` tag. Returns `Some(L)` for
/// `L ∈ {1..=7}`, else `None`.
pub fn parse_b_level(tag: &str) -> Option<i32> {
    let rest = tag.strip_prefix('B')?;
    // Take leading digits as level
    let (lvl_str, rest) = rest.split_at(rest.chars().take_while(|c| c.is_ascii_digit()).count());
    let lvl: i32 = lvl_str.parse().ok()?;
    if !(1..=7).contains(&lvl) {
        return None;
    }
    // Require '(' then 2 digits then ')'
    let rest = rest.strip_prefix('(')?;
    if rest.len() < 3 {
        return None;
    }
    let (digits, tail) = rest.split_at(2);
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if !tail.starts_with(')') {
        return None;
    }
    Some(lvl)
}

/// Open an EEGLAB recording from its `.set` file.
///
/// Locates the `.fdt` companion via `EEG.data` (a string filename inside
/// the `.set` file). Falls back to `<stem>.fdt` next to `.set` if `data` is
/// missing or doesn't end in `.fdt`.
pub fn open_raw<P: AsRef<Path>>(set_path: P) -> Result<RawSet> {
    let set_path = set_path.as_ref().to_path_buf();
    let f = File::open(&set_path).with_context(|| format!("open {}", set_path.display()))?;
    let mat = mat5::read_mat_v5(f)?;

    // EEGLAB writes the recording as a top-level struct. Conventionally named
    // "EEG", but tolerate other names — if there is exactly one top-level
    // struct, use it.
    let eeg = mat
        .get("EEG")
        .or_else(|| {
            let structs: Vec<&MatValue> = mat
                .vars
                .values()
                .filter(|v| matches!(v, MatValue::Struct(_)))
                .collect();
            if structs.len() == 1 {
                Some(structs[0])
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("no top-level `EEG` struct in {}", set_path.display()))?;
    let eeg_fields = match eeg {
        MatValue::Struct(m) => m,
        _ => bail!("`EEG` is not a struct in {}", set_path.display()),
    };

    let n_chan = field_scalar_usize(eeg_fields, "nbchan")?;
    let pnts = field_scalar_usize(eeg_fields, "pnts")?;
    let trials = field_scalar_usize(eeg_fields, "trials").unwrap_or(1).max(1);
    let sfreq = field_scalar_f64(eeg_fields, "srate")?;

    // `EEG.data` may be a string (filename) or a numeric matrix (in-file data).
    // We only handle the external-filename case here.
    let fdt_path: PathBuf = match eeg_fields.get("data") {
        Some(MatValue::Str(name)) => {
            if name.ends_with(".fdt") || name.ends_with(".dat") {
                set_path.with_file_name(name)
            } else {
                set_path.with_extension("fdt")
            }
        }
        _ => set_path.with_extension("fdt"),
    };
    if !fdt_path.exists() {
        bail!(".fdt companion not found at {}", fdt_path.display());
    }

    let channels = parse_chanlocs(eeg_fields, n_chan);

    Ok(RawSet {
        set_path,
        fdt_path,
        n_chan,
        n_samples: pnts,
        n_trials: trials,
        sfreq,
        channels,
    })
}

// ── Field helpers ────────────────────────────────────────────────────────────

fn field_scalar_f64(
    fields: &std::collections::HashMap<String, MatValue>,
    name: &str,
) -> Result<f64> {
    let v = fields
        .get(name)
        .ok_or_else(|| anyhow!("EEG.{name} not found"))?;
    v.as_scalar()
        .ok_or_else(|| anyhow!("EEG.{name} is not a scalar (got {v:?})"))
}

fn field_scalar_usize(
    fields: &std::collections::HashMap<String, MatValue>,
    name: &str,
) -> Result<usize> {
    let x = field_scalar_f64(fields, name)?;
    if x < 0.0 || !x.is_finite() {
        bail!("EEG.{name} = {x} is not a non-negative integer");
    }
    Ok(x as usize)
}

fn parse_chanlocs(
    fields: &std::collections::HashMap<String, MatValue>,
    n_chan: usize,
) -> Vec<EeglabChannel> {
    // EEGLAB stores chanlocs as a 1×N struct array, each with labels.X/Y/Z.
    let mut out: Vec<EeglabChannel> = (0..n_chan)
        .map(|i| EeglabChannel {
            label: format!("ch{}", i + 1),
            xyz: [0.0; 3],
        })
        .collect();
    let Some(v) = fields.get("chanlocs") else {
        return out;
    };
    let elems: &Vec<std::collections::HashMap<String, MatValue>> = match v {
        MatValue::StructArray { elems, .. } => elems,
        MatValue::Struct(m) => {
            // Single-element struct (rare for chanlocs but tolerated).
            if let Some(c) = mk_channel(m) {
                out[0] = c;
            }
            return out;
        }
        _ => return out,
    };
    for (i, m) in elems.iter().take(n_chan).enumerate() {
        if let Some(c) = mk_channel(m) {
            out[i] = c;
        }
    }
    out
}

fn mk_channel(m: &std::collections::HashMap<String, MatValue>) -> Option<EeglabChannel> {
    let label = m
        .get("labels")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // EEGLAB convention: X = anterior+, Y = left+, Z = up+. Units are mm
    // unless `chaninfo.unit` says otherwise — for our purposes we leave the
    // raw values and let the caller scale if needed.
    let x = m.get("X").and_then(|v| v.as_scalar()).unwrap_or(0.0) as f32;
    let y = m.get("Y").and_then(|v| v.as_scalar()).unwrap_or(0.0) as f32;
    let z = m.get("Z").and_then(|v| v.as_scalar()).unwrap_or(0.0) as f32;
    Some(EeglabChannel {
        label,
        xyz: [x, y, z],
    })
}
