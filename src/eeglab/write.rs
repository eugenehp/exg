//! EEGLAB `.set + .fdt` writer.
//!
//! Inverse of [`super::open_raw`]: serializes a [`RawSet`] back to a
//! MAT v5 `.set` file plus a raw-f32 little-endian `.fdt` companion.
//!
//! The `EEG` struct emitted contains the minimum subset of fields required
//! for the file to round-trip through this reader and through EEGLAB /
//! `mne.io.read_epochs_eeglab`. Fields the caller doesn't supply are written
//! as empty arrays (which EEGLAB tolerates).
//!
//! ## Limitations
//!
//! - Continuous (`trials = 1`) and epoched (`trials > 1`) recordings are both
//!   supported; the writer infers from `RawSet::n_trials`.
//! - Events, ICA matrices, and reject structs are not emitted (left empty);
//!   round-tripping a file does not preserve them.
//! - Output is always **little-endian** MAT v5, no compression.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::{mat5::*, RawSet};

// ── Public API ───────────────────────────────────────────────────────────────

/// Write a [`RawSet`] to a `.set + .fdt` pair.
///
/// `set_path` must have the `.set` extension; the companion `.fdt` is
/// written next to it with the same stem. `data` should be the
/// `[n_chan, n_times_total]` array previously returned by
/// [`RawSet::read_all_data`] (or freshly generated to match the struct's
/// dimensions). The trial layout in `.fdt` is "channel index fastest" within
/// each sample, matching how [`RawSet::read_all_data`] reads it back.
pub fn write_raw<P: AsRef<Path>>(
    raw: &RawSet,
    data: &ndarray::Array2<f32>,
    set_path: P,
) -> Result<()> {
    let set_path = set_path.as_ref().to_path_buf();
    if set_path.extension().and_then(|s| s.to_str()) != Some("set") {
        anyhow::bail!(
            "write_raw: path must end in .set (got {})",
            set_path.display()
        );
    }
    let fdt_path: PathBuf = set_path.with_extension("fdt");

    // ── 1. Validate dims ─────────────────────────────────────────────────────
    let (n_ch, n_t) = (data.nrows(), data.ncols());
    let expected_t = raw.n_samples * raw.n_trials;
    if n_ch != raw.n_chan {
        anyhow::bail!(
            "write_raw: data has {n_ch} channels, RawSet says {}",
            raw.n_chan
        );
    }
    if n_t != expected_t {
        anyhow::bail!(
            "write_raw: data has {n_t} time-points, expected n_samples × n_trials = {} × {} = {}",
            raw.n_samples,
            raw.n_trials,
            expected_t,
        );
    }

    // ── 2. Write .fdt (sample-major, channel-minor: matches read path) ──────
    {
        let f =
            File::create(&fdt_path).with_context(|| format!("create {}", fdt_path.display()))?;
        let mut w = BufWriter::new(f);
        for t in 0..n_t {
            for c in 0..n_ch {
                w.write_all(&data[[c, t]].to_le_bytes())?;
            }
        }
        w.flush()?;
    }

    // ── 3. Write .set ────────────────────────────────────────────────────────
    let fdt_name = fdt_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("data.fdt")
        .to_string();

    let mut writer = ElementWriter::new();
    writer.write_header(); // 128-byte MAT v5 header

    // EEGLAB main struct
    let mut eeg = StructBuilder::new();
    eeg.set_str(
        "setname",
        &raw.set_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("eeg"),
    );
    eeg.set_str(
        "filename",
        set_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("out.set"),
    );
    eeg.set_str(
        "filepath",
        set_path
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default()
            .as_str(),
    );
    eeg.set_scalar("nbchan", raw.n_chan as f64);
    eeg.set_scalar("pnts", raw.n_samples as f64);
    eeg.set_scalar("trials", raw.n_trials as f64);
    eeg.set_scalar("srate", raw.sfreq);
    eeg.set_scalar("xmin", 0.0);
    eeg.set_scalar("xmax", (raw.n_samples - 1) as f64 / raw.sfreq);
    eeg.set_str("data", &fdt_name);
    eeg.set_str("ref", "average");
    eeg.set_chanlocs(&raw.channels);

    writer.write_top_level_struct("EEG", &eeg);

    let f = File::create(&set_path).with_context(|| format!("create {}", set_path.display()))?;
    let mut w = BufWriter::new(f);
    w.write_all(&writer.bytes)?;
    w.flush()?;
    Ok(())
}

// ── MAT v5 element writer ────────────────────────────────────────────────────

struct ElementWriter {
    bytes: Vec<u8>,
}

impl ElementWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(64 * 1024),
        }
    }

    fn write_header(&mut self) {
        // 116-byte description + 8 bytes subsys offset/version + 2 bytes
        // version + 2 bytes endian marker. The header takes exactly 128
        // bytes; we don't care what's in the first 124 as long as the last
        // 4 carry version (0x0100) and marker 'MI' as a little-endian u16.
        let mut hdr = [0u8; 128];
        let desc = b"MATLAB 5.0 MAT-file, written by exg (Rust) - little-endian, uncompressed";
        hdr[..desc.len()].copy_from_slice(desc);
        for b in &mut hdr[desc.len()..116] {
            *b = b' ';
        }
        // subsystem-specific data offset: 8 zero bytes
        // version: 0x0100 at offset 124..126
        hdr[124] = 0x00;
        hdr[125] = 0x01;
        // endian marker: 'IM' at offset 126..128 reads as u16 0x4D49 LE = 'MI'
        hdr[126] = b'I';
        hdr[127] = b'M';
        self.bytes.extend_from_slice(&hdr);
    }

    /// Append a top-level miMATRIX containing one named struct.
    fn write_top_level_struct(&mut self, name: &str, st: &StructBuilder) {
        let body = build_struct_body(name, st);
        self.write_normal_element(MI_MATRIX, &body);
    }

    /// Write a normal (non-small) element: 8-byte tag + body + pad to 8.
    fn write_normal_element(&mut self, ty: u32, body: &[u8]) {
        self.bytes.extend_from_slice(&ty.to_le_bytes());
        self.bytes
            .extend_from_slice(&(body.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(body);
        align8(&mut self.bytes);
    }
}

fn align8(buf: &mut Vec<u8>) {
    while buf.len() % 8 != 0 {
        buf.push(0);
    }
}

/// Build a normal subelement (used inside another element's body).
fn emit_subelement(out: &mut Vec<u8>, ty: u32, body: &[u8]) {
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    align8(out);
}

/// Build the miMATRIX body for an mxSTRUCT_CLASS variable.
fn build_struct_body(name: &str, st: &StructBuilder) -> Vec<u8> {
    let mut body = Vec::new();
    // (1) Array flags: 8-byte uint32 element
    let mut af = Vec::with_capacity(8);
    af.extend_from_slice(&(MX_STRUCT_CLASS as u32).to_le_bytes());
    af.extend_from_slice(&0u32.to_le_bytes());
    emit_subelement(&mut body, MI_UINT32, &af);
    // (2) Dimensions: int32 [1, 1]
    let mut dims = Vec::with_capacity(8);
    dims.extend_from_slice(&1i32.to_le_bytes());
    dims.extend_from_slice(&1i32.to_le_bytes());
    emit_subelement(&mut body, MI_INT32, &dims);
    // (3) Array name: int8 — use the normal form (not small) for simplicity
    emit_subelement(&mut body, MI_INT8, name.as_bytes());
    // (4) Field name length: int32, value FIELD_LEN
    const FIELD_LEN: usize = 32;
    emit_subelement(&mut body, MI_INT32, &(FIELD_LEN as i32).to_le_bytes());
    // (5) Field names: int8, concatenated FIELD_LEN-byte buckets
    let mut names_block = Vec::with_capacity(st.fields.len() * FIELD_LEN);
    for (name, _) in &st.fields {
        let mut bucket = vec![0u8; FIELD_LEN];
        let bytes = name.as_bytes();
        let n = bytes.len().min(FIELD_LEN - 1);
        bucket[..n].copy_from_slice(&bytes[..n]);
        names_block.extend_from_slice(&bucket);
    }
    emit_subelement(&mut body, MI_INT8, &names_block);
    // (6) For each field, write one miMATRIX subelement.
    for (_name, val) in &st.fields {
        let inner_body = build_value_body(val);
        emit_subelement(&mut body, MI_MATRIX, &inner_body);
    }
    body
}

/// Build the miMATRIX body for an arbitrary `MatLike` value.
fn build_value_body(val: &MatLike) -> Vec<u8> {
    let mut body = Vec::new();
    match val {
        MatLike::Empty => {
            // Array flags (mxDOUBLE_CLASS), dims [0,0], empty name, empty data.
            let mut af = Vec::with_capacity(8);
            af.extend_from_slice(&(MX_DOUBLE_CLASS as u32).to_le_bytes());
            af.extend_from_slice(&0u32.to_le_bytes());
            emit_subelement(&mut body, MI_UINT32, &af);
            let mut dims = Vec::with_capacity(8);
            dims.extend_from_slice(&0i32.to_le_bytes());
            dims.extend_from_slice(&0i32.to_le_bytes());
            emit_subelement(&mut body, MI_INT32, &dims);
            emit_subelement(&mut body, MI_INT8, &[]);
            emit_subelement(&mut body, MI_DOUBLE, &[]);
        }
        MatLike::Scalar(x) => {
            let mut af = Vec::with_capacity(8);
            af.extend_from_slice(&(MX_DOUBLE_CLASS as u32).to_le_bytes());
            af.extend_from_slice(&0u32.to_le_bytes());
            emit_subelement(&mut body, MI_UINT32, &af);
            let mut dims = Vec::with_capacity(8);
            dims.extend_from_slice(&1i32.to_le_bytes());
            dims.extend_from_slice(&1i32.to_le_bytes());
            emit_subelement(&mut body, MI_INT32, &dims);
            emit_subelement(&mut body, MI_INT8, &[]);
            emit_subelement(&mut body, MI_DOUBLE, &x.to_le_bytes());
        }
        MatLike::Str(s) => {
            // mxCHAR_CLASS with dim [1, len], int8 body.
            let mut af = Vec::with_capacity(8);
            af.extend_from_slice(&(MX_CHAR_CLASS as u32).to_le_bytes());
            af.extend_from_slice(&0u32.to_le_bytes());
            emit_subelement(&mut body, MI_UINT32, &af);
            let mut dims = Vec::with_capacity(8);
            dims.extend_from_slice(&1i32.to_le_bytes());
            dims.extend_from_slice(&(s.len() as i32).to_le_bytes());
            emit_subelement(&mut body, MI_INT32, &dims);
            emit_subelement(&mut body, MI_INT8, &[]);
            // EEGLAB writes strings as miUINT16 character codes inside mxCHAR;
            // but miUINT8 (with type tag MI_UTF8) also round-trips through MNE.
            // We pick miUINT8 for simplicity.
            emit_subelement(&mut body, MI_UTF8, s.as_bytes());
        }
        MatLike::StructArray { dims, elems } => {
            let mut af = Vec::with_capacity(8);
            af.extend_from_slice(&(MX_STRUCT_CLASS as u32).to_le_bytes());
            af.extend_from_slice(&0u32.to_le_bytes());
            emit_subelement(&mut body, MI_UINT32, &af);
            let mut dims_buf = Vec::with_capacity(4 * dims.len());
            for d in dims {
                dims_buf.extend_from_slice(&(*d as i32).to_le_bytes());
            }
            emit_subelement(&mut body, MI_INT32, &dims_buf);
            emit_subelement(&mut body, MI_INT8, &[]);
            const FIELD_LEN: usize = 32;
            emit_subelement(&mut body, MI_INT32, &(FIELD_LEN as i32).to_le_bytes());
            // Determine field names from the first element (all elements must
            // share the same field schema in MATLAB struct arrays).
            let field_names: Vec<&str> = elems
                .first()
                .map(|e| e.fields.iter().map(|(n, _)| n.as_str()).collect())
                .unwrap_or_default();
            let mut names_block = Vec::with_capacity(field_names.len() * FIELD_LEN);
            for fname in &field_names {
                let mut bucket = vec![0u8; FIELD_LEN];
                let bytes = fname.as_bytes();
                let n = bytes.len().min(FIELD_LEN - 1);
                bucket[..n].copy_from_slice(&bytes[..n]);
                names_block.extend_from_slice(&bucket);
            }
            emit_subelement(&mut body, MI_INT8, &names_block);
            for el in elems {
                for fname in &field_names {
                    let v = el
                        .fields
                        .iter()
                        .find(|(n, _)| n == fname)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(MatLike::Empty);
                    let inner = build_value_body(&v);
                    emit_subelement(&mut body, MI_MATRIX, &inner);
                }
            }
        }
    }
    body
}

// ── Writer-side value types (kept distinct from the reader-side `MatValue`
// to keep the write path simple and tightly scoped).

#[derive(Clone)]
enum MatLike {
    Empty,
    Scalar(f64),
    Str(String),
    StructArray {
        dims: Vec<usize>,
        elems: Vec<StructBuilder>,
    },
}

#[derive(Clone, Default)]
struct StructBuilder {
    fields: Vec<(String, MatLike)>,
}

impl StructBuilder {
    fn new() -> Self {
        Self::default()
    }
    fn set_scalar(&mut self, name: &str, x: f64) {
        self.fields.push((name.to_string(), MatLike::Scalar(x)));
    }
    fn set_str(&mut self, name: &str, s: &str) {
        self.fields
            .push((name.to_string(), MatLike::Str(s.to_string())));
    }
    fn set_chanlocs(&mut self, chans: &[super::EeglabChannel]) {
        let mut elems: Vec<StructBuilder> = Vec::with_capacity(chans.len());
        for ch in chans {
            let mut sb = StructBuilder::new();
            sb.set_str("labels", &ch.label);
            sb.set_scalar("X", ch.xyz[0] as f64);
            sb.set_scalar("Y", ch.xyz[1] as f64);
            sb.set_scalar("Z", ch.xyz[2] as f64);
            elems.push(sb);
        }
        self.fields.push((
            "chanlocs".to_string(),
            MatLike::StructArray {
                dims: vec![1, chans.len()],
                elems,
            },
        ));
    }
}
