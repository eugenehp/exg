//! Minimal MAT v5 reader — just enough to parse an EEGLAB `.set` file.
//!
//! EEGLAB `.set` files are MATLAB v5 .mat files whose top-level element is a
//! single named struct `EEG` containing the fields we care about (`nbchan`,
//! `pnts`, `srate`, `trials`, `data`, `chanlocs`, …). This reader supports
//! the subset of data types those fields actually use:
//!
//! - `miMATRIX` (the only top-level element form we need to walk)
//! - `mxSTRUCT_CLASS` (the `EEG` envelope and `chanlocs` elements)
//! - `mxDOUBLE_CLASS` (scalar metadata: nbchan, pnts, srate, trials, …)
//! - `mxCHAR_CLASS`   (string fields: `data` → external .fdt filename)
//! - `mxCELL_CLASS`   (skipped — we don't need it for raw extraction)
//!
//! Compressed (`miCOMPRESSED`) elements are **not** supported — EEGLAB saves
//! its `.set` files uncompressed by default, which is what we need here.
//! Sparse arrays, objects, and the v7.3 HDF5 variant are also out of scope.
//!
//! Endian: detected from the 2-byte indicator at offset 126 of the header.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// ── MAT v5 data type codes ───────────────────────────────────────────────────

pub const MI_INT8: u32 = 1;
pub const MI_UINT8: u32 = 2;
pub const MI_INT16: u32 = 3;
pub const MI_UINT16: u32 = 4;
pub const MI_INT32: u32 = 5;
pub const MI_UINT32: u32 = 6;
pub const MI_SINGLE: u32 = 7;
pub const MI_DOUBLE: u32 = 9;
pub const MI_INT64: u32 = 12;
pub const MI_UINT64: u32 = 13;
pub const MI_MATRIX: u32 = 14;
pub const MI_COMPRESSED: u32 = 15;
pub const MI_UTF8: u32 = 16;
pub const MI_UTF16: u32 = 17;

// ── mxClass codes (inside an miMATRIX) ───────────────────────────────────────

pub const MX_CELL_CLASS: u8 = 1;
pub const MX_STRUCT_CLASS: u8 = 2;
pub const MX_OBJECT_CLASS: u8 = 3;
pub const MX_CHAR_CLASS: u8 = 4;
pub const MX_SPARSE_CLASS: u8 = 5;
pub const MX_DOUBLE_CLASS: u8 = 6;
pub const MX_SINGLE_CLASS: u8 = 7;
pub const MX_INT8_CLASS: u8 = 8;
pub const MX_UINT8_CLASS: u8 = 9;
pub const MX_INT16_CLASS: u8 = 10;
pub const MX_UINT16_CLASS: u8 = 11;
pub const MX_INT32_CLASS: u8 = 12;
pub const MX_UINT32_CLASS: u8 = 13;

// ── Parsed-value representation ──────────────────────────────────────────────

/// A subset of MAT v5 values — only the kinds an EEGLAB `.set` file uses.
#[derive(Debug, Clone)]
pub enum MatValue {
    /// Numeric array (mxDOUBLE / mxSINGLE / integer classes), reduced to f64.
    /// `dims` is row-major; the value buffer is column-major (MATLAB native).
    Numeric { dims: Vec<usize>, data: Vec<f64> },
    /// String (mxCHAR_CLASS reduced to UTF-8).
    Str(String),
    /// Empty 0×0 array.
    Empty,
    /// Struct (mxSTRUCT_CLASS with dims [1,1]) — field name → value.
    Struct(HashMap<String, MatValue>),
    /// Struct array (mxSTRUCT_CLASS with N>1 dims). Outer Vec is element-major
    /// (column-major flatten of the dims), each element is a field map.
    StructArray {
        dims: Vec<usize>,
        elems: Vec<HashMap<String, MatValue>>,
    },
    /// Cell array — kept as opaque element list.
    Cell {
        dims: Vec<usize>,
        elems: Vec<MatValue>,
    },
    /// Anything we deliberately skipped (with the byte size we walked past).
    Unsupported { class: u8, dims: Vec<usize> },
}

impl MatValue {
    /// If this is a scalar `Numeric`, return the single value.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            MatValue::Numeric { dims, data } if data.len() == 1 && dims.iter().all(|&d| d == 1) => {
                Some(data[0])
            }
            _ => None,
        }
    }
    /// If this is a `Str`, return the slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MatValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    /// If this is a numeric array, return a flat slice.
    pub fn as_numeric(&self) -> Option<&[f64]> {
        match self {
            MatValue::Numeric { data, .. } => Some(data.as_slice()),
            _ => None,
        }
    }
}

// ── Endian-aware byte reader ─────────────────────────────────────────────────

struct LeReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> LeReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            bail!("MAT5: short read (skip {n} @ {})", self.pos);
        }
        self.pos += n;
        Ok(())
    }
    fn read_u16(&mut self) -> Result<u16> {
        if self.remaining() < 2 {
            bail!("MAT5: short read (u16)");
        }
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }
    fn read_u32(&mut self) -> Result<u32> {
        if self.remaining() < 4 {
            bail!("MAT5: short read (u32)");
        }
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }
    fn read_f64(&mut self) -> Result<f64> {
        if self.remaining() < 8 {
            bail!("MAT5: short read (f64)");
        }
        let v = f64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!("MAT5: short read ({n} bytes @ {})", self.pos);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    /// Align position up to the next 8-byte boundary (after a data element body).
    fn align8(&mut self) {
        let pad = (8 - (self.pos % 8)) % 8;
        self.pos = (self.pos + pad).min(self.buf.len());
    }
}

// ── Element header ───────────────────────────────────────────────────────────

/// A "data element" header: type + size, plus a flag for the small-data
/// 4-byte form (where the body is inlined in the high 4 bytes of the tag).
struct ElemHeader {
    ty: u32,
    size: u32,
    small: bool,
    inline_bytes: Option<[u8; 4]>, // present iff small
}

fn read_elem_header(r: &mut LeReader) -> Result<ElemHeader> {
    let first = r.read_u32()?;
    // Small Data Element format: when the upper 16 bits are non-zero, the
    // element fits in 4 bytes. Layout (little-endian read of u32):
    //   bits  0..15 → type
    //   bits 16..31 → size (≤4)
    // Body is the next 4 bytes (no separate u32 size word).
    let upper = (first >> 16) & 0xFFFF;
    if upper != 0 {
        let ty = first & 0xFFFF;
        let size = upper;
        if size > 4 {
            bail!("MAT5: small element with size {size} > 4");
        }
        let inline = r.read_bytes(4)?.try_into().unwrap();
        Ok(ElemHeader {
            ty,
            size,
            small: true,
            inline_bytes: Some(inline),
        })
    } else {
        let size = r.read_u32()?;
        Ok(ElemHeader {
            ty: first,
            size,
            small: false,
            inline_bytes: None,
        })
    }
}

/// Skip past an element body (for unsupported types).
fn skip_elem_body(r: &mut LeReader, h: &ElemHeader) -> Result<()> {
    if h.small {
        return Ok(());
    }
    r.skip(h.size as usize)?;
    r.align8();
    Ok(())
}

// ── Per-type body readers ────────────────────────────────────────────────────

fn read_numeric_body(r: &mut LeReader, h: &ElemHeader) -> Result<Vec<f64>> {
    if let Some(inline) = h.inline_bytes {
        // Small data: type-specific cast into f64
        return cast_to_f64(h.ty, &inline[..h.size as usize]);
    }
    let bytes = r.read_bytes(h.size as usize)?.to_vec();
    r.align8();
    cast_to_f64(h.ty, &bytes)
}

fn cast_to_f64(ty: u32, bytes: &[u8]) -> Result<Vec<f64>> {
    match ty {
        MI_INT8 => Ok(bytes.iter().map(|&b| b as i8 as f64).collect()),
        MI_UINT8 => Ok(bytes.iter().map(|&b| b as f64).collect()),
        MI_INT16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect()),
        MI_UINT16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64)
            .collect()),
        MI_INT32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()),
        MI_UINT32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()),
        MI_SINGLE => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()),
        MI_DOUBLE => Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        other => bail!("MAT5: cast_to_f64: unsupported numeric type {other}"),
    }
}

fn read_string_body(r: &mut LeReader, h: &ElemHeader) -> Result<String> {
    let raw_bytes: Vec<u8> = if let Some(inline) = h.inline_bytes {
        inline[..h.size as usize].to_vec()
    } else {
        let b = r.read_bytes(h.size as usize)?.to_vec();
        r.align8();
        b
    };
    match h.ty {
        MI_UINT8 | MI_INT8 | MI_UTF8 => Ok(String::from_utf8_lossy(&raw_bytes).to_string()),
        MI_UINT16 | MI_UTF16 => {
            let words: Vec<u16> = raw_bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            Ok(String::from_utf16_lossy(&words))
        }
        other => bail!("MAT5: string body: unsupported encoding type {other}"),
    }
}

// ── Top-level miMATRIX walker ────────────────────────────────────────────────

/// Read the body of an miMATRIX element (header already consumed).
/// `h.size` bounds how many bytes belong to this matrix in the parent stream.
fn read_matrix_body(r: &mut LeReader, h: &ElemHeader) -> Result<(String, MatValue)> {
    if h.small {
        bail!("MAT5: miMATRIX cannot be a small element");
    }
    let end = r.pos + h.size as usize;
    // 1) Array flags (always 8 bytes of u32)
    let af = read_elem_header(r)?;
    if af.ty != MI_UINT32 || af.size != 8 {
        bail!(
            "MAT5: expected array-flags subelement (uint32, size 8); got ty={} size={}",
            af.ty,
            af.size
        );
    }
    let flags_u32 = r.read_u32()?;
    let _nzmax = r.read_u32()?;
    let class = (flags_u32 & 0xff) as u8;
    let _flags = ((flags_u32 >> 8) & 0xff) as u8;
    // 2) Dimensions array (i32)
    let dh = read_elem_header(r)?;
    if dh.ty != MI_INT32 {
        bail!("MAT5: expected dims subelement (int32); got ty={}", dh.ty);
    }
    let dims_count = (dh.size / 4) as usize;
    let mut dims = Vec::with_capacity(dims_count);
    if let Some(inline) = dh.inline_bytes {
        for c in inline.chunks_exact(4).take(dims_count) {
            dims.push(i32::from_le_bytes(c.try_into().unwrap()) as usize);
        }
    } else {
        for _ in 0..dims_count {
            dims.push(r.read_i32()? as usize);
        }
        r.align8();
    }
    // 3) Array name (int8)
    let nh = read_elem_header(r)?;
    let name = read_string_body(r, &nh)?;

    // 4) Per-class body
    let value = match class {
        MX_DOUBLE_CLASS | MX_SINGLE_CLASS | MX_INT8_CLASS | MX_UINT8_CLASS | MX_INT16_CLASS
        | MX_UINT16_CLASS | MX_INT32_CLASS | MX_UINT32_CLASS => {
            // Real-part data only (we ignore complex)
            let ph = read_elem_header(r)?;
            let data = read_numeric_body(r, &ph)?;
            // (Complex part, if present, would follow. We ignore it.)
            if dims.iter().product::<usize>() == 0 {
                MatValue::Empty
            } else {
                MatValue::Numeric {
                    dims: dims.clone(),
                    data,
                }
            }
        }
        MX_CHAR_CLASS => {
            let ph = read_elem_header(r)?;
            let s = read_string_body(r, &ph)?;
            MatValue::Str(s)
        }
        MX_STRUCT_CLASS => {
            // Field name length (i32 scalar)
            let lh = read_elem_header(r)?;
            let field_len: usize = if let Some(inline) = lh.inline_bytes {
                i32::from_le_bytes(inline) as usize
            } else {
                let n = r.read_i32()? as usize;
                r.align8();
                n
            };
            // Field name array (int8, packed strings of length `field_len`)
            let fh = read_elem_header(r)?;
            let names_block: Vec<u8> = if let Some(inline) = fh.inline_bytes {
                inline[..fh.size as usize].to_vec()
            } else {
                let v = r.read_bytes(fh.size as usize)?.to_vec();
                r.align8();
                v
            };
            let n_fields = names_block.len() / field_len;
            let field_names: Vec<String> = (0..n_fields)
                .map(|i| {
                    let slice = &names_block[i * field_len..(i + 1) * field_len];
                    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
                    String::from_utf8_lossy(&slice[..end]).to_string()
                })
                .collect();
            // Number of struct elements = product of dims
            let nelem = dims.iter().product::<usize>().max(1);
            // For each element, read one miMATRIX per field (in field order)
            let mut elems = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                let mut map = HashMap::with_capacity(n_fields);
                for fname in &field_names {
                    let inner_h = read_elem_header(r)?;
                    if inner_h.ty != MI_MATRIX {
                        bail!(
                            "MAT5: struct field {fname}: expected miMATRIX, got ty={}",
                            inner_h.ty
                        );
                    }
                    if inner_h.size == 0 {
                        map.insert(fname.clone(), MatValue::Empty);
                        continue;
                    }
                    let (_n, v) = read_matrix_body(r, &inner_h)?;
                    map.insert(fname.clone(), v);
                }
                elems.push(map);
            }
            if nelem == 1 && dims.iter().all(|&d| d == 1) {
                MatValue::Struct(elems.into_iter().next().unwrap())
            } else {
                MatValue::StructArray {
                    dims: dims.clone(),
                    elems,
                }
            }
        }
        MX_CELL_CLASS => {
            let nelem = dims.iter().product::<usize>();
            let mut elems = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                let inner_h = read_elem_header(r)?;
                if inner_h.size == 0 {
                    elems.push(MatValue::Empty);
                } else if inner_h.ty == MI_MATRIX {
                    let (_n, v) = read_matrix_body(r, &inner_h)?;
                    elems.push(v);
                } else {
                    // Inline numeric/string in a cell — uncommon but handle it.
                    skip_elem_body(r, &inner_h)?;
                    elems.push(MatValue::Unsupported {
                        class: 0,
                        dims: vec![],
                    });
                }
            }
            MatValue::Cell {
                dims: dims.clone(),
                elems,
            }
        }
        _ => {
            // Skip the remaining bytes of this matrix
            let remaining = end.saturating_sub(r.pos);
            r.skip(remaining)?;
            MatValue::Unsupported {
                class,
                dims: dims.clone(),
            }
        }
    };
    // Realign and skip any trailing bytes within this matrix's footprint.
    if r.pos < end {
        let remaining = end - r.pos;
        r.skip(remaining)?;
    }
    Ok((name, value))
}

// ── Public entry ─────────────────────────────────────────────────────────────

/// Parsed MAT v5 file: top-level named variables.
pub struct MatFile {
    pub vars: HashMap<String, MatValue>,
}

impl MatFile {
    pub fn get(&self, name: &str) -> Option<&MatValue> {
        self.vars.get(name)
    }
}

/// Read a MAT v5 file from any `Read + Seek` source.
///
/// Loads the file fully into memory — the EEGLAB `.set` files we care about
/// here are sub-MB, so this trades a small allocation for a much simpler
/// walker over a single contiguous slice.
pub fn read_mat_v5<R: Read + Seek>(mut r: R) -> Result<MatFile> {
    let mut all = Vec::new();
    r.read_to_end(&mut all).context("MAT5: read file")?;
    if all.len() < 128 {
        bail!("MAT5: file too short ({} bytes)", all.len());
    }

    // Header: 124 bytes description + 4 bytes (subsys offset, version) +
    // 2 bytes version (we read as u16) + 2 bytes endian marker.
    // Endian indicator: bytes 126..128 read as a little-endian u16 should
    // equal 0x4d49 ('MI'). If it's 0x494d ('IM'), the file is big-endian.
    let marker = u16::from_le_bytes([all[126], all[127]]);
    let big_endian = match marker {
        0x4d49 => false, // 'IM' in raw bytes → little-endian
        0x494d => true,  // 'MI' in raw bytes → big-endian
        other => bail!("MAT5: bad endian marker 0x{other:04x}"),
    };
    if big_endian {
        bail!("MAT5: big-endian .mat files are not supported by this minimal reader");
    }

    let mut rd = LeReader::new(&all[128..]);
    let mut vars: HashMap<String, MatValue> = HashMap::new();
    while rd.remaining() > 0 {
        let h = read_elem_header(&mut rd)?;
        if h.ty == MI_COMPRESSED {
            bail!("MAT5: compressed elements are not supported (re-save without compression in EEGLAB)");
        }
        if h.ty != MI_MATRIX {
            // Skip unknown top-level elements gracefully
            skip_elem_body(&mut rd, &h)?;
            continue;
        }
        if h.size == 0 {
            continue;
        }
        let (name, value) = read_matrix_body(&mut rd, &h)?;
        vars.insert(name, value);
    }
    Ok(MatFile { vars })
}
