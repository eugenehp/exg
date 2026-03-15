//! FIFF tag I/O.
//!
//! A tag is the smallest structural unit of a FIF file.
//! On-disk layout (always big-endian):
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │  kind : i32  │  type : u32  │  size : i32  │ next : i32 │  ← 16 bytes
//! ├──────────────────────────────────────────────────────┤
//! │  <size bytes of payload data>                        │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! `next == 0` means the next tag follows immediately (pos + 16 + size).
//! `next  > 0` means seek to byte offset `next`.
//! `next == -1` means there is no next tag (end of sequence).
use std::io::{Read, Seek, SeekFrom};
use anyhow::{bail, Context, Result};

use super::constants::*;

// ── Tag header ────────────────────────────────────────────────────────────

/// Lightweight tag header — no payload loaded yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagHeader {
    pub kind: i32,
    pub ftype: u32,   // "type" is a Rust keyword
    pub size: i32,
    pub next: i32,
    pub pos: u64,     // byte offset of the header in the file
}

impl TagHeader {
    /// Byte position of the first payload byte.
    #[inline]
    pub fn data_pos(&self) -> u64 {
        self.pos + 16
    }

    /// Position of the NEXT tag header (or `None` if this is the last tag).
    pub fn next_pos(&self) -> Option<u64> {
        if self.next == FIFFV_NEXT_SEQ {
            Some(self.pos + 16 + self.size as u64)
        } else if self.next > 0 {
            Some(self.next as u64)
        } else {
            None // FIFFV_NEXT_NONE (-1) or any other negative
        }
    }
}

/// Read only the 16-byte tag header at the given file position.
pub fn read_tag_header<R: Read + Seek>(reader: &mut R, pos: u64) -> Result<TagHeader> {
    reader.seek(SeekFrom::Start(pos))
        .with_context(|| format!("seek to tag header @ {pos:#x}"))?;
    let mut buf = [0u8; 16];
    reader.read_exact(&mut buf)
        .with_context(|| format!("read tag header @ {pos:#x}"))?;
    Ok(TagHeader {
        kind:  i32::from_be_bytes(buf[0..4].try_into().unwrap()),
        ftype: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
        size:  i32::from_be_bytes(buf[8..12].try_into().unwrap()),
        next:  i32::from_be_bytes(buf[12..16].try_into().unwrap()),
        pos,
    })
}

// ── Payload readers (stateless, called with a reader pointing just after
//   the 16-byte header, i.e. at `tag.data_pos()`) ──────────────────────────

/// Read a single big-endian i32 payload.
pub fn read_i32<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<i32> {
    seek_data(reader, tag)?;
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_be_bytes(buf))
}

/// Read a single big-endian f32 payload.
pub fn read_f32<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<f32> {
    seek_data(reader, tag)?;
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_be_bytes(buf))
}

/// Read a single big-endian f64 payload.
pub fn read_f64<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<f64> {
    seek_data(reader, tag)?;
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_be_bytes(buf))
}

/// Read a Latin-1 string payload.
pub fn read_string<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<String> {
    seek_data(reader, tag)?;
    let n = tag.size.max(0) as usize;
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf)?;
    // FIFF strings are ISO-8859-1 / Latin-1; safe to decode byte-by-byte.
    Ok(buf.iter().map(|&b| b as char).collect())
}

/// Read a big-endian i32 array (one or more ints).
pub fn read_i32_array<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<Vec<i32>> {
    seek_data(reader, tag)?;
    let n = tag.size as usize / 4;
    let mut out = vec![0i32; n];
    let mut buf = [0u8; 4];
    for v in &mut out {
        reader.read_exact(&mut buf)?;
        *v = i32::from_be_bytes(buf);
    }
    Ok(out)
}

/// Read a big-endian f32 array.
pub fn read_f32_array<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<Vec<f32>> {
    seek_data(reader, tag)?;
    let n = tag.size as usize / 4;
    let mut out = vec![0f32; n];
    let mut buf = [0u8; 4];
    for v in &mut out {
        reader.read_exact(&mut buf)?;
        *v = f32::from_be_bytes(buf);
    }
    Ok(out)
}

/// Read a big-endian f64 array.
pub fn read_f64_array<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<Vec<f64>> {
    seek_data(reader, tag)?;
    let n = tag.size as usize / 8;
    let mut out = vec![0f64; n];
    let mut buf = [0u8; 8];
    for v in &mut out {
        reader.read_exact(&mut buf)?;
        *v = f64::from_be_bytes(buf);
    }
    Ok(out)
}

/// Read the entire payload as raw bytes (useful for ch_info struct parsing).
pub fn read_raw_bytes<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<Vec<u8>> {
    seek_data(reader, tag)?;
    let n = tag.size.max(0) as usize;
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

// ── Directory tag (FIFFT_DIR_ENTRY_STRUCT) ────────────────────────────────

/// Read a directory of tag headers embedded in a `FIFF_DIR_POINTER` tag.
/// Each entry is a 16-byte structure identical to a tag header, but the
/// `next` field stores the real file position of that tag.
pub fn read_directory<R: Read + Seek>(
    reader: &mut R,
    tag: &TagHeader,
) -> Result<Vec<TagHeader>> {
    if tag.ftype != FIFFT_DIR_ENTRY_STRUCT {
        bail!("expected FIFFT_DIR_ENTRY_STRUCT, got {}", tag.ftype);
    }
    let n = tag.size.max(0) as usize / 16;
    let mut entries = Vec::with_capacity(n);
    seek_data(reader, tag)?;
    let mut buf = [0u8; 16];
    for _ in 0..n {
        reader.read_exact(&mut buf)?;
        let kind  = i32::from_be_bytes(buf[0..4].try_into().unwrap());
        let ftype = u32::from_be_bytes(buf[4..8].try_into().unwrap());
        let size  = i32::from_be_bytes(buf[8..12].try_into().unwrap());
        let pos   = u32::from_be_bytes(buf[12..16].try_into().unwrap()) as u64;
        entries.push(TagHeader { kind, ftype, size, next: FIFFV_NEXT_NONE, pos });
    }
    Ok(entries)
}

// ── Helpers ───────────────────────────────────────────────────────────────

#[inline]
fn seek_data<R: Read + Seek>(reader: &mut R, tag: &TagHeader) -> Result<()> {
    reader
        .seek(SeekFrom::Start(tag.data_pos()))
        .with_context(|| format!("seek to tag data @ {:#x}", tag.data_pos()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_tag_bytes(kind: i32, ftype: u32, size: i32, next: i32) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&kind.to_be_bytes());
        b[4..8].copy_from_slice(&ftype.to_be_bytes());
        b[8..12].copy_from_slice(&size.to_be_bytes());
        b[12..16].copy_from_slice(&next.to_be_bytes());
        b
    }

    #[test]
    fn round_trip_i32_tag() {
        let header = make_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, FIFFV_NEXT_SEQ);
        let payload = 42_i32.to_be_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&header);
        buf.extend_from_slice(&payload);

        let mut cursor = Cursor::new(buf);
        let tag = read_tag_header(&mut cursor, 0).unwrap();
        assert_eq!(tag.kind,  FIFF_NCHAN);
        assert_eq!(tag.ftype, FIFFT_INT);
        assert_eq!(tag.size,  4);
        assert_eq!(read_i32(&mut cursor, &tag).unwrap(), 42);
    }

    #[test]
    fn round_trip_f32_tag() {
        let header = make_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, FIFFV_NEXT_SEQ);
        let payload = 256_f32.to_be_bytes();
        let mut buf: Vec<u8> = header.to_vec();
        buf.extend_from_slice(&payload);

        let mut cursor = Cursor::new(buf);
        let tag = read_tag_header(&mut cursor, 0).unwrap();
        let v = read_f32(&mut cursor, &tag).unwrap();
        approx::assert_abs_diff_eq!(v, 256.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn next_pos_sequential() {
        let tag = TagHeader { kind: 1, ftype: 3, size: 8, next: 0, pos: 100 };
        assert_eq!(tag.next_pos(), Some(124)); // 100 + 16 + 8
    }

    #[test]
    fn next_pos_explicit() {
        let tag = TagHeader { kind: 1, ftype: 3, size: 8, next: 5000, pos: 100 };
        assert_eq!(tag.next_pos(), Some(5000));
    }

    #[test]
    fn next_pos_none() {
        let tag = TagHeader { kind: 1, ftype: 3, size: 8, next: -1, pos: 100 };
        assert_eq!(tag.next_pos(), None);
    }

    #[test]
    fn round_trip_string_tag() {
        let text = b"hello";
        let header = make_tag_bytes(FIFF_COMMENT, FIFFT_STRING, text.len() as i32, -1);
        let mut buf: Vec<u8> = header.to_vec();
        buf.extend_from_slice(text);
        let mut cursor = Cursor::new(buf);
        let tag = read_tag_header(&mut cursor, 0).unwrap();
        assert_eq!(read_string(&mut cursor, &tag).unwrap(), "hello");
    }
}
