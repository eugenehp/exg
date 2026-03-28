//! XDF (Extensible Data Format) file writer.
//!
//! Implements the XDF 1.0 specification:
//! <https://github.com/sccn/xdf/wiki/Specifications>
//!
//! Chunk layout:  `[NumLengthBytes] [Length] [Tag(u16)] [Content…]`

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

/// XDF chunk tags
#[repr(u16)]
#[derive(Clone, Copy)]
enum ChunkTag {
    FileHeader = 1,
    StreamHeader = 2,
    Samples = 3,
    ClockOffset = 4,
    Boundary = 5,
    StreamFooter = 6,
}

/// XDF file writer (thread-safe).
pub struct XdfWriter {
    inner: Mutex<BufWriter<File>>,
}

impl XdfWriter {
    /// Create a new XDF file and write the magic header + FileHeader chunk.
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Magic bytes: "XDF:"
        w.write_all(b"XDF:")?;

        // FileHeader chunk
        let content = "<?xml version=\"1.0\"?><info><version>1.0</version></info>";
        write_chunk(&mut w, ChunkTag::FileHeader, None, content.as_bytes())?;
        w.flush()?;

        Ok(XdfWriter {
            inner: Mutex::new(w),
        })
    }

    /// Write a StreamHeader chunk.
    pub fn write_stream_header(&self, stream_id: u32, xml_content: &str) -> anyhow::Result<()> {
        let mut w = self.inner.lock().unwrap();
        write_chunk(
            &mut *w,
            ChunkTag::StreamHeader,
            Some(stream_id),
            xml_content.as_bytes(),
        )?;
        w.flush()?;
        Ok(())
    }

    /// Write a StreamFooter chunk.
    pub fn write_stream_footer(&self, stream_id: u32, xml_content: &str) -> anyhow::Result<()> {
        let mut w = self.inner.lock().unwrap();
        write_chunk(
            &mut *w,
            ChunkTag::StreamFooter,
            Some(stream_id),
            xml_content.as_bytes(),
        )?;
        w.flush()?;
        Ok(())
    }

    /// Write a ClockOffset chunk.
    pub fn write_clock_offset(
        &self,
        stream_id: u32,
        collection_time: f64,
        offset_value: f64,
    ) -> anyhow::Result<()> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&collection_time.to_le_bytes());
        buf.extend_from_slice(&offset_value.to_le_bytes());
        let mut w = self.inner.lock().unwrap();
        write_chunk(&mut *w, ChunkTag::ClockOffset, Some(stream_id), &buf)?;
        Ok(())
    }

    /// Write a Boundary chunk.
    pub fn write_boundary(&self) -> anyhow::Result<()> {
        let boundary_uuid: [u8; 16] = [
            0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83,
            0xCB, 0xE4,
        ];
        let mut w = self.inner.lock().unwrap();
        write_chunk(&mut *w, ChunkTag::Boundary, None, &boundary_uuid)?;
        Ok(())
    }

    /// Write a Samples chunk with numeric data.
    pub fn write_samples_numeric<T: NumericSample>(
        &self,
        stream_id: u32,
        timestamps: &[f64],
        data: &[T],
        n_channels: u32,
    ) -> anyhow::Result<()> {
        if timestamps.is_empty() {
            return Ok(());
        }
        let n_samples = timestamps.len();
        assert_eq!(data.len(), n_samples * n_channels as usize);

        let mut payload = Vec::with_capacity(n_samples * (9 + n_channels as usize * T::SIZE));
        write_varlen_int(&mut payload, n_samples as u64);
        for (i, &ts) in timestamps.iter().enumerate() {
            if ts == 0.0 {
                payload.push(0);
            } else {
                payload.push(8);
                payload.extend_from_slice(&ts.to_le_bytes());
            }
            let offset = i * n_channels as usize;
            for j in 0..n_channels as usize {
                data[offset + j].write_le(&mut payload);
            }
        }

        let mut w = self.inner.lock().unwrap();
        write_chunk(&mut *w, ChunkTag::Samples, Some(stream_id), &payload)?;
        Ok(())
    }

    /// Write a Samples chunk with string data.
    #[allow(dead_code)]
    pub fn write_samples_string(
        &self,
        stream_id: u32,
        timestamps: &[f64],
        data: &[String],
        n_channels: u32,
    ) -> anyhow::Result<()> {
        if timestamps.is_empty() {
            return Ok(());
        }
        let n_samples = timestamps.len();
        assert_eq!(data.len(), n_samples * n_channels as usize);

        let mut payload = Vec::with_capacity(256);
        write_varlen_int(&mut payload, n_samples as u64);
        for (i, &ts) in timestamps.iter().enumerate() {
            if ts == 0.0 {
                payload.push(0);
            } else {
                payload.push(8);
                payload.extend_from_slice(&ts.to_le_bytes());
            }
            let offset = i * n_channels as usize;
            for j in 0..n_channels as usize {
                let s = &data[offset + j];
                write_varlen_int(&mut payload, s.len() as u64);
                payload.extend_from_slice(s.as_bytes());
            }
        }

        let mut w = self.inner.lock().unwrap();
        write_chunk(&mut *w, ChunkTag::Samples, Some(stream_id), &payload)?;
        Ok(())
    }

    /// Current file size in bytes.
    #[allow(dead_code)]
    pub fn file_size(&self) -> u64 {
        let w = self.inner.lock().unwrap();
        w.get_ref().metadata().map(|m| m.len()).unwrap_or(0)
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn write_chunk<W: Write>(
    w: &mut W,
    tag: ChunkTag,
    stream_id: Option<u32>,
    content: &[u8],
) -> std::io::Result<()> {
    // length = tag(2) + optional stream_id(4) + content
    let length = 2 + if stream_id.is_some() { 4 } else { 0 } + content.len();
    write_chunk_length(w, length as u64)?;
    w.write_all(&(tag as u16).to_le_bytes())?;
    if let Some(sid) = stream_id {
        w.write_all(&sid.to_le_bytes())?;
    }
    w.write_all(content)?;
    Ok(())
}

fn write_chunk_length<W: Write>(w: &mut W, length: u64) -> std::io::Result<()> {
    if length < 256 {
        w.write_all(&[1])?; // NumLengthBytes = 1
        w.write_all(&[length as u8])?;
    } else if length < (1u64 << 32) {
        w.write_all(&[4])?;
        w.write_all(&(length as u32).to_le_bytes())?;
    } else {
        w.write_all(&[8])?;
        w.write_all(&length.to_le_bytes())?;
    }
    Ok(())
}

fn write_varlen_int(buf: &mut Vec<u8>, val: u64) {
    if val < 256 {
        buf.push(1);
        buf.push(val as u8);
    } else if val < (1u64 << 32) {
        buf.push(4);
        buf.extend_from_slice(&(val as u32).to_le_bytes());
    } else {
        buf.push(8);
        buf.extend_from_slice(&val.to_le_bytes());
    }
}

// ── numeric sample trait ─────────────────────────────────────────────

pub trait NumericSample: Copy {
    const SIZE: usize;
    fn write_le(&self, buf: &mut Vec<u8>);
}

impl NumericSample for f32 {
    const SIZE: usize = 4;
    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}
impl NumericSample for f64 {
    const SIZE: usize = 8;
    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}
impl NumericSample for i32 {
    const SIZE: usize = 4;
    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}
impl NumericSample for i16 {
    const SIZE: usize = 2;
    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}
impl NumericSample for i64 {
    const SIZE: usize = 8;
    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
}
impl NumericSample for i8 {
    const SIZE: usize = 1;
    fn write_le(&self, buf: &mut Vec<u8>) {
        buf.push(*self as u8);
    }
}
