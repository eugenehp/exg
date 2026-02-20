//! Measurement info (MNE's `Info` struct) read from a FIF file.
//!
//! We read only the fields needed for EEG preprocessing; MEG-specific fields
//! (projections, CTF compensations, HPI, …) are intentionally omitted.
use std::io::{Read, Seek};
use anyhow::{bail, Result};

use super::constants::*;
use super::tag::*;
use super::tree::Node;

// ── Channel info ─────────────────────────────────────────────────────────

/// Channel info, parsed from a `FIFFT_CH_INFO_STRUCT` (30) tag.
///
/// On-disk layout (big-endian, 100 bytes total):
/// ```text
///  4  scanno       i32
///  4  logno        i32
///  4  kind         i32
///  4  range        f32
///  4  cal          f32
///  4  coil_type    i32
/// 48  loc          12 × f32
///  4  unit         i32
///  4  unit_mul     i32
/// 16  ch_name      16 × u8 (null-padded Latin-1)
/// ─────────────────
/// 100 bytes
/// ```
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    pub scan_no:   i32,
    pub log_no:    i32,
    pub kind:      i32,
    pub range:     f32,
    pub cal:       f32,
    pub coil_type: i32,
    /// Position + orientation: `[x, y, z, nx0, ny0, nz0, …]` in metres.
    pub loc:       [f32; 12],
    pub unit:      i32,
    pub unit_mul:  i32,
    pub name:      String,
}

impl ChannelInfo {
    /// Calibration factor applied to raw integer/float samples: `cal × range`.
    #[inline]
    pub fn calibration(&self) -> f64 {
        (self.cal as f64) * (self.range as f64)
    }

    /// Parse from the 96-byte payload of a FIFFT_CH_INFO_STRUCT tag.
    ///
    /// Layout: scanno(4) + logno(4) + kind(4) + range(4) + cal(4) +
    ///         coil_type(4) + loc(48) + unit(4) + unit_mul(4) + ch_name(16) = 96
    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        if raw.len() < 96 {
            bail!("ch_info payload too short: {} bytes (need 96)", raw.len());
        }
        let scan_no   = i32::from_be_bytes(raw[0..4].try_into().unwrap());
        let log_no    = i32::from_be_bytes(raw[4..8].try_into().unwrap());
        let kind      = i32::from_be_bytes(raw[8..12].try_into().unwrap());
        let range     = f32::from_be_bytes(raw[12..16].try_into().unwrap());
        let cal       = f32::from_be_bytes(raw[16..20].try_into().unwrap());
        let coil_type = i32::from_be_bytes(raw[20..24].try_into().unwrap());
        let mut loc   = [0f32; 12];
        for (i, v) in loc.iter_mut().enumerate() {
            *v = f32::from_be_bytes(raw[24 + i * 4..24 + i * 4 + 4].try_into().unwrap());
        }
        let unit      = i32::from_be_bytes(raw[72..76].try_into().unwrap());
        let unit_mul  = i32::from_be_bytes(raw[76..80].try_into().unwrap());
        // Channel name: null-terminated Latin-1, 16 bytes
        let name_bytes = &raw[80..96];
        let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let name = name_bytes[..end].iter().map(|&b| b as char).collect();
        Ok(ChannelInfo { scan_no, log_no, kind, range, cal, coil_type, loc, unit, unit_mul, name })
    }
}

// ── Measurement info ─────────────────────────────────────────────────────

/// Measurement metadata extracted from `FIFFB_MEAS_INFO`.
///
/// Corresponds to MNE's `Info` object, restricted to fields we actually use.
#[derive(Debug, Clone)]
pub struct MeasInfo {
    pub n_chan:     usize,
    pub sfreq:     f64,
    pub lowpass:   Option<f64>,
    pub highpass:  Option<f64>,
    pub line_freq: Option<f64>,
    pub chs:       Vec<ChannelInfo>,
    pub bad_ch_names: Vec<String>,
    pub experimenter: Option<String>,
    pub description:  Option<String>,
}

impl MeasInfo {
    /// Calibration array `[n_chan]`: `cal[i] = chs[i].cal * chs[i].range`.
    pub fn cals(&self) -> Vec<f64> {
        self.chs.iter().map(|c| c.calibration()).collect()
    }

    /// Channel names in order.
    pub fn ch_names(&self) -> Vec<&str> {
        self.chs.iter().map(|c| c.name.as_str()).collect()
    }
}

/// Read `MeasInfo` from an open FIF file given the tree.
pub fn read_meas_info<R: Read + Seek>(reader: &mut R, tree: &Node) -> Result<MeasInfo> {
    // Navigate to FIFFB_MEAS_INFO
    let meas_node = tree
        .find_block(FIFFB_MEAS)
        .ok_or_else(|| anyhow::anyhow!("FIFFB_MEAS block not found"))?;
    let info_node = meas_node
        .find_block(FIFFB_MEAS_INFO)
        .ok_or_else(|| anyhow::anyhow!("FIFFB_MEAS_INFO block not found"))?;

    let mut n_chan     = None::<usize>;
    let mut sfreq      = None::<f64>;
    let mut lowpass    = None::<f64>;
    let mut highpass   = None::<f64>;
    let mut line_freq  = None::<f64>;
    let mut chs        = Vec::<ChannelInfo>::new();
    let mut bad_ch_names = Vec::<String>::new();
    let mut experimenter = None::<String>;
    let mut description  = None::<String>;

    for ent in &info_node.entries {
        match ent.kind {
            FIFF_NCHAN => {
                n_chan = Some(read_i32(reader, ent)? as usize);
            }
            FIFF_SFREQ => {
                sfreq = Some(read_f32(reader, ent)? as f64);
            }
            FIFF_LOWPASS => {
                let v = read_f32(reader, ent)?;
                if v.is_finite() {
                    lowpass = Some(v as f64);
                }
            }
            FIFF_HIGHPASS => {
                let v = read_f32(reader, ent)?;
                if v.is_finite() {
                    highpass = Some(v as f64);
                }
            }
            FIFF_LINE_FREQ => {
                let v = read_f32(reader, ent)?;
                if v.is_finite() {
                    line_freq = Some(v as f64);
                }
            }
            FIFF_CH_INFO => {
                let raw = read_raw_bytes(reader, ent)?;
                chs.push(ChannelInfo::from_bytes(&raw)?);
            }
            FIFF_BAD_CHS => {
                // List of bad channel names separated by colons.
                let s = read_string(reader, ent)?;
                bad_ch_names = s
                    .split(':')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            FIFF_EXPERIMENTER => {
                experimenter = Some(read_string(reader, ent)?);
            }
            FIFF_DESCRIPTION => {
                description = Some(read_string(reader, ent)?);
            }
            _ => {}
        }
    }

    let n_chan = n_chan.ok_or_else(|| anyhow::anyhow!("FIFF_NCHAN not found"))?;
    let sfreq  = sfreq.ok_or_else(|| anyhow::anyhow!("FIFF_SFREQ not found"))?;

    if chs.len() != n_chan {
        bail!("expected {n_chan} ch_info structs, got {}", chs.len());
    }

    Ok(MeasInfo { n_chan, sfreq, lowpass, highpass, line_freq, chs, bad_ch_names, experimenter, description })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ch_info_from_bytes_basic() {
        // Construct a minimal 96-byte payload.
        let mut raw = vec![0u8; 96];
        // kind = 2 (FIFFV_EEG_CH)
        raw[8..12].copy_from_slice(&2_i32.to_be_bytes());
        // range = 1.0
        raw[12..16].copy_from_slice(&1_f32.to_be_bytes());
        // cal = 2.0
        raw[16..20].copy_from_slice(&2_f32.to_be_bytes());
        // loc[0] = 0.5
        raw[24..28].copy_from_slice(&0.5_f32.to_be_bytes());
        // name = "Fp1\0..."
        raw[80..84].copy_from_slice(b"Fp1\0");

        let ch = ChannelInfo::from_bytes(&raw).unwrap();
        assert_eq!(ch.kind, 2);
        approx::assert_abs_diff_eq!(ch.range, 1.0_f32, epsilon = 1e-7);
        approx::assert_abs_diff_eq!(ch.cal, 2.0_f32, epsilon = 1e-7);
        approx::assert_abs_diff_eq!(ch.loc[0], 0.5_f32, epsilon = 1e-7);
        approx::assert_abs_diff_eq!(ch.calibration() as f32, 2.0, epsilon = 1e-6);
        assert_eq!(ch.name, "Fp1");
    }

    #[test]
    fn ch_info_too_short() {
        assert!(ChannelInfo::from_bytes(&[0u8; 95]).is_err());
    }
}
