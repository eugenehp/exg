/// pipeline_steps: read a FIF file, run each preprocessing step, write every
/// intermediate array to a safetensors file for comparison against Python/MNE.
///
/// Output keys:
///   raw           [C, T_orig]   f32  raw FIF data (cast from f64)
///   resample      [C, T]        f32  after resample to 256 Hz
///   hp            [C, T]        f32  after 0.5 Hz highpass FIR
///   ref           [C, T]        f32  after average reference
///   zscore        [C, T]        f32  after global z-score
///   epoch_N       [C, 1280]     f32  after epoch + baseline (N = 0, 1, …)
///   final_N       [C, 1280]     f32  epoch_N / data_norm
///   n_epochs      [1]           i32
///   zscore_mean   [1]           f32
///   zscore_std    [1]           f32
use anyhow::Result;
use clap::Parser;
use ndarray::Array2;
use std::path::PathBuf;

use exg::{
    epoch::epoch_and_baseline,
    filter::{apply_fir_zero_phase, design_highpass},
    fiff::raw::open_raw,
    io::StWriter,
    normalize::zscore_global_inplace,
    reference::average_reference_inplace,
    resample::resample,
};

#[derive(Parser, Debug)]
#[command(name = "pipeline_steps")]
struct Args {
    /// Input FIF file.
    #[arg(long)]
    fif: PathBuf,

    /// Output safetensors path.
    #[arg(long)]
    output: PathBuf,

    /// Target sampling rate (Hz).
    #[arg(long, default_value_t = 256.0_f32)]
    sfreq: f32,

    /// Highpass cutoff (Hz).
    #[arg(long, default_value_t = 0.5_f32)]
    hp: f32,

    /// Epoch duration (s).
    #[arg(long, default_value_t = 5.0_f32)]
    epoch_dur: f32,

    /// Data normalisation divisor.
    #[arg(long, default_value_t = 10.0_f32)]
    data_norm: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let epoch_samples = (args.epoch_dur * args.sfreq) as usize;

    // ── 1. Read FIF ────────────────────────────────────────────────────────
    let t_fif = now();
    let raw_fif = open_raw(&args.fif)?;
    let src_sfreq = raw_fif.info.sfreq as f32;
    let data_f64: Array2<f64> = raw_fif.read_all_data()?;
    let raw_f32: Array2<f32> = data_f64.mapv(|v| v as f32);
    let ms_fif = t_fif.elapsed().as_secs_f64() * 1000.0;
    let (n_ch, _n_t_orig) = raw_f32.dim();

    // ── 2. Resample ────────────────────────────────────────────────────────
    let t_rs = now();
    let data_rs = resample(&raw_f32, src_sfreq, args.sfreq)?;
    let ms_rs = t_rs.elapsed().as_secs_f64() * 1000.0;

    // ── 3. Highpass filter ─────────────────────────────────────────────────
    let t_hp = now();
    let mut data_hp = data_rs.clone();
    let h = design_highpass(args.hp, args.sfreq);
    apply_fir_zero_phase(&mut data_hp, &h)?;
    let ms_hp = t_hp.elapsed().as_secs_f64() * 1000.0;

    // ── 4. Average reference ───────────────────────────────────────────────
    let t_ref = now();
    let mut data_ref = data_hp.clone();
    average_reference_inplace(&mut data_ref);
    let ms_ref = t_ref.elapsed().as_secs_f64() * 1000.0;

    // ── 5. Z-score ─────────────────────────────────────────────────────────
    let t_z = now();
    let mut data_z = data_ref.clone();
    let (mean, std) = zscore_global_inplace(&mut data_z);
    let ms_z = t_z.elapsed().as_secs_f64() * 1000.0;

    // ── 6. Epoch + baseline ────────────────────────────────────────────────
    let t_ep = now();
    let epochs = epoch_and_baseline(&data_z, epoch_samples);
    let n_epochs = epochs.len();
    let ms_ep = t_ep.elapsed().as_secs_f64() * 1000.0;

    // Print internal step timings to stderr (parsed by compare.py).
    // Format: "TIMING fif=Xms resample=Xms hp=Xms ref=Xms zscore=Xms epoch=Xms"
    eprintln!(
        "TIMING fif={ms_fif:.4}ms resample={ms_rs:.4}ms hp={ms_hp:.4}ms \
         ref={ms_ref:.4}ms zscore={ms_z:.4}ms epoch={ms_ep:.4}ms",
    );
    eprintln!(
        "  {n_ch} ch  src_sfreq={src_sfreq} Hz  {n_epochs} epochs"
    );

    // ── 7. Write output ────────────────────────────────────────────────────
    eprintln!("Writing → {}", args.output.display());
    let mut w = StWriter::new();

    // Store each step at f32 precision (FIF buffers are f32 on disk).
    w.add_f32_arr2("raw",     &raw_f32);
    w.add_f32_arr2("resample", &data_rs);
    w.add_f32_arr2("hp",       &data_hp);
    w.add_f32_arr2("ref",      &data_ref);
    w.add_f32_arr2("zscore",   &data_z);

    for (i, ep) in epochs.iter().enumerate() {
        w.add_f32_arr2(&format!("epoch_{i}"), ep);
        let final_ep = ep.mapv(|v| v / args.data_norm);
        w.add_f32_arr2(&format!("final_{i}"), &final_ep);
    }

    w.add_i32("n_epochs",    &[n_epochs as i32],     &[1]);
    w.add_f32("zscore_mean", &[mean],                 &[1]);
    w.add_f32("zscore_std",  &[std],                  &[1]);
    w.write(&args.output)?;

    eprintln!("Done.");
    Ok(())
}

/// Return `std::time::Instant::now()` (used for internal timing).
#[inline(always)]
fn now() -> std::time::Instant { std::time::Instant::now() }
