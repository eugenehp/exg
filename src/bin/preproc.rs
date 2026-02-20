use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use exg::{preprocess, PipelineConfig, io::{RawData, write_batch}};

#[derive(Parser)]
#[command(name = "preproc", about = "EEG preprocessing pipeline (Rust/Burn)")]
struct Args {
    /// raw.safetensors from scripts/read_raw.py
    #[arg(long)]
    input: PathBuf,

    /// batch.safetensors output path
    #[arg(long)]
    output: PathBuf,

    /// Data normalisation divisor (default: 10.0)
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Highpass cutoff in Hz (default: 0.5)
    #[arg(long, default_value_t = 0.5)]
    hp_freq: f32,

    /// Channel names to zero out (comma-separated)
    #[arg(long, default_value = "")]
    bad_channels: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let raw = RawData::load(&args.input)?;
    println!("Loaded {} ch × {} samples @ {} Hz",
        raw.data.nrows(), raw.data.ncols(), raw.sfreq);

    let bad: Vec<String> = if args.bad_channels.is_empty() {
        vec![]
    } else {
        args.bad_channels.split(',').map(str::to_string).collect()
    };

    let cfg = PipelineConfig {
        target_sfreq: 256.0,
        hp_freq: args.hp_freq,
        epoch_dur: 5.0,
        data_norm: args.data_norm,
        bad_channels: bad,
    };

    let epochs = preprocess(raw.data, raw.chan_pos, raw.sfreq, &cfg)?;
    println!("Produced {} epochs", epochs.len());

    let (eeg_list, pos_list): (Vec<_>, Vec<_>) = epochs.into_iter().unzip();
    write_batch(&eeg_list, &pos_list, &args.output)?;
    println!("Written → {}", args.output.display());

    Ok(())
}
