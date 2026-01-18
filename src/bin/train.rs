#![recursion_limit = "256"]
use anyhow::{Context, Result};
use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    config::Config,
};
use clap::Parser;
use std::path::PathBuf;
use zi2zi_burn::training::TrainingConfig;

#[derive(Parser, Debug)]
#[command(about = "Train zi2zi with Burn")]
struct Args {
    #[arg(long)]
    experiment_dir: PathBuf,
    #[arg(long, default_value = "config.json")]
    config: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = TrainingConfig::load(&args.config)
        .with_context(|| format!("failed to load config from {}", args.config.display()))?;

    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    let device = CudaDevice::default();

    zi2zi_burn::training::train::<AutodiffBackend>(&args.experiment_dir, config, device)?;
    Ok(())
}
