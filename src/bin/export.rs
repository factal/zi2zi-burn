#![recursion_limit = "256"]
use anyhow::{Context, Result};
use burn::{
    backend::{Cuda, cuda::CudaDevice},
    record::{CompactRecorder, Recorder},
    config::Config,
    module::Module,
};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use zi2zi_burn::training::TrainingConfig;

#[derive(Parser, Debug)]
#[command(about = "Export generator weights")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    save_dir: PathBuf,
    #[arg(long, default_value = "gen_model")]
    model_name: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    type Backend = Cuda<f32, i32>;
    let device = CudaDevice::default();

    let model_dir = resolve_model_dir(&args.model_dir);
    let config_path = model_dir.join("config.json");
    let training_config =
        TrainingConfig::load(&config_path).context("failed to load config.json")?;
    let model_config = training_config.model;

    let mut generator = model_config.init_generator::<Backend>(&device);
    let record = CompactRecorder::new()
        .load(model_dir.join("generator"), &device)
        .context("failed to load generator weights")?;
    generator = generator.load_record(record);

    fs::create_dir_all(&args.save_dir)?;
    let save_path = args.save_dir.join(&args.model_name);
    generator
        .clone()
        .save_file(save_path, &CompactRecorder::new())?;
    Ok(())
}

/// Resolve a checkpoint directory from an experiment or checkpoint root.
fn resolve_model_dir(model_dir: &Path) -> PathBuf {
    if model_dir.join("config.json").exists() && model_dir.join("generator").exists() {
        return model_dir.to_path_buf();
    }
    let checkpoint_root = if model_dir.join("checkpoint").is_dir() {
        model_dir.join("checkpoint")
    } else {
        model_dir.to_path_buf()
    };
    let mut candidates = Vec::new();
    if let Ok(entries) = fs::read_dir(&checkpoint_root) {
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            if path.join("config.json").exists() && path.join("generator").exists() {
                candidates.push(path);
            }
        }
    }
    candidates.sort_by_key(|p| latest_mtime(p));
    candidates.last().cloned().unwrap_or_else(|| model_dir.to_path_buf())
}

fn latest_mtime(path: &Path) -> std::time::SystemTime {
    let generator = path.join("generator");
    if let Ok(meta) = fs::metadata(&generator) {
        if let Ok(time) = meta.modified() {
            return time;
        }
    }
    fs::metadata(path).and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH)
}
