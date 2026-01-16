use anyhow::{Context, Result};
use clap::Parser;
use glob::glob;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use serde_pickle::{to_writer, SerOptions};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Package training examples into pickled files")]
struct Args {
    #[arg(long)]
    dir: PathBuf,
    #[arg(long)]
    save_dir: PathBuf,
    #[arg(long, default_value_t = 0.1)]
    split_ratio: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// (label, jpeg_bytes) tuple written to the pickle stream.
#[derive(Serialize)]
struct PickledExample(i64, Vec<u8>);

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.save_dir.exists() {
        fs::create_dir_all(&args.save_dir).with_context(|| {
            format!("failed to create {}", args.save_dir.display())
        })?;
    }

    let train_path = args.save_dir.join("train.obj");
    let val_path = args.save_dir.join("val.obj");
    let mut train_file = BufWriter::new(
        File::create(&train_path).with_context(|| format!("failed to create {}", train_path.display()))?,
    );
    let mut val_file = BufWriter::new(
        File::create(&val_path).with_context(|| format!("failed to create {}", val_path.display()))?,
    );

    let mut rng = StdRng::seed_from_u64(args.seed);
    let options = SerOptions::default();
    let mut paths: Vec<_> = glob(&format!("{}/*.jpg", args.dir.display()))?
        .filter_map(Result::ok)
        .collect();
    paths.sort();

    for path in paths {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("invalid filename")?;
        let label_str = file_name
            .split('_')
            .next()
            .context("filename must include label prefix")?;
        let label: i64 = label_str.parse().with_context(|| {
            format!("failed to parse label from {file_name}")
        })?;
        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let example = PickledExample(label, bytes);
        let r: f64 = rng.gen_range(0.0..1.0);
        if r < args.split_ratio {
            to_writer(&mut val_file, &example, options.clone())?;
        } else {
            to_writer(&mut train_file, &example, options.clone())?;
        }
    }

    Ok(())
}
