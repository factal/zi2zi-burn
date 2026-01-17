#![recursion_limit = "256"]
use anyhow::{Context, Result};
use burn_cuda::{Cuda, CudaDevice};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::record::{CompactRecorder, Recorder};
use burn::config::Config;
use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs;
use std::path::{Path, PathBuf};
use zi2zi_burn::data::{build_batch, load_pickled_examples, DataConfig, Example};
use zi2zi_burn::model::split_real;
use zi2zi_burn::training::TrainingConfig;
use zi2zi_burn::utils::{compile_frames_to_gif, concat_images_horiz, merge_images, save_concat_images, tensor_to_images};

#[derive(Parser, Debug)]
#[command(about = "Inference for zi2zi")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long)]
    source_obj: PathBuf,
    #[arg(long)]
    embedding_ids: String,
    #[arg(long)]
    save_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    swap_ab: bool,
    #[arg(long, default_value_t = false)]
    interpolate: bool,
    #[arg(long, default_value_t = 10)]
    steps: usize,
    #[arg(long)]
    output_gif: Option<String>,
    #[arg(long, default_value_t = false)]
    uroboros: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    type Backend = Cuda<f32, i32>;
    let device = CudaDevice::default();

    let model_dir = resolve_model_dir(&args.model_dir, args.batch_size);
    if model_dir != args.model_dir {
        println!(
            "resolved model_dir {} -> {}",
            args.model_dir.display(),
            model_dir.display()
        );
    }

    let config_path = model_dir.join("config.json");
    let training_config =
        TrainingConfig::load(&config_path).context("failed to load config.json")?;
    let model_config = training_config.model;

    let mut generator = model_config.init_generator::<Backend>(&device);
    let record = CompactRecorder::new()
        .load(model_dir.join("generator"), &device)
        .context("failed to load generator weights")?;
    generator = generator.load_record(record);

    let data_config = DataConfig {
        image_size: model_config.image_size as u32,
        input_channels: model_config.input_channels,
        output_channels: model_config.output_channels,
        swap_ab: args.swap_ab,
    };

    let examples = load_pickled_examples(&args.source_obj)?;
    if examples.is_empty() {
        return Err(anyhow::anyhow!("no examples in {}", args.source_obj.display()));
    }

    fs::create_dir_all(&args.save_dir)?;
    let embedding_ids = parse_ids(&args.embedding_ids, model_config.embedding_num)?;
    let mut rng = StdRng::seed_from_u64(0);

    if !args.interpolate {
        infer_batches(
            &generator,
            &examples,
            &data_config,
            &embedding_ids,
            &args.save_dir,
            args.batch_size,
            &mut rng,
            &device,
        )?;
    } else {
        if embedding_ids.len() < 2 {
            return Err(anyhow::anyhow!("need at least two embedding ids to interpolate"));
        }
        let mut chain = embedding_ids.clone();
        if args.uroboros {
            chain.push(chain[0]);
        }
        for pair in chain.windows(2) {
            let start = pair[0];
            let end = pair[1];
            interpolate_chain(
                &generator,
                &examples,
                &data_config,
                start,
                end,
                args.steps,
                &args.save_dir,
                args.batch_size,
                &mut rng,
                &device,
            )?;
        }
        if let Some(gif_name) = &args.output_gif {
            compile_frames_to_gif(&args.save_dir, &args.save_dir.join(gif_name))?;
        }
    }

    Ok(())
}

/// Parse and validate a comma-separated list of embedding ids.
fn parse_ids(ids: &str, max: usize) -> Result<Vec<i64>> {
    let parsed: Vec<i64> = ids
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if parsed.is_empty() {
        return Err(anyhow::anyhow!("embedding_ids must not be empty"));
    }
    if parsed.iter().any(|&id| id < 0 || id >= max as i64) {
        return Err(anyhow::anyhow!(
            "embedding_ids must be in [0, {max})"
        ));
    }
    Ok(parsed)
}

/// Resolve a checkpoint directory from an experiment or checkpoint root.
fn resolve_model_dir(model_dir: &Path, batch_size: usize) -> PathBuf {
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

    if candidates.is_empty() {
        return model_dir.to_path_buf();
    }

    let batch_tag = format!("batch_{batch_size}");
    let preferred: Vec<_> = candidates
        .iter()
        .filter(|p| p.file_name().map(|s| s.to_string_lossy().contains(&batch_tag)).unwrap_or(false))
        .cloned()
        .collect();
    if !preferred.is_empty() {
        candidates = preferred;
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

/// Run inference over the dataset and save tiled result images.
fn infer_batches<B: Backend>(
    generator: &zi2zi_burn::model::Generator<B>,
    examples: &[Example],
    data_config: &DataConfig,
    embedding_ids: &[i64],
    save_dir: &Path,
    batch_size: usize,
    rng: &mut StdRng,
    device: &B::Device,
) -> Result<()> {
    let total_batches = (examples.len() + batch_size - 1) / batch_size;
    let mut buffer: Vec<image::RgbImage> = Vec::new();
    let mut count = 0usize;

    for batch_idx in 0..total_batches {
        let batch_refs = select_batch(examples, batch_size, batch_idx);
        let batch = build_batch::<B>(&batch_refs, data_config, false, rng, device)?;
        let labels_vec = build_labels(embedding_ids, batch_size, rng);
        let labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(labels_vec, [batch_size]),
            device,
        );

        let (real_a, real_b) = split_real(
            batch.images.clone(),
            data_config.input_channels,
            data_config.output_channels,
        );
        let (fake_b, _) = generator.forward(real_a.clone(), labels);
        let merged = format_pair(real_a, real_b, fake_b, batch_size, true)?;
        buffer.push(merged);
        if buffer.len() == 10 {
            let path = save_dir.join(format!("inferred_{count:04}.png"));
            save_concat_images(&buffer, &path)?;
            buffer.clear();
            count += 1;
        }
    }

    if !buffer.is_empty() {
        let path = save_dir.join(format!("inferred_{count:04}.png"));
        save_concat_images(&buffer, &path)?;
    }

    Ok(())
}

/// Interpolate between two embeddings and save frames for each step.
fn interpolate_chain<B: Backend>(
    generator: &zi2zi_burn::model::Generator<B>,
    examples: &[Example],
    data_config: &DataConfig,
    start_id: i64,
    end_id: i64,
    steps: usize,
    save_dir: &Path,
    batch_size: usize,
    rng: &mut StdRng,
    device: &B::Device,
) -> Result<()> {
    let steps = steps.max(1);
    let total_batches = (examples.len() + batch_size - 1) / batch_size;
    for step_idx in 0..=steps {
        let alpha = step_idx as f64 / steps as f64;
        let mut buffer: Vec<image::RgbImage> = Vec::new();
        for batch_idx in 0..total_batches {
            let batch_refs = select_batch(examples, batch_size, batch_idx);
            let batch = build_batch::<B>(&batch_refs, data_config, false, rng, device)?;
            let (real_a, real_b) = split_real(
                batch.images.clone(),
                data_config.input_channels,
                data_config.output_channels,
            );
            let (fake_b, _) = generator.forward_interpolated(real_a.clone(), start_id, end_id, alpha);
            let merged = format_pair(real_a, real_b, fake_b, batch_size, false)?;
            buffer.push(merged);
        }
        let filename = format!("frame_{start_id:02}_{end_id:02}_step_{step_idx:02}.png");
        save_concat_images(&buffer, &save_dir.join(filename))?;
    }
    Ok(())
}

/// Assemble a single grid image for (input, real, fake) or (real, fake).
fn format_pair<B: Backend>(
    real_a: Tensor<B, 4>,
    real_b: Tensor<B, 4>,
    fake_b: Tensor<B, 4>,
    batch_size: usize,
    include_input: bool,
) -> Result<image::RgbImage> {
    let input_imgs = tensor_to_images(real_a)?;
    let real_imgs = tensor_to_images(real_b)?;
    let fake_imgs = tensor_to_images(fake_b)?;
    let merged_real = merge_images(&real_imgs, batch_size, 1)?;
    let merged_fake = merge_images(&fake_imgs, batch_size, 1)?;
    if include_input {
        let merged_input = merge_images(&input_imgs, batch_size, 1)?;
        concat_images_horiz(&[merged_input, merged_real, merged_fake])
    } else {
        concat_images_horiz(&[merged_real, merged_fake])
    }
}

/// Select a batch with wraparound to avoid short final batches.
fn select_batch<'a>(examples: &'a [Example], batch_size: usize, batch_idx: usize) -> Vec<&'a Example> {
    let mut batch = Vec::with_capacity(batch_size);
    let start = batch_idx * batch_size;
    for offset in 0..batch_size {
        let idx = (start + offset) % examples.len();
        batch.push(&examples[idx]);
    }
    batch
}

/// Build a label list for a batch, randomly sampling from the provided ids.
fn build_labels(ids: &[i64], batch_size: usize, rng: &mut StdRng) -> Vec<i64> {
    if ids.len() == 1 {
        return vec![ids[0]; batch_size];
    }
    (0..batch_size)
        .map(|_| {
            let idx = rng.gen_range(0..ids.len());
            ids[idx]
        })
        .collect()
}
