#![recursion_limit = "256"]
use anyhow::{Context, Result};
use burn_cuda::{Cuda, CudaDevice};
use burn::module::Ignored;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::record::{CompactRecorder, Recorder};
use burn::config::Config;
use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use zi2zi_burn::data::{build_batch, load_pickled_examples, DataConfig, Example};
use zi2zi_burn::model::{Discriminator, Generator, LossConfig, ModelConfig, split_real};
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

#[derive(Module, Debug)]
struct Zi2ziGan<B: Backend> {
    generator: Generator<B>,
    discriminator: Discriminator<B>,
    model_config: Ignored<ModelConfig>,
    loss_config: Ignored<LossConfig>,
}

impl<B: Backend> Zi2ziGan<B> {
    fn new(model_config: ModelConfig, loss_config: LossConfig, device: &B::Device) -> Self {
        let generator = model_config.init_generator(device);
        let discriminator = model_config.init_discriminator(device);
        Self {
            generator,
            discriminator,
            model_config: Ignored(model_config),
            loss_config: Ignored(loss_config),
        }
    }
}

#[derive(Deserialize)]
struct TrainingState {
    step: usize,
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
    let TrainingConfig {
        model: model_config,
        loss: loss_config,
        ..
    } = training_config;

    let checkpoint_path = resolve_checkpoint_path(&model_dir)
        .context("failed to resolve model checkpoint")?;
    let mut model = Zi2ziGan::<Backend>::new(model_config.clone(), loss_config, &device);
    let record = CompactRecorder::new()
        .load(checkpoint_path, &device)
        .context("failed to load model checkpoint")?;
    model = model.load_record(record);
    let generator = model.generator;

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
    if is_model_dir(model_dir) {
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
            if is_model_dir(&path) && has_model_checkpoints(&path) {
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

fn is_model_dir(path: &Path) -> bool {
    path.join("config.json").exists() && path.join("checkpoint").is_dir()
}

fn has_model_checkpoints(path: &Path) -> bool {
    let checkpoint_dir = path.join("checkpoint");
    if !checkpoint_dir.is_dir() {
        return false;
    }
    fs::read_dir(&checkpoint_dir)
        .map(|entries| {
            entries.flatten().any(|entry| is_model_checkpoint_file(&entry.path()))
        })
        .unwrap_or(false)
}

fn is_model_checkpoint_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let file_name = path.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
    file_name.starts_with("model-") && path.extension().map(|s| s == "mpk").unwrap_or(false)
}

fn latest_mtime(path: &Path) -> std::time::SystemTime {
    let checkpoint_dir = path.join("checkpoint");
    if let Ok(entries) = fs::read_dir(&checkpoint_dir) {
        let mut latest = std::time::SystemTime::UNIX_EPOCH;
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if !is_model_checkpoint_file(&entry_path) {
                continue;
            }
            if let Ok(time) = entry_path.metadata().and_then(|m| m.modified()) {
                if time > latest {
                    latest = time;
                }
            }
        }
        if latest != std::time::SystemTime::UNIX_EPOCH {
            return latest;
        }
    }
    fs::metadata(path).and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH)
}

fn resolve_checkpoint_path(model_dir: &Path) -> Result<PathBuf> {
    let checkpoint_dir = model_dir.join("checkpoint");
    if !checkpoint_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "checkpoint directory not found: {}",
            checkpoint_dir.display()
        ));
    }

    if let Some(state) = read_training_state(model_dir) {
        let candidate = checkpoint_dir.join(format!("model-{}", state.step));
        if candidate.with_extension("mpk").exists() {
            return Ok(candidate);
        }
    }

    let mut best_step: Option<(usize, PathBuf)> = None;
    let mut best_time: Option<(std::time::SystemTime, PathBuf)> = None;
    if let Ok(entries) = fs::read_dir(&checkpoint_dir) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if !is_model_checkpoint_file(&entry_path) {
                continue;
            }
            if let Some(step) = checkpoint_step(&entry_path) {
                let update = match best_step {
                    Some((best, _)) => step > best,
                    None => true,
                };
                if update {
                    best_step = Some((step, entry_path.clone()));
                }
            }
            if let Ok(time) = entry_path.metadata().and_then(|m| m.modified()) {
                let update = match best_time {
                    Some((best, _)) => time > best,
                    None => true,
                };
                if update {
                    best_time = Some((time, entry_path.clone()));
                }
            }
        }
    }

    if let Some((_, path)) = best_step {
        return Ok(path.with_extension(""));
    }
    if let Some((_, path)) = best_time {
        return Ok(path.with_extension(""));
    }

    Err(anyhow::anyhow!(
        "no model checkpoints found in {}",
        checkpoint_dir.display()
    ))
}

fn read_training_state(model_dir: &Path) -> Option<TrainingState> {
    let state_path = model_dir.join("state.json");
    let contents = fs::read_to_string(state_path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn checkpoint_step(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_string_lossy();
    let step = stem.strip_prefix("model-")?;
    step.parse::<usize>().ok()
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
