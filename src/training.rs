use crate::data::{build_batch, load_pickled_examples, DataConfig, Example};
use crate::model::{compute_losses, LossConfig, ModelConfig};
use crate::utils::{merge_images, save_concat_images, tensor_to_images};
use anyhow::{Context, Result};
use burn::config::Config;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Training configuration loaded from `config.json`.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub loss: LossConfig,
    pub data_dir: String,
    pub experiment_id: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub schedule: usize,
    pub min_learning_rate: f64,
    pub resume: bool,
    pub flip_labels: bool,
    pub swap_ab: bool,
    pub sample_steps: usize,
    pub checkpoint_steps: usize,
    pub seed: u64,
    pub optimizer_gen: AdamConfig,
    pub optimizer_disc: AdamConfig,
    pub fine_tune: Option<Vec<i64>>,
}

/// Persisted training state for resume support.
#[derive(Serialize, Deserialize, Default)]
struct TrainingState {
    step: usize,
    epoch: usize,
    learning_rate: f64,
}

/// Train zi2zi models with Burn, handling checkpoints and sampling.
pub fn train<B: AutodiffBackend>(
    experiment_dir: &Path,
    config: TrainingConfig,
    device: B::Device,
) -> Result<()> {
    let checkpoint_dir = experiment_dir.join("checkpoint");
    let sample_dir = experiment_dir.join("sample");
    let log_dir = experiment_dir.join("logs");
    std::fs::create_dir_all(&checkpoint_dir)?;
    std::fs::create_dir_all(&sample_dir)?;
    std::fs::create_dir_all(&log_dir)?;

    let model_id = format!("experiment_{}_batch_{}", config.experiment_id, config.batch_size);
    let model_dir = checkpoint_dir.join(model_id);
    std::fs::create_dir_all(&model_dir)?;
    config.save(model_dir.join("config.json"))?;

    let data_dir = resolve_data_dir(experiment_dir, &config.data_dir);
    let train_path = data_dir.join("train.obj");
    let val_path = data_dir.join("val.obj");

    let mut train_examples = load_pickled_examples(&train_path)
        .with_context(|| format!("failed to load {}", train_path.display()))?;
    let mut val_examples = load_pickled_examples(&val_path)
        .with_context(|| format!("failed to load {}", val_path.display()))?;

    if let Some(ids) = &config.fine_tune {
        train_examples.retain(|ex| ids.contains(&ex.label));
        val_examples.retain(|ex| ids.contains(&ex.label));
    }

    println!(
        "train examples -> {}, val examples -> {}",
        train_examples.len(),
        val_examples.len()
    );
    if train_examples.is_empty() {
        return Err(anyhow::anyhow!("no training examples found"));
    }

    let data_config = DataConfig {
        image_size: config.model.image_size as u32,
        input_channels: config.model.input_channels,
        output_channels: config.model.output_channels,
        swap_ab: config.swap_ab,
    };

    B::seed(&device, config.seed);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut generator = config.model.init_generator::<B>(&device);
    let mut discriminator = config.model.init_discriminator::<B>(&device);
    let mut optim_gen =
        config
            .optimizer_gen
            .init::<B, crate::model::generator::Generator<B>>();
    let mut optim_disc =
        config
            .optimizer_disc
            .init::<B, crate::model::discriminator::Discriminator<B>>();

    let mut state = TrainingState {
        learning_rate: config.learning_rate,
        ..Default::default()
    };
    if config.resume {
        if let Some(loaded) = load_checkpoint(
            &model_dir,
            &device,
            &mut generator,
            &mut discriminator,
            &mut optim_gen,
            &mut optim_disc,
        )? {
            state = loaded;
        }
    }

    let mut val_index = 0usize;
    let start_time = Instant::now();

    for epoch in state.epoch..config.num_epochs {
        if (epoch + 1) % config.schedule == 0 {
            let next_lr = (state.learning_rate / 2.0).max(config.min_learning_rate);
            if next_lr != state.learning_rate {
                println!(
                    "decay learning rate from {:.5} to {:.5}",
                    state.learning_rate, next_lr
                );
                state.learning_rate = next_lr;
            }
        }

        train_examples.shuffle(&mut rng);
        let total_batches = (train_examples.len() + config.batch_size - 1) / config.batch_size;

        for batch_idx in 0..total_batches {
            state.step += 1;
            let batch_refs = select_batch(&train_examples, config.batch_size, batch_idx);
            let batch = build_batch::<B>(&batch_refs, &data_config, true, &mut rng, &device)?;

            let labels = batch
                .labels
                .to_data()
                .to_vec::<i32>()
                .context("failed to read labels")?;
            let mut shuffled_labels = labels.clone();
            if config.flip_labels {
                shuffled_labels.shuffle(&mut rng);
            }

            let embedding_ids = Tensor::<B, 1, Int>::from_data(
                TensorData::new(labels.clone(), [labels.len()]),
                &device,
            );

            let no_target = if config.flip_labels {
                let shuffled = Tensor::<B, 1, Int>::from_data(
                    TensorData::new(shuffled_labels, [labels.len()]),
                    &device,
                );
                Some((batch.images.clone(), shuffled))
            } else {
                None
            };

            let losses_d = compute_losses(
                &generator,
                &discriminator,
                &config.model,
                &config.loss,
                batch.images.clone(),
                embedding_ids.clone(),
                no_target.clone(),
            );
            let grads_d = losses_d.d_loss.backward();
            let grads_d = GradientsParams::from_grads(grads_d, &discriminator);
            discriminator = optim_disc.step(state.learning_rate, discriminator, grads_d);

            let losses_g = compute_losses(
                &generator,
                &discriminator,
                &config.model,
                &config.loss,
                batch.images.clone(),
                embedding_ids.clone(),
                no_target.clone(),
            );
            let grads_g = losses_g.g_loss.backward();
            let grads_g = GradientsParams::from_grads(grads_g, &generator);
            generator = optim_gen.step(state.learning_rate, generator, grads_g);

            // Match the original TF1 loop: apply a second generator update per batch.
            let losses_g2 = compute_losses(
                &generator,
                &discriminator,
                &config.model,
                &config.loss,
                batch.images,
                embedding_ids,
                no_target,
            );
            let grads_g2 = losses_g2.g_loss.backward();
            let grads_g2 = GradientsParams::from_grads(grads_g2, &generator);
            generator = optim_gen.step(state.learning_rate, generator, grads_g2);

            let elapsed = start_time.elapsed().as_secs_f32();
            println!(
                "Epoch: [{:2}], [{:4}/{:4}] time: {:.4}, d_loss: {:.5}, g_loss: {:.5}, category_loss: {:.5}, cheat_loss: {:.5}, const_loss: {:.5}, l1_loss: {:.5}, tv_loss: {:.5}",
                epoch,
                batch_idx,
                total_batches,
                elapsed,
                losses_g2.d_loss.into_scalar(),
                losses_g2.g_loss.into_scalar(),
                losses_g2.category_loss.into_scalar(),
                losses_g2.cheat_loss.into_scalar(),
                losses_g2.const_loss.into_scalar(),
                losses_g2.l1_loss.into_scalar(),
                losses_g2.tv_loss.into_scalar(),
            );

            if state.step % config.sample_steps == 0 {
                validate_model(
                    &generator,
                    &val_examples,
                    &data_config,
                    &config,
                    &model_dir,
                    &mut val_index,
                    &mut rng,
                    &device,
                    epoch,
                    state.step,
                )?;
            }

            if state.step % config.checkpoint_steps == 0 {
                println!("Checkpoint: save checkpoint step {}", state.step);
                save_checkpoint(
                    &model_dir,
                    &generator,
                    &discriminator,
                    &optim_gen,
                    &optim_disc,
                    &state,
                )?;
            }
        }

        state.epoch = epoch + 1;
    }

    println!("Checkpoint: last checkpoint step {}", state.step);
    save_checkpoint(
        &model_dir,
        &generator,
        &discriminator,
        &optim_gen,
        &optim_disc,
        &state,
    )?;

    Ok(())
}

/// Resolve `data_dir` relative to the experiment directory if needed.
fn resolve_data_dir(experiment_dir: &Path, data_dir: &str) -> PathBuf {
    let candidate = PathBuf::from(data_dir);
    if candidate.is_relative() {
        experiment_dir.join(candidate)
    } else {
        candidate
    }
}

/// Select a batch with wraparound to avoid short final batches.
fn select_batch<'a>(
    examples: &'a [Example],
    batch_size: usize,
    batch_idx: usize,
) -> Vec<&'a Example> {
    let mut batch = Vec::with_capacity(batch_size);
    let start = batch_idx * batch_size;
    for offset in 0..batch_size {
        let idx = (start + offset) % examples.len();
        batch.push(&examples[idx]);
    }
    batch
}

/// Save model, optimizer, and state to the checkpoint directory.
fn save_checkpoint<B, OG, OD>(
    model_dir: &Path,
    generator: &crate::model::generator::Generator<B>,
    discriminator: &crate::model::discriminator::Discriminator<B>,
    optim_gen: &OG,
    optim_disc: &OD,
    state: &TrainingState,
) -> Result<()>
where
    B: AutodiffBackend,
    OG: Optimizer<crate::model::generator::Generator<B>, B>,
    OD: Optimizer<crate::model::discriminator::Discriminator<B>, B>,
{
    let recorder = CompactRecorder::new();
    generator
        .clone()
        .save_file(model_dir.join("generator"), &recorder)?;
    discriminator
        .clone()
        .save_file(model_dir.join("discriminator"), &recorder)?;
    recorder.record(optim_gen.to_record(), model_dir.join("optimizer_gen"))?;
    recorder.record(optim_disc.to_record(), model_dir.join("optimizer_disc"))?;
    let state_path = model_dir.join("state.json");
    let state_json = serde_json::to_string_pretty(state)?;
    std::fs::write(state_path, state_json)?;
    Ok(())
}

/// Load model, optimizer, and state if a checkpoint exists.
fn load_checkpoint<B, OG, OD>(
    model_dir: &Path,
    device: &B::Device,
    generator: &mut crate::model::generator::Generator<B>,
    discriminator: &mut crate::model::discriminator::Discriminator<B>,
    optim_gen: &mut OG,
    optim_disc: &mut OD,
) -> Result<Option<TrainingState>>
where
    B: AutodiffBackend,
    OG: Optimizer<crate::model::generator::Generator<B>, B>,
    OD: Optimizer<crate::model::discriminator::Discriminator<B>, B>,
{
    let recorder = CompactRecorder::new();
    let gen_path = model_dir.join("generator");
    let disc_path = model_dir.join("discriminator");
    if gen_path.exists() {
        let record = recorder.load(gen_path, device)?;
        *generator = generator.clone().load_record(record);
    }
    if disc_path.exists() {
        let record = recorder.load(disc_path, device)?;
        *discriminator = discriminator.clone().load_record(record);
    }

    let opt_gen_path = model_dir.join("optimizer_gen");
    if opt_gen_path.exists() {
        let record = recorder.load(opt_gen_path, device)?;
        *optim_gen = optim_gen.clone().load_record(record);
    }
    let opt_disc_path = model_dir.join("optimizer_disc");
    if opt_disc_path.exists() {
        let record = recorder.load(opt_disc_path, device)?;
        *optim_disc = optim_disc.clone().load_record(record);
    }

    let state_path = model_dir.join("state.json");
    if state_path.exists() {
        let contents = std::fs::read_to_string(state_path)?;
        let state: TrainingState = serde_json::from_str(&contents)?;
        return Ok(Some(state));
    }
    Ok(None)
}

/// Run a validation step and save merged sample images.
fn validate_model<B: AutodiffBackend>(
    generator: &crate::model::generator::Generator<B>,
    val_examples: &[Example],
    data_config: &DataConfig,
    config: &TrainingConfig,
    model_dir: &Path,
    val_index: &mut usize,
    rng: &mut StdRng,
    device: &B::Device,
    epoch: usize,
    step: usize,
) -> Result<()> {
    if val_examples.is_empty() {
        return Ok(());
    }

    let batch_refs = select_val_batch(val_examples, config.batch_size, val_index);
    let batch = build_batch::<B>(&batch_refs, data_config, false, rng, device)?;

    let labels_vec = batch
        .labels
        .to_data()
        .to_vec::<i32>()
        .context("failed to read labels")?;
    let labels = Tensor::<B, 1, Int>::from_data(
        TensorData::new(labels_vec.clone(), [labels_vec.len()]),
        device,
    );

    let (real_a, real_b) = crate::model::split_real(
        batch.images.clone(),
        config.model.input_channels,
        config.model.output_channels,
    );
    let (fake_b, _) = generator.forward(real_a.clone(), labels);

    let input_imgs = tensor_to_images(real_a)?;
    let fake_imgs = tensor_to_images(fake_b)?;
    let real_imgs = tensor_to_images(real_b)?;

    let merged_input = merge_images(&input_imgs, config.batch_size, 1)?;
    let merged_fake = merge_images(&fake_imgs, config.batch_size, 1)?;
    let merged_real = merge_images(&real_imgs, config.batch_size, 1)?;
    let merged_pair = vec![merged_input, merged_real, merged_fake];

    let sample_dir = model_dir.join("samples");
    std::fs::create_dir_all(&sample_dir)?;
    let labels_str = labels_vec
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("_");
    let filename = format!("sample_{:02}_{:04}_{}.png", epoch, step, labels_str);
    save_concat_images(&merged_pair, &sample_dir.join(filename))?;
    Ok(())
}

/// Select a rolling validation batch to vary samples over time.
fn select_val_batch<'a>(
    examples: &'a [Example],
    batch_size: usize,
    start: &mut usize,
) -> Vec<&'a Example> {
    let mut batch = Vec::with_capacity(batch_size);
    for offset in 0..batch_size {
        let idx = (*start + offset) % examples.len();
        batch.push(&examples[idx]);
    }
    *start = (*start + batch_size) % examples.len();
    batch
}
