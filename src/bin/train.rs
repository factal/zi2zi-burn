#![recursion_limit = "256"]
use anyhow::{Context, Result};
use burn::backend::{Autodiff, WebGpu};
use burn::config::Config;
use burn::backend::wgpu::{RuntimeOptions, init_setup};
use burn::backend::wgpu::graphics::AutoGraphicsApi;
use burn_cuda::{Cuda, CudaDevice};
use clap::Parser;
use std::path::PathBuf;
use zi2zi_burn::training::TrainingConfig;
use zi2zi_burn::model::ModelConfig;

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
    let mut config = TrainingConfig::load(&args.config)
        .with_context(|| format!("failed to load config from {}", args.config.display()))?;

    type Backend = WebGpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
    let max_storage_buffer_binding_size =
        setup.device.limits().max_storage_buffer_binding_size as u64;
    adjust_batch_size_for_wgpu(&mut config, max_storage_buffer_binding_size)?;

    zi2zi_burn::training::train::<AutodiffBackend>(&args.experiment_dir, config, device)?;
    Ok(())
}

fn adjust_batch_size_for_wgpu(
    config: &mut TrainingConfig,
    max_storage_buffer_binding_size: u64,
) -> Result<()> {
    let elem_bytes = std::mem::size_of::<f32>() as u64;
    let per_sample_bytes = estimate_max_conv_workspace_bytes(&config.model, elem_bytes);
    if per_sample_bytes == 0 {
        return Ok(());
    }

    let safe_limit = max_storage_buffer_binding_size.saturating_sub(1);
    let max_batch = (safe_limit / per_sample_bytes) as usize;
    if max_batch == 0 {
        return Err(anyhow::anyhow!(
            "WGPU max storage buffer size ({max_storage_buffer_binding_size} bytes) is too small for a single sample (estimated {per_sample_bytes} bytes). Reduce image_size or model dims."
        ));
    }

    if config.batch_size > max_batch {
        println!(
            "wgpu max storage buffer size {} bytes; estimated max conv workspace per sample {} bytes. lowering batch_size from {} to {}.",
            max_storage_buffer_binding_size,
            per_sample_bytes,
            config.batch_size,
            max_batch
        );
        config.batch_size = max_batch;
    }

    Ok(())
}

fn estimate_max_conv_workspace_bytes(model: &ModelConfig, elem_bytes: u64) -> u64 {
    estimate_max_conv_workspace_elems(model).saturating_mul(elem_bytes)
}

fn estimate_max_conv_workspace_elems(model: &ModelConfig) -> u64 {
    let kernel_area = 16u64;
    let mut max_elems = 0u64;

    // Generator encoder conv2d layers.
    let mut size = model.image_size as u64;
    let mut in_channels = model.input_channels as u64;
    let enc_out_channels = [
        model.generator_dim as u64,
        model.generator_dim as u64 * 2,
        model.generator_dim as u64 * 4,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
    ];
    let mut enc_sizes = Vec::with_capacity(enc_out_channels.len());
    for &out_channels in &enc_out_channels {
        size = conv_out(size, 4, 2, 1);
        enc_sizes.push(size);
        let elems = in_channels * size * size * kernel_area;
        if elems > max_elems {
            max_elems = elems;
        }
        in_channels = out_channels;
    }

    // Generator decoder conv_transpose2d layers (col2im workspace).
    let dec_out_channels = [
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 8,
        model.generator_dim as u64 * 4,
        model.generator_dim as u64 * 2,
        model.generator_dim as u64,
        model.output_channels as u64,
    ];
    for (idx, &out_channels) in dec_out_channels.iter().enumerate() {
        let input_size = enc_sizes
            .get(enc_sizes.len().saturating_sub(1 + idx))
            .copied()
            .unwrap_or(1);
        let elems = out_channels * input_size * input_size * kernel_area;
        if elems > max_elems {
            max_elems = elems;
        }
    }

    // Discriminator conv2d layers.
    let mut disc_size = model.image_size as u64;
    let mut disc_in_channels = (model.input_channels + model.output_channels) as u64;
    let disc_out_channels = [
        model.discriminator_dim as u64,
        model.discriminator_dim as u64 * 2,
        model.discriminator_dim as u64 * 4,
        model.discriminator_dim as u64 * 8,
    ];
    let disc_strides = [2u64, 2u64, 2u64, 1u64];
    for (&out_channels, &stride) in disc_out_channels.iter().zip(disc_strides.iter()) {
        disc_size = conv_out(disc_size, 4, stride, 1);
        let elems = disc_in_channels * disc_size * disc_size * kernel_area;
        if elems > max_elems {
            max_elems = elems;
        }
        disc_in_channels = out_channels;
    }

    max_elems
}

fn conv_out(input: u64, kernel: u64, stride: u64, padding: u64) -> u64 {
    (input + 2 * padding - (kernel - 1) - 1) / stride + 1
}
