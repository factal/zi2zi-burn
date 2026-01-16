use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use image::RgbImage;
use rand::Rng;
use serde::Deserialize;
use serde_pickle::{DeOptions, Deserializer, Error as PickleError};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// A single training example loaded from the pickle stream.
#[derive(Debug, Clone)]
pub struct Example {
    pub label: i64,
    pub bytes: Vec<u8>,
}

/// (label, jpeg_bytes) tuple stored in the pickle stream.
#[derive(Debug, Deserialize)]
struct PickledExample(i64, Vec<u8>);

/// Settings for decoding and assembling image batches.
#[derive(Debug, Clone)]
pub struct DataConfig {
    pub image_size: u32,
    pub input_channels: usize,
    pub output_channels: usize,
    pub swap_ab: bool,
}

/// A batch of concatenated images and label ids.
#[derive(Clone, Debug)]
pub struct Zi2ziBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

/// Load a pickle stream containing (label, jpeg_bytes) tuples.
pub fn load_pickled_examples(path: &Path) -> Result<Vec<Example>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut de = Deserializer::new(BufReader::new(file), DeOptions::default());
    let mut examples = Vec::new();

    loop {
        de.reset_memo();
        match PickledExample::deserialize(&mut de) {
            Ok(PickledExample(label, bytes)) => examples.push(Example { label, bytes }),
            Err(PickleError::Io(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "failed to decode pickle stream at {}: {err}",
                    path.display()
                ));
            }
        }
    }

    Ok(examples)
}

/// Build a normalized batch from examples, optionally applying augmentation.
pub fn build_batch<B: Backend>(
    examples: &[&Example],
    config: &DataConfig,
    augment: bool,
    rng: &mut impl Rng,
    device: &B::Device,
) -> Result<Zi2ziBatch<B>> {
    let batch_size = examples.len();
    let image_size = config.image_size as usize;
    let channels = config.input_channels + config.output_channels;
    let mut batch = Vec::with_capacity(batch_size * channels * image_size * image_size);
    let mut labels = Vec::with_capacity(batch_size);

    for &example in examples {
        // Decode -> split -> optional augment, then normalize to CHW in [-1, 1].
        let img = decode_image(&example.bytes)?;
        let (mut img_a, mut img_b) = split_image(&img)?;
        if config.swap_ab {
            std::mem::swap(&mut img_a, &mut img_b);
        }

        if img_a.width() != config.image_size || img_a.height() != config.image_size {
            img_a = resize_image(&img_a, config.image_size, config.image_size);
            img_b = resize_image(&img_b, config.image_size, config.image_size);
        }

        if augment {
            let (aug_a, aug_b) = augment_pair(&img_a, &img_b, rng);
            img_a = aug_a;
            img_b = aug_b;
        }

        let mut a = image_to_chw(&img_a);
        let mut b = image_to_chw(&img_b);
        batch.append(&mut a);
        batch.append(&mut b);
        labels.push(example.label);
    }

    let images = Tensor::<B, 4>::from_data(
        TensorData::new(
            batch,
            [batch_size, channels, image_size, image_size],
        ),
        device,
    );
    let labels = Tensor::<B, 1, Int>::from_data(
        TensorData::new(labels, [batch_size]),
        device,
    );

    Ok(Zi2ziBatch { images, labels })
}

/// Decode raw image bytes into an RGB image.
fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    let img = image::load_from_memory(bytes)
        .context("failed to decode image bytes")?
        .to_rgb8();
    Ok(img)
}

/// Split a paired image into left and right halves.
fn split_image(img: &RgbImage) -> Result<(RgbImage, RgbImage)> {
    let (width, height) = img.dimensions();
    if width % 2 != 0 {
        return Err(anyhow::anyhow!(
            "image width must be even to split, got {width}"
        ));
    }
    let half = width / 2;
    let left = image::imageops::crop_imm(img, 0, 0, half, height).to_image();
    let right = image::imageops::crop_imm(img, half, 0, half, height).to_image();
    Ok((left, right))
}

fn resize_image(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    image::imageops::resize(img, width, height, image::imageops::FilterType::CatmullRom)
}

/// Apply paired augmentation (resize + random crop) while keeping alignment.
fn augment_pair(img_a: &RgbImage, img_b: &RgbImage, rng: &mut impl Rng) -> (RgbImage, RgbImage) {
    let width = img_a.width();
    let height = img_a.height();
    // Mirror the original pipeline: random upscale then crop back to the original size.
    let scale: f32 = rng.gen_range(1.0..=1.2);
    let new_w = (width as f32 * scale).ceil() as u32;
    let new_h = (height as f32 * scale).ceil() as u32;
    let new_w = new_w.max(width);
    let new_h = new_h.max(height);

    let resized_a = resize_image(img_a, new_w, new_h);
    let resized_b = resize_image(img_b, new_w, new_h);

    let max_x = new_w.saturating_sub(width);
    let max_y = new_h.saturating_sub(height);
    let shift_x = if max_x == 0 { 0 } else { rng.gen_range(0..=max_x) };
    let shift_y = if max_y == 0 { 0 } else { rng.gen_range(0..=max_y) };

    let cropped_a = image::imageops::crop_imm(&resized_a, shift_x, shift_y, width, height).to_image();
    let cropped_b = image::imageops::crop_imm(&resized_b, shift_x, shift_y, width, height).to_image();

    (cropped_a, cropped_b)
}

/// Convert RGB image data to CHW floats normalized to [-1, 1].
fn image_to_chw(img: &RgbImage) -> Vec<f32> {
    let (width, height) = img.dimensions();
    let hw = (width * height) as usize;
    let mut out = vec![0.0f32; hw * 3];

    // Convert HWC u8 to CHW f32 in [-1, 1].
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y).0;
            let idx = (y * width + x) as usize;
            out[idx] = (pixel[0] as f32 / 127.5) - 1.0;
            out[hw + idx] = (pixel[1] as f32 / 127.5) - 1.0;
            out[2 * hw + idx] = (pixel[2] as f32 / 127.5) - 1.0;
        }
    }

    out
}
