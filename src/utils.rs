use anyhow::{Context, Result};
use burn::prelude::*;
use gif::{Encoder, Frame, Repeat};
use image::{GenericImage, Rgb, RgbImage};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Map [-1, 1] normalized values back to [0, 1].
pub fn scale_back(value: f32) -> f32 {
    (value + 1.0) * 0.5
}

/// Convert a BCHW tensor in [-1, 1] to a vector of RGB images.
pub fn tensor_to_images<B: Backend>(tensor: Tensor<B, 4>) -> Result<Vec<RgbImage>> {
    let data = tensor.to_data().convert::<f32>();
    let shape = data.shape.clone();
    if shape.len() != 4 {
        return Err(anyhow::anyhow!(
            "expected rank-4 tensor for images, got shape {shape:?}"
        ));
    }

    let batch = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "expected 3 channels for images, got {channels}"
        ));
    }

    let values = data
        .to_vec::<f32>()
        .context("failed to read tensor data as f32")?;
    let hw = height * width;
    let mut images = Vec::with_capacity(batch);

    for b in 0..batch {
        let base = b * channels * hw;
        let mut img = RgbImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let r = scale_back(values[base + idx]).clamp(0.0, 1.0) * 255.0;
                let g = scale_back(values[base + hw + idx]).clamp(0.0, 1.0) * 255.0;
                let b = scale_back(values[base + 2 * hw + idx]).clamp(0.0, 1.0) * 255.0;
                img.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([r as u8, g as u8, b as u8]),
                );
            }
        }
        images.push(img);
    }

    Ok(images)
}

/// Merge images into a fixed grid (rows x cols).
pub fn merge_images(images: &[RgbImage], rows: usize, cols: usize) -> Result<RgbImage> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("no images to merge"));
    }
    let width = images[0].width();
    let height = images[0].height();
    let mut out = RgbImage::new(width * cols as u32, height * rows as u32);

    for (idx, img) in images.iter().enumerate() {
        let row = idx / cols;
        let col = idx % cols;
        if row >= rows {
            break;
        }
        out.copy_from(img, (col as u32) * width, (row as u32) * height)
            .context("failed to copy image into grid")?;
    }

    Ok(out)
}

/// Concatenate images horizontally.
pub fn concat_images_horiz(images: &[RgbImage]) -> Result<RgbImage> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("no images to concatenate"));
    }
    let height = images[0].height();
    let total_width: u32 = images.iter().map(|img| img.width()).sum();
    let mut out = RgbImage::new(total_width, height);

    let mut offset_x = 0;
    for img in images {
        out.copy_from(img, offset_x, 0)
            .context("failed to concatenate image")?;
        offset_x += img.width();
    }

    Ok(out)
}

/// Save a horizontal concatenation of images to disk.
pub fn save_concat_images(images: &[RgbImage], path: &Path) -> Result<()> {
    let output = concat_images_horiz(images)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    output
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

/// Compile a list of PNG frames into an animated GIF.
pub fn compile_frames_to_gif_from_paths(
    frame_paths: &[PathBuf],
    gif_path: &Path,
) -> Result<()> {
    if frame_paths.is_empty() {
        return Err(anyhow::anyhow!(
            "no frames provided for {}",
            gif_path.display()
        ));
    }

    let first = image::open(&frame_paths[0])
        .with_context(|| format!("failed to open {}", frame_paths[0].display()))?
        .to_rgb8();
    let (width, height) = first.dimensions();
    // Downscale to ~33% to match the original Python GIF output size.
    let target_width = ((width as f32) * 0.33).max(1.0).round() as u32;
    let target_height = ((height as f32) * 0.33).max(1.0).round() as u32;

    let mut file = File::create(gif_path)
        .with_context(|| format!("failed to create {}", gif_path.display()))?;
    let mut encoder = Encoder::new(&mut file, target_width as u16, target_height as u16, &[])?;
    encoder.set_repeat(Repeat::Infinite)?;

    for frame_path in frame_paths {
        let img = image::open(&frame_path)
            .with_context(|| format!("failed to open {}", frame_path.display()))?
            .to_rgb8();
        let resized = image::imageops::resize(
            &img,
            target_width,
            target_height,
            image::imageops::FilterType::Nearest,
        );
        let mut frame = Frame::from_rgb(
            target_width as u16,
            target_height as u16,
            resized.as_raw(),
        );
        frame.delay = 10;
        encoder.write_frame(&frame)?;
    }

    Ok(())
}

/// Compile PNG frames in a directory into an animated GIF.
pub fn compile_frames_to_gif(frame_dir: &Path, gif_path: &Path) -> Result<()> {
    let mut frames: Vec<_> = glob::glob(&format!("{}/**/*.png", frame_dir.display()))?
        .filter_map(Result::ok)
        .collect();
    frames.sort();

    if frames.is_empty() {
        return Err(anyhow::anyhow!("no png frames found in {}", frame_dir.display()));
    }

    compile_frames_to_gif_from_paths(&frames, gif_path)
}
