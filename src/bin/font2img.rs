use anyhow::{Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImage, Rgb, RgbImage};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use rusttype::{point, Font, Scale};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Convert fonts to paired training images")]
struct Args {
    #[arg(long)]
    src_font: PathBuf,
    #[arg(long)]
    dst_font: PathBuf,
    #[arg(long, default_value_t = false)]
    filter: bool,
    #[arg(long, default_value = "CN")]
    charset: String,
    #[arg(long, default_value_t = false)]
    shuffle: bool,
    #[arg(long, default_value_t = 150)]
    char_size: u32,
    #[arg(long, default_value_t = 256)]
    canvas_size: u32,
    #[arg(long, default_value_t = 20)]
    x_offset: i32,
    #[arg(long, default_value_t = 20)]
    y_offset: i32,
    #[arg(long, default_value_t = 1000)]
    sample_count: usize,
    #[arg(long)]
    sample_dir: PathBuf,
    #[arg(long, default_value_t = 0)]
    label: i64,
    #[arg(long, default_value = "./charset/cjk.json")]
    charset_json: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// JSON structure for the bundled CJK character sets.
#[derive(Deserialize)]
struct CjkCharset {
    gbk: Vec<String>,
    jp: Vec<String>,
    kr: Vec<String>,
    #[serde(rename = "gb2312_t")]
    gb2312_t: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut charset = load_charset(&args)?;
    if args.shuffle {
        charset.shuffle(&mut rng);
    }

    let src_font = load_font(&args.src_font)?;
    let dst_font = load_font(&args.dst_font)?;

    if !args.sample_dir.exists() {
        fs::create_dir_all(&args.sample_dir).with_context(|| {
            format!("failed to create {}", args.sample_dir.display())
        })?;
    }

    let filter_hashes = if args.filter {
        filter_recurring_hash(
            &charset,
            &dst_font,
            args.char_size,
            args.canvas_size,
            args.x_offset,
            args.y_offset,
            &mut rng,
        )?
    } else {
        HashSet::new()
    };

    let mut count = 0usize;
    for ch in charset {
        if count >= args.sample_count {
            break;
        }
        if let Some(example) = draw_example(
            ch,
            &src_font,
            &dst_font,
            args.char_size,
            args.canvas_size,
            args.x_offset,
            args.y_offset,
            &filter_hashes,
        )? {
            let file_name = format!("{}_{}.jpg", args.label, format!("{count:04}"));
            let path = args.sample_dir.join(file_name);
            DynamicImage::ImageRgb8(example).save(&path).with_context(|| {
                format!("failed to save {}", path.display())
            })?;
            count += 1;
            if count % 100 == 0 {
                println!("processed {count} chars");
            }
        }
    }

    Ok(())
}

/// Load a font file into a rusttype Font.
fn load_font(path: &Path) -> Result<Font<'static>> {
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let font = Font::try_from_vec(data).context("failed to parse font file")?;
    Ok(font)
}

/// Load the character list from a built-in CJK set or a custom file.
fn load_charset(args: &Args) -> Result<Vec<char>> {
    match args.charset.as_str() {
        "CN" | "JP" | "KR" | "CN_T" => {
            let contents = fs::read_to_string(&args.charset_json).with_context(|| {
                format!("failed to read {}", args.charset_json.display())
            })?;
            let cjk: CjkCharset = serde_json::from_str(&contents)
                .context("failed to parse charset json")?;
            let list = match args.charset.as_str() {
                "CN" => cjk.gbk,
                "JP" => cjk.jp,
                "KR" => cjk.kr,
                "CN_T" => cjk.gb2312_t,
                _ => unreachable!(),
            };
            Ok(list.into_iter().filter_map(|s| s.chars().next()).collect())
        }
        other => {
            let line = fs::read_to_string(other)
                .with_context(|| format!("failed to read charset file {other}"))?;
            Ok(line.trim_end_matches('\n').chars().collect())
        }
    }
}

/// Render a paired target/source sample for a single character.
fn draw_example(
    ch: char,
    src_font: &Font,
    dst_font: &Font,
    char_size: u32,
    canvas_size: u32,
    x_offset: i32,
    y_offset: i32,
    filter_hashes: &HashSet<u64>,
) -> Result<Option<RgbImage>> {
    let dst_img = draw_single_char(ch, dst_font, char_size, canvas_size, x_offset, y_offset);
    let dst_hash = hash_image(&dst_img);
    if filter_hashes.contains(&dst_hash) {
        return Ok(None);
    }
    let src_img = draw_single_char(ch, src_font, char_size, canvas_size, x_offset, y_offset);
    let mut out = RgbImage::new(canvas_size * 2, canvas_size);
    // Keep left=target (dst), right=source (src) to match split_real ordering.
    out.copy_from(&dst_img, 0, 0)?;
    out.copy_from(&src_img, canvas_size, 0)?;
    Ok(Some(out))
}

/// Render a single character onto a white canvas.
fn draw_single_char(
    ch: char,
    font: &Font,
    char_size: u32,
    canvas_size: u32,
    x_offset: i32,
    y_offset: i32,
) -> RgbImage {
    let mut image = RgbImage::from_pixel(canvas_size, canvas_size, Rgb([255, 255, 255]));
    let scale = Scale::uniform(char_size as f32);
    let v_metrics = font.v_metrics(scale);
    let glyph = font
        .glyph(ch)
        .scaled(scale)
        .positioned(point(
            x_offset as f32,
            y_offset as f32 + v_metrics.ascent,
        ));

    if let Some(bb) = glyph.pixel_bounding_box() {
        glyph.draw(|x, y, v| {
            let px = x as i32 + bb.min.x;
            let py = y as i32 + bb.min.y;
            if px < 0 || py < 0 {
                return;
            }
            let (px, py) = (px as u32, py as u32);
            if px >= canvas_size || py >= canvas_size {
                return;
            }
            let intensity = (255.0 * (1.0 - v)) as u8;
            let pixel = image.get_pixel_mut(px, py);
            pixel.0 = [
                pixel.0[0].min(intensity),
                pixel.0[1].min(intensity),
                pixel.0[2].min(intensity),
            ];
        });
    }

    image
}

/// Hash raw image bytes for duplicate detection.
fn hash_image(img: &RgbImage) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    img.as_raw().hash(&mut hasher);
    hasher.finish()
}

/// Sample characters and return hashes that appear too frequently.
fn filter_recurring_hash(
    charset: &[char],
    font: &Font,
    char_size: u32,
    canvas_size: u32,
    x_offset: i32,
    y_offset: i32,
    rng: &mut StdRng,
) -> Result<HashSet<u64>> {
    let mut sample = charset.to_vec();
    sample.shuffle(rng);
    let sample_len = sample.len().min(2000);
    let mut counts: HashMap<u64, usize> = HashMap::new();

    for ch in sample.iter().take(sample_len) {
        let img = draw_single_char(*ch, font, char_size, canvas_size, x_offset, y_offset);
        let h = hash_image(&img);
        *counts.entry(h).or_insert(0) += 1;
    }

    Ok(counts
        .into_iter()
        .filter_map(|(hash, count)| (count > 2).then_some(hash))
        .collect())
}
