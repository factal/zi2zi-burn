use anyhow::{bail, Context, Result};
use clap::Parser;
use image::{DynamicImage, GenericImage, Rgb, RgbImage};
use owned_ttf_parser::{AsFaceRef, OutlineBuilder, OwnedFace};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use tiny_skia::{Color, FillRule, Paint, Path as SkiaPath, PathBuilder, Pixmap, Transform};


#[derive(Parser, Debug)]
#[command(about = "Convert fonts to paired training images")]
struct Args {
    #[arg(long)]
    src_font: PathBuf,
    #[arg(long)]
    dst_font: PathBuf,
    #[arg(long, default_value_t = false)]
    filter: bool,
    #[arg(long, default_value = "JP")]
    charset: String,
    #[arg(long, default_value_t = false)]
    shuffle: bool,
    #[arg(long, default_value_t = 192)]
    char_size: u32,
    #[arg(long, default_value_t = 256)]
    canvas_size: u32,
    #[arg(long, default_value_t = 32)]
    x_offset: i32,
    #[arg(long, default_value_t = 32)]
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

struct FontFace {
    face: OwnedFace,
    ascender: f32,
    height: f32,
}

struct GlyphPathBuilder {
    builder: PathBuilder,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
}

impl GlyphPathBuilder {
    fn new(scale: f32, offset_x: f32, offset_y: f32) -> Self {
        Self {
            builder: PathBuilder::new(),
            scale,
            offset_x,
            offset_y,
        }
    }

    fn map_point(&self, x: f32, y: f32) -> (f32, f32) {
        (self.offset_x + x * self.scale, self.offset_y - y * self.scale)
    }

    fn finish(self) -> Option<SkiaPath> {
        self.builder.finish()
    }
}

impl OutlineBuilder for GlyphPathBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.map_point(x, y);
        self.builder.move_to(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.map_point(x, y);
        self.builder.line_to(x, y);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        let (x1, y1) = self.map_point(x1, y1);
        let (x2, y2) = self.map_point(x2, y2);
        self.builder.quad_to(x1, y1, x2, y2);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        let (x1, y1) = self.map_point(x1, y1);
        let (x2, y2) = self.map_point(x2, y2);
        let (x3, y3) = self.map_point(x3, y3);
        self.builder.cubic_to(x1, y1, x2, y2, x3, y3);
    }

    fn close(&mut self) {
        let _ = self.builder.close();
    }
}

struct LabeledFontPath {
    label: i64,
    path: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut charset = load_charset(&args)?;
    if args.shuffle {
        charset.shuffle(&mut rng);
    }

    let src_font = load_font(&args.src_font)?;
    let dst_font_paths = collect_dst_font_paths(&args.dst_font, args.label)?;

    if !args.sample_dir.exists() {
        fs::create_dir_all(&args.sample_dir).with_context(|| {
            format!("failed to create {}", args.sample_dir.display())
        })?;
    }

    for dst_entry in &dst_font_paths {
        let dst_font = load_font(&dst_entry.path)?;
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
        for ch in &charset {
            if count >= args.sample_count {
                break;
            }
            let ch = *ch;
            let src_has = has_glyph(&src_font, ch);
            let dst_has = has_glyph(&dst_font, ch);
            if !src_has || !dst_has {
                let missing = match (src_has, dst_has) {
                    (false, false) => "src_font and dst_font",
                    (false, true) => "src_font",
                    (true, false) => "dst_font",
                    (true, true) => unreachable!(),
                };
                eprintln!(
                    "warning: skipping '{}' (U+{:04X}) missing in {}",
                    ch,
                    ch as u32,
                    missing
                );
                continue;
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
                let file_name = format!("{}_{}.png", dst_entry.label, format!("{count:04}"));
                let path = args.sample_dir.join(file_name);
                DynamicImage::ImageRgb8(example).save(&path).with_context(|| {
                    format!("failed to save {}", path.display())
                })?;
                count += 1;
                if count % 100 == 0 {
                    println!("label {} processed {count} chars", dst_entry.label);
                }
            }
        }
    }

    if args.dst_font.is_dir() {
        let mut log = String::new();
        for entry in &dst_font_paths {
            log.push_str(&format!("{}\t{}\n", entry.label, entry.path.display()));
        }
        let log_path = args.sample_dir.join("label_map.txt");
        fs::write(&log_path, log).with_context(|| {
            format!("failed to write {}", log_path.display())
        })?;
    }

    Ok(())
}

fn collect_dst_font_paths(dst_font: &Path, base_label: i64) -> Result<Vec<LabeledFontPath>> {
    if dst_font.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(dst_font)
            .with_context(|| format!("failed to read {}", dst_font.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_font_file(path))
            .collect();
        entries.sort_by(|a, b| {
            a.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase()
                .cmp(
                    &b.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase(),
                )
        });
        if entries.is_empty() {
            bail!("no font files found in {}", dst_font.display());
        }
        Ok(entries
            .into_iter()
            .enumerate()
            .map(|(idx, path)| LabeledFontPath {
                label: base_label + idx as i64,
                path,
            })
            .collect())
    } else {
        Ok(vec![LabeledFontPath {
            label: base_label,
            path: dst_font.to_path_buf(),
        }])
    }
}

fn is_font_file(path: &Path) -> bool {
    let ext = match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => ext.to_ascii_lowercase(),
        None => return false,
    };
    matches!(ext.as_str(), "ttf" | "otf" | "ttc" | "otc")
}

/// Load a font file into a font face that supports CFF/CFF2 outlines.
fn load_font(path: &Path) -> Result<FontFace> {
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let face = OwnedFace::from_vec(data, 0).context("failed to parse font file")?;
    let face_ref = face.as_face_ref();
    let ascender = face_ref.ascender() as f32;
    let descender = face_ref.descender() as f32;
    let height = (ascender - descender).max(1.0); // Match rusttype scale_for_pixel_height.
    Ok(FontFace {
        face,
        ascender,
        height,
    })
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

fn has_glyph(font: &FontFace, ch: char) -> bool {
    font.face
        .as_face_ref()
        .glyph_index(ch)
        .map(|id| id.0 != 0)
        .unwrap_or(false)
}

/// Render a paired target/source sample for a single character.
fn draw_example(
    ch: char,
    src_font: &FontFace,
    dst_font: &FontFace,
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
    font: &FontFace,
    char_size: u32,
    canvas_size: u32,
    x_offset: i32,
    y_offset: i32,
) -> RgbImage {
    let mut pixmap = match Pixmap::new(canvas_size, canvas_size) {
        Some(pixmap) => pixmap,
        None => return RgbImage::from_pixel(canvas_size, canvas_size, Rgb([255, 255, 255])),
    };
    pixmap.fill(Color::from_rgba8(255, 255, 255, 255));

    let face = font.face.as_face_ref();
    let Some(glyph_id) = face.glyph_index(ch) else {
        return pixmap_to_rgb(pixmap);
    };
    if glyph_id.0 == 0 {
        return pixmap_to_rgb(pixmap);
    }

    let scale = char_size as f32 / font.height;
    let baseline_y = y_offset as f32 + font.ascender * scale;
    let mut builder = GlyphPathBuilder::new(scale, x_offset as f32, baseline_y);
    if face.outline_glyph(glyph_id, &mut builder).is_none() {
        return pixmap_to_rgb(pixmap);
    }

    if let Some(path) = builder.finish() {
        let mut paint = Paint::default();
        paint.set_color(Color::from_rgba8(0, 0, 0, 255));
        let _ = pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }

    pixmap_to_rgb(pixmap)
}

fn pixmap_to_rgb(pixmap: Pixmap) -> RgbImage {
    let (width, height) = (pixmap.width(), pixmap.height());
    let data = pixmap.data();
    let mut image = RgbImage::new(width, height);
    for (idx, pixel) in image.pixels_mut().enumerate() {
        let base = idx * 4;
        pixel.0 = [data[base], data[base + 1], data[base + 2]];
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
    font: &FontFace,
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
        if !has_glyph(font, *ch) {
            continue;
        }
        let img = draw_single_char(*ch, font, char_size, canvas_size, x_offset, y_offset);
        let h = hash_image(&img);
        *counts.entry(h).or_insert(0) += 1;
    }

    Ok(counts
        .into_iter()
        .filter_map(|(hash, count)| (count > 2).then_some(hash))
        .collect())
}
