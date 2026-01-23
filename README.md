# zi2zi-burn

A Rust/Burn port of [kaonashi-tyc/zi2zi](https://github.com/kaonashi-tyc/zi2zi).

## Binaries

CLI binaries (via `cargo run --bin <name>`):

- `train`: train with `config.json`
- `infer`: run inference or interpolation
- `export`: export generator weights
- `font2img`: render paired images from fonts
- `package`: package PNGs into pickle streams

## Requirements

- Rust (stable)
- CUDA (for GPU training/inference)
- Python 3 (for dataset scripts)

## Prepare Dataset

The recommended directory structure is as follows:

```
dataset/
├── images/
│   ├── {LABEL}_XXXX.png
│   └── ...
├── dst_fonts/
│   └── (target fonts here)
└── src_fonts/
    └── (source fonts here)

experiment/
├── checkpoint/
└── data/
    ├── train.obj
    └── val.obj
```

1. Define the character set to learn:

```bash
python ./scripts/generate_charset.py jp.txt -o charset.json
```

Each input text file should contain only the characters you want to include. You can pass multiple files to make a different character set.

2. Render paired samples from fonts:

```bash
cargo run --bin font2img --release -- \
  --src-font ./dataset/src_fonts/src.ttf \
  --dst-font ./dataset/dst_fonts \
  --sample-dir ./dataset/images \
  --sample-count 1000 --shuffle
```

Notes:

- If `--dst-font` is a directory, labels are assigned automatically and a `label_map.txt` is written to `--sample-dir`. When a single font file is specified, use `--labels`.
- If you passed multiple text files to `generate_charset.py`, you can switch character sets with `--charset`.
- For small `--sample-count`, enable `--shuffle` to avoid repeated characters.
- Output files are `{LABEL}_XXXX.png` with left=target (dst) and right=source (src).

3. Package into pickle streams:

```bash
cargo run --bin package --release -- \
  --dir ./dataset/images \
  --save-dir ./experiment/data
```

This creates `train.obj` and `val.obj` (pickle stream of `(label, png_bytes)` tuples) in `experiment/data`.

## Config

Training reads a Burn config file. Example `config.json`:

```json
{
  "model": {
    "image_size": 256,
    "input_channels": 3,
    "output_channels": 3,
    "generator_dim": 64,
    "discriminator_dim": 64,
    "embedding_num": 40,
    "embedding_dim": 128,
    "inst_norm": false
  },
  "loss": {
    "l1_penalty": 100.0,
    "lconst_penalty": 15.0,
    "ltv_penalty": 0.0,
    "lcategory_penalty": 1.0
  },
  "data_dir": "data",
  "experiment_id": 0,
  "num_epochs": 40,
  "batch_size": 16,
  "learning_rate": 0.001,
  "schedule": 10,
  "min_learning_rate": 0.0002,
  "resume": true,
  "flip_labels": false,
  "swap_ab": false,
  "sample_steps": 50,
  "checkpoint_steps": 500,
  "seed": 42,
  "optimizer_gen": {
    "beta_1": 0.5,
    "beta_2": 0.999,
    "epsilon": 1e-5,
    "weight_decay": null,
    "grad_clipping": null
  },
  "optimizer_disc": {
    "beta_1": 0.5,
    "beta_2": 0.999,
    "epsilon": 1e-5,
    "weight_decay": null,
    "grad_clipping": null
  },
  "fine_tune": null
}
```

Notes:

- `data_dir` is resolved relative to `--experiment-dir` if it is a relative path.
- `fine_tune` is a list of label ids (e.g. `[0, 1, 2]`) or `null`.

## Train

```bash
cargo run --bin train --release -- \
  --experiment-dir ./experiment \
  --config ./config.json
```

Checkpoints are stored under
`experiment_dir/checkpoint/experiment_{id}_batch_{batch}`.

## Inference

```bash
cargo run --bin infer --release -- \
  --model-dir ./experiment/checkpoint/experiment_{id}_batch_{batch} \
  --source-obj ./experiment/data/val.obj \
  --embedding-ids 0 \
  --save-dir out
```

Image input:

```bash
cargo run --bin infer --release -- \
  --model-dir ./experiment/checkpoint/experiment_{id}_batch_{batch} \
  --source-image ./path/to/image_or_dir \
  --embedding-ids 0 \
  --save-dir out
```

Notes:

- `--source-image` accepts a single image file or a directory of images.
- Images must match the model `image_size` (e.g. 256x256, see `experiment_{id}_batch_{batch}/config.json`); mismatched sizes error out.

Interpolation:

```bash
cargo run --bin infer --release -- \
  --model-dir ./experiment/checkpoint/experiment_{id}_batch_{batch} \
  --source-obj ./experiment/data/val.obj \
  --embedding-ids 0,1,2 \
  --interpolate --steps 10 --output-gif interp.gif \
  --save-dir out
```

## Export

```bash
cargo run --bin export --release -- \
  --model-dir ./experiment/checkpoint/experiment_{id}_batch_{batch} \
  --save-dir ./export \
  --model-name gen_model
```
