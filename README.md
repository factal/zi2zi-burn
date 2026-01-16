# zi2zi-burn

Rust/Burn port of the zi2zi pix2pix-style font transfer pipeline. It keeps the
original data flow (paired half images, label embeddings) and exposes
training/inference CLIs under `src/bin`.

## Binaries

- `train`: train with `config.json`
- `infer`: run inference or interpolation
- `export`: export generator weights
- `font2img`: render paired images from fonts
- `package`: package PNGs into pickle streams

## Requirements

- Rust stable
- WGPU compatible device (default backend is `WebGpu`)

## Data Pipeline

1. Render paired samples from fonts:

```bash
cargo run --bin font2img -- \
  --src-font path/to/src.ttf \
  --dst-font path/to/dst.ttf \
  --sample-dir ./data/png \
  --label 0
```

This writes `LABEL_XXXX.png` with left=target (dst) and right=source (src).

2. Package to pickle streams:

```bash
cargo run --bin package -- \
  --dir ./data/png \
  --save-dir ./data
```

This creates `train.obj` and `val.obj` (pickle stream of `(label, png_bytes)`
tuples).

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
  "num_epochs": 100,
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
cargo run --bin train -- \
  --experiment-dir ./experiments/exp1 \
  --config ./config.json
```

Checkpoints are stored under
`experiment_dir/checkpoint/experiment_{id}_batch_{batch}`.

## Inference

```bash
cargo run --bin infer -- \
  --model-dir ./experiments/exp1 \
  --source-obj ./data/val.obj \
  --embedding-ids 0 \
  --save-dir ./out
```

Interpolation:

```bash
cargo run --bin infer -- \
  --model-dir ./experiments/exp1 \
  --source-obj ./data/val.obj \
  --embedding-ids 0,1,2 \
  --interpolate --steps 10 --output-gif interp.gif \
  --save-dir ./out
```

## Export

```bash
cargo run --bin export -- \
  --model-dir ./experiments/exp1 \
  --save-dir ./export \
  --model-name gen_model
```

## Notes

- `charset/cjk.json` is large; use `--charset` to select a subset or pass a
  one-line charset file.
- The current data pipeline assumes RGB images (3 channels).
