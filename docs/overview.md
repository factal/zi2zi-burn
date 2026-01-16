# Overview

This repository is a Rust/Burn port of the zi2zi pix2pix-style font transfer
pipeline. It keeps the original data format (paired half images + label
embeddings) and exposes CLI entry points for training, inference, export, and
data preparation.

## Repository layout

- `src/lib.rs` exposes the library modules used by the CLI binaries.
- `src/data.rs` handles loading pickled examples and building normalized batches.
- `src/model/` contains the generator, discriminator, and conditional instance
  norm layers.
- `src/training.rs` implements the training loop, checkpointing, and sampling.
- `src/utils.rs` contains image/tensor utilities and GIF assembly helpers.
- `src/bin/train.rs` trains a model from a Burn config file.
- `src/bin/infer.rs` runs inference and interpolation over embeddings.
- `src/bin/export.rs` exports generator weights from a checkpoint.
- `src/bin/font2img.rs` renders paired JPGs from fonts.
- `src/bin/package.rs` packages JPGs into pickle streams.
- `charset/` stores large character sets for font rendering.
- `datasets/` contains example fonts and sample images.

## Data format and pipeline

- `font2img` renders paired images. The left half is the target (dst), and the
  right half is the source (src).
- `package` writes a pickle stream of `(label, jpg_bytes)` tuples into
  `train.obj` and `val.obj`.
- `data::build_batch` decodes each JPEG, splits the halves, optionally applies
  augmentation, and normalizes the tensors to CHW in the `[-1, 1]` range.

## Model architecture

- Generator: U-Net with 8 encoder/decoder stages. A style embedding is
  concatenated at the bottleneck; decoder blocks use batch norm or conditional
  instance norm when `inst_norm` is enabled.
- Discriminator: 4 convolutional blocks feeding two heads (adversarial logits
  and category logits).

## Training loop

- `training::train` loads pickled examples, shuffles, builds batches, and runs a
  discriminator step plus two generator steps per batch (matching the original
  TF loop).
- Losses include adversarial, category classification, L1 reconstruction,
  const loss on encoder features, and optional total variation.
- Checkpoints are stored under
  `experiment_dir/checkpoint/experiment_{id}_batch_{batch}` along with
  `samples/` outputs.

## Configuration notes

- `data_dir` is resolved relative to `--experiment-dir` when given a relative
  path.
- `swap_ab` swaps the paired halves when building batches.
- `flip_labels` enables the no-target loss branch by shuffling labels.
- `inst_norm` toggles conditional instance norm in the generator decoder.

## Inference and export

- `infer` runs fixed embedding IDs or interpolates across IDs
  (`--interpolate`, `--steps`, `--uroboros`) and can compile frames into a GIF.
- `export` saves generator weights from a checkpoint into a standalone file.

## Tensor shapes and conventions

- Images are RGB, normalized to `[-1, 1]` and stored as
  `[batch, channels, height, width]`.
- Batches are channel-concatenated as `[left | right]`. With the default
  left=target, right=source convention, `split_real` returns the right half as
  the generator input and the left half as the target.
