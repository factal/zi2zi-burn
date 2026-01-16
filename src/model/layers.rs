use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;

/// Conditional instance normalization driven by label embeddings.
#[derive(Module, Debug)]
pub struct ConditionalInstanceNorm<B: Backend> {
    scale: Embedding<B>,
    shift: Embedding<B>,
    #[module(ignore)]
    epsilon: f64,
}

impl<B: Backend> ConditionalInstanceNorm<B> {
    /// Create the scale/shift embedding tables.
    pub fn new(num_labels: usize, num_channels: usize, epsilon: f64, device: &B::Device) -> Self {
        let scale = EmbeddingConfig::new(num_labels, num_channels).init(device);
        let shift = EmbeddingConfig::new(num_labels, num_channels).init(device);
        Self {
            scale,
            shift,
            epsilon,
        }
    }

    /// Apply conditional instance normalization for the given labels.
    pub fn forward(&self, x: Tensor<B, 4>, ids: Tensor<B, 1, Int>) -> Tensor<B, 4> {
        let (scale, shift) = self.lookup(ids);
        self.forward_with_scale_shift(x, scale, shift)
    }

    /// Look up per-label scale and shift vectors.
    pub fn lookup(&self, ids: Tensor<B, 1, Int>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let ids = ids.unsqueeze_dim::<2>(1);
        let scale = self.scale.forward(ids.clone()).squeeze_dim::<2>(1);
        let shift = self.shift.forward(ids).squeeze_dim::<2>(1);
        (scale, shift)
    }

    /// Apply instance norm using explicit scale and shift tensors.
    pub fn forward_with_scale_shift(
        &self,
        x: Tensor<B, 4>,
        scale: Tensor<B, 2>,
        shift: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let mean = x.clone().mean_dims(&[2, 3]);
        let var = x.clone().sub(mean.clone()).square().mean_dims(&[2, 3]);
        let norm = (x - mean).div((var + self.epsilon).sqrt());

        let batch = norm.dims()[0];
        let channels = norm.dims()[1];
        let scale = scale.reshape([batch, channels, 1, 1]);
        let shift = shift.reshape([batch, channels, 1, 1]);
        norm.mul(scale).add(shift)
    }
}
