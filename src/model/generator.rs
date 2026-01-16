use crate::model::layers::ConditionalInstanceNorm;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::activation::{leaky_relu, relu};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::nn::PaddingConfig2d;

/// Configuration for the U-Net generator with label embeddings.
#[derive(Config, Debug)]
pub struct GeneratorConfig {
    pub generator_dim: usize,
    pub embedding_num: usize,
    pub embedding_dim: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    #[config(default = false)]
    pub inst_norm: bool,
}

/// Scale/shift parameters used by conditional instance norm.
#[derive(Clone, Debug)]
pub struct CondParams<B: Backend> {
    pub scale: Tensor<B, 2>,
    pub shift: Tensor<B, 2>,
}

/// Normalization layer wrapper for decoder blocks.
#[derive(Module, Debug)]
pub struct DecoderNorm<B: Backend> {
    batch: Option<BatchNorm<B>>,
    cond: Option<ConditionalInstanceNorm<B>>,
}

impl<B: Backend> DecoderNorm<B> {
    /// Create a decoder norm backed by batch normalization.
    fn new_batch(num_channels: usize, device: &B::Device) -> Self {
        Self {
            batch: Some(BatchNormConfig::new(num_channels).init(device)),
            cond: None,
        }
    }

    /// Create a decoder norm backed by conditional instance normalization.
    fn new_cond(num_labels: usize, num_channels: usize, device: &B::Device) -> Self {
        Self {
            batch: None,
            cond: Some(ConditionalInstanceNorm::new(
                num_labels,
                num_channels,
                1e-5,
                device,
            )),
        }
    }

    /// Apply the configured normalization mode.
    fn forward(
        &self,
        x: Tensor<B, 4>,
        ids: Option<&Tensor<B, 1, Int>>,
        params: Option<&CondParams<B>>,
    ) -> Tensor<B, 4> {
        match (&self.cond, &self.batch, params, ids) {
            (Some(cond), _, Some(params), _) => {
                cond.forward_with_scale_shift(x, params.scale.clone(), params.shift.clone())
            }
            (Some(cond), _, None, Some(ids)) => cond.forward(x, ids.clone()),
            (_, Some(batch), _, _) => batch.forward(x),
            _ => x,
        }
    }

    /// Look up conditional normalization parameters for the given label ids.
    fn lookup(&self, ids: &Tensor<B, 1, Int>) -> Option<CondParams<B>> {
        let cond = self.cond.as_ref()?;
        let (scale, shift) = cond.lookup(ids.clone());
        Some(CondParams { scale, shift })
    }
}

/// U-Net generator that conditions on label embeddings.
#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    embedding: Embedding<B>,
    enc_convs: Vec<Conv2d<B>>,
    enc_bns: Vec<BatchNorm<B>>,
    dec_convs: Vec<ConvTranspose2d<B>>,
    dec_norms: Vec<DecoderNorm<B>>,
    dropout: Dropout,
    #[module(ignore)]
    embedding_dim: usize,
    #[module(ignore)]
    inst_norm: bool,
}

impl GeneratorConfig {
    /// Initialize generator layers on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        let embedding = EmbeddingConfig::new(self.embedding_num, self.embedding_dim).init(device);

        let mut enc_convs = Vec::with_capacity(8);
        enc_convs.push(enc_conv(self.input_channels, self.generator_dim, true, device));
        enc_convs.push(enc_conv(self.generator_dim, self.generator_dim * 2, false, device));
        enc_convs.push(enc_conv(self.generator_dim * 2, self.generator_dim * 4, false, device));
        enc_convs.push(enc_conv(self.generator_dim * 4, self.generator_dim * 8, false, device));
        enc_convs.push(enc_conv(self.generator_dim * 8, self.generator_dim * 8, false, device));
        enc_convs.push(enc_conv(self.generator_dim * 8, self.generator_dim * 8, false, device));
        enc_convs.push(enc_conv(self.generator_dim * 8, self.generator_dim * 8, false, device));
        enc_convs.push(enc_conv(self.generator_dim * 8, self.generator_dim * 8, false, device));

        let enc_bns = vec![
            BatchNormConfig::new(self.generator_dim * 2).init(device),
            BatchNormConfig::new(self.generator_dim * 4).init(device),
            BatchNormConfig::new(self.generator_dim * 8).init(device),
            BatchNormConfig::new(self.generator_dim * 8).init(device),
            BatchNormConfig::new(self.generator_dim * 8).init(device),
            BatchNormConfig::new(self.generator_dim * 8).init(device),
            BatchNormConfig::new(self.generator_dim * 8).init(device),
        ];

        let mut dec_convs = Vec::with_capacity(8);
        dec_convs.push(dec_conv(
            self.generator_dim * 8 + self.embedding_dim,
            self.generator_dim * 8,
            false,
            device,
        ));
        dec_convs.push(dec_conv(self.generator_dim * 16, self.generator_dim * 8, false, device));
        dec_convs.push(dec_conv(self.generator_dim * 16, self.generator_dim * 8, false, device));
        dec_convs.push(dec_conv(self.generator_dim * 16, self.generator_dim * 8, false, device));
        dec_convs.push(dec_conv(self.generator_dim * 16, self.generator_dim * 4, false, device));
        dec_convs.push(dec_conv(self.generator_dim * 8, self.generator_dim * 2, false, device));
        dec_convs.push(dec_conv(self.generator_dim * 4, self.generator_dim, false, device));
        dec_convs.push(dec_conv(self.generator_dim * 2, self.output_channels, true, device));

        let mut dec_norms = Vec::with_capacity(7);
        let norm_channels = [
            self.generator_dim * 8,
            self.generator_dim * 8,
            self.generator_dim * 8,
            self.generator_dim * 8,
            self.generator_dim * 4,
            self.generator_dim * 2,
            self.generator_dim,
        ];
        for &channels in &norm_channels {
            let norm = if self.inst_norm {
                DecoderNorm::new_cond(self.embedding_num, channels, device)
            } else {
                DecoderNorm::new_batch(channels, device)
            };
            dec_norms.push(norm);
        }

        let dropout = DropoutConfig::new(0.5).init();

        Generator {
            embedding,
            enc_convs,
            enc_bns,
            dec_convs,
            dec_norms,
            dropout,
            embedding_dim: self.embedding_dim,
            inst_norm: self.inst_norm,
        }
    }
}

impl<B: Backend> Generator<B> {
    /// Forward pass that returns (fake_b, encoded_real_a).
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
        embedding_ids: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (encoded, enc_layers) = self.encode(images);
        let style = self.lookup_embedding(embedding_ids.clone());
        let batch = encoded.dims()[0];
        let style = style.reshape([batch, self.embedding_dim, 1, 1]);
        let x = Tensor::cat(vec![encoded.clone(), style], 1);
        let output = self.decode(x, &enc_layers, Some(&embedding_ids), None);
        (output, encoded)
    }

    /// Forward pass using a precomputed style embedding and optional norm params.
    pub fn forward_with_params(
        &self,
        images: Tensor<B, 4>,
        style_embedding: Tensor<B, 2>,
        params: Option<&[CondParams<B>]>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (encoded, enc_layers) = self.encode(images);
        let batch = encoded.dims()[0];
        let style = style_embedding.reshape([batch, self.embedding_dim, 1, 1]);
        let x = Tensor::cat(vec![encoded.clone(), style], 1);
        let output = self.decode(x, &enc_layers, None, params);
        (output, encoded)
    }

    /// Forward pass with interpolated style parameters between two label ids.
    pub fn forward_interpolated(
        &self,
        images: Tensor<B, 4>,
        start_id: i64,
        end_id: i64,
        alpha: f64,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let device = images.device();
        let batch_size = images.dims()[0];
        let ids_start = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![start_id], [1]), &device);
        let ids_end = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![end_id], [1]), &device);
        let style_start = self.lookup_embedding(ids_start.clone());
        let style_end = self.lookup_embedding(ids_end.clone());
        let style = lerp(style_start, style_end, alpha).repeat(&[batch_size, 1]);

        let params = if self.inst_norm {
            Some(self.interpolate_params(&ids_start, &ids_end, alpha, batch_size))
        } else {
            None
        };

        self.forward_with_params(images, style, params.as_deref())
    }

    /// Encode only, used by the const loss.
    pub fn encode_only(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode(images).0
    }

    fn encode(&self, images: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut layers = Vec::with_capacity(8);
        let mut x = self.enc_convs[0].forward(images);
        layers.push(x.clone());
        for idx in 1..self.enc_convs.len() {
            x = leaky_relu(x, 0.2);
            x = self.enc_convs[idx].forward(x);
            x = self.enc_bns[idx - 1].forward(x);
            layers.push(x.clone());
        }
        (x, layers)
    }

    fn decode(
        &self,
        mut x: Tensor<B, 4>,
        enc_layers: &[Tensor<B, 4>],
        embedding_ids: Option<&Tensor<B, 1, Int>>,
        params: Option<&[CondParams<B>]>,
    ) -> Tensor<B, 4> {
        // U-Net decoder: early dropout and skip connections for spatial detail.
        for idx in 0..self.dec_convs.len() {
            x = relu(x);
            x = self.dec_convs[idx].forward(x);
            if idx != self.dec_convs.len() - 1 {
                let norm_params = params.and_then(|p| p.get(idx));
                x = self.dec_norms[idx].forward(x, embedding_ids, norm_params);
            }
            if idx < 3 {
                x = self.dropout.forward(x);
            }
            if idx < 7 {
                let skip = enc_layers[6 - idx].clone();
                x = Tensor::cat(vec![x, skip], 1);
            }
        }
        x.tanh()
    }

    fn lookup_embedding(&self, ids: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let ids = ids.unsqueeze_dim::<2>(1);
        self.embedding.forward(ids).squeeze_dim::<2>(1)
    }

    fn interpolate_params(
        &self,
        start_ids: &Tensor<B, 1, Int>,
        end_ids: &Tensor<B, 1, Int>,
        alpha: f64,
        batch_size: usize,
    ) -> Vec<CondParams<B>> {
        self.dec_norms
            .iter()
            .map(|norm| {
                let start = norm.lookup(start_ids).expect("missing conditional norm");
                let end = norm.lookup(end_ids).expect("missing conditional norm");
                let scale = lerp(start.scale, end.scale, alpha).repeat(&[batch_size, 1]);
                let shift = lerp(start.shift, end.shift, alpha).repeat(&[batch_size, 1]);
                CondParams { scale, shift }
            })
            .collect()
    }
}

fn lerp<B: Backend>(start: Tensor<B, 2>, end: Tensor<B, 2>, alpha: f64) -> Tensor<B, 2> {
    start.mul_scalar(1.0 - alpha).add(end.mul_scalar(alpha))
}

fn enc_conv<B: Backend>(
    in_channels: usize,
    out_channels: usize,
    bias: bool,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([in_channels, out_channels], [4, 4])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .with_bias(bias)
        .init(device)
}

fn dec_conv<B: Backend>(
    in_channels: usize,
    out_channels: usize,
    bias: bool,
    device: &B::Device,
) -> ConvTranspose2d<B> {
    ConvTranspose2dConfig::new([in_channels, out_channels], [4, 4])
        .with_stride([2, 2])
        .with_padding([1, 1])
        .with_bias(bias)
        .init(device)
}
