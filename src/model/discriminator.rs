use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::leaky_relu;
use burn::prelude::*;

/// Configuration for the PatchGAN-style discriminator.
#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    pub discriminator_dim: usize,
    pub embedding_num: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub image_size: usize,
}

/// Discriminator with adversarial and category classification heads.
#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    convs: Vec<Conv2d<B>>,
    bns: Vec<BatchNorm<B>>,
    fc_adv: Linear<B>,
    fc_cls: Linear<B>,
}

impl DiscriminatorConfig {
    /// Initialize the discriminator layers on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        let in_channels = self.input_channels + self.output_channels;
        let mut convs = Vec::with_capacity(4);
        convs.push(conv(in_channels, self.discriminator_dim, true, device, 2));
        convs.push(conv(
            self.discriminator_dim,
            self.discriminator_dim * 2,
            false,
            device,
            2,
        ));
        convs.push(conv(
            self.discriminator_dim * 2,
            self.discriminator_dim * 4,
            false,
            device,
            2,
        ));
        convs.push(conv(
            self.discriminator_dim * 4,
            self.discriminator_dim * 8,
            false,
            device,
            1,
        ));

        let bns = vec![
            BatchNormConfig::new(self.discriminator_dim * 2).init(device),
            BatchNormConfig::new(self.discriminator_dim * 4).init(device),
            BatchNormConfig::new(self.discriminator_dim * 8).init(device),
        ];

        let mut size = self.image_size;
        size = conv_out(size, 4, 2, 1);
        size = conv_out(size, 4, 2, 1);
        size = conv_out(size, 4, 2, 1);
        size = conv_out(size, 4, 1, 1);
        let flat_dim = size * size * self.discriminator_dim * 8;

        let fc_adv = LinearConfig::new(flat_dim, 1).init(device);
        let fc_cls = LinearConfig::new(flat_dim, self.embedding_num).init(device);

        Discriminator {
            convs,
            bns,
            fc_adv,
            fc_cls,
        }
    }
}

impl<B: Backend> Discriminator<B> {
    /// Forward pass returning (adversarial_logits, category_logits).
    pub fn forward(&self, images: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = leaky_relu(self.convs[0].forward(images), 0.2);
        x = leaky_relu(self.bns[0].forward(self.convs[1].forward(x)), 0.2);
        x = leaky_relu(self.bns[1].forward(self.convs[2].forward(x)), 0.2);
        x = leaky_relu(self.bns[2].forward(self.convs[3].forward(x)), 0.2);

        let [batch, channels, height, width] = x.dims();
        let flat = x.reshape([batch, channels * height * width]);
        let logits = self.fc_adv.forward(flat.clone());
        let category_logits = self.fc_cls.forward(flat);
        (logits, category_logits)
    }
}

fn conv<B: Backend>(
    in_channels: usize,
    out_channels: usize,
    bias: bool,
    device: &B::Device,
    stride: usize,
) -> Conv2d<B> {
    Conv2dConfig::new([in_channels, out_channels], [4, 4])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .with_bias(bias)
        .init(device)
}

fn conv_out(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding - (kernel - 1) - 1) / stride + 1
}
