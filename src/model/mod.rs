pub mod discriminator;
pub mod generator;
pub mod layers;

use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::prelude::*;
use discriminator::DiscriminatorConfig;

pub use discriminator::Discriminator;
pub use discriminator::DiscriminatorConfig as DiscConfig;
pub use generator::{CondParams, Generator, GeneratorConfig};

/// Hyperparameters for the generator and discriminator.
#[derive(Config, Debug)]
pub struct ModelConfig {
    pub image_size: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub generator_dim: usize,
    pub discriminator_dim: usize,
    pub embedding_num: usize,
    pub embedding_dim: usize,
    #[config(default = false)]
    pub inst_norm: bool,
}

/// Weighting for each loss term used during training.
#[derive(Config, Debug)]
pub struct LossConfig {
    #[config(default = 100.0)]
    pub l1_penalty: f64,
    #[config(default = 15.0)]
    pub lconst_penalty: f64,
    #[config(default = 0.0)]
    pub ltv_penalty: f64,
    #[config(default = 1.0)]
    pub lcategory_penalty: f64,
}

/// Collected loss values and intermediate tensors for logging or sampling.
#[derive(Debug)]
pub struct Zi2ziLosses<B: Backend> {
    pub d_loss: Tensor<B, 1>,
    pub g_loss: Tensor<B, 1>,
    pub const_loss: Tensor<B, 1>,
    pub l1_loss: Tensor<B, 1>,
    pub category_loss: Tensor<B, 1>,
    pub cheat_loss: Tensor<B, 1>,
    pub tv_loss: Tensor<B, 1>,
    pub fake_b: Tensor<B, 4>,
    pub real_b: Tensor<B, 4>,
}

impl ModelConfig {
    pub fn generator_config(&self) -> GeneratorConfig {
        GeneratorConfig::new(
            self.generator_dim,
            self.embedding_num,
            self.embedding_dim,
            self.input_channels,
            self.output_channels,
        )
        .with_inst_norm(self.inst_norm)
    }

    pub fn discriminator_config(&self) -> DiscriminatorConfig {
        DiscriminatorConfig::new(
            self.discriminator_dim,
            self.embedding_num,
            self.input_channels,
            self.output_channels,
            self.image_size,
        )
    }

    pub fn init_generator<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        self.generator_config().init(device)
    }

    pub fn init_discriminator<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        self.discriminator_config().init(device)
    }
}

/// Split a channel-concatenated batch into (right_half, left_half).
///
/// The data pipeline stores paired images as [left | right]. With the default
/// convention left=target and right=source, this returns (source, target).
pub fn split_real<B: Backend>(
    real_data: Tensor<B, 4>,
    input_channels: usize,
    output_channels: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let real_b = real_data.clone().slice_dim(1, 0..input_channels);
    let real_a = real_data.slice_dim(1, input_channels..(input_channels + output_channels));
    (real_a, real_b)
}

/// Compute generator/discriminator losses for a batch, with optional no-target loss.
pub fn compute_losses<B: Backend>(
    generator: &Generator<B>,
    discriminator: &Discriminator<B>,
    model_config: &ModelConfig,
    loss_config: &LossConfig,
    real_data: Tensor<B, 4>,
    embedding_ids: Tensor<B, 1, Int>,
    no_target: Option<(Tensor<B, 4>, Tensor<B, 1, Int>)>,
) -> Zi2ziLosses<B> {
    let device = real_data.device();
    let (real_a, real_b) = split_real(
        real_data,
        model_config.input_channels,
        model_config.output_channels,
    );

    let (fake_b, encoded_real_a) = generator.forward(real_a.clone(), embedding_ids.clone());
    let real_ab = Tensor::cat(vec![real_a.clone(), real_b.clone()], 1);
    let fake_ab = Tensor::cat(vec![real_a.clone(), fake_b.clone()], 1);

    let (real_logits, real_category_logits) = discriminator.forward(real_ab);
    let (fake_logits, fake_category_logits) = discriminator.forward(fake_ab.clone());

    let encoded_fake_b = generator.encode_only(fake_b.clone());
    let const_loss = encoded_real_a
        .sub(encoded_fake_b)
        .square()
        .mean()
        .mul_scalar(loss_config.lconst_penalty);

    let true_labels = embedding_ids.clone().one_hot::<2>(model_config.embedding_num);
    let category_loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);
    let real_category_loss = category_loss_fn.forward(real_category_logits, true_labels.clone());
    let fake_category_loss = category_loss_fn.forward(fake_category_logits, true_labels.clone());
    let category_loss = real_category_loss
        .clone()
        .add(fake_category_loss.clone())
        .mul_scalar(loss_config.lcategory_penalty);

    let adv_loss_fn = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device);
    let ones = Tensor::<B, 2, Int>::ones(real_logits.dims(), &device);
    let zeros = Tensor::<B, 2, Int>::zeros(fake_logits.dims(), &device);
    let d_loss_real = adv_loss_fn.forward(real_logits, ones);
    let d_loss_fake = adv_loss_fn.forward(fake_logits.clone(), zeros);

    let l1_loss = fake_b
        .clone()
        .sub(real_b.clone())
        .abs()
        .mean()
        .mul_scalar(loss_config.l1_penalty);
    let tv_loss = total_variation_loss(fake_b.clone(), model_config.image_size)
        .mul_scalar(loss_config.ltv_penalty);
    let cheat_loss = adv_loss_fn.forward(
        fake_logits.clone(),
        Tensor::<B, 2, Int>::ones(fake_logits.dims(), &device),
    );

    let mut d_loss = d_loss_real
        .clone()
        .add(d_loss_fake.clone())
        .add(category_loss.clone().div_scalar(2.0));
    let mut g_loss = cheat_loss
        .clone()
        .add(l1_loss.clone())
        .add(fake_category_loss.clone().mul_scalar(loss_config.lcategory_penalty))
        .add(const_loss.clone())
        .add(tv_loss.clone());

    if let Some((no_target_data, no_target_ids)) = no_target {
        let (no_target_a, _) = split_real(
            no_target_data,
            model_config.input_channels,
            model_config.output_channels,
        );
        let (no_target_b, encoded_no_target_a) =
            generator.forward(no_target_a.clone(), no_target_ids.clone());
        let no_target_labels = no_target_ids.one_hot::<2>(model_config.embedding_num);
        let no_target_ab = Tensor::cat(vec![no_target_a.clone(), no_target_b.clone()], 1);
        let (no_target_logits, no_target_category_logits) =
            discriminator.forward(no_target_ab);
        let encoded_no_target_b = generator.encode_only(no_target_b);
        let no_target_const_loss = encoded_no_target_a
            .sub(encoded_no_target_b)
            .square()
            .mean()
            .mul_scalar(loss_config.lconst_penalty);
        let no_target_category_loss = category_loss_fn
            .forward(no_target_category_logits, no_target_labels)
            .mul_scalar(loss_config.lcategory_penalty);

        let no_target_dims = no_target_logits.dims();
        let d_loss_no_target = adv_loss_fn.forward(
            no_target_logits.clone(),
            Tensor::<B, 2, Int>::zeros(no_target_dims, &device),
        );
        let no_target_cheat = adv_loss_fn.forward(
            no_target_logits,
            Tensor::<B, 2, Int>::ones(no_target_dims, &device),
        );

        d_loss = d_loss_real
            .clone()
            .add(d_loss_fake.clone())
            .add(d_loss_no_target)
            .add(category_loss.clone().add(no_target_category_loss.clone()).div_scalar(3.0));
        g_loss = cheat_loss
            .clone()
            .add(no_target_cheat)
            .div_scalar(2.0)
            .add(l1_loss.clone())
            .add(
                fake_category_loss
                    .mul_scalar(loss_config.lcategory_penalty)
                    .add(no_target_category_loss)
                    .div_scalar(2.0),
            )
            .add(const_loss.clone().add(no_target_const_loss).div_scalar(2.0))
            .add(tv_loss.clone());
    }

    Zi2ziLosses {
        d_loss,
        g_loss,
        const_loss,
        l1_loss,
        category_loss,
        cheat_loss,
        tv_loss,
        fake_b,
        real_b,
    }
}

/// Total variation regularizer for spatial smoothness.
fn total_variation_loss<B: Backend>(fake_b: Tensor<B, 4>, image_size: usize) -> Tensor<B, 1> {
    let height = image_size;
    let width = image_size;
    let diff_h = fake_b
        .clone()
        .slice_dim(2, 1..height)
        .sub(fake_b.clone().slice_dim(2, 0..(height - 1)));
    let diff_w = fake_b
        .clone()
        .slice_dim(3, 1..width)
        .sub(fake_b.slice_dim(3, 0..(width - 1)));
    l2_loss(diff_h)
        .div_scalar(width as f64)
        .add(l2_loss(diff_w).div_scalar(width as f64))
}

/// Convenience L2 loss used by the TV term (0.5 * sum of squares).
fn l2_loss<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 1> {
    tensor.square().sum().mul_scalar(0.5)
}
