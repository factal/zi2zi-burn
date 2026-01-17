use crate::data::{build_batch, load_pickled_examples, DataConfig, Example};
use crate::model::{compute_losses, LossConfig, ModelConfig};
use crate::utils::{merge_images, save_concat_images, tensor_to_images};
use anyhow::{Context, Result};
use burn::config::Config;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Ignored, ModuleVisitor, Param};
use burn::optim::{AdamConfig, GradientsParams, MultiGradientsParams, Optimizer};
use burn::optim::lr_scheduler::LrScheduler;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::*;
use burn::record::{CompactRecorder, Record};
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{
    EventProcessorTraining, InferenceStep, ItemLazy, Learner, LearnerEvent, LearnerItem,
    LearningComponentsMarker, SupervisedLearningStrategy, SupervisedTraining,
    SupervisedTrainingEventProcessor, TrainOutput, TrainStep, TrainingComponents,
    TrainingStrategy, TrainLoader, ValidLoader,
};
use burn::train::checkpoint::{CheckpointingAction, CheckpointingStrategy};
use burn::train::metric::{
    Adaptor, Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, SerializedEntry,
};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::store::EventStoreClient;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Training configuration loaded from `config.json`.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub loss: LossConfig,
    pub data_dir: String,
    pub experiment_id: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub schedule: usize,
    pub min_learning_rate: f64,
    pub resume: bool,
    pub flip_labels: bool,
    pub swap_ab: bool,
    pub sample_steps: usize,
    pub checkpoint_steps: usize,
    pub seed: u64,
    pub optimizer_gen: AdamConfig,
    pub optimizer_disc: AdamConfig,
    pub fine_tune: Option<Vec<i64>>,
}

type Zi2ziComponents<B> =
    LearningComponentsMarker<B, Zi2ziLrScheduler, Zi2ziGan<B>, Zi2ziOptimizer<B>>;

/// Persisted training state for resume support.
#[derive(Serialize, Deserialize, Default, Clone)]
struct TrainingState {
    step: usize,
    epoch: usize,
    learning_rate: f64,
}

#[derive(Clone)]
struct Zi2ziDataset {
    examples: Vec<Arc<Example>>,
    len: usize,
}

impl Zi2ziDataset {
    fn new(examples: Vec<Arc<Example>>, batch_size: usize, wrap: bool) -> Self {
        let len = if wrap && !examples.is_empty() {
            let batches = (examples.len() + batch_size - 1) / batch_size;
            batches * batch_size
        } else {
            examples.len()
        };
        Self { examples, len }
    }
}

impl Dataset<Arc<Example>> for Zi2ziDataset {
    fn get(&self, index: usize) -> Option<Arc<Example>> {
        if self.examples.is_empty() {
            return None;
        }
        let idx = if self.len == self.examples.len() {
            index
        } else {
            index % self.examples.len()
        };
        self.examples.get(idx).cloned()
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone)]
struct Zi2ziBatcher {
    config: DataConfig,
    augment: bool,
    flip_labels: bool,
    rng: Arc<Mutex<StdRng>>,
}

impl Zi2ziBatcher {
    fn new(config: DataConfig, augment: bool, flip_labels: bool, seed: u64) -> Self {
        Self {
            config,
            augment,
            flip_labels,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }
}

impl<B: Backend> Batcher<B, Arc<Example>, Zi2ziTrainInput<B>> for Zi2ziBatcher {
    fn batch(&self, items: Vec<Arc<Example>>, device: &B::Device) -> Zi2ziTrainInput<B> {
        let refs: Vec<&Example> = items.iter().map(|item| item.as_ref()).collect();
        let mut rng = self.rng.lock().expect("rng lock poisoned");
        let batch = build_batch::<B>(&refs, &self.config, self.augment, &mut *rng, device)
            .expect("failed to build batch");
        let embedding_ids = batch.labels.clone();
        let no_target = if self.flip_labels {
            let labels = embedding_ids
                .to_data()
                .to_vec::<i32>()
                .expect("failed to read labels");
            let mut shuffled = labels.clone();
            shuffled.shuffle(&mut *rng);
            let shuffled = Tensor::<B, 1, Int>::from_data(
                TensorData::new(shuffled, [labels.len()]),
                device,
            );
            Some((batch.images.clone(), shuffled))
        } else {
            None
        };

        Zi2ziTrainInput {
            images: batch.images,
            embedding_ids,
            no_target,
            mode: Zi2ziTrainMode::Generator,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Zi2ziTrainMode {
    Discriminator,
    Generator,
}

#[derive(Clone, Debug)]
struct Zi2ziTrainInput<B: Backend> {
    images: Tensor<B, 4>,
    embedding_ids: Tensor<B, 1, Int>,
    no_target: Option<(Tensor<B, 4>, Tensor<B, 1, Int>)>,
    mode: Zi2ziTrainMode,
}

impl<B: Backend> Zi2ziTrainInput<B> {
    fn with_mode(mut self, mode: Zi2ziTrainMode) -> Self {
        self.mode = mode;
        self
    }
}

#[derive(Clone, Debug)]
struct Zi2ziMetrics {
    batch_size: usize,
    d_loss: f64,
    g_loss: f64,
    category_loss: f64,
    cheat_loss: f64,
    const_loss: f64,
    l1_loss: f64,
    tv_loss: f64,
}

impl ItemLazy for Zi2ziMetrics {
    type ItemSync = Zi2ziMetrics;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<Zi2ziMetrics> for Zi2ziMetrics {
    fn adapt(&self) -> Zi2ziMetrics {
        self.clone()
    }
}

#[derive(Module, Debug)]
struct Zi2ziGan<B: Backend> {
    generator: crate::model::generator::Generator<B>,
    discriminator: crate::model::discriminator::Discriminator<B>,
    model_config: Ignored<ModelConfig>,
    loss_config: Ignored<LossConfig>,
}

impl<B: Backend> Zi2ziGan<B> {
    fn new(model_config: ModelConfig, loss_config: LossConfig, device: &B::Device) -> Self {
        let generator = model_config.init_generator(device);
        let discriminator = model_config.init_discriminator(device);
        Self {
            generator,
            discriminator,
            model_config: Ignored(model_config),
            loss_config: Ignored(loss_config),
        }
    }
}

impl<B: AutodiffBackend> TrainStep for Zi2ziGan<B> {
    type Input = Zi2ziTrainInput<B>;
    type Output = Zi2ziMetrics;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let batch_size = item.embedding_ids.dims()[0];
        let losses = compute_losses(
            &self.generator,
            &self.discriminator,
            &self.model_config,
            &self.loss_config,
            item.images,
            item.embedding_ids,
            item.no_target,
        );

        let metrics = Zi2ziMetrics {
            batch_size,
            d_loss: losses.d_loss.clone().into_scalar().elem::<f64>(),
            g_loss: losses.g_loss.clone().into_scalar().elem::<f64>(),
            category_loss: losses.category_loss.clone().into_scalar().elem::<f64>(),
            cheat_loss: losses.cheat_loss.clone().into_scalar().elem::<f64>(),
            const_loss: losses.const_loss.clone().into_scalar().elem::<f64>(),
            l1_loss: losses.l1_loss.clone().into_scalar().elem::<f64>(),
            tv_loss: losses.tv_loss.clone().into_scalar().elem::<f64>(),
        };

        let grads = match item.mode {
            Zi2ziTrainMode::Discriminator => losses.d_loss.backward(),
            Zi2ziTrainMode::Generator => losses.g_loss.backward(),
        };

        match item.mode {
            Zi2ziTrainMode::Discriminator => TrainOutput::new(&self.discriminator, grads, metrics),
            Zi2ziTrainMode::Generator => TrainOutput::new(&self.generator, grads, metrics),
        }
    }
}

impl<B: Backend> InferenceStep for Zi2ziGan<B> {
    type Input = Zi2ziTrainInput<B>;
    type Output = Zi2ziMetrics;

    fn step(&self, item: Self::Input) -> Self::Output {
        let batch_size = item.embedding_ids.dims()[0];
        let losses = compute_losses(
            &self.generator,
            &self.discriminator,
            &self.model_config,
            &self.loss_config,
            item.images,
            item.embedding_ids,
            item.no_target,
        );

        Zi2ziMetrics {
            batch_size,
            d_loss: losses.d_loss.into_scalar().elem::<f64>(),
            g_loss: losses.g_loss.into_scalar().elem::<f64>(),
            category_loss: losses.category_loss.into_scalar().elem::<f64>(),
            cheat_loss: losses.cheat_loss.into_scalar().elem::<f64>(),
            const_loss: losses.const_loss.into_scalar().elem::<f64>(),
            l1_loss: losses.l1_loss.into_scalar().elem::<f64>(),
            tv_loss: losses.tv_loss.into_scalar().elem::<f64>(),
        }
    }
}

#[derive(Clone, Copy)]
enum Zi2ziLossKind {
    D,
    G,
    Category,
    Cheat,
    Const,
    L1,
    Tv,
}

#[derive(Clone)]
struct Zi2ziLossMetric {
    kind: Zi2ziLossKind,
    name: MetricName,
    state: NumericMetricState,
}

impl Zi2ziLossMetric {
    fn new(kind: Zi2ziLossKind, name: &str) -> Self {
        Self {
            kind,
            name: Arc::new(name.to_string()),
            state: NumericMetricState::default(),
        }
    }

    fn value(&self, metrics: &Zi2ziMetrics) -> f64 {
        match self.kind {
            Zi2ziLossKind::D => metrics.d_loss,
            Zi2ziLossKind::G => metrics.g_loss,
            Zi2ziLossKind::Category => metrics.category_loss,
            Zi2ziLossKind::Cheat => metrics.cheat_loss,
            Zi2ziLossKind::Const => metrics.const_loss,
            Zi2ziLossKind::L1 => metrics.l1_loss,
            Zi2ziLossKind::Tv => metrics.tv_loss,
        }
    }
}

impl Metric for Zi2ziLossMetric {
    type Input = Zi2ziMetrics;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(&mut self, metrics: &Self::Input, _metadata: &burn::train::metric::MetricMetadata) -> SerializedEntry {
        self.state.update(
            self.value(metrics),
            metrics.batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl Numeric for Zi2ziLossMetric {
    fn value(&self) -> burn::train::metric::NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> burn::train::metric::NumericEntry {
        self.state.running_value()
    }
}

struct GradsExtractor<'a, B: AutodiffBackend> {
    source: &'a mut GradientsParams,
    target: &'a mut GradientsParams,
    _phantom: PhantomData<B>,
}

impl<'a, B: AutodiffBackend> GradsExtractor<'a, B> {
    fn new(source: &'a mut GradientsParams, target: &'a mut GradientsParams) -> Self {
        Self {
            source,
            target,
            _phantom: PhantomData,
        }
    }
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradsExtractor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.source.remove::<B::InnerBackend, D>(param.id) {
            self.target.register::<B::InnerBackend, D>(param.id, grad);
        }
    }
}

#[derive(Clone)]
struct Zi2ziOptimizer<B: AutodiffBackend> {
    optim_gen: OptimizerAdaptor<burn::optim::Adam, crate::model::generator::Generator<B>, B>,
    optim_disc: OptimizerAdaptor<burn::optim::Adam, crate::model::discriminator::Discriminator<B>, B>,
}

type Zi2ziOptimRecordGen<B> = <OptimizerAdaptor<
    burn::optim::Adam,
    crate::model::generator::Generator<B>,
    B,
> as Optimizer<crate::model::generator::Generator<B>, B>>::Record;
type Zi2ziOptimRecordDisc<B> = <OptimizerAdaptor<
    burn::optim::Adam,
    crate::model::discriminator::Discriminator<B>,
    B,
> as Optimizer<crate::model::discriminator::Discriminator<B>, B>>::Record;

impl<B: AutodiffBackend> Zi2ziOptimizer<B> {
    fn new(
        optim_gen: OptimizerAdaptor<burn::optim::Adam, crate::model::generator::Generator<B>, B>,
        optim_disc: OptimizerAdaptor<burn::optim::Adam, crate::model::discriminator::Discriminator<B>, B>,
    ) -> Self {
        Self { optim_gen, optim_disc }
    }
}

impl<B: AutodiffBackend> Optimizer<Zi2ziGan<B>, B> for Zi2ziOptimizer<B> {
    type Record = (Zi2ziOptimRecordGen<B>, Zi2ziOptimRecordDisc<B>);

    fn step(&mut self, lr: f64, module: Zi2ziGan<B>, grads: GradientsParams) -> Zi2ziGan<B> {
        let (gen_grads, disc_grads) = split_grads(&module, grads);
        let Zi2ziGan {
            generator,
            discriminator,
            model_config,
            loss_config,
        } = module;

        let generator = if gen_grads.is_empty() {
            generator
        } else {
            self.optim_gen.step(lr, generator, gen_grads)
        };

        let discriminator = if disc_grads.is_empty() {
            discriminator
        } else {
            self.optim_disc.step(lr, discriminator, disc_grads)
        };

        Zi2ziGan {
            generator,
            discriminator,
            model_config,
            loss_config,
        }
    }

    fn step_multi(
        &mut self,
        lr: f64,
        module: Zi2ziGan<B>,
        mut grads: MultiGradientsParams,
    ) -> Zi2ziGan<B> {
        if grads.grads.is_empty() {
            return module;
        }
        let (grads, _) = grads.grads.remove(0);
        self.step(lr, module, grads)
    }

    fn to_record(&self) -> Self::Record {
        (self.optim_gen.to_record(), self.optim_disc.to_record())
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.optim_gen = self.optim_gen.load_record(record.0);
        self.optim_disc = self.optim_disc.load_record(record.1);
        self
    }
}

fn split_grads<B: AutodiffBackend>(
    model: &Zi2ziGan<B>,
    mut grads: GradientsParams,
) -> (GradientsParams, GradientsParams) {
    let mut gen_grads = GradientsParams::new();
    let mut extractor = GradsExtractor::<B>::new(&mut grads, &mut gen_grads);
    model.generator.visit(&mut extractor);

    let mut disc_grads = GradientsParams::new();
    let mut extractor = GradsExtractor::<B>::new(&mut grads, &mut disc_grads);
    model.discriminator.visit(&mut extractor);

    (gen_grads, disc_grads)
}

#[derive(Clone, Debug)]
struct Zi2ziLrScheduler {
    lr: f64,
    min_lr: f64,
    schedule: usize,
    epoch: usize,
}

#[derive(Record, Clone)]
struct Zi2ziLrRecord {
    lr: f64,
    epoch: usize,
}

impl Zi2ziLrScheduler {
    fn new(lr: f64, schedule: usize, min_lr: f64, epoch: usize) -> Self {
        Self {
            lr,
            min_lr,
            schedule,
            epoch,
        }
    }
}

impl LrScheduler for Zi2ziLrScheduler {
    type Record<B: Backend> = Zi2ziLrRecord;

    fn step(&mut self) -> f64 {
        self.epoch += 1;
        if self.schedule > 0 && self.epoch % self.schedule == 0 {
            self.lr = (self.lr / 2.0).max(self.min_lr);
        }
        self.lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        Zi2ziLrRecord {
            lr: self.lr,
            epoch: self.epoch,
        }
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.lr = record.lr;
        self.epoch = record.epoch;
        self
    }
}

#[derive(Clone)]
struct AlwaysSaveCheckpoints;

impl CheckpointingStrategy for AlwaysSaveCheckpoints {
    fn checkpointing(
        &mut self,
        _epoch: usize,
        _collector: &EventStoreClient,
    ) -> Vec<CheckpointingAction> {
        vec![CheckpointingAction::Save]
    }
}

struct Zi2ziTrainingStrategy<B: AutodiffBackend> {
    config: TrainingConfig,
    data_config: DataConfig,
    model_dir: PathBuf,
    state_path: PathBuf,
    val_examples: Vec<Arc<Example>>,
    start_state: TrainingState,
    device: B::Device,
}

impl<B: AutodiffBackend> Zi2ziTrainingStrategy<B> {
    fn new(
        config: TrainingConfig,
        data_config: DataConfig,
        model_dir: PathBuf,
        state_path: PathBuf,
        val_examples: Vec<Arc<Example>>,
        start_state: TrainingState,
        device: B::Device,
    ) -> Self {
        Self {
            config,
            data_config,
            model_dir,
            state_path,
            val_examples,
            start_state,
            device,
        }
    }
}

impl<B: AutodiffBackend> SupervisedLearningStrategy<Zi2ziComponents<B>>
    for Zi2ziTrainingStrategy<B>
{
    fn fit(
        &self,
        mut training_components: TrainingComponents<Zi2ziComponents<B>>,
        mut learner: Learner<Zi2ziComponents<B>>,
        dataloader_train: TrainLoader<Zi2ziComponents<B>>,
        dataloader_valid: ValidLoader<Zi2ziComponents<B>>,
        _starting_epoch: usize,
    ) -> (Zi2ziGan<B>, SupervisedTrainingEventProcessor<Zi2ziComponents<B>>) {
        let mut state = self.start_state.clone();
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        if self.config.resume && state.step > 0 {
            if let Some(checkpointer) = training_components.checkpointer.as_ref() {
                learner = checkpointer.load_checkpoint(learner, &self.device, state.step);
            }
        }

        for epoch in state.epoch..self.config.num_epochs {
            let lr_prev = state.learning_rate;
            learner.lr_step();
            state.learning_rate = learner.lr_current();

            if state.learning_rate != lr_prev {
                println!(
                    "decay learning rate from {:.5} to {:.5}",
                    lr_prev, state.learning_rate
                );
            }

            let epoch_for_metrics = epoch + 1;
            let mut iterator = dataloader_train.iter();
            let mut iteration = 0;

            while let Some(batch) = iterator.next() {
                iteration += 1;
                state.step += 1;

                let progress = iterator.progress();

                let output_d = learner.train_step(
                    batch
                        .clone()
                        .with_mode(Zi2ziTrainMode::Discriminator),
                );
                learner.optimizer_step(output_d.grads);

                let output_g = learner.train_step(
                    batch
                        .clone()
                        .with_mode(Zi2ziTrainMode::Generator),
                );
                learner.optimizer_step(output_g.grads);

                let output_g2 = learner.train_step(batch);
                learner.optimizer_step(output_g2.grads);

                let item = LearnerItem::new(
                    output_g2.item,
                    progress,
                    epoch_for_metrics,
                    self.config.num_epochs,
                    iteration,
                    Some(learner.lr_current()),
                );
                training_components
                    .event_processor
                    .process_train(LearnerEvent::ProcessedItem(item));

                if self.config.sample_steps > 0
                    && state.step % self.config.sample_steps == 0
                {
                    validate_model(
                        &learner.model().generator,
                        &self.val_examples,
                        &self.data_config,
                        &self.config,
                        &self.model_dir,
                        &mut rng,
                        &self.device,
                        epoch,
                        state.step,
                    )
                    .expect("failed to save validation samples");
                }

                if self.config.checkpoint_steps > 0
                    && state.step % self.config.checkpoint_steps == 0
                {
                    if let Some(checkpointer) = training_components.checkpointer.as_mut() {
                        checkpointer.checkpoint(
                            &learner,
                            state.step,
                            &training_components.event_store,
                        );
                    }
                    write_training_state(&self.state_path, &state);
                }

                if training_components.interrupter.should_stop() {
                    break;
                }
            }

            training_components
                .event_processor
                .process_train(LearnerEvent::EndEpoch(epoch_for_metrics));

            if !self.val_examples.is_empty() && !training_components.interrupter.should_stop() {
                run_validation_epoch(
                    &learner,
                    epoch_for_metrics,
                    self.config.num_epochs,
                    &dataloader_valid,
                    &mut training_components.event_processor,
                    &training_components.interrupter,
                );
            }

            state.epoch = epoch + 1;

            if training_components.interrupter.should_stop() {
                break;
            }
        }

        if let Some(checkpointer) = training_components.checkpointer.as_mut() {
            checkpointer.checkpoint(
                &learner,
                state.step,
                &training_components.event_store,
            );
        }
        write_training_state(&self.state_path, &state);

        (learner.model(), training_components.event_processor)
    }
}

fn run_validation_epoch<B: AutodiffBackend>(
    learner: &Learner<Zi2ziComponents<B>>,
    epoch: usize,
    epoch_total: usize,
    dataloader_valid: &Arc<dyn DataLoader<B::InnerBackend, Zi2ziTrainInput<B::InnerBackend>>>,
    processor: &mut SupervisedTrainingEventProcessor<Zi2ziComponents<B>>,
    interrupter: &burn::train::Interrupter,
) {
    let model = learner.model().valid();
    let mut iterator = dataloader_valid.iter();
    let mut iteration = 0;

    while let Some(item) = iterator.next() {
        let progress = iterator.progress();
        iteration += 1;

        let output = model.step(item);
        let item = LearnerItem::new(output, progress, epoch, epoch_total, iteration, None);
        processor.process_valid(LearnerEvent::ProcessedItem(item));

        if interrupter.should_stop() {
            break;
        }
    }

    processor.process_valid(LearnerEvent::EndEpoch(epoch));
}

fn write_training_state(state_path: &Path, state: &TrainingState) {
    let state_json = serde_json::to_string_pretty(state)
        .expect("failed to serialize training state");
    std::fs::write(state_path, state_json).expect("failed to write training state");
}

/// Train zi2zi models with Burn, handling checkpoints and sampling.
pub fn train<B: AutodiffBackend>(
    experiment_dir: &Path,
    config: TrainingConfig,
    device: B::Device,
) -> Result<()> {
    let checkpoint_dir = experiment_dir.join("checkpoint");
    std::fs::create_dir_all(&checkpoint_dir)?;

    let model_id = format!("experiment_{}_batch_{}", config.experiment_id, config.batch_size);
    let model_dir = checkpoint_dir.join(model_id);
    std::fs::create_dir_all(&model_dir)?;
    config.save(model_dir.join("config.json"))?;

    let state_path = model_dir.join("state.json");
    let mut state = TrainingState {
        learning_rate: config.learning_rate,
        ..Default::default()
    };
    if config.resume && state_path.exists() {
        let contents = std::fs::read_to_string(&state_path)?;
        state = serde_json::from_str(&contents)?;
    }

    let data_dir = resolve_data_dir(experiment_dir, &config.data_dir);
    let train_path = data_dir.join("train.obj");
    let val_path = data_dir.join("val.obj");

    let mut train_examples = load_pickled_examples(&train_path)
        .with_context(|| format!("failed to load {}", train_path.display()))?;
    let mut val_examples = load_pickled_examples(&val_path)
        .with_context(|| format!("failed to load {}", val_path.display()))?;

    if let Some(ids) = &config.fine_tune {
        train_examples.retain(|ex| ids.contains(&ex.label));
        val_examples.retain(|ex| ids.contains(&ex.label));
    }

    println!(
        "train examples -> {}, val examples -> {}",
        train_examples.len(),
        val_examples.len()
    );
    if train_examples.is_empty() {
        return Err(anyhow::anyhow!("no training examples found"));
    }

    let data_config = DataConfig {
        image_size: config.model.image_size as u32,
        input_channels: config.model.input_channels,
        output_channels: config.model.output_channels,
        swap_ab: config.swap_ab,
    };

    B::seed(&device, config.seed);

    let model = Zi2ziGan::new(config.model.clone(), config.loss.clone(), &device);
    let optim_gen = config
        .optimizer_gen
        .init::<B, crate::model::generator::Generator<B>>();
    let optim_disc = config
        .optimizer_disc
        .init::<B, crate::model::discriminator::Discriminator<B>>();
    let optimizer = Zi2ziOptimizer::new(optim_gen, optim_disc);
    let scheduler = Zi2ziLrScheduler::new(
        state.learning_rate,
        config.schedule,
        config.min_learning_rate,
        state.epoch,
    );

    let learner = Learner::<Zi2ziComponents<B>>::new(model, optimizer, scheduler);

    let train_examples = train_examples
        .into_iter()
        .map(Arc::new)
        .collect::<Vec<_>>();
    let val_examples = val_examples
        .into_iter()
        .map(Arc::new)
        .collect::<Vec<_>>();

    let train_dataset = Zi2ziDataset::new(train_examples, config.batch_size, true);
    let valid_dataset = Zi2ziDataset::new(val_examples.clone(), config.batch_size, false);

    let train_batcher =
        Zi2ziBatcher::new(data_config.clone(), true, config.flip_labels, config.seed);
    let valid_batcher = Zi2ziBatcher::new(data_config.clone(), false, false, config.seed);

    let train_loader =
        DataLoaderBuilder::<B, Arc<Example>, Zi2ziTrainInput<B>>::new(train_batcher)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .set_device(device.clone())
            .build(train_dataset);
    let valid_loader = DataLoaderBuilder::<
        B::InnerBackend,
        Arc<Example>,
        Zi2ziTrainInput<B::InnerBackend>,
    >::new(valid_batcher)
    .batch_size(config.batch_size)
    .set_device(device.clone())
    .build(valid_dataset);

    let num_epochs = config.num_epochs;
    let strategy = Zi2ziTrainingStrategy::new(
        config,
        data_config,
        model_dir.clone(),
        state_path,
        val_examples,
        state,
        device,
    );
    let strategy = Arc::new(strategy);

    let mut training = SupervisedTraining::new(model_dir, train_loader, valid_loader)
        .with_training_strategy(TrainingStrategy::Custom(strategy.clone()))
        .with_checkpointing_strategy(AlwaysSaveCheckpoints)
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(num_epochs);

    let metrics = vec![
        Zi2ziLossMetric::new(Zi2ziLossKind::D, "d_loss"),
        Zi2ziLossMetric::new(Zi2ziLossKind::G, "g_loss"),
        Zi2ziLossMetric::new(Zi2ziLossKind::Category, "category_loss"),
        Zi2ziLossMetric::new(Zi2ziLossKind::Cheat, "cheat_loss"),
        Zi2ziLossMetric::new(Zi2ziLossKind::Const, "const_loss"),
        Zi2ziLossMetric::new(Zi2ziLossKind::L1, "l1_loss"),
        Zi2ziLossMetric::new(Zi2ziLossKind::Tv, "tv_loss"),
    ];

    for metric in metrics {
        training = training.metric_train_numeric(metric.clone());
        training = training.metric_valid_numeric(metric);
    }

    training.launch(learner);

    Ok(())
}

/// Resolve `data_dir` relative to the experiment directory if needed.
fn resolve_data_dir(experiment_dir: &Path, data_dir: &str) -> PathBuf {
    let candidate = PathBuf::from(data_dir);
    if candidate.is_relative() {
        experiment_dir.join(candidate)
    } else {
        candidate
    }
}

/// Run a validation step and save merged sample images.
fn validate_model<B: AutodiffBackend>(
    generator: &crate::model::generator::Generator<B>,
    val_examples: &[Arc<Example>],
    data_config: &DataConfig,
    config: &TrainingConfig,
    model_dir: &Path,
    rng: &mut StdRng,
    device: &B::Device,
    epoch: usize,
    step: usize,
) -> Result<()> {
    if val_examples.is_empty() {
        return Ok(());
    }

    let batch_refs = select_val_batch(val_examples, config.batch_size, rng);
    if batch_refs.is_empty() {
        return Ok(());
    }
    let batch_size = batch_refs.len();
    let batch = build_batch::<B>(&batch_refs, data_config, false, rng, device)?;

    let labels_vec = batch
        .labels
        .to_data()
        .to_vec::<i32>()
        .context("failed to read labels")?;
    let labels = Tensor::<B, 1, Int>::from_data(
        TensorData::new(labels_vec.clone(), [labels_vec.len()]),
        device,
    );

    let (real_a, real_b) = crate::model::split_real(
        batch.images.clone(),
        config.model.input_channels,
        config.model.output_channels,
    );
    let (fake_b, _) = generator.forward(real_a.clone(), labels);

    let input_imgs = tensor_to_images(real_a)?;
    let fake_imgs = tensor_to_images(fake_b)?;
    let real_imgs = tensor_to_images(real_b)?;

    let merged_input = merge_images(&input_imgs, batch_size, 1)?;
    let merged_fake = merge_images(&fake_imgs, batch_size, 1)?;
    let merged_real = merge_images(&real_imgs, batch_size, 1)?;
    let merged_pair = vec![merged_input, merged_real, merged_fake];

    let sample_dir = model_dir.join("samples");
    std::fs::create_dir_all(&sample_dir)?;
    let labels_str = labels_vec
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("_");
    let filename = format!("sample_{:02}_{:04}_{}.png", epoch, step, labels_str);
    save_concat_images(&merged_pair, &sample_dir.join(filename))?;
    Ok(())
}

/// Select a random validation batch with unique labels.
fn select_val_batch<'a>(
    examples: &'a [Arc<Example>],
    batch_size: usize,
    rng: &mut StdRng,
) -> Vec<&'a Example> {
    if examples.is_empty() || batch_size == 0 {
        return Vec::new();
    }

    let mut batch = Vec::with_capacity(batch_size);
    let mut seen = HashSet::new();
    let mut indices: Vec<usize> = (0..examples.len()).collect();
    indices.shuffle(rng);

    for idx in indices {
        let example = examples[idx].as_ref();
        if seen.insert(example.label) {
            batch.push(example);
            if batch.len() == batch_size {
                break;
            }
        }
    }

    batch
}
