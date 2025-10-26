use crate::dataset::{HousingBatch, NUM_FEATURES};
use burn::{
    nn::{
        Gelu, Linear, LinearConfig, Relu, Sigmoid,
        loss::{MseLoss, Reduction::Mean},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(
    Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, clap::ValueEnum,
)]
pub enum ActivationFunction {
    Relu,
    Gelu,
    Sigmoid,
}

#[derive(Module, Debug, Clone)]
pub enum ActivationFn {
    Relu(Relu),
    Gelu(Gelu),
    Sigmoid(Sigmoid),
}

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    hidden_layers: Vec<Linear<B>>,
    output_layer: Linear<B>,
    activation_fn: ActivationFn,
}

#[derive(Config, Debug)]
pub struct RegressionModelConfig {
    #[config(default = 64)]
    pub hidden_size: usize,
    #[config(default = 2)]
    pub num_hidden_layers: usize,
    #[config(default = "ActivationFunction::Relu")]
    pub activation_fn: ActivationFunction,
    #[config(default = true)]
    pub use_bias: bool,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let mut hidden_layers = Vec::new();

        // First hidden layer: input -> hidden
        let first_layer = LinearConfig::new(NUM_FEATURES, self.hidden_size)
            .with_bias(self.use_bias)
            .init(device);
        hidden_layers.push(first_layer);

        // Additional hidden layers: hidden -> hidden
        for _ in 1..self.num_hidden_layers {
            let layer = LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(self.use_bias)
                .init(device);
            hidden_layers.push(layer);
        }

        // Output layer: hidden -> output
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(self.use_bias)
            .init(device);

        let activation_fn = match self.activation_fn {
            ActivationFunction::Gelu => ActivationFn::Gelu(Gelu::new()),
            ActivationFunction::Relu => ActivationFn::Relu(Relu::new()),
            ActivationFunction::Sigmoid => ActivationFn::Sigmoid(Sigmoid::new()),
        };

        RegressionModel {
            hidden_layers,
            output_layer,
            activation_fn: activation_fn,
        }
    }
}

impl<B: Backend> RegressionModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // Forward through all hidden layers with specified activation
        for layer in &self.hidden_layers {
            x = layer.forward(x);
            x = match self.activation_fn {
                ActivationFn::Relu(ref relu) => relu.forward(x),
                ActivationFn::Gelu(ref gelu) => gelu.forward(x),
                ActivationFn::Sigmoid(ref sigmoid) => sigmoid.forward(x),
            };
        }

        // Final output layer (no activation)
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: HousingBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze_dim(1);
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<HousingBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: HousingBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<HousingBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: HousingBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
