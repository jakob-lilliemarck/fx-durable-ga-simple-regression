use clap::Parser;
use serde::Serialize;
use tracing::Level;

mod dataset;
mod model;
mod training;

type Backend = burn::backend::Autodiff<burn::backend::ndarray::NdArray>;

const EPOCHS: usize = 5;

#[derive(Debug, Parser)]
#[clap(
    name = "example-simple-regression",
    version = "1.0",
    author = "Jakob",
    about = "Regression motherf*cker!"
)]
enum Cli {
    Regression {
        #[clap(long, required = true)]
        hidden_size: usize,
        #[clap(long, required = true)]
        num_hidden_layers: usize,
        #[clap(long, value_enum, required = true)]
        activation_fn: model::ActivationFunction,
        #[clap(long, default_value_t = false)]
        use_bias: bool,
        #[clap(long, required = true)]
        learning_rate: f64,
    },
}

#[derive(Serialize)]
struct ResultOutput {
    validation_loss: f32,
}

fn main() {
    let format = tracing_subscriber::fmt::format::Format::default()
        .with_level(true)
        .with_target(true)
        .with_thread_ids(false)
        .with_ansi(true)
        .pretty();

    tracing_subscriber::fmt()
        .event_format(format)
        .with_max_level(Level::INFO)
        .init();

    match Cli::parse() {
        Cli::Regression {
            hidden_size,
            num_hidden_layers,
            activation_fn,
            use_bias,
            learning_rate,
        } => {
            tracing::info!(
                message = "Cli::TrainRegressionModel",
                hidden_size = hidden_size,
                num_hidden_layers = num_hidden_layers,
                ?activation_fn,
                use_bias = use_bias,
                learning_rate = learning_rate
            );

            let device: <Backend as burn::prelude::Backend>::Device = Default::default();

            let (dataloader_train, dataloader_valid) =
                training::create_dataloaders::<Backend>(device, 128, 1337);

            let model_config = model::RegressionModelConfig {
                hidden_size,
                num_hidden_layers,
                activation_fn,
                use_bias,
                learning_rate,
            };

            let validation_loss = training::train_silent(
                model_config,
                device,
                EPOCHS,
                &dataloader_train,
                &dataloader_valid,
            );

            println!(
                "{}",
                serde_json::to_string(&ResultOutput { validation_loss }).unwrap()
            );
        }
    }
}
