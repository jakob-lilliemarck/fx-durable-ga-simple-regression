use clap::Parser;
use serde::Serialize;
use tracing::Level;

mod dataset;
mod model;
mod training;

type Backend = burn::backend::Autodiff<burn::backend::ndarray::NdArray>;
const EPOCHS: usize = 50;

/// Train a simple regression model (single-command CLI)
#[derive(Debug, Parser)]
#[command(
    name = "example-simple-regression",
    version = "1.0",
    author = "Jakob",
    about = "Regression motherf*cker!"
)]
struct Cli {
    /// Hidden layer size
    #[arg(long, required = true)]
    hidden_size: usize,

    /// Number of hidden layers
    #[arg(long, required = true)]
    num_hidden_layers: usize,

    /// Activation function
    #[arg(long, value_enum, required = true)]
    activation_fn: model::ActivationFunction,

    /// Whether to use bias
    #[arg(long, default_value_t = false)]
    use_bias: bool,

    /// Learning rate
    #[arg(long, required = true)]
    learning_rate: f64,
}

#[derive(Serialize)]
struct ResultOutput {
    validation_loss: f32,
}

fn main() {
    // --- Initialize tracing
    let format = tracing_subscriber::fmt::format::Format::default()
        .with_level(true)
        .with_target(true)
        .with_thread_ids(false)
        .with_ansi(true)
        .pretty();

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr) // send all logs to stderr
        .event_format(format)
        .with_max_level(Level::INFO)
        .init();

    // --- Parse CLI arguments
    let args = Cli::parse();

    tracing::info!(
        message = "Cli::TrainRegressionModel",
        hidden_size = args.hidden_size,
        num_hidden_layers = args.num_hidden_layers,
        ?args.activation_fn,
        use_bias = args.use_bias,
        learning_rate = args.learning_rate
    );

    // --- Setup and train
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();

    let (dataloader_train, dataloader_valid) =
        training::create_dataloaders::<Backend>(device, 128, 1337);

    let model_config = model::RegressionModelConfig {
        hidden_size: args.hidden_size,
        num_hidden_layers: args.num_hidden_layers,
        activation_fn: args.activation_fn,
        use_bias: args.use_bias,
        learning_rate: args.learning_rate,
    };

    let validation_loss = training::train_silent(
        model_config,
        device,
        EPOCHS,
        &dataloader_train,
        &dataloader_valid,
    );

    // --- Print result JSON for GA to capture
    println!(
        "{}",
        serde_json::to_string(&ResultOutput { validation_loss }).unwrap()
    );
}
