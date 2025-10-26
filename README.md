This crate provides a minimal CLI to run a single training run over a sythetic dataset minimicing Californa Housing.
It's used as an example of `fx-durable-ga` demonstrating how to setup a training loop with multiple subsequent training runs while managing memory usage.

Currently the Burn ML framework, which is used for this example, does not properly free allocated memory between training runs. Running the training as a standalone program invoked through `tokio::process:Command` provides a simple to ensure memory is properly freed after each run is completed.

If you wish to run the example from `fx-durable-ga` you must first clone this repository and then call:
```sh
cargo install --path .
```

Invoking a training run looks like this:
```sh
fx-example-regression --hidden-size 16 --num-hidden-layers 64 --activation-fn relu --learning-rate 0.001
```

Training a model does not create a model file. Since this repository is only used to search and tune hyperparameters it outputs loss only - which is the value the GA-optimization example attempts to minimize.
