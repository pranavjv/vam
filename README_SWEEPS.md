# VADAM Hyperparameter Optimization with Weights & Biases

This project implements a hyperparameter optimization system for the VADAM (Velocity-Adaptive Momentum) optimizer using Weights & Biases (W&B) sweeps. The system allows you to find the optimal hyperparameters for VADAM across different model architectures and datasets.

## Prerequisites

1. Install the required packages:
   ```bash
   pip install torch torchvision torchtext==0.6.0 wandb matplotlib numpy
   ```

2. Create a Weights & Biases account and sign in:
   ```bash
   wandb login
   ```

## Directory Structure

- `sweep_vadam.py`: Main script to set up and run W&B sweeps
- `run_sweep_agent.py`: Script to run individual sweep agents
- `analyze_sweeps.py`: Script to analyze sweep results and extract best hyperparameters
- `run_optimized_benchmark.py`: Script to run benchmarks with optimized hyperparameters
- `sweep_results/`: Directory where sweep configurations and results are stored
- `optimized_results/`: Directory where optimized benchmark results are stored

## Running Hyperparameter Optimization

### Option 1: Full Sweep across All Models and Datasets

To run a complete sweep across all model and dataset combinations:

```bash
python sweep_vadam.py
```

This will:
1. Create W&B sweeps for each model/dataset combination
2. Run 10 experiments for each sweep using Bayesian optimization
3. Save results to `sweep_results/` directory

### Option 2: Individual Sweep for a Specific Model and Dataset

To create and run a sweep for a specific model and dataset:

```bash
python run_sweep_agent.py --model SimpleCNN --dataset CIFAR10 --count 20
```

Arguments:
- `--model`: Model architecture (SimpleCNN, MLPModel, TransformerModel)
- `--dataset`: Dataset (CIFAR10, IMDB, WikiText2)
- `--count`: Number of experiments to run (default: 10)

### Option 3: Continue an Existing Sweep

To continue an existing sweep using a sweep ID:

```bash
python run_sweep_agent.py --sweep_id <SWEEP_ID> --count 10
```

## Analyzing Sweep Results

After running sweeps, analyze the results to find the best hyperparameters:

```bash
python analyze_sweeps.py
```

This will:
1. Gather all sweep results
2. Identify the best hyperparameters for each model/dataset
3. Generate a markdown report in `sweep_results/optimization_report.md`
4. Save best configurations to `sweep_results/best_configs.json`

You can also generate commands to run benchmarks with the optimized parameters:

```bash
python analyze_sweeps.py --generate_commands
```

## Running Benchmarks with Optimized Parameters

After finding the optimal hyperparameters, run benchmarks to compare optimized VADAM against Adam:

```bash
python run_optimized_benchmark.py --model SimpleCNN --dataset CIFAR10 --load_params --epochs 10
```

Arguments:
- `--model`: Model architecture (SimpleCNN, MLPModel, TransformerModel)
- `--dataset`: Dataset (CIFAR10, IMDB, WikiText2)
- `--load_params`: Load the best parameters from previous sweeps
- `--epochs`: Number of epochs to train (default: 10)
- `--use_wandb`: Track results with W&B
- `--dataset_size`: Size of dataset ('small' or 'full')

You can also specify VADAM parameters manually:

```bash
python run_optimized_benchmark.py --model SimpleCNN --dataset CIFAR10 \
    --eta 0.003 --beta1 0.9 --beta2 0.999 --beta3 1.2 \
    --power 2 --normgrad --lr_cutoff 15 --weight_decay 0.0001 --eps 1e-8
```

## Hyperparameters to Optimize

VADAM has several specific hyperparameters that can be optimized:

- `eta`: Maximum learning rate (similar to lr in Adam)
- `beta1`: Exponential decay rate for the first moment estimate
- `beta2`: Exponential decay rate for the second moment estimate
- `beta3`: Controls adaptive learning rate scaling factor
- `power`: Exponent used in gradient norm calculation
- `normgrad`: Whether to normalize gradients
- `lr_cutoff`: Learning rate cutoff threshold
- `weight_decay`: Weight decay factor
- `eps`: Small constant for numerical stability

## Visualizing Results

The system automatically generates comparison plots between optimized VADAM and Adam:

- Loss curves
- Accuracy or perplexity curves
- Test performance comparison

These visualizations are saved to the `optimized_results/` directory.

## Example Workflow

1. Run hyperparameter optimization:
   ```bash
   python sweep_vadam.py
   ```

2. Analyze results:
   ```bash
   python analyze_sweeps.py --generate_commands
   ```

3. Run benchmarks with optimized parameters:
   ```bash
   python run_optimized_benchmark.py --model SimpleCNN --dataset CIFAR10 --load_params --epochs 10 --use_wandb
   ```

4. Repeat for other model/dataset combinations:
   ```bash
   python run_optimized_benchmark.py --model MLPModel --dataset IMDB --load_params --epochs 10 --use_wandb
   python run_optimized_benchmark.py --model TransformerModel --dataset WikiText2 --load_params --epochs 10 --use_wandb
   ```

5. Check the `optimized_results/` directory for detailed comparisons and visualizations.

## Tips for Effective Hyperparameter Optimization

1. Start with small datasets and fewer epochs for faster iterations
2. Use the Bayesian optimization method for efficient search
3. Run multiple sweep agents in parallel for faster results
4. Increase the number of experiments for more thorough optimization 