# VAM Optimizer Benchmarking

This repository contains code to benchmark the VAM (Variational Adam) optimizer against standard Adam optimizer across different model architectures and datasets.

## Models Implemented

- **SimpleCNN**: A basic CNN model for image classification
- **TransformerModel**: A transformer model for language modeling/sentence completion
- **MLPModel**: A simple MLP model for text classification
- **RLPolicy**: A policy network for reinforcement learning tasks

## Datasets

- **CIFAR10**: Image classification dataset
- **WikiText2**: Text dataset for language modeling (sentence completion)
- **IMDB**: Text classification dataset for sentiment analysis
- **HalfCheetah**: Reinforcement learning environment for continuous control

## Requirements

```
torch
torchvision
torchtext
nltk
matplotlib
numpy
wandb
gymnasium
scikit-learn
tqdm
datasets
pillow
```

## Usage

To run the benchmark comparison between VAM and Adam optimizers:

```bash
python run_benchmarks.py
```

This will:
1. Run benchmarks for all model/dataset combinations
2. Compare performance between VAM and Adam optimizers
3. Save results and plots in the `benchmark_results` directory

### Reinforcement Learning Benchmark

To run the RL benchmark comparing VADAM and Adam:

```bash
python rl_benchmark.py
```

This will:
1. Train policy networks on the HalfCheetah environment using both VADAM and Adam
2. Compare performance metrics including mean rewards and training time
3. Generate plots showing the training progress

To test just the HalfCheetah environment with a specific optimizer:

```bash
python test_halfcheetah.py --optimizer VADAM  # or ADAM
```

### Visualizing HalfCheetah Performance

To visualize the performance of trained policies in the HalfCheetah environment:

```bash
python render_halfcheetah.py
```

This will:
1. Load or train models for both VADAM and Adam optimizers
2. Record animations of the HalfCheetah agent's performance
3. Save individual GIFs and a side-by-side comparison to the `animations` directory

Additional options:
```bash
# Visualize a specific optimizer
python render_halfcheetah.py --optimizer VADAM  # or ADAM

# Force training new models instead of loading existing ones
python render_halfcheetah.py --train

# Specify device to use
python render_halfcheetah.py --device cpu  # or cuda, mps

# Change number of episodes to record
python render_halfcheetah.py --episodes 5
```

## Customization

You can modify the benchmark parameters in `run_benchmarks.py`:

```python
base_params = {
    'device': 'mps',  # Use 'cuda' if available, otherwise 'cpu'
    'dataset_size': 'small',  # Use 'small' for faster runs or 'full' for complete dataset
    'batch_size': 32,
    'max_seq_len': 128,
    'embed_dim': 256,
    'hidden_dim': 512,
    'lr': 0.001,
    'epochs': 3  # Increase for better results
}
```

For RL benchmarks, you can customize parameters in `rl_benchmark.py`:

```python
run_rl_benchmark(
    model='RLPolicy',
    dataset='HalfCheetah',
    hidden_dim=256,
    epochs=30,
    batch_size=64,
    seed=42
)
```

## Single Model Benchmark

To benchmark a specific model and dataset:

```python
from benchmarker import Benchmarker

# Define parameters
params = {
    'model': 'TransformerModel',
    'device': 'mps',
    'dataset': 'WikiText2',
    'dataset_size': 'small',
    'optimizer': 'VADAM',
    'batch_size': 32,
    'max_seq_len': 128,
    'embed_dim': 256,
    'hidden_dim': 512,
    'lr': 0.001,
    'epochs': 5
}

# Run benchmark
benchmark = Benchmarker(params)
results = benchmark.run()
print(results)
```

## Results

The benchmark will generate:
- JSON files with detailed metrics
- Plot images comparing loss curves and performance metrics
- Console output with training progress

For reinforcement learning, the plots will show:
- Mean reward per epoch for both optimizers
- Training loss curves
- Animations of agent performance (when using `render_halfcheetah.py`)

## About the Optimizers

- **Adam**: Standard Adam optimizer from PyTorch
- **VADAM (Variational Adam)**: Implementation of the Variational Adam optimizer which combines variational inference techniques with Adam optimization
