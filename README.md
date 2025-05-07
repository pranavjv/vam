# VAM Optimizer Benchmarking

This repository contains code to benchmark the VAM (Variational Adam) optimizer against standard Adam optimizer across different model architectures and datasets.

## Models Implemented

- **SimpleCNN**: A basic CNN model for image classification
- **TransformerModel**: A transformer model for language modeling/sentence completion
- **MLPModel**: A simple MLP model for text classification

## Datasets

- **CIFAR10**: Image classification dataset
- **WikiText2**: Text dataset for language modeling (sentence completion)
- **IMDB**: Text classification dataset for sentiment analysis

## Requirements

```
torch
torchvision
torchtext
nltk
matplotlib
numpy
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

## About the Optimizers

- **Adam**: Standard Adam optimizer from PyTorch
- **VAM (Variational Adam)**: Implementation of the Variational Adam optimizer which combines variational inference techniques with Adam optimization
