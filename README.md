# VADAM Benchmarking Framework

This repository contains a comprehensive benchmarking framework for comparing the ADAM and VADAM optimizers across various deep learning tasks.

## Project Structure

The project is organized into separate modules for each benchmark task:

```
vam/
├── benchmarker.py              # Core benchmarking functionality
├── VADAM.py                    # VADAM optimizer implementation
├── architectures.py            # Common model architectures
├── run_benchmarks.py           # Main script to run all benchmarks
├── run_sweep_agent.py          # Script to run hyperparameter sweeps
│
├── cnn_image_classification/   # CNN on CIFAR10
│   ├── cnn_benchmark.py        # CNN benchmark script
│   └── sweep_cnn.py            # W&B sweep configuration for CNN
│
├── transformer_language_modeling/  # Transformer on WikiText2
│   ├── transformer_benchmark.py    # Transformer benchmark script
│   └── sweep_transformer.py        # W&B sweep configuration for Transformer
│
├── rl_halfcheetah/             # RL on HalfCheetah environment
│   ├── rl_benchmark.py         # RL benchmark script
│   └── sweep_rl.py             # W&B sweep configuration for RL
│
├── diffusion_mnist/            # Diffusion model on MNIST
│   ├── diffusion_benchmark.py  # Diffusion benchmark script
│   └── sweep_diffusion.py      # W&B sweep configuration for Diffusion
│
├── data/                       # Dataset storage
├── benchmark_results/          # Output files from benchmarks
├── diffusion_samples/          # Generated samples from diffusion models
└── sweep_results/              # W&B sweep configurations and results
```

## Benchmarks

The framework includes the following benchmarks:

1. **CNN for Image Classification**: Trains a simple CNN on the CIFAR10 dataset for image classification.
2. **Transformer for Language Modeling**: Trains a transformer model on the WikiText2 dataset for language modeling.
3. **RL for Control**: Trains a policy network on the HalfCheetah environment for reinforcement learning.
4. **Diffusion Model**: Trains a diffusion model on the MNIST dataset for image generation.

## Running Benchmarks

To run all benchmarks:

```bash
python run_benchmarks.py
```

To run a specific benchmark:

```bash
python run_benchmarks.py --model CNN  # Options: CNN, Transformer, RL, Diffusion
```

## Hyperparameter Sweeps

The framework integrates with Weights & Biases for hyperparameter optimization. For each benchmark, we can sweep:

- For Adam: only the learning rate
- For VADAM: learning rate (eta), beta3 and lr_cutoff parameters

To run a sweep:

```bash
python run_sweep_agent.py --model CNN --optimizer vadam  # Options for model: CNN, Transformer, RL, Diffusion
                                                         # Options for optimizer: adam, vadam, both
```

## Results and Analysis

Benchmark results are saved in the `benchmark_results` directory. Each benchmark run produces:

- JSON files with detailed metrics
- PNG files with performance charts
- Summary text files comparing ADAM and VADAM

## Requirements

The major dependencies for this project are:

- PyTorch
- NumPy
- Matplotlib
- Weights & Biases
- NLTK (for text processing)
- OpenAI Gym (for RL environments)

See `requirements.txt` for a complete list of dependencies.
