# VRADAM Benchmarking Framework

This repository contains a comprehensive benchmarking framework for comparing the ADAM and VRADAM optimizers across various deep learning tasks.

## Project Structure

The project is organized into separate modules for each benchmark task:

```
vam/
├── benchmarker.py              # Core benchmarking functionality
├── VRADAM.py                    # VRADAM optimizer implementation
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


## Hyperparameter Sweeps

The framework integrates with Weights & Biases for hyperparameter optimization. For each benchmark, we can sweep:

- For Adam: only the learning rate
- For VRADAM: learning rate (eta), beta3 and lr_cutoff parameters

To run a sweep:

```bash
python run_sweep_agent.py --model CNN --optimizer vradam  # Options for model: CNN, Transformer, RL, Diffusion
                                                         # Options for optimizer: adam, vradam, both
```

### Individual Sweep Commands

To run sweeps for specific models, use the following commands. These examples use the ADAM optimizer and a small number of runs (`--count 1`) for demonstration purposes.

**CNN Sweep:**
```bash
python run_sweep_agent_cnn.py --optimizer_name VRADAM --model DeeperCNN --dataset CIFAR10 --count 1
```

**Diffusion Model Sweep:**
```bash
python run_sweep_agent_diffusion.py --optimizer VRADAM --count 1
```

**GFlowNet Sweep:**
```bash
python run_sweep_agent_gflownet.py --optimizer_name VRADAM --count 1
```

**Transformer Sweep:**
```bash
python run_sweep_agent_transformer.py --optimizer_name VRADAM --count 1
```

## Results and Analysis

Benchmark results are saved in the `benchmark_results` directory. Each benchmark run produces:

- JSON files with detailed metrics
- PNG files with performance charts
- Summary text files comparing ADAM and VRADAM

## Requirements

The major dependencies for this project are:

- PyTorch
- NumPy
- Matplotlib
- Weights & Biases
- NLTK (for text processing)
- OpenAI Gym (for RL environments)

See `requirements.txt` for a complete list of dependencies.
