import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarker import Benchmarker

def plot_rl_results(results, title, filename):
    """Plot and save comparison charts for RL optimizer performance"""
    plt.figure(figsize=(16, 12))
    
    # Plot training loss curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f"{name} - Train")
    
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot reward curves
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        plt.plot(result['mean_rewards'], label=f"{name}")
    
    plt.title("Mean Reward Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Bar chart comparing final training metrics
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    
    # Get train metrics
    train_losses = [results[name]['final_train_loss'] for name in names]
    train_rewards = [results[name]['final_train_acc'] for name in names]  # Mean reward is stored as "train_acc"
    
    x = range(len(names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], train_losses, width, label='Final Train Loss')
    plt.bar([i + width/2 for i in x], train_rewards, width, label='Final Mean Reward')
    
    plt.title("Training Performance")
    plt.xlabel("Optimizer")
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Bar chart comparing test metrics
    plt.subplot(2, 2, 4)
    
    # Get test metrics
    test_rewards = [results[name]['test_acc'] for name in names]  # Test mean reward is stored as "test_acc"
    
    plt.bar(x, test_rewards, width, label='Test Mean Reward')
    
    # Add training time
    for i, name in enumerate(names):
        plt.text(i, 0.05, f"Time: {results[name]['train_time']:.1f}s", 
                 ha='center', va='bottom', fontsize=10)
    
    plt.title("Test Performance")
    plt.xlabel("Optimizer")
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def compare_optimizers(base_params):
    """Run RL benchmarks comparing ADAM and VADAM optimizers"""
    results = {}
    
    # Run benchmark with Adam optimizer
    print("\n" + "="*50)
    print(f"Running RL on HalfCheetah with ADAM optimizer")
    print("="*50 + "\n")
    
    adam_params = base_params.copy()
    adam_params['optimizer'] = 'ADAM'
    adam_params.update({
        'lr': base_params.get('lr', 0.01),
        'beta1': base_params.get('adam_beta1', 0.9),
        'beta2': base_params.get('adam_beta2', 0.999),
        'eps': base_params.get('adam_eps', 1e-8),
        'weight_decay': base_params.get('adam_weight_decay', 0)
    })
    benchmark_adam = Benchmarker(adam_params)
    adam_results = benchmark_adam.run()
    results['ADAM'] = adam_results
    
    # Run benchmark with VADAM optimizer
    print("\n" + "="*50)
    print(f"Running RL on HalfCheetah with VADAM optimizer")
    print("="*50 + "\n")
    
    vadam_params = base_params.copy()
    vadam_params['optimizer'] = 'VADAM'
    vadam_params.update({
        'eta': base_params.get('vadam_eta', base_params.get('lr', 0.1)),
        'beta1': base_params.get('vadam_beta1', 0.9),
        'beta2': base_params.get('vadam_beta2', 0.999),
        'beta3': base_params.get('vadam_beta3', 0.8),
        'eps': base_params.get('vadam_eps', 1e-8),
        'weight_decay': base_params.get('vadam_weight_decay', 0),
        'power': base_params.get('vadam_power', 2),
        'normgrad': base_params.get('vadam_normgrad', False),
        'lr_cutoff': base_params.get('vadam_lr_cutoff', 19)
    })
    benchmark_vadam = Benchmarker(vadam_params)
    vadam_results = benchmark_vadam.run()
    results['VADAM'] = vadam_results
    
    # Save results to file
    output_dir = '../benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"HalfCheetah_RL_{timestamp}.json")
    
    # Clean up results for JSON serialization
    clean_results = {}
    for optimizer, result in results.items():
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                clean_result[key] = value
            elif isinstance(value, (torch.Tensor, getattr(torch, 'ndarray', type(None)))):
                clean_result[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                clean_result[key] = str(value)
        clean_results[optimizer] = clean_result
    
    with open(result_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Plot results
    plot_file = os.path.join(output_dir, f"HalfCheetah_RL_{timestamp}.png")
    plot_rl_results(results, "Optimizer Comparison: RL on HalfCheetah", plot_file)
    
    # Print head-to-head summary
    print("\n" + "="*50)
    print(f"SUMMARY: RL on HalfCheetah")
    print("="*50)
    
    # Compare key metrics
    print(f"ADAM Final Train Loss: {results['ADAM']['final_train_loss']:.6f}")
    print(f"VADAM Final Train Loss: {results['VADAM']['final_train_loss']:.6f}")
    
    print(f"ADAM Final Mean Reward: {results['ADAM']['final_train_acc']:.2f}")
    print(f"VADAM Final Mean Reward: {results['VADAM']['final_train_acc']:.2f}")
    
    print(f"ADAM Test Mean Reward: {results['ADAM']['test_acc']:.2f}")
    print(f"VADAM Test Mean Reward: {results['VADAM']['test_acc']:.2f}")
    
    print(f"ADAM Training Time: {results['ADAM']['train_time']:.2f}s")
    print(f"VADAM Training Time: {results['VADAM']['train_time']:.2f}s")
    
    return results

def run_rl_benchmark(epochs=30, hidden_dim=256, batch_size=64, seed=42):
    """Run benchmark for RL HalfCheetah task"""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set up base parameters
    base_params = {
        'model': 'RLPolicy',
        'dataset': 'HalfCheetah',
        'dataset_size': 'small',
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'embed_dim': 300,  # Not used for RL but required by API
        'max_seq_len': 256,  # Not used for RL but required by API
        'epochs': epochs,
        
        # RL specific parameters
        'gamma': 0.99,  # Discount factor
        'entropy_coef': 0.01,  # Entropy coefficient for exploration
        
        # General optimization parameters
        'lr': 0.01,
        
        # Adam specific parameters
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8,
        'adam_weight_decay': 0,
        
        # VADAM specific parameters
        'vadam_eta': 0.01,
        'vadam_beta1': 0.9,
        'vadam_beta2': 0.999,
        'vadam_beta3': 1.0,
        'vadam_eps': 1e-8,
        'vadam_weight_decay': 0,
        'vadam_power': 2,
        'vadam_normgrad': False,
        'vadam_lr_cutoff': 19
    }
    
    return compare_optimizers(base_params)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RL benchmark comparing VADAM and Adam')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for policy network')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Determine the best available device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    print("\nStarting RL benchmark...")
    start_time = time.time()
    
    results = run_rl_benchmark(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.2f} seconds") 