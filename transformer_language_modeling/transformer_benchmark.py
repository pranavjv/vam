import torch
import matplotlib.pyplot as plt
import os
import time
import sys
import json
from datetime import datetime
import nltk

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarker import Benchmarker

# Make sure NLTK data is downloaded for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def plot_transformer_results(results, title, filename):
    """Plot and save comparison charts for transformer optimizer performance"""
    plt.figure(figsize=(16, 12))
    
    # Plot training and validation loss curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f"{name} - Train")
        if 'val_losses' in result and result['val_losses']:
            plt.plot(result['val_losses'], linestyle='--', label=f"{name} - Val")
    
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot perplexity curves
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if 'train_perplexities' in result and result['train_perplexities']:
            plt.plot(result['train_perplexities'], label=f"{name} - Train")
        if 'val_perplexities' in result and result['val_perplexities']:
            plt.plot(result['val_perplexities'], linestyle='--', label=f"{name} - Val")
    
    plt.title("Perplexity Curves (Lower is Better)")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Bar chart comparing final training metrics
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    
    # Get train metrics
    train_losses = [results[name]['final_train_loss'] for name in names]
    train_ppls = [results[name]['final_train_perplexity'] for name in names]
    
    x = range(len(names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], train_losses, width, label='Final Train Loss')
    
    # Normalize perplexity for better visualization
    max_ppl = max(train_ppls)
    norm_train_ppls = [ppl/max_ppl for ppl in train_ppls]
    plt.bar([i + width/2 for i in x], norm_train_ppls, width, label='Normalized Train PPL')
    
    # Add actual perplexity values as text
    for i, ppl in enumerate(train_ppls):
        plt.text(i + width/2, norm_train_ppls[i] + 0.05, f"{ppl:.2f}", 
                 ha='center', va='bottom', fontsize=10)
    
    plt.title("Training Performance")
    plt.xlabel("Optimizer")
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Bar chart comparing test metrics
    plt.subplot(2, 2, 4)
    
    # Get test metrics
    test_losses = [results[name]['test_loss'] for name in names]
    test_ppls = [results[name]['test_perplexity'] for name in names]
    
    plt.bar([i - width/2 for i in x], test_losses, width, label='Test Loss')
    
    # Normalize perplexity for better visualization
    max_ppl = max(test_ppls)
    norm_test_ppls = [ppl/max_ppl for ppl in test_ppls]
    plt.bar([i + width/2 for i in x], norm_test_ppls, width, label='Normalized Test PPL')
    
    # Add actual perplexity values as text
    for i, ppl in enumerate(test_ppls):
        plt.text(i + width/2, norm_test_ppls[i] + 0.05, f"{ppl:.2f}", 
                 ha='center', va='bottom', fontsize=10)
    
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
    """Run transformer benchmarks comparing ADAM and VRADAM optimizers"""
    results = {}
    
    # Run benchmark with Adam optimizer
    print("\n" + "="*50)
    print(f"Running Transformer on WikiText2 with ADAM optimizer")
    print("="*50 + "\n")
    
    adam_params = base_params.copy()
    adam_params['optimizer'] = 'ADAM'
    adam_params.update({
        'lr': base_params.get('lr', 0.001),
        'beta1': base_params.get('adam_beta1', 0.9),
        'beta2': base_params.get('adam_beta2', 0.999),
        'eps': base_params.get('adam_eps', 1e-8),
        'weight_decay': base_params.get('adam_weight_decay', 0)
    })
    benchmark_adam = Benchmarker(adam_params)
    adam_results = benchmark_adam.run()
    results['ADAM'] = adam_results
    
    # Run benchmark with VRADAM optimizer
    print("\n" + "="*50)
    print(f"Running Transformer on WikiText2 with VRADAM optimizer")
    print("="*50 + "\n")
    
    vradam_params = base_params.copy()
    vradam_params['optimizer'] = 'VRADAM'
    vradam_params.update({
        'eta': base_params.get('vradam_eta', base_params.get('lr', 0.001)),
        'beta1': base_params.get('vradam_beta1', 0.9),
        'beta2': base_params.get('vradam_beta2', 0.999),
        'beta3': base_params.get('vradam_beta3', 1.0),
        'eps': base_params.get('vradam_eps', 1e-8),
        'weight_decay': base_params.get('vradam_weight_decay', 0),
        'power': base_params.get('vradam_power', 2),
        'normgrad': base_params.get('vradam_normgrad', False),
        'lr_cutoff': base_params.get('vradam_lr_cutoff', 19)
    })
    benchmark_vradam = Benchmarker(vradam_params)
    vradam_results = benchmark_vradam.run()
    results['VRADAM'] = vradam_results
    
    # Save results to file
    output_dir = '../benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"WikiText2_Transformer_{timestamp}.json")
    
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
    plot_file = os.path.join(output_dir, f"WikiText2_Transformer_{timestamp}.png")
    plot_transformer_results(results, "Optimizer Comparison: Transformer on WikiText2", plot_file)
    
    # Print head-to-head summary
    print("\n" + "="*50)
    print(f"SUMMARY: Transformer on WikiText2")
    print("="*50)
    
    # Compare key metrics
    print(f"ADAM Test Loss: {results['ADAM']['test_loss']:.6f}")
    print(f"VRADAM Test Loss: {results['VRADAM']['test_loss']:.6f}")
    
    print(f"ADAM Test Perplexity: {results['ADAM']['test_perplexity']:.2f}")
    print(f"VRADAM Test Perplexity: {results['VRADAM']['test_perplexity']:.2f}")
    
    print(f"ADAM Training Time: {results['ADAM']['train_time']:.2f}s")
    print(f"VRADAM Training Time: {results['VRADAM']['train_time']:.2f}s")
    
    return results

def run_transformer_benchmark():
    """Run benchmark for transformer language modeling"""
    # Set up base parameters
    base_params = {
        'model': 'TransformerModel',
        'dataset': 'WikiText2',
        'dataset_size': 'small',
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        'batch_size': 32,
        'max_seq_len': 128,
        'embed_dim': 256,
        'hidden_dim': 512,
        
        # General optimization parameters
        'lr': 0.001,
        'epochs': 3,
        
        # Adam specific parameters
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8,
        'adam_weight_decay': 0,
        
        # VRADAM specific parameters
        'vradam_eta': 0.001,
        'vradam_beta1': 0.9,
        'vradam_beta2': 0.999,
        'vradam_beta3': 0.8,
        'vradam_eps': 1e-8,
        'vradam_weight_decay': 0,
        'vradam_power': 2,
        'vradam_normgrad': False,
        'vradam_lr_cutoff': 15
    }
    
    return compare_optimizers(base_params)

if __name__ == "__main__":
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
    
    print("\nStarting transformer language modeling benchmark...")
    start_time = time.time()
    
    results = run_transformer_benchmark()
    
    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.2f} seconds") 