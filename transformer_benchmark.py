import torch
import matplotlib.pyplot as plt
import os
import time
import math
import json
from benchmarker import Benchmarker
from datetime import datetime

def plot_transformer_results(results):
    """Plot the transformer benchmark results with a focus on loss and perplexity"""
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training/validation loss curves
    ax = axs[0, 0]
    for name, result in results.items():
        ax.plot(result['train_losses'], label=f"{name} - Train")
        if result['val_losses']:
            ax.plot(result['val_losses'], linestyle='--', label=f"{name} - Val")
    
    ax.set_title("Loss Curves", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Perplexity curves
    ax = axs[0, 1]
    for name, result in results.items():
        if 'train_perplexities' in result and result['train_perplexities']:
            ax.plot(result['train_perplexities'], label=f"{name} - Train")
        if 'val_perplexities' in result and result['val_perplexities']:
            ax.plot(result['val_perplexities'], linestyle='--', label=f"{name} - Val")
    
    ax.set_title("Perplexity Curves (Lower is Better)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Test metrics comparison
    ax = axs[1, 0]
    names = list(results.keys())
    x = range(len(names))
    width = 0.35
    
    # Get test metrics
    test_losses = [results[name]['test_loss'] for name in names]
    ax.bar([i - width/2 for i in x], test_losses, width, label='Test Loss')
    
    if any('test_perplexity' in result for result in results.values()):
        test_ppls = [results[name].get('test_perplexity', 0) for name in names]
        # Normalize perplexity for better visualization alongside loss
        max_ppl = max(test_ppls)
        norm_ppls = [ppl/max_ppl for ppl in test_ppls]
        ax.bar([i + width/2 for i in x], norm_ppls, width, label='Normalized Test PPL')
        
        # Add perplexity values as text labels
        for i, ppl in enumerate(test_ppls):
            ax.text(i + width/2, norm_ppls[i] + 0.05, f"PPL: {ppl:.2f}", 
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_title("Test Performance", fontsize=14)
    ax.set_xlabel("Optimizer")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Training metrics comparison
    ax = axs[1, 1]
    
    # Get training metrics
    train_times = [results[name]['train_time'] for name in names]
    final_train_losses = [results[name]['final_train_loss'] for name in names]
    
    # Side-by-side bars for time and final loss
    width = 0.35
    ax.bar([i - width/2 for i in x], final_train_losses, width, label='Final Train Loss')
    
    # Use a secondary y-axis for time to better show the comparison
    ax2 = ax.twinx()
    ax2.bar([i + width/2 for i in x], train_times, width, color='orange', label='Training Time (s)')
    
    # Add training time values as text labels
    for i, time_val in enumerate(train_times):
        ax2.text(i + width/2, time_val + 0.5, f"{time_val:.1f}s", 
                 ha='center', va='bottom', fontsize=10)
    
    ax.set_title("Training Metrics", fontsize=14)
    ax.set_xlabel("Optimizer")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Time (seconds)")
    
    # Add both legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.suptitle("Transformer Language Model - Optimizer Comparison", fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    plt.savefig('transformer_benchmark_results.png', dpi=300)
    print("Plot saved to transformer_benchmark_results.png")
    
    # Show the plot if in an interactive environment
    plt.show()

def run_transformer_benchmark():
    """Run benchmark comparing ADAM and VADAM optimizers on a Transformer model"""
    
    # Define parameters for the benchmark
    params = {
        'model': 'TransformerModel',
        'device': 'mps',  # Will automatically fall back to CPU if MPS is not available
        'dataset': 'WikiText2',
        'dataset_size': 'large',  # Use 'small' for faster demonstration
        'batch_size': 32,
        'max_seq_len': 128,
        'embed_dim': 256,
        'hidden_dim': 512,
        
        # General optimization parameters
        'lr': 0.001,  # Used for Adam
        'epochs': 3,  # Use a small number for demonstration
        
        # Common hyperparameters
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0,
        
        # VADAM specific hyperparameters
        'eta': 0.001,  # Max learning rate for VADAM
        'beta3': 0.8,  # 
        'power': 2,
        'normgrad': False,
        'lr_cutoff': 15  
    }
    
    # Results dictionary to store benchmark results
    results = {}
    
    # Run benchmark with Adam optimizer
    print("\n" + "="*50)
    print("Running benchmark with Adam optimizer")
    print("="*50 + "\n")
    
    adam_params = params.copy()
    adam_params['optimizer'] = 'ADAM'
    benchmark_adam = Benchmarker(adam_params)
    adam_results = benchmark_adam.run()
    results['ADAM'] = adam_results
    
    # Run benchmark with VADAM optimizer
    print("\n" + "="*50)
    print("Running benchmark with VADAM optimizer")
    print("="*50 + "\n")
    
    vadam_params = params.copy()
    vadam_params['optimizer'] = 'VADAM'
    benchmark_vadam = Benchmarker(vadam_params)
    vadam_results = benchmark_vadam.run()
    results['VADAM'] = vadam_results
    
    # Save results to JSON
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"transformer_benchmark_{timestamp}.json")
    
    # Clean the results for JSON serialization
    clean_results = {}
    for optimizer, result in results.items():
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                clean_result[key] = value
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                clean_result[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                clean_result[key] = str(value)
        clean_results[optimizer] = clean_result
    
    with open(result_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Results saved to {result_file}")
    
    # Plot the results
    plot_transformer_results(results)
    
    # Write detailed summary
    summary_file = os.path.join(output_dir, f"transformer_benchmark_{timestamp}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("TRANSFORMER BENCHMARK: ADAM vs VADAM\n")
        f.write("="*50 + "\n\n")
        
        for name, result in results.items():
            f.write(f"OPTIMIZER: {name}\n")
            f.write("-"*30 + "\n")
            
            # Write training metrics
            f.write(f"Training Time: {result['train_time']:.2f} seconds\n")
            f.write(f"Final Training Loss: {result['final_train_loss']:.6f}\n")
            
            if 'final_train_perplexity' in result and result['final_train_perplexity'] is not None:
                f.write(f"Final Training Perplexity: {result['final_train_perplexity']:.2f}\n")
            
            # Write test metrics
            f.write(f"Test Loss: {result['test_loss']:.6f}\n")
            if 'test_perplexity' in result and result['test_perplexity'] is not None:
                f.write(f"Test Perplexity: {result['test_perplexity']:.2f}\n")
            
            # Calculate generalization gap
            if 'final_train_perplexity' in result and 'test_perplexity' in result:
                ppl_gap = result['test_perplexity'] - result['final_train_perplexity']
                gap_percent = (ppl_gap / result['final_train_perplexity']) * 100
                f.write(f"Generalization Gap (PPL): {ppl_gap:.2f} ({gap_percent:.1f}%)\n")
            
            f.write("\n")
        
        # Add head-to-head comparison
        f.write("HEAD-TO-HEAD COMPARISON\n")
        f.write("-"*30 + "\n")
        
        # Calculate performance differences
        adam = results['ADAM']
        vadam = results['VADAM']
        
        # Time efficiency
        time_diff = vadam['train_time'] - adam['train_time']
        time_percent = (time_diff / adam['train_time']) * 100
        f.write(f"Training Time: VADAM is {time_percent:.1f}% {'slower' if time_percent > 0 else 'faster'} ({time_diff:.2f}s)\n")
        
        # Loss improvement
        loss_diff = vadam['test_loss'] - adam['test_loss']
        loss_percent = (loss_diff / adam['test_loss']) * 100
        f.write(f"Test Loss: VADAM is {abs(loss_percent):.1f}% {'worse' if loss_percent > 0 else 'better'} ({loss_diff:.6f})\n")
        
        # Perplexity improvement
        if 'test_perplexity' in adam and 'test_perplexity' in vadam:
            ppl_diff = vadam['test_perplexity'] - adam['test_perplexity']
            ppl_percent = (ppl_diff / adam['test_perplexity']) * 100
            f.write(f"Test Perplexity: VADAM is {abs(ppl_percent):.1f}% {'worse' if ppl_percent > 0 else 'better'} ({ppl_diff:.2f})\n")
        
        # Overall assessment
        f.write("\nOVERALL ASSESSMENT:\n")
        if 'test_perplexity' in adam and 'test_perplexity' in vadam:
            if vadam['test_perplexity'] < adam['test_perplexity']:
                if vadam['train_time'] <= adam['train_time']:
                    f.write("VADAM outperforms ADAM in both perplexity and training efficiency.\n")
                else:
                    f.write(f"VADAM achieves better perplexity than ADAM ({abs(ppl_percent):.1f}% lower) but at a cost of {time_percent:.1f}% longer training time.\n")
            else:
                if vadam['train_time'] >= adam['train_time']:
                    f.write("ADAM outperforms VADAM in both perplexity and training efficiency.\n")
                else:
                    f.write(f"ADAM achieves better perplexity than VADAM ({abs(ppl_percent):.1f}% lower) but VADAM is {abs(time_percent):.1f}% faster in training.\n")
    
    print(f"Detailed summary saved to {summary_file}")
    
    # Print terminal summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"ADAM Test Loss: {adam['test_loss']:.6f}")
    print(f"VADAM Test Loss: {vadam['test_loss']:.6f}")
    
    if 'test_perplexity' in adam and 'test_perplexity' in vadam:
        print(f"ADAM Test Perplexity: {adam['test_perplexity']:.2f}")
        print(f"VADAM Test Perplexity: {vadam['test_perplexity']:.2f}")
        
        ppl_diff = vadam['test_perplexity'] - adam['test_perplexity']
        ppl_percent = (ppl_diff / adam['test_perplexity']) * 100
        print(f"Perplexity Difference: {ppl_diff:.2f} ({ppl_percent:.1f}%)")
    
    print(f"ADAM Training Time: {adam['train_time']:.2f}s")
    print(f"VADAM Training Time: {vadam['train_time']:.2f}s")
    
    time_diff = vadam['train_time'] - adam['train_time']
    time_percent = (time_diff / adam['train_time']) * 100
    print(f"Training Time Difference: {time_diff:.2f}s ({time_percent:.1f}%)")
    
    return results

if __name__ == "__main__":
    # Ensure we have a device that can run the model
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    print("\nStarting Transformer Benchmark with ADAM vs VADAM...")
    start_time = time.time()
    
    try:
        results = run_transformer_benchmark()
        
        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.2f} seconds")
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise 