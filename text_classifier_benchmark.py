import torch
import matplotlib.pyplot as plt
import os
import time
import json
import numpy as np
from benchmarker import Benchmarker
from datetime import datetime

def plot_classifier_results(results):
    """Plot the text classifier benchmark results with focus on loss and accuracy"""
    
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
    
    # Accuracy curves
    ax = axs[0, 1]
    for name, result in results.items():
        if 'train_accs' in result and result['train_accs']:
            ax.plot(result['train_accs'], label=f"{name} - Train")
        if 'val_accs' in result and result['val_accs']:
            ax.plot(result['val_accs'], linestyle='--', label=f"{name} - Val")
    
    ax.set_title("Accuracy Curves", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
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
    
    if any('test_acc' in result for result in results.values()):
        test_accs = [results[name].get('test_acc', 0) for name in names]
        ax.bar([i + width/2 for i in x], test_accs, width, label='Test Accuracy')
        
        # Add accuracy values as text labels
        for i, acc in enumerate(test_accs):
            ax.text(i + width/2, acc + 0.02, f"{acc:.3f}", 
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
    final_train_accs = [results[name]['final_train_acc'] for name in names]
    
    # Side-by-side bars for accuracy and time
    ax.bar([i - width/2 for i in x], final_train_accs, width, label='Final Train Accuracy')
    
    # Use a secondary y-axis for time to better show the comparison
    ax2 = ax.twinx()
    ax2.bar([i + width/2 for i in x], train_times, width, color='orange', label='Training Time (s)')
    
    # Add training time and accuracy values as text labels
    for i, acc in enumerate(final_train_accs):
        ax.text(i - width/2, acc + 0.02, f"{acc:.3f}", 
                ha='center', va='bottom', fontsize=10)
        
    for i, time_val in enumerate(train_times):
        ax2.text(i + width/2, time_val + 0.5, f"{time_val:.1f}s", 
                 ha='center', va='bottom', fontsize=10)
    
    ax.set_title("Training Metrics", fontsize=14)
    ax.set_xlabel("Optimizer")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Time (seconds)")
    
    # Add both legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.suptitle("Text Classification - Optimizer Comparison", fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    plt.savefig('text_classifier_benchmark_results.png', dpi=300)
    print("Plot saved to text_classifier_benchmark_results.png")
    
    # Show the plot if in an interactive environment
    plt.show()

def run_text_classifier_benchmark():
    """Run benchmark comparing ADAM and VADAM optimizers on the MLP model for text classification"""
    
    # Define parameters for the benchmark
    params = {
        'model': 'MLPModel',
        'device': 'mps',  # Will automatically fall back to CPU if MPS is not available
        'dataset': 'IMDB',
        'dataset_size': 'small',  # Use 'small' for faster demonstration
        'batch_size': 64,
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
        'weight_decay': 1e-5,  # Small weight decay to prevent overfitting
        
        # VADAM specific hyperparameters
        'eta': 0.001,  # Max learning rate for VADAM
        'beta3': 1.2,  # Slightly higher than default (1.0) for text classification
        'power': 2,
        'normgrad': True,
        'lr_cutoff': 10  # Lower cutoff for more aggressive learning rate adaptation
    }
    
    # Results dictionary to store benchmark results
    results = {}
    
    # Run benchmark with Adam optimizer
    print("\n" + "="*50)
    print("Running text classification benchmark with Adam optimizer")
    print("="*50 + "\n")
    
    adam_params = params.copy()
    adam_params['optimizer'] = 'ADAM'
    benchmark_adam = Benchmarker(adam_params)
    adam_results = benchmark_adam.run()
    results['ADAM'] = adam_results
    
    # Run benchmark with VADAM optimizer
    print("\n" + "="*50)
    print("Running text classification benchmark with VADAM optimizer")
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
    result_file = os.path.join(output_dir, f"text_classifier_benchmark_{timestamp}.json")
    
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
    plot_classifier_results(results)
    
    # Write detailed summary
    summary_file = os.path.join(output_dir, f"text_classifier_benchmark_{timestamp}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("TEXT CLASSIFICATION BENCHMARK: ADAM vs VADAM\n")
        f.write("="*50 + "\n\n")
        
        for name, result in results.items():
            f.write(f"OPTIMIZER: {name}\n")
            f.write("-"*30 + "\n")
            
            # Write training metrics
            f.write(f"Training Time: {result['train_time']:.2f} seconds\n")
            f.write(f"Final Training Loss: {result['final_train_loss']:.6f}\n")
            f.write(f"Final Training Accuracy: {result['final_train_acc']:.4f}\n")
            
            # Write test metrics
            f.write(f"Test Loss: {result['test_loss']:.6f}\n")
            f.write(f"Test Accuracy: {result['test_acc']:.4f}\n")
            
            # Calculate generalization gap
            acc_gap = result['test_acc'] - result['final_train_acc']
            f.write(f"Generalization Gap (Accuracy): {acc_gap:.4f}\n")
            
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
        
        # Accuracy improvement
        acc_diff = vadam['test_acc'] - adam['test_acc']
        acc_percent = acc_diff * 100  # Converting to percentage points
        f.write(f"Test Accuracy: VADAM is {abs(acc_percent):.2f} percentage points {'better' if acc_diff > 0 else 'worse'} ({acc_diff:.4f})\n")
        
        # Overall assessment
        f.write("\nOVERALL ASSESSMENT:\n")
        if vadam['test_acc'] > adam['test_acc']:
            if vadam['train_time'] <= adam['train_time']:
                f.write("VADAM outperforms ADAM in both accuracy and training efficiency.\n")
            else:
                f.write(f"VADAM achieves better accuracy than ADAM ({abs(acc_percent):.2f} percentage points higher) but at a cost of {time_percent:.1f}% longer training time.\n")
        else:
            if vadam['train_time'] >= adam['train_time']:
                f.write("ADAM outperforms VADAM in both accuracy and training efficiency.\n")
            else:
                f.write(f"ADAM achieves better accuracy than VADAM ({abs(acc_percent):.2f} percentage points higher) but VADAM is {abs(time_percent):.1f}% faster in training.\n")
    
    print(f"Detailed summary saved to {summary_file}")
    
    # Print terminal summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"ADAM Test Loss: {adam['test_loss']:.6f}")
    print(f"VADAM Test Loss: {vadam['test_loss']:.6f}")
    print(f"ADAM Test Accuracy: {adam['test_acc']:.4f}")
    print(f"VADAM Test Accuracy: {vadam['test_acc']:.4f}")
    
    acc_diff = vadam['test_acc'] - adam['test_acc']
    print(f"Accuracy Difference: {acc_diff:.4f} ({acc_diff*100:.2f} percentage points)")
    
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
    
    print("\nStarting Text Classification Benchmark with ADAM vs VADAM...")
    start_time = time.time()
    
    try:
        results = run_text_classifier_benchmark()
        
        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.2f} seconds")
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise 