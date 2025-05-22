import torch
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from benchmarker import Benchmarker
from datetime import datetime
from diffusion_benchmark import compare_optimizers as compare_diffusion_optimizers

# Import benchmark modules
from cnn_image_classification.cnn_benchmark import run_cnn_benchmark
from transformer_language_modeling.transformer_benchmark import run_transformer_benchmark
from rl_halfcheetah.rl_benchmark import run_rl_benchmark
from diffusion_mnist.diffusion_benchmark import run_diffusion_benchmark

def plot_results(results, title, filename):
    """Plot and save comprehensive comparison charts for optimizer performance"""
    plt.figure(figsize=(16, 12))
    
    # 1. Plot training and validation loss curves
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
    
    # 2. Plot accuracy curves if available
    has_accuracy = any('train_accs' in result and result['train_accs'] for result in results.values())
    if has_accuracy:
        plt.subplot(2, 2, 2)
        for name, result in results.items():
            if 'train_accs' in result and result['train_accs']:
                plt.plot(result['train_accs'], label=f"{name} - Train")
            if 'val_accs' in result and result['val_accs']:
                plt.plot(result['val_accs'], linestyle='--', label=f"{name} - Val")
        
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Plot perplexity curves if available (for language modeling)
    has_perplexity = any('train_perplexities' in result and result['train_perplexities'] for result in results.values())
    if has_perplexity:
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
    
    # 4. Bar chart comparing final training metrics
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    
    # Get train metrics
    train_losses = []
    for name in names:
        if results[name]['final_train_loss'] is not None:
            train_losses.append(results[name]['final_train_loss'])
        else:
            train_losses.append(0)
    
    # Handle different types of metrics depending on task
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x, train_losses, width, label='Final Train Loss')
    
    if has_accuracy:
        train_accs = []
        for name in names:
            acc = results[name].get('final_train_acc', 0)
            train_accs.append(acc if acc is not None else 0)
        plt.bar(x + width, train_accs, width, label='Final Train Accuracy')
    elif has_perplexity:
        train_ppls = []
        for name in names:
            ppl = results[name].get('final_train_perplexity', 0)
            train_ppls.append(ppl if ppl is not None else 0)
        # Scale perplexity for visualization if needed
        scaled_ppls = [min(ppl/10, 10) for ppl in train_ppls]  # Cap at 10 for visualization
        plt.bar(x + width, scaled_ppls, width, label='Final Train Perplexity (Scaled)')
    
    plt.title("Training Performance")
    plt.xlabel("Optimizer")
    plt.xticks(x + width/2, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 5. Bar chart comparing test metrics
    plt.subplot(2, 2, 4)
    
    # Get test metrics
    test_losses = []
    for name in names:
        if results[name]['test_loss'] is not None:
            test_losses.append(results[name]['test_loss'])
        else:
            test_losses.append(0)
    
    plt.bar(x, test_losses, width, label='Test Loss')
    
    if has_accuracy:
        test_accs = []
        for name in names:
            acc = results[name].get('test_acc', 0)
            test_accs.append(acc if acc is not None else 0)
        plt.bar(x + width, test_accs, width, label='Test Accuracy')
    elif has_perplexity:
        test_ppls = []
        for name in names:
            ppl = results[name].get('test_perplexity', 0)
            test_ppls.append(ppl if ppl is not None else 0)
        # Scale perplexity for visualization
        scaled_ppls = [min(ppl/10, 10) for ppl in test_ppls]  # Cap at 10 for visualization
        plt.bar(x + width, scaled_ppls, width, label='Test Perplexity (Scaled)')
    
    # Add training time
    train_times = []
    for name in names:
        time_val = results[name]['train_time']
        train_times.append(time_val if time_val is not None else 0)
    
    max_time = max(train_times) if train_times else 1.0
    normalized_times = [t/max_time for t in train_times] 
    plt.bar(x + 2*width, normalized_times, width, label='Normalized Time')
    
    plt.title("Test Performance")
    plt.xlabel("Optimizer")
    plt.xticks(x + width, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    
    # Create a separate text summary file with detailed metrics
    summary_file = os.path.splitext(filename)[0] + '_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"OPTIMIZER COMPARISON: {title}\n")
        f.write("="*50 + "\n\n")
        
        for name in names:
            result = results[name]
            f.write(f"OPTIMIZER: {name}\n")
            f.write("-"*30 + "\n")
            f.write(f"Training Time: {result['train_time']:.2f} seconds\n")
            f.write(f"Final Train Loss: {result['final_train_loss']:.6f}\n")
            
            if 'final_train_acc' in result and result['final_train_acc'] is not None:
                f.write(f"Final Train Accuracy: {result['final_train_acc']:.4f}\n")
            if 'final_train_perplexity' in result and result['final_train_perplexity'] is not None:
                f.write(f"Final Train Perplexity: {result['final_train_perplexity']:.2f}\n")
                
            f.write(f"Test Loss: {result['test_loss']:.6f}\n")
            if 'test_acc' in result and result['test_acc'] is not None:
                f.write(f"Test Accuracy: {result['test_acc']:.4f}\n")
            if 'test_perplexity' in result and result['test_perplexity'] is not None:
                f.write(f"Test Perplexity: {result['test_perplexity']:.2f}\n")
                
            # Calculate improvement percentages
            if 'test_acc' in result and result['test_acc'] is not None:
                f.write(f"Generalization Gap (Acc): {result['test_acc'] - result['final_train_acc']:.4f}\n")
            if 'test_perplexity' in result and result['test_perplexity'] is not None:
                ppl_gap = result['test_perplexity'] - result['final_train_perplexity']
                f.write(f"Generalization Gap (PPL): {ppl_gap:.2f}\n")
                
            f.write("\n")
            
        # Add head-to-head comparison
        if len(names) == 2:
            f.write("\nHEAD-TO-HEAD COMPARISON\n")
            f.write("-"*30 + "\n")
            adam_result = results['ADAM']
            vradam_result = results['VRADAM']
            
            time_diff = vradam_result['train_time'] - adam_result['train_time']
            time_percent = (time_diff / adam_result['train_time']) * 100
            f.write(f"Training Time Difference: {time_diff:.2f}s ({time_percent:.1f}%)\n")
            
            loss_diff = vradam_result['test_loss'] - adam_result['test_loss']
            loss_percent = (loss_diff / adam_result['test_loss']) * 100
            f.write(f"Test Loss Difference: {loss_diff:.6f} ({loss_percent:.1f}%)\n")
            
            if 'test_acc' in adam_result and 'test_acc' in vradam_result:
                acc_diff = vradam_result['test_acc'] - adam_result['test_acc']
                acc_percent = acc_diff * 100  # percentage points
                f.write(f"Test Accuracy Difference: {acc_diff:.4f} ({acc_percent:.1f} percentage points)\n")
                
            if 'test_perplexity' in adam_result and 'test_perplexity' in vradam_result:
                ppl_diff = vradam_result['test_perplexity'] - adam_result['test_perplexity']
                ppl_percent = (ppl_diff / adam_result['test_perplexity']) * 100
                f.write(f"Test Perplexity Difference: {ppl_diff:.2f} ({ppl_percent:.1f}%)\n")
    
    print(f"Detailed summary saved to {summary_file}")

def compare_optimizers(base_params, dataset, model_type):
    """Run benchmarks comparing ADAM and VRADAM optimizers"""
    results = {}
    
    # Update params for this specific comparison
    params = base_params.copy()
    params.update({
        'dataset': dataset,
        'model': model_type
    })
    
    # Run benchmark with Adam optimizer
    print("\n" + "="*50)
    print(f"Running {model_type} on {dataset} with ADAM optimizer")
    print("="*50 + "\n")
    
    adam_params = params.copy()
    adam_params['optimizer'] = 'ADAM'
    # ADAM specific parameters (using defaults if not specified)
    adam_params.update({
        'lr': params.get('lr', 0.001),
        'beta1': params.get('adam_beta1', 0.9),
        'beta2': params.get('adam_beta2', 0.999),
        'eps': params.get('adam_eps', 1e-8),
        'weight_decay': params.get('adam_weight_decay', 0)
    })
    benchmark_adam = Benchmarker(adam_params)
    adam_results = benchmark_adam.run()
    results['ADAM'] = adam_results
    
    # Run benchmark with VRADAM optimizer
    print("\n" + "="*50)
    print(f"Running {model_type} on {dataset} with VRADAM optimizer")
    print("="*50 + "\n")
    
    vradam_params = params.copy()
    vradam_params['optimizer'] = 'VRADAM'
    # VRADAM specific parameters (using defaults if not specified)
    vradam_params.update({
        'eta': params.get('vradam_eta', params.get('lr', 0.001)),  # Use eta if provided, otherwise fall back to lr
        'beta1': params.get('vradam_beta1', 0.9),
        'beta2': params.get('vradam_beta2', 0.999),
        'beta3': params.get('vradam_beta3', 1.0),
        'eps': params.get('vradam_eps', 1e-8),
        'weight_decay': params.get('vradam_weight_decay', 0),
        'power': params.get('vradam_power', 2),
        'normgrad': params.get('vradam_normgrad', True),
        'lr_cutoff': params.get('vradam_lr_cutoff', 19)
    })
    benchmark_vradam = Benchmarker(vradam_params)
    vradam_results = benchmark_vradam.run()
    results['VRADAM'] = vradam_results
    
    # Save results to file
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{dataset}_{model_type}_{timestamp}.json")
    
    # Clean up results for JSON serialization
    clean_results = {}
    for optimizer, result in results.items():
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                clean_result[key] = value
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                clean_result[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                clean_result[key] = str(value)
        clean_results[optimizer] = clean_result
    
    with open(result_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Plot results
    plot_file = os.path.join(output_dir, f"{dataset}_{model_type}_{timestamp}.png")
    plot_results(results, f"Optimizer Comparison: {model_type} on {dataset}", plot_file)
    
    # Print head-to-head summary
    print("\n" + "="*50)
    print(f"SUMMARY: {model_type} on {dataset}")
    print("="*50)
    
    # Compare key metrics
    print(f"ADAM Test Loss: {results['ADAM']['test_loss']:.6f}")
    print(f"VRADAM Test Loss: {results['VRADAM']['test_loss']:.6f}")
    
    if results['ADAM'].get('test_acc') is not None:
        print(f"ADAM Test Accuracy: {results['ADAM']['test_acc']:.4f}")
        print(f"VRADAM Test Accuracy: {results['VRADAM']['test_acc']:.4f}")
    
    if results['ADAM'].get('test_perplexity') is not None:
        print(f"ADAM Test Perplexity: {results['ADAM']['test_perplexity']:.2f}")
        print(f"VRADAM Test Perplexity: {results['VRADAM']['test_perplexity']:.2f}")
        
    print(f"ADAM Training Time: {results['ADAM']['train_time']:.2f}s")
    print(f"VRADAM Training Time: {results['VRADAM']['train_time']:.2f}s")
    
    return results

def run_all_benchmarks(device=None):
    """Run benchmarks for all configured models and datasets"""
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS (Apple Silicon)")
        else:
            device = 'cpu'
            print("Using CPU")
    
    all_results = {}
    
    # Create benchmark_results directory if it doesn't exist
    os.makedirs('benchmark_results', exist_ok=True)
    
    print("\nStarting benchmarks...\n")
    print("This will compare ADAM and VRADAM optimizers across multiple models and datasets")
    print("Results will be saved to the benchmark_results directory\n")
    
    total_start_time = time.time()
    
    # Run CNN on CIFAR10
    print(f"\n{'='*50}")
    print(f"Running benchmark for CNN on CIFAR10")
    print(f"{'='*50}\n")
    try:
        cnn_results = run_cnn_benchmark()
        all_results["CIFAR10_CNN"] = cnn_results
    except Exception as e:
        print(f"Error running CNN benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Run Transformer on WikiText2
    print(f"\n{'='*50}")
    print(f"Running benchmark for Transformer on WikiText2")
    print(f"{'='*50}\n")
    try:
        transformer_results = run_transformer_benchmark()
        all_results["WikiText2_Transformer"] = transformer_results
    except Exception as e:
        print(f"Error running Transformer benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Run RL on HalfCheetah
    print(f"\n{'='*50}")
    print(f"Running benchmark for RL on HalfCheetah")
    print(f"{'='*50}\n")
    try:
        rl_results = run_rl_benchmark(epochs=10)  # Reduced epochs for faster benchmarking
        all_results["HalfCheetah_RL"] = rl_results
    except Exception as e:
        print(f"Error running RL benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Run Diffusion Model on MNIST
    print(f"\n{'='*50}")
    print(f"Running benchmark for Diffusion Model on MNIST")
    print(f"{'='*50}\n")
    try:
        diffusion_results = run_diffusion_benchmark()
        all_results["MNIST_Diffusion"] = diffusion_results
    except Exception as e:
        print(f"Error running Diffusion benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Create a summary comparison of all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join('benchmark_results', f"optimizer_summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("OPTIMIZER BENCHMARK SUMMARY\n")
        f.write("==========================\n\n")
        
        for config_name, results in all_results.items():
            f.write(f"Model Configuration: {config_name}\n")
            f.write("-" * 50 + "\n")
            
            # Extract key metrics
            adam = results['ADAM']
            vradam = results['VRADAM']
            
            # Determine which metrics to report based on task type
            f.write(f"Test Loss: ADAM = {adam['test_loss']:.6f}, VRADAM = {vradam['test_loss']:.6f}\n")
            
            if 'test_acc' in adam and adam['test_acc'] is not None:
                f.write(f"Test Accuracy: ADAM = {adam['test_acc']:.4f}, VRADAM = {vradam['test_acc']:.4f}\n")
                
            if 'test_perplexity' in adam and adam['test_perplexity'] is not None:
                f.write(f"Test Perplexity: ADAM = {adam['test_perplexity']:.2f}, VRADAM = {vradam['test_perplexity']:.2f}\n")

            if 'final_fid_score' in adam and adam['final_fid_score'] is not None:
                f.write(f"FID Score: ADAM = {adam['final_fid_score']:.4f}, VRADAM = {vradam['final_fid_score']:.4f}\n")
                
            f.write(f"Training Time: ADAM = {adam['train_time']:.2f}s, VRADAM = {vradam['train_time']:.2f}s\n")
            
            # Calculate percentage improvements
            if adam['test_loss'] is not None and vradam['test_loss'] is not None:
                loss_diff = (vradam['test_loss'] - adam['test_loss']) / max(adam['test_loss'], 1e-8) * 100
                f.write(f"Loss Improvement: {-loss_diff:.2f}% ({'better' if loss_diff < 0 else 'worse'} for VRADAM)\n")
            
            if 'test_acc' in adam and adam['test_acc'] is not None and 'test_acc' in vradam and vradam['test_acc'] is not None:
                acc_diff = vradam['test_acc'] - adam['test_acc']
                f.write(f"Accuracy Improvement: {acc_diff*100:.2f} percentage points ({'better' if acc_diff > 0 else 'worse'} for VRADAM)\n")
                
            if 'test_perplexity' in adam and adam['test_perplexity'] is not None and 'test_perplexity' in vradam and vradam['test_perplexity'] is not None:
                ppl_diff = (vradam['test_perplexity'] - adam['test_perplexity']) / max(adam['test_perplexity'], 1e-8) * 100
                f.write(f"Perplexity Improvement: {-ppl_diff:.2f}% ({'better' if ppl_diff < 0 else 'worse'} for VRADAM)\n")
                
            if adam['train_time'] is not None and vradam['train_time'] is not None:
                time_diff = (vradam['train_time'] - adam['train_time']) / max(adam['train_time'], 1e-8) * 100
                f.write(f"Time Efficiency: {-time_diff:.2f}% ({'faster' if time_diff < 0 else 'slower'} for VRADAM)\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
    print(f"\nFull benchmark summary written to {summary_file}")
    
    total_time = time.time() - total_start_time
    print(f"\nAll benchmarks completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return all_results

def run_specific_benchmark(model_name):
    """Run a specific benchmark"""
    if model_name.upper() == 'CNN':
        print(f"\n{'='*50}")
        print(f"Running benchmark for CNN on CIFAR10")
        print(f"{'='*50}\n")
        return run_cnn_benchmark()
    elif model_name.upper() == 'TRANSFORMER':
        print(f"\n{'='*50}")
        print(f"Running benchmark for Transformer on WikiText2")
        print(f"{'='*50}\n")
        return run_transformer_benchmark()
    elif model_name.upper() == 'RL':
        print(f"\n{'='*50}")
        print(f"Running benchmark for RL on HalfCheetah")
        print(f"{'='*50}\n")
        return run_rl_benchmark()
    elif model_name.upper() == 'DIFFUSION':
        print(f"\n{'='*50}")
        print(f"Running benchmark for Diffusion Model on MNIST")
        print(f"{'='*50}\n")
        return run_diffusion_benchmark()
    else:
        raise ValueError(f"Unknown benchmark: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks comparing ADAM and VRADAM")
    parser.add_argument("--model", type=str, choices=["CNN", "Transformer", "RL", "Diffusion", "all"],
                      help="Which model to benchmark (default: all)", default="all")
    
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
    
    start_time = time.time()
    
    if args.model.lower() == 'all':
        all_results = run_all_benchmarks(device)
    else:
        results = run_specific_benchmark(args.model)
    
    total_time = time.time() - start_time
    print(f"\nBenchmarks completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)") 