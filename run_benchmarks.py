import torch
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from benchmarker import Benchmarker
from datetime import datetime
from diffusion_benchmark import compare_optimizers as compare_diffusion_optimizers

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
            vadam_result = results['VADAM']
            
            time_diff = vadam_result['train_time'] - adam_result['train_time']
            time_percent = (time_diff / adam_result['train_time']) * 100
            f.write(f"Training Time Difference: {time_diff:.2f}s ({time_percent:.1f}%)\n")
            
            loss_diff = vadam_result['test_loss'] - adam_result['test_loss']
            loss_percent = (loss_diff / adam_result['test_loss']) * 100
            f.write(f"Test Loss Difference: {loss_diff:.6f} ({loss_percent:.1f}%)\n")
            
            if 'test_acc' in adam_result and 'test_acc' in vadam_result:
                acc_diff = vadam_result['test_acc'] - adam_result['test_acc']
                acc_percent = acc_diff * 100  # percentage points
                f.write(f"Test Accuracy Difference: {acc_diff:.4f} ({acc_percent:.1f} percentage points)\n")
                
            if 'test_perplexity' in adam_result and 'test_perplexity' in vadam_result:
                ppl_diff = vadam_result['test_perplexity'] - adam_result['test_perplexity']
                ppl_percent = (ppl_diff / adam_result['test_perplexity']) * 100
                f.write(f"Test Perplexity Difference: {ppl_diff:.2f} ({ppl_percent:.1f}%)\n")
    
    print(f"Detailed summary saved to {summary_file}")

def compare_optimizers(base_params, dataset, model_type):
    """Run benchmarks comparing ADAM and VADAM optimizers"""
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
    
    # Run benchmark with VADAM optimizer
    print("\n" + "="*50)
    print(f"Running {model_type} on {dataset} with VADAM optimizer")
    print("="*50 + "\n")
    
    vadam_params = params.copy()
    vadam_params['optimizer'] = 'VADAM'
    # VADAM specific parameters (using defaults if not specified)
    vadam_params.update({
        'eta': params.get('vadam_eta', params.get('lr', 0.001)),  # Use eta if provided, otherwise fall back to lr
        'beta1': params.get('vadam_beta1', 0.9),
        'beta2': params.get('vadam_beta2', 0.999),
        'beta3': params.get('vadam_beta3', 1.0),
        'eps': params.get('vadam_eps', 1e-8),
        'weight_decay': params.get('vadam_weight_decay', 0),
        'power': params.get('vadam_power', 2),
        'normgrad': params.get('vadam_normgrad', True),
        'lr_cutoff': params.get('vadam_lr_cutoff', 19)
    })
    benchmark_vadam = Benchmarker(vadam_params)
    vadam_results = benchmark_vadam.run()
    results['VADAM'] = vadam_results
    
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
    print(f"VADAM Test Loss: {results['VADAM']['test_loss']:.6f}")
    
    if results['ADAM'].get('test_acc') is not None:
        print(f"ADAM Test Accuracy: {results['ADAM']['test_acc']:.4f}")
        print(f"VADAM Test Accuracy: {results['VADAM']['test_acc']:.4f}")
    
    if results['ADAM'].get('test_perplexity') is not None:
        print(f"ADAM Test Perplexity: {results['ADAM']['test_perplexity']:.2f}")
        print(f"VADAM Test Perplexity: {results['VADAM']['test_perplexity']:.2f}")
        
    print(f"ADAM Training Time: {results['ADAM']['train_time']:.2f}s")
    print(f"VADAM Training Time: {results['VADAM']['train_time']:.2f}s")
    
    return results

def run_all_benchmarks():
    """Run benchmarks for all configured models and datasets"""
    # Set up base parameters
    base_params = {
        'device': 'mps',  # Use 'cuda' if available, otherwise 'cpu'
        'dataset_size': 'small',  # Use 'small' for faster runs during testing
        'batch_size': 32,
        'max_seq_len': 128,
        'embed_dim': 256,
        'hidden_dim': 512,
        
        # General optimization parameters
        'lr': 0.001,  # Used for Adam
        'epochs': 3,  # Use a small number of epochs for quick testing
        
        # Adam specific parameters
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8,
        'adam_weight_decay': 0,
        
        # VADAM specific parameters
        'vadam_eta': 0.001,  # Max learning rate for VADAM
        'vadam_beta1': 0.9,
        'vadam_beta2': 0.999,
        'vadam_beta3': 1.0,
        'vadam_eps': 1e-8,
        'vadam_weight_decay': 0,
        'vadam_power': 2,
        'vadam_normgrad': True,
        'vadam_lr_cutoff': 19
    }
    
    # Define benchmark configurations
    configs = [
        # Image classification
        {'dataset': 'CIFAR10', 'model': 'SimpleCNN'},
        
        # Text classification
        {'dataset': 'IMDB', 'model': 'MLPModel'},
        
        # Language modeling / Sentence completion
        {'dataset': 'WikiText2', 'model': 'TransformerModel'}
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Running benchmark for {config['model']} on {config['dataset']}")
        print(f"{'='*50}\n")
        
        try:
            results = compare_optimizers(base_params, config['dataset'], config['model'])
            all_results[f"{config['dataset']}_{config['model']}"] = results
        except Exception as e:
            print(f"Error running benchmark for {config['model']} on {config['dataset']}: {e}")
    
    # Run diffusion model benchmark with MNIST
    print(f"\n{'='*50}")
    print(f"Running benchmark for Diffusion Model on MNIST")
    print(f"{'='*50}\n")
    
    try:
        # Diffusion model specific parameters
        diffusion_params = {
            'device': base_params['device'],
            'dataset_size': 'small',
            'batch_size': 64,
            'lr': 0.0002,  # Specific learning rate for diffusion
            'epochs': 5,  # Reduced epochs for diffusion model
            'unet_base_channels': 64,  # Enhanced model architecture
            'unet_time_embed_dim': 128,  # Enhanced model architecture
            'num_timesteps': 200,  # Reduced timesteps for faster benchmarking
            'beta_min': 1e-4,
            'beta_max': 0.02,
            'sample_every': 1,  # Generate samples every epoch
            'eval_batch_size': 16,
            'fid_batch_size': 32,
            'fid_num_samples': 250,  # Use fewer samples for faster FID calculation
            
            # Optimizer specific parameters
            'adam_beta1': base_params['adam_beta1'],
            'adam_beta2': base_params['adam_beta2'],
            'adam_eps': base_params['adam_eps'],
            'adam_weight_decay': base_params['adam_weight_decay'],
            'vadam_eta': 0.0002,  # Specific for diffusion
            'vadam_beta1': base_params['vadam_beta1'],
            'vadam_beta2': base_params['vadam_beta2'],
            'vadam_beta3': base_params['vadam_beta3'],
            'vadam_eps': base_params['vadam_eps'],
            'vadam_weight_decay': base_params['vadam_weight_decay'],
            'vadam_power': base_params['vadam_power'],
            'vadam_normgrad': base_params['vadam_normgrad'],
            'vadam_lr_cutoff': base_params['vadam_lr_cutoff']
        }
        
        diffusion_results = compare_diffusion_optimizers(diffusion_params)
        all_results["MNIST_DiffusionModel"] = diffusion_results
    except Exception as e:
        print(f"Error running benchmark for Diffusion Model on MNIST: {e}")
    
    # Create a summary comparison of all results
    output_dir = 'benchmark_results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"optimizer_summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("OPTIMIZER BENCHMARK SUMMARY\n")
        f.write("==========================\n\n")
        
        for config_name, results in all_results.items():
            f.write(f"Model Configuration: {config_name}\n")
            f.write("-" * 50 + "\n")
            
            # Extract key metrics
            adam = results['ADAM']
            vadam = results['VADAM']
            
            # Determine which metrics to report based on task type
            f.write(f"Test Loss: ADAM = {adam['test_loss']:.6f}, VADAM = {vadam['test_loss']:.6f}\n")
            
            if 'test_acc' in adam and adam['test_acc'] is not None:
                f.write(f"Test Accuracy: ADAM = {adam['test_acc']:.4f}, VADAM = {vadam['test_acc']:.4f}\n")
                
            if 'test_perplexity' in adam and adam['test_perplexity'] is not None:
                f.write(f"Test Perplexity: ADAM = {adam['test_perplexity']:.2f}, VADAM = {vadam['test_perplexity']:.2f}\n")
                
            f.write(f"Training Time: ADAM = {adam['train_time']:.2f}s, VADAM = {vadam['train_time']:.2f}s\n")
            
            # Calculate percentage improvements
            if adam['test_loss'] is not None and vadam['test_loss'] is not None:
                loss_diff = (vadam['test_loss'] - adam['test_loss']) / max(adam['test_loss'], 1e-8) * 100
                f.write(f"Loss Improvement: {-loss_diff:.2f}% ({'better' if loss_diff < 0 else 'worse'} for VADAM)\n")
            
            if 'test_acc' in adam and adam['test_acc'] is not None and 'test_acc' in vadam and vadam['test_acc'] is not None:
                acc_diff = vadam['test_acc'] - adam['test_acc']
                f.write(f"Accuracy Improvement: {acc_diff*100:.2f} percentage points ({'better' if acc_diff > 0 else 'worse'} for VADAM)\n")
                
            if 'test_perplexity' in adam and adam['test_perplexity'] is not None and 'test_perplexity' in vadam and vadam['test_perplexity'] is not None:
                ppl_diff = (vadam['test_perplexity'] - adam['test_perplexity']) / max(adam['test_perplexity'], 1e-8) * 100
                f.write(f"Perplexity Improvement: {-ppl_diff:.2f}% ({'better' if ppl_diff < 0 else 'worse'} for VADAM)\n")
                
            if adam['train_time'] is not None and vadam['train_time'] is not None:
                time_diff = (vadam['train_time'] - adam['train_time']) / max(adam['train_time'], 1e-8) * 100
                f.write(f"Time Efficiency: {-time_diff:.2f}% ({'faster' if time_diff < 0 else 'slower'} for VADAM)\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
    print(f"\nFull benchmark summary written to {summary_file}")
    return all_results

if __name__ == "__main__":
    # Determine the best available device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Create benchmark_results directory if it doesn't exist
    os.makedirs('benchmark_results', exist_ok=True)
    
    print("\nStarting benchmarks...\n")
    print("This will compare ADAM and VADAM optimizers across multiple models and datasets")
    print("Results will be saved to the benchmark_results directory\n")
    
    start_time = time.time()
    
    all_results = run_all_benchmarks()
    
    total_time = time.time() - start_time
    print(f"\nAll benchmarks completed in {total_time:.2f} seconds") 