import argparse
import wandb
import json
import os
import torch
from benchmarker import Benchmarker
import matplotlib.pyplot as plt

def run_benchmarks_with_optimized_params(
    model, dataset, eta=None, beta1=None, beta2=None, beta3=None, 
    power=None, normgrad=None, lr_cutoff=None, weight_decay=None, eps=None,
    dataset_size="full", epochs=10, use_wandb=True):
    """
    Run benchmarks with optimized hyperparameters for VADAM
    
    Args:
        model: Model architecture (SimpleCNN, MLPModel, TransformerModel)
        dataset: Dataset (CIFAR10, IMDB, WikiText2)
        eta, beta1, beta2, beta3, power, normgrad, lr_cutoff, weight_decay, eps: VADAM parameters
        dataset_size: Size of dataset ('small' or 'full')
        epochs: Number of epochs to train
        use_wandb: Whether to use W&B for tracking
    """
    # Base parameters
    params = {
        'model': model,
        'dataset': dataset,
        'dataset_size': dataset_size,
        'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer': 'VADAM',
        'batch_size': 64,
        'max_seq_len': 128,
        'embed_dim': 256,
        'hidden_dim': 512,
        'epochs': epochs
    }
    
    # Add VADAM specific parameters if provided
    vadam_params = {}
    if eta is not None:
        vadam_params['eta'] = eta
    if beta1 is not None:
        vadam_params['beta1'] = beta1
    if beta2 is not None:
        vadam_params['beta2'] = beta2
    if beta3 is not None:
        vadam_params['beta3'] = beta3
    if power is not None:
        vadam_params['power'] = power
    if normgrad is not None:
        vadam_params['normgrad'] = normgrad
    if lr_cutoff is not None:
        vadam_params['lr_cutoff'] = lr_cutoff
    if weight_decay is not None:
        vadam_params['weight_decay'] = weight_decay
    if eps is not None:
        vadam_params['eps'] = eps
    
    # Update params with VADAM parameters
    params.update(vadam_params)
    
    # Initialize W&B if needed
    if use_wandb:
        wandb.init(
            project=f"vadam-optimized-{model}-{dataset}",
            config=params,
            name=f"optimized_{model}_{dataset}"
        )
    
    print(f"Running benchmark with optimized parameters for {model} on {dataset}:")
    for param, value in vadam_params.items():
        print(f"  - {param}: {value}")
    
    # Run the benchmark with optimized parameters
    benchmark = Benchmarker(params)
    results = benchmark.run()
    
    # Log final results to W&B if enabled
    if use_wandb:
        if benchmark.task_type == "language_modeling":
            wandb.log({
                "final_train_loss": results['final_train_loss'],
                "final_train_perplexity": results['final_train_perplexity'],
                "test_loss": results['test_loss'],
                "test_perplexity": results['test_perplexity'],
                "train_time": results['train_time']
            })
        else:  # Classification
            wandb.log({
                "final_train_loss": results['final_train_loss'],
                "final_train_acc": results['final_train_acc'],
                "test_loss": results['test_loss'],
                "test_acc": results['test_acc'],
                "train_time": results['train_time']
            })
    
    # Compare with default Adam
    print("\nComparing with default Adam optimizer...")
    
    # Use same parameters but with Adam optimizer
    adam_params = params.copy()
    adam_params['optimizer'] = 'ADAM'
    adam_params['lr'] = params.get('eta', 0.001)  # Use eta as lr
    
    # Remove VADAM specific parameters
    for param in ['eta', 'beta3', 'power', 'normgrad', 'lr_cutoff']:
        if param in adam_params:
            del adam_params[param]
    
    # Run Adam benchmark
    benchmark_adam = Benchmarker(adam_params)
    adam_results = benchmark_adam.run()
    
    # Save both results to JSON file
    os.makedirs("optimized_results", exist_ok=True)
    result_file = f"optimized_results/{model}_{dataset}_comparison.json"
    
    comparison = {
        "model": model,
        "dataset": dataset,
        "vadam_params": params,
        "vadam_results": results,
        "adam_params": adam_params,
        "adam_results": adam_results
    }
    
    with open(result_file, 'w') as f:
        # Convert any tensors to lists
        json.dump(comparison, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"Results saved to {result_file}")
    
    # Create comparison plot
    create_comparison_plot(results, adam_results, model, dataset, benchmark.task_type)
    
    # Close W&B run if used
    if use_wandb:
        wandb.finish()
    
    return results, adam_results

def create_comparison_plot(vadam_results, adam_results, model, dataset, task_type):
    """Create a plot comparing VADAM and Adam results"""
    plt.figure(figsize=(15, 10))
    
    # Training loss plot
    plt.subplot(2, 2, 1)
    plt.plot(vadam_results['train_losses'], 'b-', label='VADAM Train Loss')
    plt.plot(adam_results['train_losses'], 'r-', label='Adam Train Loss')
    if 'val_losses' in vadam_results and vadam_results['val_losses']:
        plt.plot(vadam_results['val_losses'], 'b--', label='VADAM Val Loss')
        plt.plot(adam_results['val_losses'], 'r--', label='Adam Val Loss')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot accuracy or perplexity based on task type
    plt.subplot(2, 2, 2)
    if task_type == "language_modeling":
        # Plot perplexity for language modeling
        plt.plot(vadam_results['train_perplexities'], 'b-', label='VADAM Train PPL')
        plt.plot(adam_results['train_perplexities'], 'r-', label='Adam Train PPL')
        if 'val_perplexities' in vadam_results and vadam_results['val_perplexities']:
            plt.plot(vadam_results['val_perplexities'], 'b--', label='VADAM Val PPL')
            plt.plot(adam_results['val_perplexities'], 'r--', label='Adam Val PPL')
        plt.title('Perplexity Comparison (Lower is Better)')
        plt.ylabel('Perplexity')
    else:
        # Plot accuracy for classification
        plt.plot(vadam_results['train_accs'], 'b-', label='VADAM Train Acc')
        plt.plot(adam_results['train_accs'], 'r-', label='Adam Train Acc')
        if 'val_accs' in vadam_results and vadam_results['val_accs']:
            plt.plot(vadam_results['val_accs'], 'b--', label='VADAM Val Acc')
            plt.plot(adam_results['val_accs'], 'r--', label='Adam Val Acc')
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Bar plot for test metrics
    plt.subplot(2, 2, 3)
    labels = ['Adam', 'VADAM']
    
    # Test loss
    loss_values = [adam_results['test_loss'], vadam_results['test_loss']]
    plt.bar([0, 1], loss_values, width=0.4, alpha=0.7, label='Test Loss')
    
    # Add values above bars
    for i, v in enumerate(loss_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.title('Test Loss Comparison')
    plt.ylabel('Loss')
    plt.xticks([0, 1], labels)
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    # Bar plot for accuracy or perplexity
    plt.subplot(2, 2, 4)
    if task_type == "language_modeling":
        # Test perplexity for language modeling
        metric_values = [adam_results['test_perplexity'], vadam_results['test_perplexity']]
        title = 'Test Perplexity Comparison (Lower is Better)'
        ylabel = 'Perplexity'
    else:
        # Test accuracy for classification
        metric_values = [adam_results['test_acc'], vadam_results['test_acc']]
        title = 'Test Accuracy Comparison (Higher is Better)'
        ylabel = 'Accuracy'
    
    plt.bar([0, 1], metric_values, width=0.4, alpha=0.7, color=['r', 'b'])
    
    # Add values above bars
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks([0, 1], labels)
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.suptitle(f'Optimized VADAM vs Adam on {dataset} with {model}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    os.makedirs("optimized_results", exist_ok=True)
    plt.savefig(f"optimized_results/{model}_{dataset}_comparison.png")
    print(f"Comparison plot saved to optimized_results/{model}_{dataset}_comparison.png")

def load_best_params_from_file(model, dataset):
    """Load best parameters from JSON file"""
    best_configs_file = "sweep_results/best_configs.json"
    if not os.path.exists(best_configs_file):
        print(f"Best configs file not found: {best_configs_file}")
        return {}
    
    with open(best_configs_file, 'r') as f:
        best_configs = json.load(f)
    
    config_key = f"{model}_{dataset}"
    if config_key not in best_configs:
        print(f"No optimized parameters found for {model} on {dataset}")
        return {}
    
    # Extract VADAM parameters
    config = best_configs[config_key]["best_config"]
    vadam_params = {k: v for k, v in config.items() 
                   if k in ['eta', 'beta1', 'beta2', 'beta3', 'power', 
                           'normgrad', 'lr_cutoff', 'weight_decay', 'eps']}
    
    return vadam_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks with optimized VADAM parameters")
    parser.add_argument("--model", type=str, required=True, 
                      choices=["SimpleCNN", "MLPModel", "TransformerModel"],
                      help="Model architecture")
    parser.add_argument("--dataset", type=str, required=True, 
                      choices=["CIFAR10", "IMDB", "WikiText2"],
                      help="Dataset")
    parser.add_argument("--dataset_size", type=str, default="full", 
                      choices=["small", "full"],
                      help="Size of dataset")
    parser.add_argument("--epochs", type=int, default=10, 
                      help="Number of epochs to train")
    parser.add_argument("--use_wandb", action="store_true", 
                      help="Use W&B for tracking")
    parser.add_argument("--load_params", action="store_true", 
                      help="Load optimized parameters from file")
    
    # VADAM parameters
    parser.add_argument("--eta", type=float, help="Learning rate for VADAM")
    parser.add_argument("--beta1", type=float, help="Beta1 for VADAM")
    parser.add_argument("--beta2", type=float, help="Beta2 for VADAM")
    parser.add_argument("--beta3", type=float, help="Beta3 for VADAM")
    parser.add_argument("--power", type=int, help="Power for VADAM")
    parser.add_argument("--normgrad", action="store_true", help="Normalize gradients for VADAM")
    parser.add_argument("--lr_cutoff", type=int, help="Learning rate cutoff for VADAM")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for VADAM")
    parser.add_argument("--eps", type=float, help="Epsilon for VADAM")
    
    args = parser.parse_args()
    
    # If load_params is set, load optimized parameters from file
    if args.load_params:
        vadam_params = load_best_params_from_file(args.model, args.dataset)
        
        # Override with command line arguments if provided
        if args.eta is not None:
            vadam_params['eta'] = args.eta
        if args.beta1 is not None:
            vadam_params['beta1'] = args.beta1
        if args.beta2 is not None:
            vadam_params['beta2'] = args.beta2
        if args.beta3 is not None:
            vadam_params['beta3'] = args.beta3
        if args.power is not None:
            vadam_params['power'] = args.power
        if args.normgrad:
            vadam_params['normgrad'] = True
        if args.lr_cutoff is not None:
            vadam_params['lr_cutoff'] = args.lr_cutoff
        if args.weight_decay is not None:
            vadam_params['weight_decay'] = args.weight_decay
        if args.eps is not None:
            vadam_params['eps'] = args.eps
    else:
        # Use command line arguments directly
        vadam_params = {
            'eta': args.eta,
            'beta1': args.beta1,
            'beta2': args.beta2,
            'beta3': args.beta3,
            'power': args.power,
            'normgrad': args.normgrad,
            'lr_cutoff': args.lr_cutoff,
            'weight_decay': args.weight_decay,
            'eps': args.eps
        }
        # Remove None values
        vadam_params = {k: v for k, v in vadam_params.items() if v is not None}
    
    # Run the benchmark with the specified parameters
    results, adam_results = run_benchmarks_with_optimized_params(
        model=args.model,
        dataset=args.dataset,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
        **vadam_params
    ) 