import wandb
import torch
import os
import time
import json
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': config.model,
            'dataset': config.dataset,
            'dataset_size': config.dataset_size,
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimizer': 'VADAM',
            'batch_size': config.batch_size,
            'max_seq_len': config.max_seq_len,
            'embed_dim': config.embed_dim,
            'hidden_dim': config.hidden_dim,
            'epochs': config.epochs,
            
            # VADAM specific parameters
            'eta': config.eta,
            'beta1': config.beta1,
            'beta2': config.beta2,
            'beta3': config.beta3,
            'power': config.power,
            'normgrad': config.normgrad,
            'lr_cutoff': config.lr_cutoff,
            'weight_decay': config.weight_decay,
            'eps': config.eps
        }
        
        # Create and run benchmarker
        print(f"Running benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Log metrics to wandb
        # Track key metrics based on model type
        if benchmark.task_type == "language_modeling":
            wandb.log({
                "final_train_loss": results['final_train_loss'],
                "final_train_perplexity": results['final_train_perplexity'],
                "test_loss": results['test_loss'],
                "test_perplexity": results['test_perplexity'],
                "train_time": results['train_time']
            })
            
            # Track training curves
            for epoch, (loss, perplexity) in enumerate(zip(results['train_losses'], results['train_perplexities'])):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_perplexity": perplexity
                })
                
        else:  # Classification tasks
            wandb.log({
                "final_train_loss": results['final_train_loss'],
                "final_train_acc": results['final_train_acc'],
                "test_loss": results['test_loss'],
                "test_acc": results['test_acc'],
                "train_time": results['train_time']
            })
            
            # Track training curves
            for epoch, (loss, acc) in enumerate(zip(results['train_losses'], results['train_accs'])):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_acc": acc
                })
        
        # Determine the optimization metric based on task
        if benchmark.task_type == "language_modeling":
            # For language modeling, lower perplexity is better
            optimization_metric = results['test_perplexity']
        else:
            # For classification, higher accuracy is better, but W&B minimizes by default
            # so we negate the accuracy
            optimization_metric = -results['test_acc']
            
        wandb.log({"optimization_metric": optimization_metric})
        
        return results

def create_sweep_config(model_type, dataset):
    """Create sweep configuration for a specific model and dataset"""
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # We'll use negative accuracy for classification
        },
        'parameters': {
            # Fixed parameters
            'model': {'value': model_type},
            'dataset': {'value': dataset},
            'dataset_size': {'value': 'small'},  # Use small dataset for faster iterations
            'epochs': {'value': 3},  # Keep epochs low for faster iterations
            
            # VADAM specific parameters to optimize
            'eta': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.01},
            'beta1': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'beta2': {'distribution': 'uniform', 'min': 0.99, 'max': 0.9999},
            'beta3': {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
            'power': {'values': [1, 2, 3]},
            'normgrad': {'values': [True, False]},
            'lr_cutoff': {'distribution': 'int_uniform', 'min': 5, 'max': 30},
            'weight_decay': {'distribution': 'uniform', 'min': 0.0, 'max': 0.01},
            'eps': {'distribution': 'uniform', 'min': 1e-9, 'max': 1e-7},
            
            # Model hyperparameters
            'batch_size': {'values': [16, 32, 64, 128]},
        }
    }
    
    # Add model-specific parameters
    if model_type in ['TransformerModel', 'MLPModel']:
        sweep_config['parameters'].update({
            'embed_dim': {'values': [128, 256, 384, 512]},
            'hidden_dim': {'values': [256, 512, 768, 1024]},
            'max_seq_len': {'values': [64, 128, 256]},
        })
    else:
        # Add default values for SimpleCNN
        sweep_config['parameters'].update({
            'embed_dim': {'value': 300},
            'hidden_dim': {'value': 512},
            'max_seq_len': {'value': 256},
        })
    
    return sweep_config

def run_sweeps():
    """Run sweeps for all model and dataset combinations"""
    # List of all model and dataset combinations to benchmark
    model_dataset_pairs = [
        ('SimpleCNN', 'CIFAR10'),
        ('MLPModel', 'IMDB'),
        ('TransformerModel', 'WikiText2')
    ]
    
    # Set up wandb project
    wandb.login()
    
    sweep_ids = []
    for model_type, dataset in model_dataset_pairs:
        print(f"\nSetting up sweep for {model_type} on {dataset}")
        
        # Create a sweep configuration
        sweep_config = create_sweep_config(model_type, dataset)
        
        # Initialize the sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"vadam-optimization-{model_type}-{dataset}"
        )
        
        sweep_ids.append((model_type, dataset, sweep_id))
        print(f"Sweep created for {model_type} on {dataset} with ID: {sweep_id}")
        
        # Create a directory for saving sweep results
        os.makedirs("sweep_results", exist_ok=True)
        
        # Save sweep configuration
        with open(f"sweep_results/sweep_config_{model_type}_{dataset}.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)
            
        # Run the sweep with agent
        wandb.agent(sweep_id, function=train_model, count=10)  # Run 10 experiments per model/dataset
    
    # Print summary of all sweeps
    print("\nSummary of all sweeps:")
    for model_type, dataset, sweep_id in sweep_ids:
        print(f"- {model_type} on {dataset}: {sweep_id}")
    
    # Save sweep IDs for future reference
    with open("sweep_results/sweep_ids.json", 'w') as f:
        json.dump([(m, d, s) for m, d, s in sweep_ids], f, indent=2)

if __name__ == "__main__":
    run_sweeps() 