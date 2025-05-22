import wandb
import torch
import os
import json
import sys
import argparse

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train CNN model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': 'SimpleCNN',
            'dataset': 'CIFAR10',
            'dataset_size': config.dataset_size,
            'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
            'optimizer': config.optimizer,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            
            # Optimizer parameters based on which optimizer is used
            'lr': config.lr if config.optimizer == 'ADAM' else None,
            'eta': config.lr if config.optimizer == 'VRADAM' else None,
            'beta1': config.beta1,
            'beta2': config.beta2,
            'eps': config.eps,
            'weight_decay': config.weight_decay,
        }
        
        # Add VRADAM specific parameters if needed
        if config.optimizer == 'VRADAM':
            params.update({
                'beta3': config.beta3 if hasattr(config, 'beta3') else 1.0,
                'power': config.power if hasattr(config, 'power') else 2, 
                'normgrad': config.normgrad if hasattr(config, 'normgrad') else True,
                'lr_cutoff': config.lr_cutoff if hasattr(config, 'lr_cutoff') else 19
            })
        
        # Create and run benchmarker
        print(f"Running benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Log metrics to wandb
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
        
        # For classification, higher accuracy is better, but W&B minimizes by default
        # so we negate the accuracy
        optimization_metric = -results['test_acc']
        wandb.log({"optimization_metric": optimization_metric})
        
        return results

def create_sweep_config(optimizer_type):
    """Create sweep configuration for CNN with specified optimizer type"""
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Using negative accuracy
        },
        'parameters': {
            # Fixed parameters
            'optimizer': {'value': optimizer_type},
            'dataset_size': {'value': 'small'},  # Use small dataset for faster iterations
            'epochs': {'value': 3},  # Keep epochs low for faster iterations
            
            # Common parameters for all optimizers
            'batch_size': {'values': [16, 32, 64, 128]},
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'eps': {'value': 1e-8},
            'weight_decay': {'value': 0.0},
        }
    }
    
    # Add optimizer-specific parameters
    if optimizer_type == 'ADAM':
        # For Adam, we only sweep learning rate
        sweep_config['parameters']['lr'] = {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        }
    else:  # VRADAM
        # For VRADAM, we sweep learning rate, beta3, and lr_cutoff
        sweep_config['parameters'].update({
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'beta3': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 2.0
            },
            'lr_cutoff': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 30
            },
            'power': {'value': 2},
            'normgrad': {'value': False}
        })
    
    return sweep_config

def run_sweeps(optimizer_type=None, count=10):
    """Run sweeps for CNN with specified optimizer"""
    # Set up wandb project
    wandb.login()
    
    if optimizer_type is None or optimizer_type.upper() == 'BOTH':
        optimizers = ['ADAM', 'VRADAM']
    else:
        optimizers = [optimizer_type.upper()]
    
    sweep_ids = []
    for opt in optimizers:
        print(f"\nSetting up sweep for CNN on CIFAR10 with {opt} optimizer")
        
        # Create a sweep configuration
        sweep_config = create_sweep_config(opt)
        
        # Initialize the sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"cnn-optimization-{opt}"
        )
        
        sweep_ids.append((opt, sweep_id))
        print(f"Sweep created with ID: {sweep_id}")
        
        # Create a directory for saving sweep results
        os.makedirs("../../sweep_results", exist_ok=True)
        
        # Save sweep configuration
        with open(f"../../sweep_results/sweep_config_CNN_{opt}.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)
            
        # Run the sweep with agent
        wandb.agent(sweep_id, function=train_model, count=count)
    
    # Print summary of all sweeps
    print("\nSummary of sweeps:")
    for opt, sweep_id in sweep_ids:
        print(f"- {opt}: {sweep_id}")
    
    # Save sweep IDs for future reference
    with open("../../sweep_results/cnn_sweep_ids.json", 'w') as f:
        json.dump([(o, s) for o, s in sweep_ids], f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run W&B sweeps for CNN image classification")
    parser.add_argument("--optimizer", type=str, choices=["adam", "vradam", "both"], default="both",
                       help="Which optimizer to sweep (default: both)")
    parser.add_argument("--count", type=int, default=10, help="Number of runs per sweep (default: 10)")
    
    args = parser.parse_args()
    
    run_sweeps(args.optimizer, args.count) 