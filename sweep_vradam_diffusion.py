import wandb
import torch
import os
import time
import json
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train diffusion model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': 'DiffusionModel',  # For diffusion tasks
            'dataset': 'MNIST',  # MNIST dataset for diffusion
            'dataset_size': config.dataset_size,
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimizer': config.optimizer_name,  # 'VRADAM', 'ADAM', 'SGD', or 'RMSPROP'
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            
            # Diffusion specific parameters
            'unet_base_channels': config.unet_base_channels,
            'unet_time_embed_dim': config.unet_time_embed_dim,
            'num_timesteps': config.num_timesteps,
            'beta_min': config.beta_min,
            'beta_max': config.beta_max,
            'use_attention': config.use_attention,
            'sample_every': config.sample_every,
            
            # Set fixed seed for consistent initialization
            'seed': config.seed
        }

        # Add optimizer specific parameters
        if config.optimizer_name == 'VRADAM':
            params['eta'] = config.eta  # VRADAM's learning rate
            params['beta1'] = config.beta1
            params['beta2'] = config.beta2
            params['beta3'] = config.beta3
            params['power'] = config.power
            params['normgrad'] = config.normgrad
            params['lr_cutoff'] = config.lr_cutoff
            params['weight_decay'] = config.weight_decay
            params['eps'] = config.eps
        elif config.optimizer_name == 'ADAM':
            params['lr'] = config.adam_lr  # Adam's learning rate
            params['beta1'] = config.adam_beta1
            params['beta2'] = config.adam_beta2
            params['weight_decay'] = config.adam_weight_decay
            params['eps'] = config.adam_eps
        elif config.optimizer_name == 'SGD':
            params['lr'] = config.sgd_lr  # SGD's learning rate
            params['momentum'] = config.sgd_momentum
            params['weight_decay'] = config.sgd_weight_decay
            params['nesterov'] = config.sgd_nesterov
        elif config.optimizer_name == 'RMSPROP':
            params['lr'] = config.rmsprop_lr  # RMSprop's learning rate
            params['alpha'] = config.rmsprop_alpha
            params['eps'] = config.rmsprop_eps
            params['weight_decay'] = config.rmsprop_weight_decay
            params['momentum'] = config.rmsprop_momentum
        
        # Create and run benchmarker
        print(f"Running diffusion benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Log metrics to wandb
        wandb.log({
            "final_train_loss": results.get('final_train_loss', 0),
            "val_loss": results.get('val_losses', [])[-1] if results.get('val_losses', []) else float('inf'),
            "test_loss": results.get('test_loss', 0),
            "train_time": results.get('train_time', 0)
        })
        
        # Track training and validation curves
        for epoch in range(len(results.get('train_losses', []))):
            metrics = {
                "epoch": epoch,
                "train_loss": results['train_losses'][epoch],
            }
            
            # Add validation metrics if available
            if epoch < len(results.get('val_losses', [])):
                metrics["val_loss"] = results['val_losses'][epoch]
            
            # Add generated samples if available
            if epoch % config.sample_every == 0 and 'samples' in results:
                metrics["samples"] = results['samples'][epoch]
            
            wandb.log(metrics)
        
        # Use validation loss as the optimization metric
        optimization_metric = results.get('val_losses', [])[-1] if results.get('val_losses', []) else float('inf')
        wandb.log({"optimization_metric": optimization_metric})
        
        return results

def create_vradam_sweep_config():
    """Create sweep configuration for VRADAM optimizer for diffusion model."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'VRADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 128},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # Diffusion specific parameters
            'unet_base_channels': {'value': 96},
            'unet_time_embed_dim': {'value': 128},
            'num_timesteps': {'value': 400},
            'beta_min': {'value': 1e-4},
            'beta_max': {'value': 0.02},
            'use_attention': {'value': True},
            'sample_every': {'value': 5},
            
            # VRADAM parameters to optimize
            'eta': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'beta3': {'distribution': 'uniform', 'min': 0.1, 'max': 2.0},
            'power': {'value': 2},
            'normgrad': {'values': [True, False]},
            'lr_cutoff': {'distribution': 'int_uniform', 'min': 5, 'max': 30},
            'weight_decay': {'value': 1e-5},
            'eps': {'value': 1e-8},
        }
    }
    
    return sweep_config

def create_adam_sweep_config():
    """Create sweep configuration for ADAM optimizer for diffusion model."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'ADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 128},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # Diffusion specific parameters
            'unet_base_channels': {'value': 96},
            'unet_time_embed_dim': {'value': 128},
            'num_timesteps': {'value': 400},
            'beta_min': {'value': 1e-4},
            'beta_max': {'value': 0.02},
            'use_attention': {'value': True},
            'sample_every': {'value': 5},
            
            # ADAM specific parameters to optimize
            'adam_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
            'adam_beta1': {'value': 0.9},
            'adam_beta2': {'value': 0.999},
            'adam_weight_decay': {'value': 1e-5},
            'adam_eps': {'value': 1e-8},
        }
    }
    
    return sweep_config

def create_sgd_sweep_config():
    """Create sweep configuration for SGD optimizer for diffusion model."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'SGD'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 128},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # Diffusion specific parameters
            'unet_base_channels': {'value': 96},
            'unet_time_embed_dim': {'value': 128},
            'num_timesteps': {'value': 400},
            'beta_min': {'value': 1e-4},
            'beta_max': {'value': 0.02},
            'use_attention': {'value': True},
            'sample_every': {'value': 5},
            
            # SGD specific parameters to optimize
            'sgd_lr': {'distribution': 'log_uniform_values', 'min': 1e-3, 'max': 1e-1},
            'sgd_momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.99},
            'sgd_weight_decay': {'value': 1e-5},
            'sgd_nesterov': {'values': [True, False]},
        }
    }
    
    return sweep_config

def create_rmsprop_sweep_config():
    """Create sweep configuration for RMSprop optimizer for diffusion model."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Lower validation loss is better
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'RMSPROP'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 50},
            'batch_size': {'value': 128},
            
            # Reproducibility
            'seed': {'value': 42},
            
            # Diffusion specific parameters
            'unet_base_channels': {'value': 96},
            'unet_time_embed_dim': {'value': 128},
            'num_timesteps': {'value': 400},
            'beta_min': {'value': 1e-4},
            'beta_max': {'value': 0.02},
            'use_attention': {'value': True},
            'sample_every': {'value': 5},
            
            # RMSprop specific parameters to optimize
            'rmsprop_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
            'rmsprop_alpha': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'rmsprop_eps': {'value': 1e-8},
            'rmsprop_weight_decay': {'value': 1e-5},
            'rmsprop_momentum': {'distribution': 'uniform', 'min': 0.0, 'max': 0.9},
        }
    }
    
    return sweep_config 