import wandb
import torch
import os
import time
import json
from benchmarker import Benchmarker

def train_model(config=None):
    """
    Train RL model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Set up parameters for benchmarker
        params = {
            'model': 'PPOPolicy',  # Use PPO for better performance
            'dataset': 'HalfCheetah',  # RL environment
            'dataset_size': config.dataset_size,
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimizer': config.optimizer_name,  # 'VADAM' or 'ADAM'
            'batch_size': config.batch_size,
            'hidden_dim': config.hidden_dim,
            'epochs': config.epochs,
            
            # RL specific parameters
            'gamma': config.gamma,  # Discount factor
            'entropy_coef': config.entropy_coef,  # Entropy coefficient for exploration
            'eps_clip': config.eps_clip,  # PPO clipping parameter
        }

        # Add optimizer specific parameters
        if config.optimizer_name == 'VADAM':
            params['eta'] = config.eta  # VADAM's learning rate
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
        
        # Create and run benchmarker
        print(f"Running RL benchmark with params: {params}")
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Get training and test rewards
        last_train_reward = results.get('mean_rewards', [])[-1] if results.get('mean_rewards', []) else 0
        test_reward = results.get('test_acc', 0)  # In RL, 'test_acc' contains the test reward
        
        # Log metrics to wandb for RL tasks
        wandb.log({
            "final_train_loss": results.get('final_train_loss', 0),
            "train_reward": last_train_reward,
            "test_reward": test_reward,
            "train_time": results.get('train_time', 0)
        })
        
        # Track training curves
        for epoch, reward in enumerate(results.get('mean_rewards', [])):
            wandb.log({
                "epoch": epoch,
                "mean_reward": reward
            })
        
        # For RL, we want to maximize reward, but W&B minimizes by default
        # so we negate the test reward (not the training reward)
        optimization_metric = -test_reward
        wandb.log({"optimization_metric": optimization_metric})
        
        return results

def create_vadam_sweep_config():
    """Create sweep configuration for VADAM optimizer for RL tasks."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Note: This is negated test reward, so minimizing means maximizing reward
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'VADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 20},  # Fewer epochs for RL
            'batch_size': {'value': 64},
            
            # RL specific parameters
            'gamma': {'value': 0.99},  # Discount factor
            'entropy_coef': {'distribution': 'uniform', 'min': 0.001, 'max': 0.05},
            'eps_clip': {'value': 0.2},  # PPO clipping parameter
            'hidden_dim': {'value': 128},
            
            # VADAM parameters to optimize
            'eta': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.001},  # Base LR for VADAM
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'beta3': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            'power': {'value': 2},
            'normgrad': {'values': [True, False]},
            'lr_cutoff': {'distribution': 'int_uniform', 'min': 5, 'max': 30},
            'weight_decay': {'value': 1e-5},
            'eps': {'value': 1e-8},
        }
    }
    
    return sweep_config

def create_adam_sweep_config():
    """Create sweep configuration for ADAM optimizer for RL tasks."""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'optimization_metric',
            'goal': 'minimize'  # Note: This is negated test reward, so minimizing means maximizing reward
        },
        'parameters': {
            # Optimizer Type
            'optimizer_name': {'value': 'ADAM'},
            
            # Fixed parameters
            'dataset_size': {'value': 'full'},
            'epochs': {'value': 20},  # Fewer epochs for RL
            'batch_size': {'value': 64},
            
            # RL specific parameters
            'gamma': {'value': 0.99},  # Discount factor
            'entropy_coef': {'distribution': 'uniform', 'min': 0.001, 'max': 0.05},
            'eps_clip': {'value': 0.2},  # PPO clipping parameter
            'hidden_dim': {'value': 128},
            
            # ADAM specific parameters to optimize (only LR)
            'adam_lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 0.001},  # Base LR for ADAM
            'adam_beta1': {'value': 0.9},
            'adam_beta2': {'value': 0.999},
            'adam_weight_decay': {'value': 1e-5},
            'adam_eps': {'value': 1e-8},
        }
    }
    
    return sweep_config 