import argparse
import wandb
import json
import os
import sys

# Add import paths for all benchmark modules
from cnn_image_classification.sweep_cnn import train_model as cnn_train_model
from cnn_image_classification.sweep_cnn import create_sweep_config as cnn_create_sweep_config

from transformer_language_modeling.sweep_transformer import train_model as transformer_train_model
from transformer_language_modeling.sweep_transformer import create_sweep_config as transformer_create_sweep_config

from rl_halfcheetah.sweep_rl import train_model as rl_train_model
from rl_halfcheetah.sweep_rl import create_sweep_config as rl_create_sweep_config

from diffusion_mnist.sweep_diffusion import train_model as diffusion_train_model
from diffusion_mnist.sweep_diffusion import create_sweep_config as diffusion_create_sweep_config

def run_sweep_agent(sweep_id=None, model_type=None, dataset=None, optimizer=None, count=10):
    """
    Run a sweep agent for a specific sweep ID or create a new sweep
    
    Args:
        sweep_id: Existing sweep ID (optional)
        model_type: Model type if creating a new sweep ('CNN', 'Transformer', 'RL', 'Diffusion')
        dataset: Dataset if creating a new sweep (for information only)
        optimizer: Optimizer to use ('adam', 'vadam', or 'both')
        count: Number of runs to perform
    """
    # Ensure wandb is logged in
    wandb.login()
    
    # If no sweep_id is provided, create a new sweep
    if sweep_id is None:
        if model_type is None:
            raise ValueError("If sweep_id is not provided, model_type must be specified")
        
        # Normalize optimizer name for consistency
        if optimizer:
            optimizer = optimizer.upper() if optimizer.upper() in ['ADAM', 'VADAM'] else optimizer.lower()
        else:
            optimizer = 'both'
            
        # Choose appropriate train_model function and create sweep config
        if model_type.upper() == 'CNN':
            train_model = cnn_train_model
            model_str = 'CNN'
            dataset_str = 'CIFAR10'
            sweep_config = cnn_create_sweep_config(optimizer.upper())
        elif model_type.upper() == 'TRANSFORMER':
            train_model = transformer_train_model
            model_str = 'Transformer'
            dataset_str = 'WikiText2'
            sweep_config = transformer_create_sweep_config(optimizer.upper())
        elif model_type.upper() == 'RL':
            train_model = rl_train_model
            model_str = 'RL'
            dataset_str = 'HalfCheetah'
            sweep_config = rl_create_sweep_config(optimizer.upper())
        elif model_type.upper() == 'DIFFUSION':
            train_model = diffusion_train_model
            model_str = 'Diffusion'
            dataset_str = 'MNIST'
            sweep_config = diffusion_create_sweep_config(optimizer.upper())
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create a sweep configuration
        print(f"Creating new sweep for {model_str} on {dataset_str} with {optimizer} optimizer")
        
        # Initialize the sweep
        project_name = f"{model_str.lower()}-optimization-{optimizer}"
        sweep_id = wandb.sweep(
            sweep_config, 
            project=project_name
        )
        
        # Save sweep configuration and ID
        os.makedirs("sweep_results", exist_ok=True)
        with open(f"sweep_results/sweep_config_{model_str}_{optimizer}.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)
            
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Using existing sweep ID: {sweep_id}")
        
        # Determine model type from sweep_id if possible, otherwise ask the user
        if model_type is None:
            raise ValueError("When using an existing sweep_id, you must specify the model_type")
        
        # Choose appropriate train_model function
        if model_type.upper() == 'CNN':
            train_model = cnn_train_model
        elif model_type.upper() == 'TRANSFORMER':
            train_model = transformer_train_model
        elif model_type.upper() == 'RL':
            train_model = rl_train_model
        elif model_type.upper() == 'DIFFUSION':
            train_model = diffusion_train_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Run the sweep agent
    print(f"Starting sweep agent to run {count} experiments")
    wandb.agent(sweep_id, function=train_model, count=count)
    
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent for optimization")
    parser.add_argument("--sweep_id", type=str, help="Existing sweep ID (optional)")
    parser.add_argument("--model", type=str, choices=["CNN", "Transformer", "RL", "Diffusion"], 
                      help="Model type if creating a new sweep")
    parser.add_argument("--optimizer", type=str, choices=["adam", "vadam", "both"], default="both",
                      help="Optimizer to use (default: both)")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to perform")
    
    args = parser.parse_args()
    
    # Run the sweep agent
    sweep_id = run_sweep_agent(
        sweep_id=args.sweep_id,
        model_type=args.model,
        optimizer=args.optimizer,
        count=args.count
    ) 