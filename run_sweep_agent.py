import argparse
import wandb
import json
import os
from sweep_vadam import train_model, create_sweep_config

def run_sweep_agent(sweep_id=None, model_type=None, dataset=None, count=10):
    """
    Run a sweep agent for a specific sweep ID or create a new sweep
    
    Args:
        sweep_id: Existing sweep ID (optional)
        model_type: Model type if creating a new sweep (SimpleCNN, MLPModel, TransformerModel)
        dataset: Dataset if creating a new sweep (CIFAR10, IMDB, WikiText2)
        count: Number of runs to perform
    """
    # Ensure wandb is logged in
    wandb.login()
    
    # If no sweep_id is provided, create a new sweep
    if sweep_id is None:
        if model_type is None or dataset is None:
            raise ValueError("If sweep_id is not provided, model_type and dataset must be specified")
        
        # Create a sweep configuration
        print(f"Creating new sweep for {model_type} on {dataset}")
        sweep_config = create_sweep_config(model_type, dataset)
        
        # Initialize the sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"vadam-optimization-{model_type}-{dataset}"
        )
        
        # Save sweep configuration and ID
        os.makedirs("sweep_results", exist_ok=True)
        with open(f"sweep_results/sweep_config_{model_type}_{dataset}.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)
            
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Using existing sweep ID: {sweep_id}")
    
    # Run the sweep agent
    print(f"Starting sweep agent to run {count} experiments")
    wandb.agent(sweep_id, function=train_model, count=count)
    
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent for VADAM optimization")
    parser.add_argument("--sweep_id", type=str, help="Existing sweep ID (optional)")
    parser.add_argument("--model", type=str, choices=["SimpleCNN", "MLPModel", "TransformerModel"], 
                      help="Model type if creating a new sweep")
    parser.add_argument("--dataset", type=str, choices=["CIFAR10", "IMDB", "WikiText2"], 
                      help="Dataset if creating a new sweep")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to perform")
    
    args = parser.parse_args()
    
    # Run the sweep agent
    sweep_id = run_sweep_agent(
        sweep_id=args.sweep_id,
        model_type=args.model,
        dataset=args.dataset,
        count=args.count
    ) 