import argparse
import wandb
import json
import os
# Import the specific config creators and the generalized train_model
from sweep_vadam import train_model, create_vadam_sweep_config, create_adam_sweep_config

def run_sweep_agent(optimizer_name, model_type, dataset, count=10):
    """
    Create and run a new W&B sweep agent for a specific optimizer, model, and dataset.
    
    Args:
        optimizer_name: Name of the optimizer ('VADAM' or 'ADAM')
        model_type: Model type (SimpleCNN, MLPModel, TransformerModel)
        dataset: Dataset (CIFAR10, IMDB, WikiText2)
        count: Number of runs to perform
    """
    # Ensure wandb is logged in
    wandb.login()
    # wandb.init(
    #     entity="team-nobu",   # ← your W&B organization / team
    # )
    
    # Select the appropriate config function and set project name
    if optimizer_name == 'VADAM':
        print(f"Creating new VADAM sweep for {model_type} on {dataset}")
        sweep_config = create_vadam_sweep_config(model_type, dataset)
        project_name = f"VADAM-optimization-{model_type}-{dataset}"
    elif optimizer_name == 'ADAM':
        print(f"Creating new ADAM sweep for {model_type} on {dataset}")
        sweep_config = create_adam_sweep_config(model_type, dataset)
        project_name = f"ADAM-optimization-{model_type}-{dataset}"
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}. Choose 'VADAM' or 'ADAM'.")

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name, 
    )
    
    # Save sweep configuration and ID locally (optional, but good practice)
    results_dir = f"sweep_results_{optimizer_name}"
    os.makedirs(results_dir, exist_ok=True)
    config_path = f"{results_dir}/sweep_config_{model_type}_{dataset}.json"
    with open(config_path, 'w') as f:
        json.dump(sweep_config, f, indent=2)
        
    print(f"Created new sweep with ID: {sweep_id} in project '{project_name}'")
    print(f"Sweep config saved to: {config_path}")
    
    # Run the sweep agent using the generalized train_model
    print(f"Starting sweep agent for {count} experiments (Sweep ID: {sweep_id}) ...")
    # Pass the generalized train_model function to the agent
    wandb.agent(sweep_id, function=train_model, count=count, entity="team-nobu")
    print(f"Sweep agent finished for sweep ID: {sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent for VADAM or ADAM optimization")
    # Removed --sweep_id, added required --optimizer_name
    parser.add_argument("--optimizer_name", type=str, required=True, choices=["VADAM", "ADAM"],
                      help="Optimizer to run the sweep for ('VADAM' or 'ADAM')")
    parser.add_argument("--model", type=str, required=True,
                      choices=["SimpleCNN", "DeeperCNN", "MLPModel", "TransformerModel"],
                      help="Model type for the sweep")
    parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR10", "IMDB", "WikiText2"], 
                      help="Dataset for the sweep")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to perform")
    
    args = parser.parse_args()
    
    # Run the sweep agent
    sweep_id = run_sweep_agent(
        optimizer_name=args.optimizer_name,
        model_type=args.model,
        dataset=args.dataset,
        count=args.count
    ) 