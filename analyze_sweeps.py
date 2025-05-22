import wandb
import pandas as pd
import json
import os
import argparse
from pprint import pprint

def extract_best_config(api, sweep_id, project_name):
    """Extract the best configuration from a completed sweep"""
    # Get sweep object
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    
    # Get all runs in this sweep
    runs = list(sweep.runs)
    
    if not runs:
        return None, "No runs found"
    
    # Find the best run based on the optimization metric
    if any("test_perplexity" in run.summary for run in runs if run.summary):
        # For language modeling tasks (lower perplexity is better)
        best_run = min(
            [run for run in runs if run.summary and "test_perplexity" in run.summary],
            key=lambda x: x.summary.get("test_perplexity", float('inf'))
        )
        metric_name = "test_perplexity"
        metric_value = best_run.summary.get("test_perplexity")
        
    else:
        # For classification tasks (higher accuracy is better)
        best_run = max(
            [run for run in runs if run.summary and "test_acc" in run.summary],
            key=lambda x: x.summary.get("test_acc", 0)
        )
        metric_name = "test_acc"
        metric_value = best_run.summary.get("test_acc")
    
    # Get the best configuration
    config = {k: v for k, v in best_run.config.items()}
    
    # Add result metrics
    result_metrics = {
        "run_id": best_run.id,
        "url": best_run.url,
        metric_name: metric_value
    }
    
    if "test_loss" in best_run.summary:
        result_metrics["test_loss"] = best_run.summary.get("test_loss")
    
    if "train_time" in best_run.summary:
        result_metrics["train_time"] = best_run.summary.get("train_time")
    
    return config, result_metrics

def analyze_all_sweeps(sweep_ids_file=None):
    """Analyze all sweeps based on a file with sweep IDs or scans all sweep_config files"""
    api = wandb.Api()
    
    # If sweep IDs file is provided, load it
    if sweep_ids_file and os.path.exists(sweep_ids_file):
        with open(sweep_ids_file, 'r') as f:
            sweep_info = json.load(f)
    else:
        # Otherwise scan the sweep_results directory for config files
        sweep_info = []
        sweep_dir = "sweep_results"
        if os.path.exists(sweep_dir):
            for filename in os.listdir(sweep_dir):
                if filename.startswith("sweep_config_") and filename.endswith(".json"):
                    parts = filename.replace("sweep_config_", "").replace(".json", "").split("_")
                    if len(parts) >= 2:
                        model_type = parts[0]
                        dataset = parts[1]
                        
                        # Look for a sweep ID file for this config
                        id_file = os.path.join(sweep_dir, f"sweep_id_{model_type}_{dataset}.txt")
                        if os.path.exists(id_file):
                            with open(id_file, 'r') as f:
                                sweep_id = f.read().strip()
                                sweep_info.append((model_type, dataset, sweep_id))
    
    if not sweep_info:
        print("No sweep information found. Please provide a sweep IDs file or ensure sweep_results directory contains configs.")
        return
    
    best_configs = {}
    
    for model_type, dataset, sweep_id in sweep_info:
        project_name = f"vradam-optimization-{model_type}-{dataset}"
        print(f"\nAnalyzing sweep for {model_type} on {dataset} (ID: {sweep_id})")
        
        try:
            config, metrics = extract_best_config(api, sweep_id, project_name)
            
            if config:
                best_configs[f"{model_type}_{dataset}"] = {
                    "model": model_type,
                    "dataset": dataset,
                    "sweep_id": sweep_id,
                    "best_config": config,
                    "metrics": metrics
                }
                
                print(f"Best configuration found with {metrics}:")
                # Print only VRADAM relevant parameters
                vradam_params = {k: v for k, v in config.items() 
                               if k in ['eta', 'beta1', 'beta2', 'beta3', 'power', 
                                        'normgrad', 'lr_cutoff', 'weight_decay', 'eps']}
                pprint(vradam_params)
            else:
                print(f"Could not extract best configuration: {metrics}")
        except Exception as e:
            print(f"Error analyzing sweep {sweep_id}: {e}")
    
    # Save all best configurations to a file
    if best_configs:
        output_file = "sweep_results/best_configs.json"
        with open(output_file, 'w') as f:
            json.dump(best_configs, f, indent=2)
        print(f"\nBest configurations saved to {output_file}")
        
        # Create a markdown report
        create_markdown_report(best_configs, "sweep_results/optimization_report.md")

def create_markdown_report(best_configs, output_file):
    """Create a markdown report with the best configurations"""
    with open(output_file, 'w') as f:
        f.write("# VRADAM Hyperparameter Optimization Report\n\n")
        f.write("## Best Configurations by Model and Dataset\n\n")
        
        for config_name, config_data in best_configs.items():
            model = config_data["model"]
            dataset = config_data["dataset"]
            metrics = config_data["metrics"]
            best_config = config_data["best_config"]
            
            f.write(f"### {model} on {dataset}\n\n")
            
            # Metrics section
            f.write("#### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric_name, metric_value in metrics.items():
                if metric_name not in ["run_id", "url"]:
                    if isinstance(metric_value, float):
                        f.write(f"| {metric_name} | {metric_value:.6f} |\n")
                    else:
                        f.write(f"| {metric_name} | {metric_value} |\n")
            f.write("\n")
            
            # Best parameters section
            f.write("#### Best VRADAM Parameters\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for param_name, param_value in sorted(best_config.items()):
                if param_name in ['eta', 'beta1', 'beta2', 'beta3', 'power', 
                                'normgrad', 'lr_cutoff', 'weight_decay', 'eps']:
                    f.write(f"| {param_name} | {param_value} |\n")
            f.write("\n")
            
            # Other parameters
            f.write("#### Other Parameters\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for param_name, param_value in sorted(best_config.items()):
                if param_name not in ['eta', 'beta1', 'beta2', 'beta3', 'power', 
                                     'normgrad', 'lr_cutoff', 'weight_decay', 'eps',
                                     'model', 'dataset', 'optimizer']:
                    f.write(f"| {param_name} | {param_value} |\n")
            f.write("\n")
            
            # Link to W&B
            f.write(f"[View Run on W&B]({metrics['url']})\n\n")
            f.write("---\n\n")
        
        f.write("## How to Use These Parameters\n\n")
        f.write("To use these optimized parameters in your benchmarks, update your configuration in `run_benchmarks.py`:\n\n")
        f.write("```python\n")
        f.write("# Example for using optimized parameters\n")
        f.write("vradam_params.update({\n")
        f.write("    'eta': <optimized_value>,\n")
        f.write("    'beta1': <optimized_value>,\n")
        f.write("    'beta2': <optimized_value>,\n")
        f.write("    'beta3': <optimized_value>,\n")
        f.write("    'power': <optimized_value>,\n")
        f.write("    'normgrad': <optimized_value>,\n")
        f.write("    'lr_cutoff': <optimized_value>,\n")
        f.write("    'weight_decay': <optimized_value>,\n")
        f.write("    'eps': <optimized_value>\n")
        f.write("})\n")
        f.write("```\n")
        
    print(f"Markdown report saved to {output_file}")

def generate_run_commands(best_configs_file):
    """Generate run commands with optimized hyperparameters"""
    if not os.path.exists(best_configs_file):
        print(f"Best configs file not found: {best_configs_file}")
        return
    
    with open(best_configs_file, 'r') as f:
        best_configs = json.load(f)
    
    output_file = "sweep_results/optimized_run_commands.sh"
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Run commands with optimized hyperparameters\n\n")
        
        for config_name, config_data in best_configs.items():
            model = config_data["model"]
            dataset = config_data["dataset"]
            best_config = config_data["best_config"]
            
            # Extract VRADAM parameters
            vradam_params = {k: v for k, v in best_config.items() 
                           if k in ['eta', 'beta1', 'beta2', 'beta3', 'power', 
                                    'normgrad', 'lr_cutoff', 'weight_decay', 'eps']}
            
            # Generate python command
            cmd = f"python run_optimized_benchmark.py --model {model} --dataset {dataset} "
            
            for param, value in vradam_params.items():
                if isinstance(value, bool):
                    if value:
                        cmd += f"--{param} "
                else:
                    cmd += f"--{param} {value} "
            
            f.write(f"# Optimized parameters for {model} on {dataset}\n")
            f.write(f"{cmd}\n\n")
    
    print(f"Optimized run commands saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze W&B sweep results for VRADAM optimization")
    parser.add_argument("--sweep_ids", type=str, help="Path to JSON file with sweep IDs")
    parser.add_argument("--generate_commands", action="store_true", 
                      help="Generate run commands with optimized hyperparameters")
    
    args = parser.parse_args()
    
    # Ensure wandb is logged in
    wandb.login()
    
    # Analyze all sweeps
    analyze_all_sweeps(args.sweep_ids)
    
    # Generate commands if requested
    if args.generate_commands:
        generate_run_commands("sweep_results/best_configs.json") 