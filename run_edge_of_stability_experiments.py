#!/usr/bin/env python
# Run edge of stability experiments for VADAM

import os
import argparse
import json
import datetime
import subprocess
from itertools import product

def run_experiment(experiment_config, output_dir):
    """
    Run a single edge of stability experiment with the given configuration
    
    Args:
        experiment_config: Dictionary with experiment configuration
        output_dir: Directory to save results
    """
    # Build command
    cmd = ["python", "edge_of_stability_analysis.py"]
    
    # Add arguments - skip experiment_id since it's not a valid parameter for the script
    for key, value in experiment_config.items():
        # Skip experiment_id as it's only used internally
        if key == 'experiment_id':
            continue
        
        if key == 'eta_values':
            # Handle list of floats
            cmd.append(f"--{key}")
            for v in value:
                cmd.append(str(v))
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Add output directory
    cmd.append("--output_dir")
    experiment_id = experiment_config.get('experiment_id', 'unknown')
    plots_dir = os.path.join(output_dir, f"plots_{experiment_id}")
    os.makedirs(plots_dir, exist_ok=True)
    cmd.append(plots_dir)
    
    # Add experiment ID to the command for logging
    print(f"\n{'='*80}")
    print(f"Running experiment {experiment_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the experiment
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save output
    output_file = os.path.join(output_dir, f"experiment_{experiment_id}_output.txt")
    with open(output_file, 'w') as f:
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n\n")
    
    # Return success/failure
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Run Edge of Stability experiments')
    parser.add_argument('--experiments', type=str, default='eos_experiments.json',
                      help='JSON file with experiment configurations')
    parser.add_argument('--output_dir', type=str, default='eos_results',
                      help='Directory to save results')
    parser.add_argument('--run_all', action='store_true',
                      help='Run all predefined experiments instead of using config file')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.run_all:
        # Predefined experiments
        
        # Experiment 1: Basic exploration with varying eta values
        basic_config = {
            'experiment_id': 'basic',
            'eta_values': [0.001, 0.005, 0.01, 0.05],
            'beta1': 0.9,
            'beta2': 0.999,
            'beta3': 1.0,
            'num_steps': 100,
            'seed': 42
        }
        
        # Experiment 2: Impact of beta3 parameter
        beta3_configs = []
        for beta3 in [0.5, 1.0, 2.0, 5.0]:
            beta3_configs.append({
                'experiment_id': f'beta3_{beta3}',
                'eta_values': [0.01],
                'beta1': 0.9,
                'beta2': 0.999,
                'beta3': beta3,
                'num_steps': 100,
                'seed': 42
            })
            
        # Experiment 3: Impact of beta1 parameter
        beta1_configs = []
        for beta1 in [0.7, 0.8, 0.9, 0.95]:
            beta1_configs.append({
                'experiment_id': f'beta1_{beta1}',
                'eta_values': [0.01],
                'beta1': beta1,
                'beta2': 0.999,
                'beta3': 1.0,
                'num_steps': 100,
                'seed': 42
            })
            
        # Experiment 4: Exploring normgrad settings
        normgrad_configs = [
            {
                'experiment_id': 'normgrad_true',
                'eta_values': [0.01],
                'beta1': 0.9,
                'beta2': 0.999,
                'beta3': 1.0,
                'num_steps': 100,
                'seed': 42,
                'normgrad': True
            },
            {
                'experiment_id': 'normgrad_false',
                'eta_values': [0.01],
                'beta1': 0.9,
                'beta2': 0.999,
                'beta3': 1.0,
                'num_steps': 100,
                'seed': 42,
                'normgrad': False
            }
        ]
        
        # Experiment 5: Long run to see convergence behavior
        long_run_config = {
            'experiment_id': 'long_run',
            'eta_values': [0.01],
            'beta1': 0.9,
            'beta2': 0.999,
            'beta3': 1.0,
            'num_steps': 500,
            'seed': 42
        }
        
        # Combine all experiments
        experiments = [basic_config] + beta3_configs + beta1_configs + normgrad_configs + [long_run_config]
        
    else:
        # Load experiments from file
        with open(args.experiments, 'r') as f:
            experiments = json.load(f)
    
    # Save experiment configurations to output directory
    with open(os.path.join(output_dir, 'experiment_configs.json'), 'w') as f:
        json.dump(experiments, f, indent=2)
    
    # Run experiments
    results = {}
    for i, config in enumerate(experiments):
        experiment_id = config.get('experiment_id', f'experiment_{i}')
        print(f"Running experiment {experiment_id} ({i+1}/{len(experiments)})")
        
        success = run_experiment(config, output_dir)
        results[experiment_id] = {
            'config': config,
            'success': success
        }
    
    # Save results summary
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report_path = os.path.join(output_dir, 'experiment_report.md')
    with open(report_path, 'w') as f:
        f.write("# Edge of Stability Experiments Report\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## Experiment Summary\n\n")
        f.write("| Experiment ID | Status | Beta1 | Beta3 | Eta Values | Steps | Plots |\n")
        f.write("|--------------|--------|-------|-------|------------|-------|-------|\n")
        
        for exp_id, result in results.items():
            config = result['config']
            status = "✅ Success" if result['success'] else "❌ Failed"
            beta1 = config.get('beta1', 'N/A')
            beta3 = config.get('beta3', 'N/A')
            eta_values = config.get('eta_values', 'N/A')
            steps = config.get('num_steps', 'N/A')
            plots_dir = f"plots_{exp_id}"
            
            f.write(f"| {exp_id} | {status} | {beta1} | {beta3} | {eta_values} | {steps} | [{plots_dir}]({plots_dir}) |\n")
        
        f.write("\n## Experiment Details\n\n")
        
        for exp_id, result in results.items():
            config = result['config']
            status = "Success" if result['success'] else "Failed"
            plots_dir = f"plots_{exp_id}"
            
            f.write(f"### Experiment: {exp_id}\n\n")
            f.write(f"**Status**: {status}\n\n")
            f.write("**Configuration**:\n")
            
            for key, value in config.items():
                if key != 'experiment_id':
                    f.write(f"- {key}: {value}\n")
            
            f.write(f"\n**Results**: See plots in the [{plots_dir}]({plots_dir}) directory\n\n")
            
            # List plot images if available
            plots_path = os.path.join(output_dir, plots_dir)
            if os.path.exists(plots_path) and result['success']:
                plot_files = [f for f in os.listdir(plots_path) if f.endswith('.png')]
                if plot_files:
                    f.write("**Available plots**:\n\n")
                    for plot_file in plot_files:
                        plot_path = os.path.join(plots_dir, plot_file)
                        f.write(f"- [{plot_file}]({plot_path})\n")
                else:
                    f.write("No plots were generated.\n")
            
            f.write("\n---\n\n")
    
    print(f"\nAll experiments completed. Report saved to {report_path}")
    print(f"Output directory: {output_dir}")
    
    # Summary of success/failure
    success_count = sum(1 for result in results.values() if result['success'])
    print(f"\nSuccessful experiments: {success_count}/{len(experiments)}")

if __name__ == "__main__":
    main() 