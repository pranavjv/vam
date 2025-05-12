import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from benchmarker import Benchmarker

def run_rl_benchmark(model='RLPolicy', dataset='HalfCheetah', hidden_dim=256, epochs=30, 
                     batch_size=64, seed=42, save_plots=True, device=None):
    """
    Run benchmark comparing VADAM and Adam on reinforcement learning tasks
    
    Args:
        model: Model architecture to use
        dataset: RL environment to use
        hidden_dim: Hidden dimension for the policy and value networks
        epochs: Number of epochs to train
        batch_size: Batch size for training
        seed: Random seed for reproducibility
        save_plots: Whether to save plots
        device: Specific device to use (if None, will auto-detect)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine device to use
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Set common parameters
    base_params = {
        'model': model,
        'dataset': dataset,
        'dataset_size': 'small',
        'device': device,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'embed_dim': 300,  # Not used for RL but required by API
        'max_seq_len': 256,  # Not used for RL but required by API
        'epochs': epochs,
        
        # RL specific parameters
        'gamma': 0.99,
        'entropy_coef': 0.01
    }
    
    results = {}
    
    # VADAM parameters
    vadam_params = base_params.copy()
    vadam_params.update({
        'optimizer': 'VADAM',
        'eta': 0.01,  # Learning rate for VADAM
        'beta1': 0.9,
        'beta2': 0.999,
        'beta3': 1.0,
        'power': 2,
        'normgrad': True,
        'lr_cutoff': 19,
        'weight_decay': 0.0,
        'eps': 1e-8
    })
    
    try:
        print("Running benchmark with VADAM optimizer...")
        vadam_benchmark = Benchmarker(vadam_params)
        vadam_results = vadam_benchmark.run()
        results['VADAM'] = vadam_results
        print("VADAM benchmark completed successfully.")
    except Exception as e:
        print(f"Error during VADAM benchmark: {e}")
        import traceback
        traceback.print_exc()
        vadam_results = None
    
    # Reset random seeds for Adam run
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Adam parameters
    adam_params = base_params.copy()
    adam_params.update({
        'optimizer': 'ADAM',
        'lr': 0.01,  # Learning rate for Adam
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.0,
        'eps': 1e-8
    })
    
    try:
        print("Running benchmark with Adam optimizer...")
        adam_benchmark = Benchmarker(adam_params)
        adam_results = adam_benchmark.run()
        results['Adam'] = adam_results
        print("Adam benchmark completed successfully.")
    except Exception as e:
        print(f"Error during Adam benchmark: {e}")
        import traceback
        traceback.print_exc()
        adam_results = None
    
    # Only print summary and create plots if both benchmarks completed successfully
    if vadam_results is not None and adam_results is not None:
        # Print summary of results
        print("\n========== BENCHMARK RESULTS ==========")
        print(f"Environment: {dataset}")
        print(f"Model: {model}")
        print(f"Epochs: {epochs}")
        
        print("\nVADAM Results:")
        print(f"Final Mean Reward: {vadam_results['final_train_acc']:.2f}")
        print(f"Test Mean Reward: {vadam_results['test_acc']:.2f}")
        print(f"Training Time: {vadam_results['train_time']:.2f} seconds")
        
        print("\nAdam Results:")
        print(f"Final Mean Reward: {adam_results['final_train_acc']:.2f}")
        print(f"Test Mean Reward: {adam_results['test_acc']:.2f}")
        print(f"Training Time: {adam_results['train_time']:.2f} seconds")
        
        # Create and save plots
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            
            # Plot training rewards
            plt.figure(figsize=(10, 6))
            plt.plot(vadam_results['mean_rewards'], label='VADAM')
            plt.plot(adam_results['mean_rewards'], label='Adam')
            plt.title(f'Mean Reward per Epoch ({dataset})')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Reward')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/rl_reward_{dataset}.png')
            
            # Plot training losses
            plt.figure(figsize=(10, 6))
            plt.plot(vadam_results['train_losses'], label='VADAM')
            plt.plot(adam_results['train_losses'], label='Adam')
            plt.title(f'Training Loss per Epoch ({dataset})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/rl_loss_{dataset}.png')
            
            print("Plots saved to 'plots' directory")
    else:
        print("\nCould not generate summary or plots due to benchmark errors.")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RL benchmark comparing VADAM and Adam')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for policy network')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--no_plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    run_rl_benchmark(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        seed=args.seed,
        save_plots=not args.no_plots,
        device=args.device
    ) 