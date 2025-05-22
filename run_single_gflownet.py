import argparse
import torch
import os
import wandb
from benchmarker import Benchmarker

def run_single_gflownet(optimizer_name, learning_rate=None, beta3=None, momentum=None, alpha=None, nesterov=True, use_wandb=True):
    """
    Run a single GFlowNet experiment with specified optimizer.
    
    Args:
        optimizer_name: Name of the optimizer ('VRADAM', 'ADAM', 'SGD', or 'RMSPROP')
        learning_rate: Learning rate for the optimizer
        beta3: Beta3 parameter for VRADAM
        momentum: Momentum parameter for SGD/RMSProp
        alpha: Alpha parameter for RMSProp
        nesterov: Whether to use Nesterov momentum (for SGD)
        use_wandb: Whether to log to wandb
    """
    print(f"Running single GFlowNet experiment with {optimizer_name} optimizer")
    
    # Common parameters for all optimizer types
    params = {
        'model': 'GFlowNetModel',
        'dataset': 'GridWorld',
        'dataset_size': 'full',
        'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer': optimizer_name,
        'batch_size': 64,
        'epochs': 10,
        
        # GFlowNet specific parameters
        'grid_size': 8,
        'hidden_dim': 128,
        'num_layers': 3,
        'action_dim': 5,
        'state_dim': 3,
        'flow_matching_weight': 1.0,
        'reward_weight': 1.0,
        'entropy_weight': 0.01,
        'reward_temp': 1.0,
        
        # Fixed seed for reproducibility
        'seed': 42,
        
        # Common optimizer params
        'weight_decay': 1e-5,
        'eps': 1e-8,
    }
    
    # Add optimizer-specific parameters
    if optimizer_name == 'VRADAM':
        # Default VRADAM parameters
        params.update({
            'eta': learning_rate if learning_rate is not None else 1e-3,  # Learning rate
            'beta1': 0.9,
            'beta2': 0.999,
            'beta3': beta3 if beta3 is not None else 1.0,
            'power': 2,
            'normgrad': True,
            'lr_cutoff': 19,
        })
    elif optimizer_name == 'ADAM':
        # Default Adam parameters
        params.update({
            'lr': learning_rate if learning_rate is not None else 1e-3,  # Learning rate
            'beta1': 0.9,
            'beta2': 0.999,
        })
    elif optimizer_name == 'SGD':
        # Default SGD parameters
        params.update({
            'lr': learning_rate if learning_rate is not None else 1e-2,  # Learning rate
            'momentum': momentum if momentum is not None else 0.9,
            'nesterov': nesterov,
        })
    elif optimizer_name == 'RMSPROP':
        # Default RMSProp parameters
        params.update({
            'lr': learning_rate if learning_rate is not None else 1e-3,  # Learning rate
            'alpha': alpha if alpha is not None else 0.99,
            'momentum': momentum if momentum is not None else 0.0,
        })
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Initialize wandb for logging (if enabled)
    if use_wandb:
        try:
            wandb.init(
                project=f"{optimizer_name}-GFlowNet-SingleRun",
                config=params
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            use_wandb = False
    
    # Create and run benchmarker
    print(f"Running GFlowNet with params: {params}")
    benchmark = Benchmarker(params)
    results = benchmark.run()
    
    # Extract validation and test flow match losses and rewards
    train_flow_match_loss = results.get('flow_match_losses', [])[-1] if results.get('flow_match_losses', []) else float('inf')
    
    # Get validation metrics
    val_flow_match_loss = None
    if 'val_losses' in results and results['val_losses']:
        val_flow_match_loss = results['val_losses'][-1]
    
    # Get test metrics from test_loss and test_acc
    test_flow_match_loss = results.get('test_loss', float('inf'))
    mean_test_reward = results.get('test_acc', 0.0)  # In benchmarker, reward mean is stored in test_acc
    max_test_reward = results.get('reward_max', 0.0)
    
    # Print detailed summary of results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    print("\nTraining Metrics:")
    print(f"  - Final train loss:        {results.get('final_train_loss', 0):.6f}")
    print(f"  - Train flow match loss:   {train_flow_match_loss:.6f}")
    
    print("\nValidation Metrics:")
    print(f"  - Validation flow match loss: {val_flow_match_loss if val_flow_match_loss is not None else 'N/A':.6f}")
    
    print("\nTest Metrics:")
    print(f"  - Test flow match loss:    {test_flow_match_loss:.6f}")
    print(f"  - Mean test reward:        {mean_test_reward:.4f}")
    print(f"  - Max test reward:         {max_test_reward:.4f}")
    
    print("\nOther Metrics:")
    print(f"  - Sample diversity:        {results.get('sample_diversity', 0):.4f}")
    print(f"  - Training time:           {results.get('train_time', 0):.2f} seconds")
    print("="*50)
    
    # Log results to wandb if enabled
    if use_wandb:
        try:
            wandb.log({
                "final_train_loss": results.get('final_train_loss', 0),
                "train_flow_match_loss": train_flow_match_loss,
                "val_flow_match_loss": val_flow_match_loss if val_flow_match_loss is not None else float('inf'),
                "test_flow_match_loss": test_flow_match_loss,
                "mean_test_reward": mean_test_reward,
                "max_test_reward": max_test_reward,
                "sample_diversity": results.get('sample_diversity', 0),
                "train_time": results.get('train_time', 0)
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single GFlowNet experiment with specified optimizer")
    parser.add_argument("--optimizer", type=str, required=True, choices=["VRADAM", "ADAM", "SGD", "RMSPROP"],
                      help="Optimizer to use ('VRADAM', 'ADAM', 'SGD', or 'RMSPROP')")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    # Optimizer-specific parameters
    parser.add_argument("--beta3", type=float, help="Beta3 parameter for VRADAM")
    parser.add_argument("--momentum", type=float, help="Momentum parameter for SGD/RMSProp")
    parser.add_argument("--alpha", type=float, help="Alpha parameter for RMSProp")
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum (for SGD)")
    
    # Wandb parameter
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Run the single experiment
    run_single_gflownet(
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        beta3=args.beta3,
        momentum=args.momentum,
        alpha=args.alpha,
        nesterov=args.nesterov,
        use_wandb=not args.no_wandb
    ) 