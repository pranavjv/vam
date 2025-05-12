import torch
import argparse
from benchmarker import Benchmarker

def test_halfcheetah_rl(optimizer_type='VADAM', device=None):
    """
    Test the HalfCheetah reinforcement learning environment
    
    Args:
        optimizer_type: 'VADAM' or 'ADAM'
        device: Specific device to use (if None, will auto-detect)
    """
    # Determine device to use
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Set up parameters for benchmarker
    params = {
        'model': 'RLPolicy',
        'dataset': 'HalfCheetah',
        'dataset_size': 'small',
        'device': device,
        'optimizer': optimizer_type,
        'batch_size': 64,
        'hidden_dim': 256,
        'embed_dim': 300,  # Not used but required by API
        'max_seq_len': 256,  # Not used but required by API
        'epochs': 10,
        
        # RL specific parameters
        'gamma': 0.99,
        'entropy_coef': 0.01
    }
    
    # Add optimizer-specific parameters
    if optimizer_type == 'VADAM':
        params.update({
            'eta': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'beta3': 1.0,
            'power': 2,
            'normgrad': True,
            'lr_cutoff': 19,
            'weight_decay': 0.0,
            'eps': 1e-8
        })
    else:  # ADAM
        params.update({
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'eps': 1e-8
        })
    
    # Create and run benchmarker
    print(f"Running HalfCheetah benchmark with {optimizer_type} optimizer")
    try:
        benchmark = Benchmarker(params)
        results = benchmark.run()
        
        # Print results
        print("\nResults:")
        print(f"Final Mean Reward: {results['final_train_acc']:.2f}")
        print(f"Test Mean Reward: {results['test_acc']:.2f}")
        print(f"Training Time: {results['train_time']:.2f} seconds")
        
        # Print training progress
        print("\nTraining Progress:")
        for epoch, (loss, reward) in enumerate(zip(results['train_losses'], results['mean_rewards'])):
            print(f"Epoch {epoch+1} | Loss: {loss:.6f} | Mean Reward: {reward:.2f}")
            
        return results
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HalfCheetah environment with a specific optimizer')
    parser.add_argument('--optimizer', type=str, default='VADAM', choices=['VADAM', 'ADAM'], 
                      help='Optimizer to use (VADAM or ADAM)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, or cpu)')
    
    args = parser.parse_args()
    
    # Run test with specified optimizer
    test_halfcheetah_rl(args.optimizer, args.device)
    
    # Uncomment to test with Adam
    # test_halfcheetah_rl('ADAM') 