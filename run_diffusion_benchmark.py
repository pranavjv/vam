import torch
import time
from diffusion_benchmark import compare_optimizers

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DIFFUSION MODEL BENCHMARK: ADAM vs VADAM on MNIST")
    print("="*60 + "\n")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Set parameters for the diffusion model benchmark
    benchmark_params = {
        'device': device,
        'dataset_size': 'large',  # Use 'small' for faster benchmarking
        'batch_size': 64,
        'lr': 0.0002,  # Standard learning rate for diffusion models
        'epochs': 30,   # Reduced for benchmarking
        'unet_base_channels': 64,  # Increased from 32 for better modeling
        'unet_time_embed_dim': 128,  # Increased from 64 for better modeling
        'num_timesteps': 200,  # Reduced from 1000 for faster sampling
        'beta_min': 1e-4,
        'beta_max': 0.02,
        'sample_every': 1,  # Generate samples every epoch
        'eval_batch_size': 16,
        'fid_batch_size': 32,
        'fid_num_samples': 250  # Reduced for faster benchmarking
    }
    
    print("\nStarting diffusion model benchmark...")
    print(f"Training for {benchmark_params['epochs']} epochs")
    print(f"Using batch size: {benchmark_params['batch_size']}")
    print(f"Model size: Base channels = {benchmark_params['unet_base_channels']}, Time embedding dim = {benchmark_params['unet_time_embed_dim']}")
    print(f"Diffusion steps: {benchmark_params['num_timesteps']}")
    print(f"FID calculation: {benchmark_params['fid_num_samples']} samples")
    
    start_time = time.time()
    
    # Run the benchmark
    results = compare_optimizers(benchmark_params)
    
    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.2f} seconds")
    
    # Summary of key findings
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    adam_result = results['ADAM']
    vadam_result = results['VADAM']
    
    loss_diff = vadam_result['test_loss'] - adam_result['test_loss']
    loss_percent = (loss_diff / adam_result['test_loss']) * 100
    
    time_diff = vadam_result['train_time'] - adam_result['train_time']
    time_percent = (time_diff / adam_result['train_time']) * 100
    
    # Compare training loss reduction (first epoch to last)
    adam_loss_reduction = (adam_result['train_losses'][0] - adam_result['train_losses'][-1]) / adam_result['train_losses'][0] * 100
    vadam_loss_reduction = (vadam_result['train_losses'][0] - vadam_result['train_losses'][-1]) / vadam_result['train_losses'][0] * 100
    
    print(f"Test Loss: ADAM = {adam_result['test_loss']:.6f}, VADAM = {vadam_result['test_loss']:.6f}")
    print(f"Loss Difference: {loss_diff:.6f} ({loss_percent:.2f}% {'better' if loss_percent < 0 else 'worse'} for VADAM)")
    
    # Compare FID scores if available
    if 'final_fid_score' in adam_result and adam_result['final_fid_score'] is not None and \
       'final_fid_score' in vadam_result and vadam_result['final_fid_score'] is not None:
        fid_diff = vadam_result['final_fid_score'] - adam_result['final_fid_score']
        fid_percent = (fid_diff / adam_result['final_fid_score']) * 100
        print(f"FID Score: ADAM = {adam_result['final_fid_score']:.4f}, VADAM = {vadam_result['final_fid_score']:.4f}")
        print(f"FID Difference: {fid_diff:.4f} ({fid_percent:.2f}% {'better' if fid_diff < 0 else 'worse'} for VADAM)")
    
    print(f"Training Time: ADAM = {adam_result['train_time']:.2f}s, VADAM = {vadam_result['train_time']:.2f}s")
    print(f"Time Difference: {time_diff:.2f}s ({time_percent:.2f}% {'faster' if time_percent < 0 else 'slower'} for VADAM)")
    print(f"Loss Reduction: ADAM = {adam_loss_reduction:.2f}%, VADAM = {vadam_loss_reduction:.2f}%")
    
    print("\nSample images have been saved to the diffusion_samples directory")
    print(f"ADAM final samples: {adam_result['final_sample_path']}")
    print(f"VADAM final samples: {vadam_result['final_sample_path']}")
    
    print("\nFull results have been saved to the benchmark_results directory")
    print("="*60) 