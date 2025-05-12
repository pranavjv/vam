import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime
import json
import torchvision.models as models
from scipy import linalg

from diffusion_model import EnhancedUNet, DiffusionModel
from VADAM import VADAM

class InceptionStatistics:
    def __init__(self, device='cuda'):
        self.device = device
        # Load pre-trained Inception v3 model
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()  # Remove final FC layer
        self.inception_model = self.inception_model.to(device)
        self.inception_model.eval()
        
        # Set up preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess_images(self, imgs):
        """Preprocess grayscale images to RGB format suitable for Inception."""
        imgs = imgs * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
        
        # Convert grayscale to RGB by repeating the channel
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        
        # Resize and normalize for Inception
        imgs = self.preprocess(imgs)
        return imgs
        
    @torch.no_grad()
    def get_features(self, images):
        """Extract features from Inception model."""
        images = self._preprocess_images(images)
        features = self.inception_model(images)
        return features
        
    @torch.no_grad()
    def calculate_statistics(self, dataloader):
        """Calculate mean and covariance of Inception features."""
        features_list = []
        
        for data in dataloader:
            if isinstance(data, (tuple, list)):
                images = data[0].to(self.device)
            else:
                images = data.to(self.device)
                
            features = self.get_features(images)
            features_list.append(features.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
        
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """Calculate FID score between two sets of statistics."""
        # Calculate squared difference between means
        diff = mu1 - mu2
        mean_term = np.sum(diff * diff)
        
        # Calculate matrix sqrt term
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Ensure covmean is real
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        trace_term = np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
        return mean_term + trace_term

class DiffusionBenchmark:
    def __init__(self, p=None):
        if p is None:
            p = {
                'device': 'cuda',  # 'cuda', 'mps', or 'cpu'
                'dataset_size': 'small',  # 'small' or 'full'
                'optimizer': 'ADAM',  # 'ADAM' or 'VADAM'
                'batch_size': 64,
                'lr': 0.0002,  # Learning rate
                'epochs': 5,  # Number of training epochs
                'unet_base_channels': 64,  # Base channels for UNet - increased
                'unet_time_embed_dim': 128,  # Time embedding dimension - increased
                'num_timesteps': 1000,  # Diffusion timesteps
                'beta_min': 1e-4,  # Min noise schedule
                'beta_max': 0.02,  # Max noise schedule
                'sample_every': 10,  # Save generated samples every n epochs
                'eval_batch_size': 16,  # Batch size for evaluation
                'fid_batch_size': 50,  # Batch size for FID evaluation
                'fid_num_samples': 1000  # Number of samples for FID evaluation
            }
        self.p = p
        
        # Determine device
        if self.p['device'] == 'mps':
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            if not torch.backends.mps.is_available():
                print("MPS is not available. Using CPU instead.")
        elif self.p['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("CUDA is not available. Using CPU instead.")
        else:
            self.device = torch.device('cpu')
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.fid_scores = []
        self.generated_samples = []
        self.train_time = None
        self.results = None
        
    def setup_data(self):
        # MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        if self.p['dataset_size'] == "small":
            # Use a small subset for quicker benchmarking
            subset_size = int(0.05 * len(train_dataset))  # 5% of training data
            val_size = int(0.01 * len(train_dataset))  # 1% of training data
            rest_size = len(train_dataset) - subset_size - val_size
            small_train, val_set, _ = random_split(
                train_dataset, 
                [subset_size, val_size, rest_size]
            )
            
            self.train_loader = DataLoader(
                small_train,
                batch_size=self.p['batch_size'],
                shuffle=True,
                num_workers=4)
            self.val_loader = DataLoader(
                val_set,
                batch_size=self.p['batch_size'],
                shuffle=False,
                num_workers=4)
        else:
            # Use the full dataset
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_set = random_split(train_dataset, [train_size, val_size])
            
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.p['batch_size'],
                shuffle=True,
                num_workers=4)
            self.val_loader = DataLoader(
                val_set,
                batch_size=self.p['batch_size'],
                shuffle=False,
                num_workers=4)
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.p['eval_batch_size'],
            shuffle=False,
            num_workers=4)
        
        # Prepare a loader for FID score calculation
        self.fid_loader = DataLoader(
            test_dataset,
            batch_size=self.p['fid_batch_size'],
            shuffle=True,
            num_workers=4)
    
    def setup_model(self):
        # Create EnhancedUNet model
        self.unet = EnhancedUNet(
            in_channels=1,  # MNIST is grayscale
            out_channels=1,  # Predict noise
            base_channels=self.p['unet_base_channels'],
            time_emb_dim=self.p['unet_time_embed_dim']
        ).to(self.device)
        
        # Create diffusion model wrapper
        self.diffusion = DiffusionModel(
            model=self.unet,
            beta_min=self.p['beta_min'],
            beta_max=self.p['beta_max'],
            num_timesteps=self.p['num_timesteps'],
            device=self.device
        )
        
        # Setup optimizer
        if self.p['optimizer'] == "VADAM":
            # VADAM specific parameters
            vadam_params = {
                'beta1': self.p.get('beta1', 0.9),
                'beta2': self.p.get('beta2', 0.999),
                'beta3': self.p.get('beta3', 1.0),
                'eta': self.p.get('eta', self.p.get('lr', 0.0002)),  # Use eta if provided, otherwise fall back to lr
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0),
                'power': self.p.get('power', 2),
                'normgrad': self.p.get('normgrad', True),
                'lr_cutoff': self.p.get('lr_cutoff', 19)
            }
            self.optimizer = VADAM(self.unet.parameters(), **vadam_params)
        elif self.p['optimizer'] == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.p.get('lr', 0.0002),
                'betas': (self.p.get('beta1', 0.9), self.p.get('beta2', 0.999)),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0)
            }
            self.optimizer = torch.optim.Adam(self.unet.parameters(), **adam_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.p['optimizer']}")
            
        # Setup Inception statistics calculator for FID
        self.inception_stats = InceptionStatistics(device=self.device)
    
    def train_epoch(self, epoch):
        self.unet.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
            
            # Calculate loss
            self.optimizer.zero_grad()
            loss = self.diffusion.p_losses(data, t)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader):
        self.unet.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                batch_size = data.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                
                # Calculate loss
                loss = self.diffusion.p_losses(data, t)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def generate_samples(self, batch_size=16):
        self.unet.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(batch_size=batch_size, img_size=28)
        return samples
    
    def compute_fid_score(self):
        """Calculate FID score between real and generated images"""
        self.unet.eval()
        
        # Calculate statistics for real images
        print(f"Computing Inception statistics for real images...")
        real_mu, real_sigma = self.inception_stats.calculate_statistics(self.fid_loader)
        
        # Generate samples for FID calculation
        print(f"Generating {self.p['fid_num_samples']} images for FID calculation...")
        generated_samples = []
        batch_size = self.p['fid_batch_size']
        num_batches = (self.p['fid_num_samples'] + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                current_batch_size = min(batch_size, self.p['fid_num_samples'] - i * batch_size)
                if current_batch_size <= 0:
                    break
                samples = self.diffusion.sample(batch_size=current_batch_size, img_size=28)
                generated_samples.append(samples)
        
        # Create a loader for generated images
        generated_loader = [(samples,) for samples in generated_samples]
        
        # Calculate statistics for generated images
        print(f"Computing Inception statistics for generated images...")
        gen_mu, gen_sigma = self.inception_stats.calculate_statistics(generated_loader)
        
        # Calculate FID score
        fid_score = self.inception_stats.calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)
        
        print(f"FID Score: {fid_score:.4f}")
        return fid_score
    
    def save_samples(self, samples, epoch, output_dir='diffusion_samples'):
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert samples to numpy arrays and denormalize
        samples_np = (samples.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        
        # Plot samples in a grid
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            if i < samples_np.shape[0]:
                ax.imshow(samples_np[i, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        optimizer_name = self.p['optimizer']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/samples_{optimizer_name}_epoch_{epoch}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    def run(self):
        """Run the benchmark and track all metrics"""
        self.setup_data()
        self.setup_model()
        
        # Create output directory for samples
        output_dir = 'diffusion_samples'
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(1, self.p['epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            # Generate and save samples every few epochs
            if epoch % self.p['sample_every'] == 0 or epoch == self.p['epochs']:
                samples = self.generate_samples(batch_size=16)
                sample_path = self.save_samples(
                    samples, 
                    epoch, 
                    output_dir=f"{output_dir}/{self.p['optimizer']}"
                )
                self.generated_samples.append({
                    'epoch': epoch,
                    'path': sample_path
                })
                
                # Compute FID score if it's the final epoch
                if epoch == self.p['epochs']:
                    try:
                        fid_score = self.compute_fid_score()
                        self.fid_scores.append({
                            'epoch': epoch,
                            'score': fid_score
                        })
                    except Exception as e:
                        print(f"Error computing FID score: {e}")
                        self.fid_scores.append({
                            'epoch': epoch,
                            'score': None,
                            'error': str(e)
                        })
        
        # Evaluate on the test set
        final_test_loss = self.evaluate(self.test_loader)
        
        # Generate final samples
        final_samples = self.generate_samples(batch_size=16)
        final_sample_path = self.save_samples(
            final_samples, 
            self.p['epochs'], 
            output_dir=f"{output_dir}/{self.p['optimizer']}_final"
        )
        
        self.train_time = time.time() - start_time
        
        # Compile results
        self.results = {
            'params': self.p,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'test_loss': final_test_loss,
            'fid_scores': self.fid_scores,
            'final_fid_score': self.fid_scores[-1]['score'] if self.fid_scores else None,
            'generated_samples': self.generated_samples,
            'final_sample_path': final_sample_path,
            'train_time': self.train_time
        }
        
        return self.results

def compare_optimizers(base_params):
    """Run diffusion model benchmarks comparing ADAM and VADAM optimizers"""
    results = {}
    
    # Run benchmark with Adam optimizer
    print("\n" + "="*50)
    print(f"Running diffusion model on MNIST with ADAM optimizer")
    print("="*50 + "\n")
    
    adam_params = base_params.copy()
    adam_params['optimizer'] = 'ADAM'
    # ADAM specific parameters
    adam_params.update({
        'lr': base_params.get('lr', 0.0002),
        'beta1': base_params.get('adam_beta1', 0.9),
        'beta2': base_params.get('adam_beta2', 0.999),
        'eps': base_params.get('adam_eps', 1e-8),
        'weight_decay': base_params.get('adam_weight_decay', 0)
    })
    benchmark_adam = DiffusionBenchmark(adam_params)
    adam_results = benchmark_adam.run()
    results['ADAM'] = adam_results
    
    # Run benchmark with VADAM optimizer
    print("\n" + "="*50)
    print(f"Running diffusion model on MNIST with VADAM optimizer")
    print("="*50 + "\n")
    
    vadam_params = base_params.copy()
    vadam_params['optimizer'] = 'VADAM'
    # VADAM specific parameters
    vadam_params.update({
        'eta': base_params.get('vadam_eta', base_params.get('lr', 0.0002)),
        'beta1': base_params.get('vadam_beta1', 0.9),
        'beta2': base_params.get('vadam_beta2', 0.999),
        'beta3': base_params.get('vadam_beta3', 1.0),
        'eps': base_params.get('vadam_eps', 1e-8),
        'weight_decay': base_params.get('vadam_weight_decay', 0),
        'power': base_params.get('vadam_power', 2),
        'normgrad': base_params.get('vadam_normgrad', True),
        'lr_cutoff': base_params.get('vadam_lr_cutoff', 19)
    })
    benchmark_vadam = DiffusionBenchmark(vadam_params)
    vadam_results = benchmark_vadam.run()
    results['VADAM'] = vadam_results
    
    # Save results to file
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"diffusion_mnist_{timestamp}.json")
    
    # Clean up results for JSON serialization
    clean_results = {}
    for optimizer, result in results.items():
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                clean_result[key] = value
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                clean_result[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                clean_result[key] = str(value)
        clean_results[optimizer] = clean_result
    
    with open(result_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Plot results
    plot_results(results, result_file.replace('.json', '.png'))
    
    # Print head-to-head summary
    print("\n" + "="*50)
    print(f"SUMMARY: Diffusion Model on MNIST")
    print("="*50)
    
    # Compare key metrics
    print(f"ADAM Test Loss: {results['ADAM']['test_loss']:.6f}")
    print(f"VADAM Test Loss: {results['VADAM']['test_loss']:.6f}")
    print(f"ADAM Training Time: {results['ADAM']['train_time']:.2f}s")
    print(f"VADAM Training Time: {results['VADAM']['train_time']:.2f}s")
    
    # Compare FID scores if available
    if results['ADAM'].get('final_fid_score') is not None and results['VADAM'].get('final_fid_score') is not None:
        print(f"ADAM FID Score: {results['ADAM']['final_fid_score']:.4f}")
        print(f"VADAM FID Score: {results['VADAM']['final_fid_score']:.4f}")
    
    return results

def plot_results(results, filename):
    """Plot and save comparison charts for optimizer performance"""
    plt.figure(figsize=(15, 12))
    
    # Plot training and validation loss curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f"{name} - Train")
        plt.plot(result['val_losses'], linestyle='--', label=f"{name} - Val")
    
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Bar chart comparing final metrics
    plt.subplot(2, 2, 2)
    names = list(results.keys())
    
    # Get metrics
    train_losses = [results[name]['final_train_loss'] for name in names]
    val_losses = [results[name]['final_val_loss'] for name in names]
    test_losses = [results[name]['test_loss'] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, train_losses, width, label='Final Train Loss')
    plt.bar(x, val_losses, width, label='Final Val Loss')
    plt.bar(x + width, test_losses, width, label='Test Loss')
    
    plt.title("Loss Comparison")
    plt.xlabel("Optimizer")
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot FID scores if available
    has_fid = all('final_fid_score' in result and result['final_fid_score'] is not None for result in results.values())
    if has_fid:
        plt.subplot(2, 2, 3)
        fid_scores = [results[name]['final_fid_score'] for name in names]
        plt.bar(x, fid_scores, width=0.4)
        plt.title("FID Score (Lower is Better)")
        plt.xlabel("Optimizer")
        plt.ylabel("FID Score")
        plt.xticks(x, names)
        
        # Add FID values as text on the bars
        for i, v in enumerate(fid_scores):
            plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
            
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        # Loss improvement over epochs if no FID
        plt.subplot(2, 2, 3)
        for name, result in results.items():
            # Calculate percentage improvement from first to last epoch
            first_loss = result['train_losses'][0]
            last_loss = result['train_losses'][-1]
            improvement = (first_loss - last_loss) / first_loss * 100
            epochs = list(range(1, len(result['train_losses']) + 1))
            
            # Plot normalized improvement
            normalized_losses = [loss / first_loss for loss in result['train_losses']]
            plt.plot(epochs, normalized_losses, marker='o', label=f"{name} (Total: {improvement:.1f}%)")
        
        plt.title("Normalized Training Loss Improvement")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (normalized to first epoch)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    # Training time comparison
    plt.subplot(2, 2, 4)
    train_times = [results[name]['train_time'] for name in names]
    
    plt.bar(x, train_times, width=0.4)
    plt.title("Training Time")
    plt.xlabel("Optimizer")
    plt.ylabel("Time (seconds)")
    plt.xticks(x, names)
    
    # Add time values as text on the bars
    for i, v in enumerate(train_times):
        plt.text(i, v + 0.1, f"{v:.1f}s", ha='center')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.suptitle("Diffusion Model Benchmark: ADAM vs VADAM on MNIST", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    
    # Create a separate text summary file with detailed metrics
    summary_file = os.path.splitext(filename)[0] + '_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("DIFFUSION MODEL BENCHMARK: ADAM vs VADAM on MNIST\n")
        f.write("="*50 + "\n\n")
        
        for name in names:
            result = results[name]
            f.write(f"OPTIMIZER: {name}\n")
            f.write("-"*30 + "\n")
            f.write(f"Training Time: {result['train_time']:.2f} seconds\n")
            f.write(f"Final Train Loss: {result['final_train_loss']:.6f}\n")
            f.write(f"Final Validation Loss: {result['final_val_loss']:.6f}\n")
            f.write(f"Test Loss: {result['test_loss']:.6f}\n")
            
            # Add FID score if available
            if 'final_fid_score' in result and result['final_fid_score'] is not None:
                f.write(f"FID Score: {result['final_fid_score']:.4f}\n")
            
            # Calculate loss reduction percentage
            first_loss = result['train_losses'][0]
            last_loss = result['train_losses'][-1]
            loss_reduction = (first_loss - last_loss) / first_loss * 100
            f.write(f"Loss Reduction: {loss_reduction:.2f}%\n")
            
            # Sample image paths
            f.write(f"Final Sample Path: {result['final_sample_path']}\n\n")
        
        # Add head-to-head comparison
        if len(names) == 2:
            f.write("\nHEAD-TO-HEAD COMPARISON\n")
            f.write("-"*30 + "\n")
            adam_result = results['ADAM']
            vadam_result = results['VADAM']
            
            time_diff = vadam_result['train_time'] - adam_result['train_time']
            time_percent = (time_diff / adam_result['train_time']) * 100
            f.write(f"Training Time Difference: {time_diff:.2f}s ({time_percent:.1f}%)\n")
            
            loss_diff = vadam_result['test_loss'] - adam_result['test_loss']
            loss_percent = (loss_diff / adam_result['test_loss']) * 100
            f.write(f"Test Loss Difference: {loss_diff:.6f} ({loss_percent:.1f}%)\n")
            
            # FID comparison if available
            if 'final_fid_score' in adam_result and adam_result['final_fid_score'] is not None and \
               'final_fid_score' in vadam_result and vadam_result['final_fid_score'] is not None:
                fid_diff = vadam_result['final_fid_score'] - adam_result['final_fid_score']
                fid_percent = (fid_diff / adam_result['final_fid_score']) * 100
                f.write(f"FID Score Difference: {fid_diff:.4f} ({fid_percent:.1f}% {'better' if fid_diff < 0 else 'worse'} for VADAM)\n")
    
    print(f"Detailed summary saved to {summary_file}")

if __name__ == "__main__":
    # Set base parameters for the diffusion model benchmark
    base_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_size': 'small',  # Use small dataset for faster benchmarking
        'batch_size': 64,
        'lr': 0.0002,
        'epochs': 10,  # Reduced epochs for benchmarking
        'unet_base_channels': 64,  # Increased from 32
        'unet_time_embed_dim': 128,  # Increased from 64
        'num_timesteps': 1000,
        'beta_min': 1e-4,
        'beta_max': 0.02,
        'sample_every': 2,  # Generate samples every n epochs
        'eval_batch_size': 16,
        'fid_batch_size': 32,
        'fid_num_samples': 500  # Number of samples for FID computation
    }
    
    print(f"Running on device: {base_params['device']}")
    print(f"Starting diffusion model benchmark: ADAM vs VADAM")
    
    # Run the benchmark comparison
    results = compare_optimizers(base_params) 