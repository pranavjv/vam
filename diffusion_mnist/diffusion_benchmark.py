import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys
import json
import argparse
from datetime import datetime
from scipy import linalg
from torchvision.utils import make_grid

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_model import EnhancedUNet, SimpleUNet, DiffusionModel, SelfAttention
from VADAM import VADAM

class InceptionStatistics:
    def __init__(self, device='cuda'):
        self.device = device
        try:
            # Load pre-trained Inception v3 model
            self.inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            self.inception_model.fc = nn.Identity()  # Remove final FC layer
            self.inception_model = self.inception_model.to(device)
            self.inception_model.eval()
            self.model_available = True
            print("Successfully loaded Inception model for FID calculation")
        except Exception as e:
            print(f"Failed to load Inception model: {e}")
            print("Using simplified metrics instead of FID")
            self.model_available = False
        
        # Set up preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess_images(self, imgs):
        """Preprocess grayscale images to RGB format suitable for Inception."""
        if not self.model_available:
            return None
            
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
        if not self.model_available:
            return None
            
        images = self._preprocess_images(images)
        features = self.inception_model(images)
        return features
        
    @torch.no_grad()
    def calculate_statistics(self, images_list):
        """Calculate mean and covariance of Inception features."""
        if not self.model_available:
            return None, None
            
        features_list = []
        
        for images in images_list:
            if isinstance(images, (tuple, list)):
                # If it's a tuple or list from a dataloader, take the first element
                images = images[0].to(self.device)
            else:
                # Otherwise just ensure it's on the correct device
                images = images.to(self.device)
                
            features = self.get_features(images)
            if features is not None:
                features_list.append(features.cpu().numpy())
        
        if not features_list:
            return None, None
            
        features = np.concatenate(features_list, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
        
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """Calculate FID score between two sets of statistics."""
        if mu1 is None or mu2 is None:
            return None
            
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
                'epochs': 10,  # Number of training epochs
                'unet_base_channels': 64,  # Base channels for UNet
                'unet_time_embed_dim': 128,  # Time embedding dimension
                'num_timesteps': 200,  # Diffusion timesteps
                'beta_min': 1e-4,  # Min noise schedule
                'beta_max': 0.02,  # Max noise schedule
                'sample_every': 1,  # Save generated samples every n epochs
                'eval_batch_size': 16,  # Batch size for evaluation
                'fid_batch_size': 32,  # Batch size for FID evaluation
                'fid_num_samples': 250,  # Number of samples for FID evaluation
                'use_attention': True,  # Whether to use attention in UNet
                'model_type': 'enhanced'  # 'enhanced' or 'simple'
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
        
        # Set random seed for reproducibility
        torch.manual_seed(self.p.get('seed', 42))
        np.random.seed(self.p.get('seed', 42))
        
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
            root='../data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root='../data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        if self.p['dataset_size'] == "small":
            # Use a small subset for quicker benchmarking
            subset_size = int(0.1 * len(train_dataset))  # 10% of training data
            val_size = int(0.02 * len(train_dataset))  # 2% of training data
            rest_size = len(train_dataset) - subset_size - val_size
            
            # Use fixed seed for consistent splits
            generator = torch.Generator().manual_seed(42)
            small_train, val_set, _ = random_split(
                train_dataset, 
                [subset_size, val_size, rest_size],
                generator=generator
            )
            
            self.train_loader = DataLoader(
                small_train,
                batch_size=self.p['batch_size'],
                shuffle=True,
                num_workers=2)
            self.val_loader = DataLoader(
                val_set,
                batch_size=self.p['batch_size'],
                shuffle=False,
                num_workers=2)
        else:
            # Use the full dataset
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            # Use fixed seed for consistent splits
            generator = torch.Generator().manual_seed(42)
            train_subset, val_set = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=generator
            )
            
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.p['batch_size'],
                shuffle=True,
                num_workers=2)
            self.val_loader = DataLoader(
                val_set,
                batch_size=self.p['batch_size'],
                shuffle=False,
                num_workers=2)
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.p['eval_batch_size'],
            shuffle=False,
            num_workers=2)
        
        # Create a separate dataloader for FID calculation
        # Using a small subset of the test set
        fid_subset_size = min(200, len(test_dataset))
        fid_indices = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(42))[:fid_subset_size]
        fid_subset = torch.utils.data.Subset(test_dataset, fid_indices)
        
        self.fid_loader = DataLoader(
            fid_subset,
            batch_size=self.p['fid_batch_size'],
            shuffle=False,
            num_workers=2
        )
    
    def setup_model(self):
        # Select UNet model based on config
        if self.p['model_type'] == 'enhanced':
            # Create EnhancedUNet model with attention if specified
            self.unet = EnhancedUNet(
                in_channels=1,  # MNIST is grayscale
                out_channels=1,  # Predict noise
                base_channels=self.p['unet_base_channels'],
                time_emb_dim=self.p['unet_time_embed_dim'],
                use_attention=self.p['use_attention']
            ).to(self.device)
            print(f"Using Enhanced UNet with attention: {self.p['use_attention']}")
        else:
            # Create SimpleUNet model (no attention)
            self.unet = SimpleUNet(
                in_channels=1,  # MNIST is grayscale
                out_channels=1,  # Predict noise
                base_channels=self.p['unet_base_channels'],
                time_emb_dim=self.p['unet_time_embed_dim']
            ).to(self.device)
            print("Using Simple UNet (no attention)")
        
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
                'eta': self.p.get('eta', self.p.get('lr', 0.0002)),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0),
                'power': self.p.get('power', 2),
                'normgrad': self.p.get('normgrad', False),
                'lr_cutoff': self.p.get('lr_cutoff', 19)
            }
            self.optimizer = VADAM(self.unet.parameters(), **vadam_params)
            print(f"Using VADAM optimizer with eta={vadam_params['eta']}, beta3={vadam_params['beta3']}")
        elif self.p['optimizer'] == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.p.get('lr', 0.0002),
                'betas': (self.p.get('beta1', 0.9), self.p.get('beta2', 0.999)),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0)
            }
            self.optimizer = torch.optim.Adam(self.unet.parameters(), **adam_params)
            print(f"Using Adam optimizer with lr={adam_params['lr']}")
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
        
        try:
            # Collect real images for FID calculation
            real_images = []
            for data, _ in self.fid_loader:
                real_images.append(data)
            
            # Generate samples for FID calculation
            print(f"Generating {self.p['fid_num_samples']} images for FID calculation...")
            generated_images = []
            batch_size = self.p['fid_batch_size']
            remaining = self.p['fid_num_samples']
            
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                samples = self.generate_samples(batch_size=current_batch)
                generated_images.append(samples)
                remaining -= current_batch
            
            # Calculate statistics for real and generated images
            print(f"Computing statistics for real images...")
            real_mu, real_sigma = self.inception_stats.calculate_statistics(real_images)
            
            print(f"Computing statistics for generated images...")
            gen_mu, gen_sigma = self.inception_stats.calculate_statistics(generated_images)
            
            # Calculate FID score
            if real_mu is not None and gen_mu is not None:
                fid_score = self.inception_stats.calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)
                print(f"FID Score: {fid_score:.4f}")
                return fid_score
            else:
                print("Could not compute FID score - invalid statistics")
                return None
        except Exception as e:
            print(f"Error computing FID score: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_samples_grid(self, samples, epoch, output_dir='../diffusion_samples'):
        """Save a grid of generated samples for visualization"""
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Denormalize samples from [-1, 1] to [0, 1]
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        
        # Create a grid of images
        grid = make_grid(samples, nrow=4)
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))
        
        # Plot the grid
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.tight_layout()
        
        # Save to file
        optimizer_name = self.p['optimizer']
        model_type = self.p['model_type']
        use_attn = "attn" if self.p.get('use_attention', False) else "no_attn"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{optimizer_name}_{model_type}_{use_attn}_epoch_{epoch}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    def run(self):
        """Run the benchmark and track all metrics"""
        self.setup_data()
        self.setup_model()
        
        # Create output directory for samples
        output_dir = '../diffusion_samples'
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
                sample_path = self.save_samples_grid(
                    samples, 
                    epoch, 
                    output_dir=output_dir
                )
                self.generated_samples.append({
                    'epoch': epoch,
                    'path': sample_path
                })
                
                # Compute FID score if it's the final epoch or every 5 epochs
                if epoch == self.p['epochs'] or epoch % 5 == 0:
                    try:
                        fid_score = self.compute_fid_score()
                        self.fid_scores.append({
                            'epoch': epoch,
                            'score': fid_score
                        })
                        print(f"Epoch {epoch} FID Score: {fid_score if fid_score is not None else 'N/A'}")
                    except Exception as e:
                        print(f"Error computing FID score: {e}")
                        self.fid_scores.append({
                            'epoch': epoch,
                            'score': None,
                            'error': str(e)
                        })
        
        # Evaluate on the test set
        final_test_loss = self.evaluate(self.test_loader)
        print(f"Final Test Loss: {final_test_loss:.6f}")
        
        # Generate final samples
        final_samples = self.generate_samples(batch_size=36)  # Generate more samples for final visualization
        final_sample_path = self.save_samples_grid(
            final_samples, 
            self.p['epochs'], 
            output_dir=output_dir
        )
        
        # Compute final FID score
        final_fid_score = self.compute_fid_score()
        
        self.train_time = time.time() - start_time
        print(f"Total training time: {self.train_time:.2f} seconds")
        
        # Compile results
        self.results = {
            'params': self.p,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'test_loss': final_test_loss,
            'fid_scores': self.fid_scores,
            'final_fid_score': final_fid_score,
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
    output_dir = '../benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"MNIST_Diffusion_{timestamp}.json")
    
    # Clean up results for JSON serialization
    clean_results = {}
    for optimizer, result in results.items():
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                clean_result[key] = value
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                clean_result[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                clean_result[key] = str(value)
        clean_results[optimizer] = clean_result
    
    with open(result_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Print head-to-head summary
    print("\n" + "="*50)
    print(f"SUMMARY: Diffusion Model on MNIST")
    print("="*50)
    
    # Compare key metrics
    print(f"ADAM Test Loss: {results['ADAM']['test_loss']:.6f}")
    print(f"VADAM Test Loss: {results['VADAM']['test_loss']:.6f}")
    
    # Compare FID scores if available
    if results['ADAM'].get('final_fid_score') is not None and results['VADAM'].get('final_fid_score') is not None:
        print(f"ADAM FID Score: {results['ADAM']['final_fid_score']:.4f}")
        print(f"VADAM FID Score: {results['VADAM']['final_fid_score']:.4f}")
    
    print(f"ADAM Training Time: {results['ADAM']['train_time']:.2f}s")
    print(f"VADAM Training Time: {results['VADAM']['train_time']:.2f}s")
    
    return results, result_file

def run_diffusion_benchmark(model_type='enhanced', use_attention=True, epochs=10, dataset_size='small'):
    """Run benchmark for diffusion model on MNIST with specified parameters"""
    # Diffusion model specific parameters
    base_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        'dataset_size': dataset_size,
        'batch_size': 64,
        'lr': 0.0002,  # Learning rate for diffusion
        'epochs': epochs,
        'unet_base_channels': 64,
        'unet_time_embed_dim': 128,
        'num_timesteps': 200,
        'beta_min': 1e-4,
        'beta_max': 0.02,
        'sample_every': 1,  # Generate samples every epoch
        'eval_batch_size': 16,
        'fid_batch_size': 32,
        'fid_num_samples': 100,  # Use fewer samples for faster FID calculation
        'use_attention': use_attention,
        'model_type': model_type,
        'seed': 42,  # For reproducibility
        
        # Adam specific parameters
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8,
        'adam_weight_decay': 0,
        
        # VADAM specific parameters
        'vadam_eta': 0.0002,
        'vadam_beta1': 0.9,
        'vadam_beta2': 0.999,
        'vadam_beta3': 1.0,
        'vadam_eps': 1e-8,
        'vadam_weight_decay': 0,
        'vadam_power': 2,
        'vadam_normgrad': True,
        'vadam_lr_cutoff': 19
    }
    
    return compare_optimizers(base_params)

def run_single_benchmark(optimizer='ADAM', model_type='enhanced', use_attention=True, epochs=10, dataset_size='small'):
    """Run benchmark for a single configuration of diffusion model on MNIST"""
    params = {
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        'dataset_size': dataset_size,
        'optimizer': optimizer,
        'batch_size': 64,
        'lr': 0.0002,  # Learning rate for diffusion
        'epochs': epochs,
        'unet_base_channels': 64,
        'unet_time_embed_dim': 128,
        'num_timesteps': 200,
        'beta_min': 1e-4,
        'beta_max': 0.02,
        'sample_every': 1,  # Generate samples every epoch
        'eval_batch_size': 16,
        'fid_batch_size': 32,
        'fid_num_samples': 100,  # Use fewer samples for faster FID calculation
        'use_attention': use_attention,
        'model_type': model_type,
        'seed': 42,  # For reproducibility
    }
    
    # Add optimizer-specific parameters
    if optimizer == 'VADAM':
        params.update({
            'eta': 0.0002,
            'beta1': 0.9,
            'beta2': 0.999,
            'beta3': 1.0, 
            'eps': 1e-8,
            'weight_decay': 0,
            'power': 2,
            'normgrad': False,
            'lr_cutoff': 19
        })
    else:  # ADAM
        params.update({
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0
        })
    
    # Run benchmark
    benchmark = DiffusionBenchmark(params)
    results = benchmark.run()
    
    # Save results to file
    output_dir = '../benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"MNIST_Diffusion_{optimizer}_{model_type}_{timestamp}.json")
    
    # Clean results for JSON serialization
    clean_result = {}
    for key, value in results.items():
        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
            clean_result[key] = value
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            clean_result[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
        else:
            clean_result[key] = str(value)
    
    with open(result_file, 'w') as f:
        json.dump(clean_result, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    return results, result_file

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run diffusion model benchmark on MNIST')
    parser.add_argument('--optimizer', type=str, choices=['ADAM', 'VADAM', 'both'], default='both',
                        help='Optimizer to use (default: both)')
    parser.add_argument('--model_type', type=str, choices=['simple', 'enhanced'], default='enhanced',
                        help='Model type to use (default: enhanced)')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention in the UNet model (default: True)')
    parser.add_argument('--no_attention', dest='use_attention', action='store_false',
                        help='Disable attention in the UNet model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--dataset_size', type=str, choices=['small', 'full'], default='small',
                        help='Dataset size to use (default: small)')
    args = parser.parse_args()
    
    # Determine the best available device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    print("\nStarting diffusion model benchmark...")
    start_time = time.time()
    
    if args.optimizer == 'both':
        print(f"Running comparison: {args.model_type} model with attention={args.use_attention}")
        results, result_file = run_diffusion_benchmark(
            model_type=args.model_type,
            use_attention=args.use_attention, 
            epochs=args.epochs,
            dataset_size=args.dataset_size
        )
    else:
        print(f"Running single benchmark: {args.optimizer} optimizer with {args.model_type} model, attention={args.use_attention}")
        results, result_file = run_single_benchmark(
            optimizer=args.optimizer,
            model_type=args.model_type,
            use_attention=args.use_attention,
            epochs=args.epochs,
            dataset_size=args.dataset_size
        )
    
    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.2f} seconds")
    print(f"Results saved to: {result_file}") 