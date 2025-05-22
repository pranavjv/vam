import wandb
import torch
import os
import json
import sys
import numpy as np
import argparse

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_model import EnhancedUNet, DiffusionModel
from VRADAM import VRADAM

class DiffusionSweepBenchmark:
    def __init__(self, config):
        self.config = config
        
        # Determine device
        if config.device == 'mps':
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            if not torch.backends.mps.is_available():
                print("MPS is not available. Using CPU instead.")
        elif config.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("CUDA is not available. Using CPU instead.")
        else:
            self.device = torch.device('cpu')
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.generated_samples = []
        self.train_time = None
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        # Use smaller subset for sweep runs
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms
        
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
        
        # Use a small subset for quick evaluation during sweep
        subset_size = int(0.05 * len(train_dataset))  # 5% of training data
        val_size = int(0.01 * len(train_dataset))  # 1% of training data
        rest_size = len(train_dataset) - subset_size - val_size
        small_train, val_set, _ = random_split(
            train_dataset, 
            [subset_size, val_size, rest_size]
        )
        
        self.train_loader = DataLoader(
            small_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2)
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2)
        
    def setup_model(self):
        # Create EnhancedUNet model
        self.unet = EnhancedUNet(
            in_channels=1,  # MNIST is grayscale
            out_channels=1,  # Predict noise
            base_channels=self.config.unet_base_channels,
            time_emb_dim=self.config.unet_time_embed_dim
        ).to(self.device)
        
        # Create diffusion model wrapper
        self.diffusion = DiffusionModel(
            model=self.unet,
            beta_min=self.config.beta_min,
            beta_max=self.config.beta_max,
            num_timesteps=self.config.num_timesteps,
            device=self.device
        )
        
        # Setup optimizer
        if self.config.optimizer == "VRADAM":
            # VRADAM specific parameters
            vradam_params = {
                'beta1': self.config.beta1,
                'beta2': self.config.beta2,
                'beta3': self.config.beta3 if hasattr(self.config, 'beta3') else 1.0,
                'eta': self.config.lr,
                'eps': self.config.eps,
                'weight_decay': self.config.weight_decay,
                'power': self.config.power if hasattr(self.config, 'power') else 2,
                'normgrad': self.config.normgrad if hasattr(self.config, 'normgrad') else True,
                'lr_cutoff': self.config.lr_cutoff if hasattr(self.config, 'lr_cutoff') else 19
            }
            self.optimizer = VRADAM(self.unet.parameters(), **vradam_params)
        elif self.config.optimizer == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.config.lr,
                'betas': (self.config.beta1, self.config.beta2),
                'eps': self.config.eps,
                'weight_decay': self.config.weight_decay
            }
            self.optimizer = torch.optim.Adam(self.unet.parameters(), **adam_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
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
    
    def generate_samples(self, batch_size=4):
        self.unet.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(batch_size=batch_size, img_size=28)
        return samples
    
    def run(self):
        """Run the benchmark and track metrics for the sweep"""
        import time
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Evaluate on test set
        test_loss = self.evaluate(self.test_loader)
        
        # Generate some samples
        samples = self.generate_samples(batch_size=4)
        
        # Convert to images for wandb logging
        sample_images = []
        for i in range(samples.shape[0]):
            sample_np = (samples[i, 0].cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            sample_images.append(wandb.Image(sample_np, caption=f"Sample {i+1}"))
        
        self.train_time = time.time() - start_time
        
        # Log final metrics
        wandb.log({
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "test_loss": test_loss,
            "train_time": self.train_time,
            "generated_samples": sample_images
        })
        
        # Return as optimization metric
        return test_loss

def train_model(config=None):
    """
    Train diffusion model with hyperparameters specified in config
    This function is called by wandb.agent
    """
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Create and run benchmark
        benchmark = DiffusionSweepBenchmark(config)
        test_loss = benchmark.run()
        
        # Return test loss as the optimization metric
        return test_loss

def create_sweep_config(optimizer_type):
    """Create sweep configuration for diffusion model with specified optimizer type"""
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'test_loss',
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'optimizer': {'value': optimizer_type},
            'device': {'value': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'},
            'seed': {'value': 42},
            'epochs': {'value': 3},  # Keep low for faster sweeps
            'batch_size': {'value': 64},
            
            # Diffusion specific parameters
            'unet_base_channels': {'value': 64},
            'unet_time_embed_dim': {'value': 128},
            'num_timesteps': {'value': 200},
            'beta_min': {'value': 1e-4},
            'beta_max': {'value': 0.02},
            
            # Common optimizer parameters
            'beta1': {'value': 0.9},
            'beta2': {'value': 0.999},
            'eps': {'value': 1e-8},
            'weight_decay': {'value': 0.0},
        }
    }
    
    # Add optimizer-specific parameters
    if optimizer_type == 'ADAM':
        # For Adam, we only sweep learning rate
        sweep_config['parameters']['lr'] = {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        }
    else:  # VRADAM
        # For VRADAM, we sweep learning rate, beta3, and lr_cutoff
        sweep_config['parameters'].update({
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'beta3': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 2.0
            },
            'lr_cutoff': {
                'distribution': 'int_uniform',
                'min': 5,
                'max': 30
            },
            'power': {'value': 2},
            'normgrad': {'value': True}
        })
    
    return sweep_config

def run_sweeps(optimizer_type=None, count=10):
    """Run sweeps for diffusion model with specified optimizer"""
    # Set up wandb project
    wandb.login()
    
    if optimizer_type is None or optimizer_type.upper() == 'BOTH':
        optimizers = ['ADAM', 'VRADAM']
    else:
        optimizers = [optimizer_type.upper()]
    
    sweep_ids = []
    for opt in optimizers:
        print(f"\nSetting up sweep for diffusion model on MNIST with {opt} optimizer")
        
        # Create a sweep configuration
        sweep_config = create_sweep_config(opt)
        
        # Initialize the sweep
        sweep_id = wandb.sweep(
            sweep_config, 
            project=f"diffusion-optimization-{opt}"
        )
        
        sweep_ids.append((opt, sweep_id))
        print(f"Sweep created with ID: {sweep_id}")
        
        # Create a directory for saving sweep results
        os.makedirs("../../sweep_results", exist_ok=True)
        
        # Save sweep configuration
        with open(f"../../sweep_results/sweep_config_Diffusion_{opt}.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)
            
        # Run the sweep with agent
        wandb.agent(sweep_id, function=train_model, count=count)
    
    # Print summary of all sweeps
    print("\nSummary of sweeps:")
    for opt, sweep_id in sweep_ids:
        print(f"- {opt}: {sweep_id}")
    
    # Save sweep IDs for future reference
    with open("../../sweep_results/diffusion_sweep_ids.json", 'w') as f:
        json.dump([(o, s) for o, s in sweep_ids], f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run W&B sweeps for diffusion model on MNIST")
    parser.add_argument("--optimizer", type=str, choices=["adam", "vradam", "both"], default="both",
                       help="Which optimizer to sweep (default: both)")
    parser.add_argument("--count", type=int, default=10, help="Number of runs per sweep (default: 10)")
    
    args = parser.parse_args()
    
    run_sweeps(args.optimizer, args.count) 