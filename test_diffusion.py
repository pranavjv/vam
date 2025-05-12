import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from diffusion_model import SimpleUNet, DiffusionModel

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a small subset of MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_loader = DataLoader(mnist, batch_size=16, shuffle=True)
    
    # Create model
    model = SimpleUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        time_emb_dim=64
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create diffusion process
    diffusion = DiffusionModel(
        model=model,
        beta_min=1e-4,
        beta_max=0.02,
        num_timesteps=100,  # Small for testing
        device=device
    )
    
    # Test forward pass with a batch
    for images, _ in test_loader:
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion process (add noise)
        noisy_images = diffusion.q_sample(images, t)
        
        # Predict noise with model
        pred_noise = model(noisy_images, t)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(pred_noise, torch.randn_like(images))
        print(f"Example loss: {loss.item()}")
        
        # Test sampling
        with torch.no_grad():
            samples = diffusion.sample(batch_size=4, img_size=28)
            
        # Plot samples
        samples_np = (samples.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        
        os.makedirs('test_samples', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples_np[i, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_samples/test_samples.png')
        plt.close()
        
        print(f"Sample images saved to test_samples/test_samples.png")
        break  # Just test one batch

if __name__ == "__main__":
    main() 