import os
import torch
import gymnasium as gym
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarker import Benchmarker
from rl_architectures import PolicyNetwork

def load_or_train_model(optimizer_type, device=None, load_existing=True, save_model=True):
    """
    Load an existing model or train a new one if not available
    
    Args:
        optimizer_type: 'VADAM' or 'ADAM'
        device: Device to use (cuda, mps, or cpu)
        load_existing: Whether to try loading an existing model
        save_model: Whether to save the model after training
        
    Returns:
        Trained policy network
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Define model path
    model_dir = "../saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"halfcheetah_{optimizer_type.lower()}.pt")
    
    # Try to load existing model if requested
    if load_existing and os.path.exists(model_path):
        print(f"Loading existing {optimizer_type} model from {model_path}")
        # Create environment to get state and action dimensions
        env = gym.make('HalfCheetah-v4')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        
        # Create policy network
        policy_net = PolicyNetwork(
            input_dim=state_dim,
            hidden_dim=256,
            output_dim=action_dim
        ).to(device)
        
        # Load state dict
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        return policy_net
    
    # Otherwise train a new model
    print(f"Training new {optimizer_type} model...")
    
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
        'epochs': 10,  # Use 30 for better results
        
        # RL specific parameters
        'gamma': 0.99,
        'entropy_coef': 0.01
    }
    
    # Add optimizer-specific parameters
    if optimizer_type == 'VADAM':
        params.update({
            'eta': 0.01,
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
            'lr': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0,
            'eps': 1e-8
        })
    
    # Create and run benchmarker
    benchmark = Benchmarker(params)
    results = benchmark.run()
    
    # Get reference to policy network
    policy_net = benchmark.policy_net
    
    # Save model if requested
    if save_model:
        print(f"Saving {optimizer_type} model to {model_path}")
        torch.save(policy_net.state_dict(), model_path)
    
    return policy_net

def record_animation(policy_net, optimizer_type, episodes=1, max_steps=1000, render_mode='rgb_array'):
    """
    Record an animation of the agent's performance
    
    Args:
        policy_net: Trained policy network
        optimizer_type: 'VADAM' or 'ADAM' (for labeling)
        episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        render_mode: Render mode for the environment
        
    Returns:
        List of frames (if render_mode is 'rgb_array')
    """
    env = gym.make('HalfCheetah-v4', render_mode=render_mode)
    frames = []
    rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action from policy
            action = policy_net.get_action(state, deterministic=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Render environment
            if render_mode == 'rgb_array':
                frame = env.render()
                frames.append(frame)
            elif render_mode == 'human':
                env.render()
                time.sleep(0.01)  # Small delay for visualization
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {ep+1} reward: {episode_reward:.2f}")
    
    env.close()
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    
    return frames, avg_reward

def save_animation(frames, filename, fps=60):
    """Save frames as an animated gif"""
    if not frames:
        print("No frames to save")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"Creating animation with {len(frames)} frames at {fps} fps")
    
    # Use matplotlib to create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()
    image = ax.imshow(frames[0])
    
    def update(frame):
        image.set_array(frame)
        return [image]
    
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=fps)
    ani.save(filename, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved to {filename}")

def compare_optimizers(device=None, load_existing=True, save_models=True, episodes=3):
    """Compare VADAM and Adam optimizers by rendering their trained policies"""
    # Load or train models
    vadam_policy = load_or_train_model('VADAM', device, load_existing, save_models)
    adam_policy = load_or_train_model('ADAM', device, load_existing, save_models)
    
    # Create output directory
    output_dir = "../animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Record animations
    print("Recording VADAM policy...")
    vadam_frames, vadam_reward = record_animation(vadam_policy, 'VADAM', episodes)
    
    print("Recording Adam policy...")
    adam_frames, adam_reward = record_animation(adam_policy, 'ADAM', episodes)
    
    # Save animations
    save_animation(vadam_frames, os.path.join(output_dir, 'halfcheetah_vadam.gif'))
    save_animation(adam_frames, os.path.join(output_dir, 'halfcheetah_adam.gif'))
    
    # Create side-by-side comparison (take min length of frames)
    min_frames = min(len(vadam_frames), len(adam_frames))
    
    if min_frames > 0:
        print("Creating side-by-side comparison...")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Set titles with rewards
        ax1.set_title(f'VADAM (Reward: {vadam_reward:.2f})')
        ax2.set_title(f'Adam (Reward: {adam_reward:.2f})')
        
        # Turn off axes
        ax1.set_axis_off()
        ax2.set_axis_off()
        
        # Create initial images
        img1 = ax1.imshow(vadam_frames[0])
        img2 = ax2.imshow(adam_frames[0])
        
        def update(i):
            img1.set_array(vadam_frames[i])
            img2.set_array(adam_frames[i])
            return [img1, img2]
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=min_frames, blit=True)
        
        # Save side-by-side animation
        writer = PillowWriter(fps=60)
        ani.save(os.path.join(output_dir, 'halfcheetah_comparison.gif'), writer=writer)
        plt.close(fig)
        
        print(f"Side-by-side comparison saved to {output_dir}/halfcheetah_comparison.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render HalfCheetah environment with trained policies')
    parser.add_argument('--optimizer', type=str, default=None, choices=['VADAM', 'ADAM', 'both'], 
                      help='Optimizer to visualize (VADAM, ADAM, or both)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--train', action='store_true', help='Force training new models instead of loading existing ones')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to record')
    
    args = parser.parse_args()
    
    if args.optimizer is None or args.optimizer == 'both':
        compare_optimizers(args.device, not args.train, True, args.episodes)
    else:
        # Load or train the specified model
        policy_net = load_or_train_model(args.optimizer, args.device, not args.train, True)
        
        # Record animation
        frames, avg_reward = record_animation(policy_net, args.optimizer, args.episodes)
        
        # Save animation
        output_dir = "../animations"
        os.makedirs(output_dir, exist_ok=True)
        save_animation(frames, os.path.join(output_dir, f'halfcheetah_{args.optimizer.lower()}.gif')) 