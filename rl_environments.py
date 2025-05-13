import torch
import numpy as np
import gymnasium as gym
from torch.utils.data import Dataset, DataLoader

class RLEnvironment:
    def __init__(self, env_name='HalfCheetah-v4', batch_size=64, device='cpu'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.batch_size = batch_size
        self.device = device
        
    def sample_batch(self, policy_net, n_steps=2048):
        """Sample batch of trajectories from the environment"""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        total_reward = 0
        
        state, _ = self.env.reset()
        for _ in range(n_steps):
            # Get action from policy network
            action = policy_net.get_action(state)
                
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            total_reward += reward
            
            # Update state
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'avg_reward': total_reward / n_steps
        }
    
    def evaluate_policy(self, policy_net, n_episodes=5):
        """Evaluate policy over multiple episodes"""
        total_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                # Get deterministic action
                action = policy_net.get_action(state, deterministic=True)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                step_count += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(step_count)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_length': np.mean(episode_lengths)
        }
    
    def get_dataloader(self, states, actions, batch_size=None):
        """Create a dataloader from collected states and actions"""
        if batch_size is None:
            batch_size = self.batch_size
            
        dataset = RLDataset(states, actions)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class RLDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx] 