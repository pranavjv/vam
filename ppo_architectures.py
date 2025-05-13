import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np

class AddBias(nn.Module):
    """Bias Layer to help exploration (makes PPO converge a lot faster)"""
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

def build_NN(input_dim, output_dim, hidden_layers, activation="ReLU"):
    """Helper function to build a neural network with specified layers"""
    layers = []
    prev_dim = input_dim
    
    # Create activation function
    if activation == "ReLU":
        act_fn = nn.ReLU()
    elif activation == "Tanh":
        act_fn = nn.Tanh()
    elif activation == "LeakyReLU":
        act_fn = nn.LeakyReLU()
    else:
        act_fn = nn.ReLU()  # Default
    
    # Build hidden layers
    for dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(act_fn)
        prev_dim = dim
    
    # Add output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)

class Critic(nn.Module):
    """Critic class for PPO algorithm that has the value function for the agent"""
    def __init__(self, input_dim, output_dim, hidden_layers, activation="ReLU"):
        super().__init__()
        self.critic = build_NN(input_dim, 1, hidden_layers, activation)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use orthogonal initialization with higher gain for the critic
            nn.init.orthogonal_(module.weight.data, gain=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, obs):
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
            
        value = self.critic(obs)
        return value[:,0]

class Actor(nn.Module):
    """Actor class for the PPO algorithm that has state-action policy"""
    def __init__(self, input_dim, output_dim, hidden_layers, activation="ReLU"):
        super().__init__()
        self.pi = build_NN(input_dim, output_dim, hidden_layers, activation)
        self.logstd = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use orthogonal initialization for better training dynamics
            nn.init.orthogonal_(module.weight.data, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state, noise=True):
        """Forward method that returns actions, log probs, and distributions"""
        # Ensure state is on the same device as the model
        device = next(self.parameters()).device
        if state.device != device:
            state = state.to(device)
            
        action_mean = self.pi(state)
        # Constrain actions to (-1, 1) with tanh for stable HalfCheetah control
        action_mean = torch.tanh(action_mean)
        
        # Use state-independent log standard deviation
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Clamp std to avoid numerical instability
        action_std = torch.clamp(action_std, min=1e-6, max=1.0)
        
        if noise:
            # Create normal distribution and sample actions
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            # Clamp actions to valid range
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(-1)
        else:
            action = action_mean
            dist = None
            log_prob = None
            
        return action, log_prob, dist

    def evaluate(self, state, action):
        """Evaluate actions given states to get log probs and entropy"""
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        if state.device != device:
            state = state.to(device)
        if action.device != device:
            action = action.to(device)
            
        action_mean = self.pi(state)
        action_mean = torch.tanh(action_mean)
        
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_std = torch.clamp(action_std, min=1e-6, max=1.0)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Calculate log probs and entropy
        action_log_probs = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return action_log_probs, entropy

    def get_action(self, state, deterministic=False):
        # For compatibility with the existing code
        device = next(self.parameters()).device
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        state = state.unsqueeze(0).to(device)
        
        with torch.no_grad():
            if deterministic:
                action_mean = self.pi(state)
                action = torch.tanh(action_mean)
            else:
                action, _, _ = self.forward(state)
        
        return action.cpu().detach().numpy()[0]

class PPOAgent(nn.Module):
    """PPO agent that contains the actor and critic networks"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, eps_clip=0.2, 
                 entropy_coef=0.01, lr=0.0003, batch_size=64, device='cpu'):
        super().__init__()
        self.device = device
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.hidden_layers = [hidden_dim, hidden_dim]
        self.activation = "ReLU"
        
        # Create actor and critic networks
        self.pi = Actor(state_dim, action_dim, self.hidden_layers, self.activation)
        self.pi_old = Actor(state_dim, action_dim, self.hidden_layers, self.activation)
        self.pi_old.load_state_dict(self.pi.state_dict())
        self.critic = Critic(state_dim, action_dim, self.hidden_layers, self.activation)
        
        # PPO parameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.grad_clip = 0.5
        self.policy_updates = 5
        self.lr = lr
        
        # Running mean and std for reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0
        
        # Optimizers
        self.actor_optim = None
        self.critic_optim = None
        
    def setup_optimizers(self, optimizer_name, optimizer_params):
        """Set up optimizers based on optimizer_name and params"""
        if optimizer_name == 'ADAM':
            # Use lower learning rates for stability
            actor_lr = self.lr
            critic_lr = self.lr * 1.5  # Slightly higher for critic
            
            self.actor_optim = torch.optim.Adam(self.pi.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        elif optimizer_name == 'VADAM':
            # This will be handled by the benchmarker
            pass
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
    def forward(self, obs, noise=True):
        """Agent step function"""
        action, log_prob, dist = self.pi.forward(obs, noise)
        return action, log_prob, dist
    
    def evaluate(self, obs, actions):
        """Evaluation function for the PPO agent"""
        log_prob, entropy = self.pi.evaluate(obs, actions)
        values = self.critic(obs)
        return log_prob, entropy, values
    
    def normalize_rewards(self, rewards):
        """Normalize rewards using running statistics"""
        # Update running statistics for reward normalization
        batch_mean = np.mean(rewards)
        batch_std = np.std(rewards)
        batch_count = len(rewards)
        
        # Update running stats
        self.reward_count += batch_count
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * batch_count / self.reward_count
        
        # Update variance
        new_m2 = np.sum((rewards - batch_mean) ** 2)
        self.reward_std = np.sqrt((self.reward_std ** 2 * (self.reward_count - batch_count) + new_m2) / self.reward_count)
        
        # Normalize rewards
        if self.reward_std > 0:
            normalized_rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        else:
            normalized_rewards = rewards - self.reward_mean
            
        return normalized_rewards
    
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(rewards)
        
        # Calculate advantages using GAE
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            running_advantage = deltas[t] + gamma * lam * (1 - dones[t]) * running_advantage
            advantages[t] = running_advantage
            
        # Compute returns
        returns = advantages + values
        
        return returns, advantages
    
    def update(self, obs, actions, rewards, next_obs, terminals, action_logprob, old_values):
        """Update function that trains the actor and critic networks"""
        # Convert numpy arrays to tensors and move to the right device
        device = next(self.parameters()).device
        obs = torch.FloatTensor(obs).to(device)
        actions = torch.FloatTensor(actions).to(device)
        
        # Normalize rewards and compute returns with GAE
        normalized_rewards = self.normalize_rewards(rewards)
        
        # Calculate values for next observations (bootstrapping)
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).to(device)
            next_values = self.critic(next_obs_tensor).cpu().numpy()
            
        # Compute returns and advantages using GAE
        returns, advantages = self.compute_gae(
            normalized_rewards,
            old_values,
            next_values,
            terminals,
            gamma=self.gamma
        )
        
        # Convert returns and advantages to tensors
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        old_action_logprob = torch.FloatTensor(action_logprob).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch size
        batch_size = min(self.batch_size, obs.size(0))
        
        # PPO update
        for _ in range(self.policy_updates):
            # Create mini-batches
            indices = torch.randperm(obs.size(0))
            
            for start_idx in range(0, obs.size(0), batch_size):
                # Get mini-batch
                idx = indices[start_idx:start_idx + batch_size]
                
                # Mini-batch data
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_old_action_logprob = old_action_logprob[idx]
                
                # Get current evaluations
                new_logprob, entropy, values = self.evaluate(mb_obs, mb_actions)
                
                # Calculate ratios and policy loss
                ratio = torch.exp(new_logprob - mb_old_action_logprob)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy bonus for exploration
                policy_loss = policy_loss - self.entropy_coef * entropy.mean()
                
                # Update actor
                self.actor_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip)
                self.actor_optim.step()
                
                # Calculate value loss with clipping
                value_loss = F.mse_loss(values, mb_returns)
                
                # Update critic
                self.critic_optim.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_optim.step()
        
        # Update old policy
        self.pi_old.load_state_dict(self.pi.state_dict())
        
        return policy_loss.item(), value_loss.item()
        
    def get_action(self, state, deterministic=False):
        """Get action for a given state (compatible with benchmarker)"""
        return self.pi.get_action(state, deterministic) 