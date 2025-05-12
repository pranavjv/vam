import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # For action distribution
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        
        # Return mean and log_std for the Gaussian policy
        return mean, self.log_std.exp()
    
    def get_action(self, state, deterministic=False):
        # Convert state to tensor and move to the device where the model is
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        state = state.unsqueeze(0).to(next(self.parameters()).device)
        
        with torch.no_grad():
            mean, std = self.forward(state)
            
            if deterministic:
                action = mean
            else:
                normal = torch.distributions.Normal(mean, std)
                action = normal.sample()
        
        # Convert back to numpy
        return action.cpu().detach().numpy()[0]

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value 