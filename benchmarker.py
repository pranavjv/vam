import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import os
import time
import math

# Import our custom text dataset implementations
import text_datasets

from VRADAM import VRADAM
import architectures


class GridWorldEnvironment:
    """
    Grid World environment for GFlowNet experiments.
    Provides methods for generating trajectories and computing rewards.
    """
    def __init__(self, grid_size=8, batch_size=64, device='cpu'):
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.device = device
        
        # Define action space: 0=up, 1=right, 2=down, 3=left, 4=stop
        self.action_dim = 5
        
        # Define state space: 3 channels (grid, position, visitation)
        self.state_dim = 3
        
        # Define reward function parameters
        self.reward_params = {
            'pattern_reward': 10.0,  # Reward for creating specific patterns
            'coverage_reward': 5.0,  # Reward for covering more of the grid
            'efficiency_reward': 3.0,  # Reward for using fewer steps
            'symmetry_reward': 4.0,  # Reward for creating symmetric patterns
        }
    
    def generate_batch(self, batch_size=None):
        """
        Generate a batch of initial states
        
        Returns:
            states: Initial states [batch_size, state_dim, grid_size, grid_size]
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Create empty grids
        states = torch.zeros(batch_size, self.state_dim, 
                           self.grid_size, self.grid_size, 
                           device=self.device)
        
        # Start at center position
        pos_x, pos_y = self.grid_size // 2, self.grid_size // 2
        states[:, 1, pos_y, pos_x] = 1.0  # Mark initial position
        
        return states
    
    def compute_rewards(self, final_states):
        """
        Compute rewards for final states
        
        Args:
            final_states: Final grid states [batch_size, state_dim, grid_size, grid_size]
            
        Returns:
            rewards: Reward values [batch_size]
        """
        batch_size = final_states.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)
        
        # Extract grid channel (channel 0)
        grids = final_states[:, 0]  # [batch_size, grid_size, grid_size]
        
        # Extract visitation map (channel 2)
        visits = final_states[:, 2]  # [batch_size, grid_size, grid_size]
        
        # Compute coverage (percentage of grid filled)
        coverage = grids.sum(dim=(1, 2)) / (self.grid_size * self.grid_size)
        
        # Compute efficiency (inverse of redundant visits)
        total_visits = visits.sum(dim=(1, 2))
        unique_visits = (visits > 0).float().sum(dim=(1, 2))
        efficiency = unique_visits / (total_visits + 1e-6)  # Avoid division by zero
        
        # Compute symmetry (horizontal and vertical)
        h_symmetry = 1.0 - (grids - torch.flip(grids, [2])).abs().mean(dim=(1, 2))
        v_symmetry = 1.0 - (grids - torch.flip(grids, [1])).abs().mean(dim=(1, 2))
        symmetry = (h_symmetry + v_symmetry) / 2.0
        
        # Compute pattern rewards (example: reward for creating specific shapes)
        # Here we reward grid cells that form lines or clusters
        pattern_scores = torch.zeros(batch_size, device=self.device)
        
        # Convolve with kernels to detect patterns
        # Example: horizontal and vertical lines
        h_kernel = torch.tensor([[1, 1, 1]], dtype=torch.float32, device=self.device).view(1, 1, 1, 3)
        v_kernel = torch.tensor([[1], [1], [1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 1)
        
        # Reshape grids for convolution
        grids_reshaped = grids.unsqueeze(1)  # [batch_size, 1, grid_size, grid_size]
        
        # Detect horizontal lines
        h_conv = F.conv2d(grids_reshaped, h_kernel, padding=(0, 1))
        h_lines = (h_conv == 3).float().sum(dim=(1, 2, 3))
        
        # Detect vertical lines
        v_conv = F.conv2d(grids_reshaped, v_kernel, padding=(1, 0))
        v_lines = (v_conv == 3).float().sum(dim=(1, 2, 3))
        
        # Combine pattern scores
        pattern_scores = h_lines + v_lines
        
        # Combine all reward components
        rewards = (
            self.reward_params['pattern_reward'] * pattern_scores / 10.0 +
            self.reward_params['coverage_reward'] * coverage +
            self.reward_params['efficiency_reward'] * efficiency +
            self.reward_params['symmetry_reward'] * symmetry
        )
        
        return rewards
    
    def evaluate_trajectories(self, trajectories):
        """
        Evaluate a batch of trajectories
        
        Args:
            trajectories: List of states in trajectories
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Get final states
        final_states = trajectories[-1]
        
        # Compute rewards
        rewards = self.compute_rewards(final_states)
        
        # Compute metrics
        metrics = {
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_max': rewards.max().item(),
            'reward_min': rewards.min().item(),
        }
        
        # Compute diversity (average pairwise distance between final grids)
        grids = final_states[:, 0]  # [batch_size, grid_size, grid_size]
        flat_grids = grids.view(grids.size(0), -1)  # [batch_size, grid_size*grid_size]
        
        # Compute pairwise Hamming distances
        if flat_grids.size(0) > 1:
            distances = []
            for i in range(flat_grids.size(0)):
                for j in range(i+1, flat_grids.size(0)):
                    dist = (flat_grids[i] != flat_grids[j]).float().mean().item()
                    distances.append(dist)
            
            metrics['diversity'] = np.mean(distances) if distances else 0.0
        else:
            metrics['diversity'] = 0.0
        
        return metrics

class Benchmarker:
    def __init__(self,p= None):
        # Default parameters
        default_params = {
            'model': 'SimpleCNN', # 'SimpleCNN', 'TransformerModel', 'MLPModel', 'RLPolicy', 'PPOPolicy', 'GFlowNetModel'
            'device': 'mps', # 'mps' or 'cpu'
            'dataset': 'CIFAR10', # 'CIFAR10', 'WikiText2', 'IMDB', 'HalfCheetah', 'GridWorld'
            'dataset_size': 'small', # 'small' or 'full'
            'optimizer': 'ADAM', # 'ADAM', 'VRADAM', 'SGD', 'RMSPROP'
            'batch_size': 128,
            'max_seq_len': 256,  # For NLP tasks
            'embed_dim': 300,    # For MLP and Transformer models
            'hidden_dim': 512,   # For MLP model
            'lr': 0.001,         # Learning rate
            'epochs': 5,         # Number of training epochs
            # Add seed for reproducible weight initialization
            'seed': 0,           # Set to an integer for reproducible initialization
            # PPO specific parameters
            'eps_clip': 0.2,     # PPO clipping parameter
            'entropy_coef': 0.01, # Entropy coefficient for PPO
            # GFlowNet specific parameters
            'grid_size': 8,      # Grid size for GFlowNet
            'num_layers': 3,     # Number of layers in GFlowNet
            'action_dim': 5,     # Action dimension for GFlowNet
            'state_dim': 3,      # State dimension for GFlowNet
            'flow_matching_weight': 1.0,  # Weight for flow matching loss
            'reward_weight': 1.0,  # Weight for reward loss
            'entropy_weight': 0.01,  # Weight for entropy regularization
            'reward_temp': 1.0,  # Temperature for reward scaling
        }

        if p is None:
            p = default_params.copy() # Use copy to avoid modifying default dict
        else:
            self.p = p

        # Ensure all required parameters are in the dict
        for key, val in default_params.items():
            if key not in self.p:
                self.p[key] = val

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

        # Initialize training metrics tracking
        self.train_losses = []
        self.train_accs = []  # Track training accuracy
        self.val_losses = []
        self.val_accs = []    # Track validation accuracy
        self.train_perplexities = []  # For language modeling
        self.val_perplexities = []    # For language modeling
        self.test_loss = None
        self.test_acc = None
        self.test_perplexity = None   # For language modeling
        self.final_train_loss = None
        self.final_train_acc = None
        self.final_train_perplexity = None  # For language modeling
        self.train_time = None
        self.results = None
        
        # For RL
        self.episode_rewards = []
        self.mean_rewards = []
        # For PPO specific metrics
        self.policy_losses = []
        self.value_losses = []
        
        # For GFlowNet specific metrics
        self.flow_match_losses = []
        self.reward_losses = []
        self.entropy = []
        self.sample_diversity = None
        self.reward_mean = None
        self.reward_max = None
        
        # --- Set Seed for Reproducible Weight Initialization ---
        seed = self.p.get('seed')
        if seed is not None:
            print(f"Setting random seed to {seed} for model initialization.")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed) # for multi-GPU
            np.random.seed(seed)
            # Ensure deterministic algorithms are used
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # If no seed, allow non-deterministic algorithms
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True # Can improve performance

    def setup_data(self):
        if self.p['dataset'] == 'CIFAR10':
            transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
        ])

            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_set = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
                                        ]))

            if self.p['dataset_size'] == "small":
                # Dataloading
                subset_size = int(0.002 * len(train_set))
                val_size = int(0.001 * len(train_set))
                rest_size = len(train_set) - subset_size - val_size
                small_train, val_set, _ = random_split(train_set, [subset_size, val_size, rest_size])

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
            elif self.p['dataset_size'] == "full":
                train_size = int(0.8 * len(train_set))
                val_size = len(train_set) - train_size
                train_subset, val_set = random_split(train_set, [train_size, val_size])
                self.train_loader = DataLoader(train_subset, batch_size=self.p['batch_size'], shuffle=True, num_workers=4)
                self.val_loader = DataLoader(val_set, batch_size=self.p['batch_size'], shuffle=False, num_workers=4)
            
            self.test_loader = DataLoader(test_set, batch_size=self.p['batch_size']*2, shuffle=False, num_workers=4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.task_type = "classification"
            
            # Check if this is a contrastive learning task
            if self.p['model'] == 'ContrastiveModel':
                # Override with contrastive learning-specific setup
                self.task_type = "contrastive_learning"
                
                # Define strong augmentations for contrastive learning
                if self.p.get('aug_strength', 'strong') == 'strong':
                    contrastive_train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])
                else:
                    # Moderate augmentations
                    contrastive_train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])
                
                # Create a wrapper for contrastive dataset that returns two augmented views of each image
                class ContrastiveDataset(torch.utils.data.Dataset):
                    def __init__(self, dataset, transform):
                        self.dataset = dataset
                        self.transform = transform
                    
                    def __getitem__(self, idx):
                        img, label = self.dataset[idx]
                        img1 = self.transform(img)
                        img2 = self.transform(img)
                        return img1, img2, label
                    
                    def __len__(self):
                        return len(self.dataset)
                
                # Use original train and val sets before splitting
                train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
                
                if self.p['dataset_size'] == "small":
                    # Use small subset for faster iterations
                    subset_size = int(0.1 * len(train_dataset))
                    val_size = int(0.02 * len(train_dataset))
                    rest_size = len(train_dataset) - subset_size - val_size
                    
                    # Create new generator with fixed seed for reproducible splits
                    generator = torch.Generator().manual_seed(self.p.get('seed', 0))
                    train_subset, val_subset, _ = random_split(train_dataset, [subset_size, val_size, rest_size], generator=generator)
                else:
                    # Use full dataset for more robust representation learning
                    train_size = int(0.9 * len(train_dataset))
                    val_size = len(train_dataset) - train_size
                    
                    # Create new generator with fixed seed for reproducible splits
                    generator = torch.Generator().manual_seed(self.p.get('seed', 0))
                    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
                
                # Wrap datasets for contrastive learning
                contrastive_train_dataset = ContrastiveDataset(train_subset, contrastive_train_transform)
                contrastive_val_dataset = ContrastiveDataset(val_subset, contrastive_train_transform)
                
                # Create data loaders
                self.train_loader = DataLoader(
                    contrastive_train_dataset,
                    batch_size=self.p['batch_size'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                
                self.val_loader = DataLoader(
                    contrastive_val_dataset,
                    batch_size=self.p['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Define contrastive loss - using NT-Xent (normalized temperature-scaled cross entropy)
                self.temperature = self.p.get('temperature', 0.5)
                
                # Keep the regular test loader for downstream classification task evaluation
            
        elif self.p['dataset'] == 'WikiText2':
            try:
                # Check if we should use the tiny dataset for quick testing
                if self.p['dataset_size'] == 'small':
                    print("Using tiny WikiText2 dataset for quick testing")
                    self.train_loader, self.val_loader, self.test_loader, self.vocab = text_datasets.get_tiny_datasets(
                        self.p['batch_size'], 'wikitext2', self.device
                    )
                else:
                    # Use the full dataset
                    self.train_loader, self.val_loader, self.test_loader, self.vocab = text_datasets.get_wikitext2_iterators(
                        self.p['batch_size'], self.p['max_seq_len'], self.device
                    )
                
                self.vocab_size = len(self.vocab)
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])
                self.task_type = "language_modeling"
                
            except Exception as e:
                print(f"Error loading WikiText2 dataset: {e}")
                # Create fallback tiny dataset
                print("Falling back to manually created tiny WikiText2 dataset")
                self.train_loader, self.val_loader, self.test_loader, self.vocab = text_datasets.get_tiny_datasets(
                    self.p['batch_size'], 'wikitext2', self.device
                )
                self.vocab_size = len(self.vocab)
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])
                self.task_type = "language_modeling"

        elif self.p['dataset'] == 'IMDB':
            try:
                # Check if we should use tiny dataset for quick testing
                if self.p['dataset_size'] == 'small':
                    print("Using tiny IMDB dataset for quick testing")
                    self.train_loader, self.val_loader, self.test_loader, self.vocab = text_datasets.get_tiny_datasets(
                        self.p['batch_size'], 'imdb', self.device
                    )
                else:
                    # Use the full dataset
                    self.train_loader, self.val_loader, self.test_loader, self.vocab = text_datasets.get_imdb_iterators(
                        self.p['batch_size'], self.device
                    )
                
                self.vocab_size = len(self.vocab)
                self.criterion = torch.nn.CrossEntropyLoss()
                self.task_type = "text_classification"
                self.num_classes = 2  # Binary classification
                
            except Exception as e:
                print(f"Error loading IMDB dataset: {e}")
                # Create fallback tiny dataset
                print("Falling back to manually created tiny IMDB dataset")
                self.train_loader, self.val_loader, self.test_loader, self.vocab = text_datasets.get_tiny_datasets(
                    self.p['batch_size'], 'imdb', self.device
                )
                self.vocab_size = len(self.vocab)
                self.criterion = torch.nn.CrossEntropyLoss()
                self.task_type = "text_classification"
                self.num_classes = 2  # Binary classification

        elif self.p['dataset'] == 'HalfCheetah':
            # For HalfCheetah reinforcement learning
            self.rl_env = rl_environments.RLEnvironment(
                env_name='HalfCheetah-v4',
                batch_size=self.p['batch_size'],
                device=self.device
            )
            self.criterion = torch.nn.MSELoss()  # For policy gradient loss
            self.task_type = "reinforcement_learning"
            
        elif self.p['dataset'] == 'GridWorld':
            # For GFlowNet grid world environment
            self.grid_env = GridWorldEnvironment(
                grid_size=self.p.get('grid_size', 8),
                batch_size=self.p.get('batch_size', 64),
                device=self.device
            )
            
            # Set task type
            self.task_type = "gflownet"
            
            # No traditional data loaders for GFlowNet
            # We'll generate data on-the-fly during training

    def setup_training(self):
        # --- Set Seed for Reproducible Weight Initialization ---
        seed = self.p.get('seed')
        if seed is not None:
            print(f"Setting random seed to {seed} for model initialization.")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed) # for multi-GPU
            np.random.seed(seed)
            # Ensure deterministic algorithms are used
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # If no seed, allow non-deterministic algorithms
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True # Can improve performance
            
        # --- Model Instantiation (AFTER setting seed) ---
        if self.p['model'] == 'SimpleCNN':
            self.model = architectures.SimpleCNN().to(self.device)
        
        elif self.p['model'] == 'ContrastiveModel':
            # Setup for contrastive learning model
            self.model = architectures.ContrastiveModel(
                embedding_dim=self.p.get('embedding_dim', 128),
                projection_dim=self.p.get('projection_dim', 64)
            ).to(self.device)
            
            # Define contrastive loss function (NT-Xent)
            self.contrastive_criterion = self._nt_xent_loss
        
        elif self.p['model'] == 'TransformerModel':
            # Setup for transformer model
            if self.p['dataset'] in ['WikiText2']:
                self.model = architectures.TransformerModel(
                    vocab_size=self.vocab_size,
                    d_model=self.p['embed_dim'],
                    nhead=8,
                    num_encoder_layers=2,
                    num_decoder_layers=2,
                    dim_feedforward=self.p['hidden_dim'],
                    dropout=0.2
                ).to(self.device)
            else:
                raise ValueError(f"TransformerModel not compatible with dataset: {self.p['dataset']}")
        
        elif self.p['model'] == 'MLPModel':
            # Setup for MLP model
            if self.p['dataset'] == 'IMDB':
                self.model = architectures.MLPModel(
                    vocab_size=self.vocab_size,
                    embed_dim=self.p['embed_dim'],
                    hidden_dim=self.p['hidden_dim'],
                    num_classes=self.num_classes,
                    dropout=0.5
                ).to(self.device)
            else:
                raise ValueError(f"MLPModel not compatible with dataset: {self.p['dataset']}")
        
        elif self.p['model'] == 'RLPolicy':
            # For reinforcement learning policy
            if self.p['dataset'] == 'HalfCheetah':
                # Initialize policy and value networks for HalfCheetah
                self.policy_net = rl_architectures.PolicyNetwork(
                    input_dim=self.rl_env.state_dim,
                    hidden_dim=self.p['hidden_dim'],
                    output_dim=self.rl_env.action_dim
                ).to(self.device)
                
                self.value_net = rl_architectures.ValueNetwork(
                    input_dim=self.rl_env.state_dim,
                    hidden_dim=self.p['hidden_dim']
                ).to(self.device)
                
                # Reference to the policy network as the main model
                self.model = self.policy_net
            else:
                raise ValueError(f"RLPolicy not compatible with dataset: {self.p['dataset']}")
        elif self.p['model'] == 'PPOPolicy':
            # For PPO policy
            if self.p['dataset'] == 'HalfCheetah':
                # Initialize PPO agent
                self.ppo_agent = PPOAgent(
                    state_dim=self.rl_env.state_dim,
                    action_dim=self.rl_env.action_dim,
                    hidden_dim=self.p['hidden_dim'],
                    lr=self.p['lr'],
                    eps_clip=self.p['eps_clip'],
                    entropy_coef=self.p['entropy_coef']
                ).to(self.device)
                
                # Reference to the PPO agent as the main model
                self.model = self.ppo_agent
            else:
                raise ValueError(f"PPOPolicy not compatible with dataset: {self.p['dataset']}")
        elif self.p['model'] == 'GFlowNetModel':
            # Setup for GFlowNet model
            self.model = architectures.GFlowNetModel(
                grid_size=self.p.get('grid_size', 8),
                hidden_dim=self.p.get('hidden_dim', 128),
                num_layers=self.p.get('num_layers', 3),
                action_dim=self.p.get('action_dim', 5),
                state_dim=self.p.get('state_dim', 3)
            ).to(self.device)
            
            # GFlowNet specific parameters
            self.flow_matching_weight = self.p.get('flow_matching_weight', 1.0)
            self.reward_weight = self.p.get('reward_weight', 1.0)
            self.entropy_weight = self.p.get('entropy_weight', 0.01)
            self.reward_temp = self.p.get('reward_temp', 1.0)
            
        else:
            raise ValueError(f"Unknown model: {self.p['model']}")

        # Setup optimizer
        if self.p['optimizer'] == "VRADAM":
            # VRADAM specific parameters
            vradam_params = {
                'beta1': self.p.get('beta1', 0.9),
                'beta2': self.p.get('beta2', 0.999),
                'beta3': self.p.get('beta3', 1.0),
                'eta': self.p.get('eta', self.p.get('lr', 0.001)),  # Use eta if provided, otherwise fall back to lr
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0),
                'power': self.p.get('power', 2),
                'normgrad': self.p.get('normgrad', True),
                'lr_cutoff': self.p.get('lr_cutoff', 19)
            }
            
            if self.p['model'] == 'RLPolicy':
                # Use VRADAM for both policy and value networks
                self.policy_optimizer = VRADAM(self.policy_net.parameters(), **vradam_params)
                self.value_optimizer = VRADAM(self.value_net.parameters(), **vradam_params)
                # Reference to policy optimizer as the main optimizer
                self.optimizer = self.policy_optimizer
            elif self.p['model'] == 'PPOPolicy':
                # Use VRADAM for the PPO agent
                self.optimizer = VRADAM(self.ppo_agent.parameters(), **vradam_params)
            else:
                self.optimizer = VRADAM(self.model.parameters(), **vradam_params)
                
        elif self.p['optimizer'] == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.p.get('lr', 0.001),
                'betas': (self.p.get('beta1', 0.9), self.p.get('beta2', 0.999)),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0)
            }
            
            if self.p['model'] == 'RLPolicy':
                # Use Adam for both policy and value networks
                self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), **adam_params)
                self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), **adam_params)
                # Reference to policy optimizer as the main optimizer
                self.optimizer = self.policy_optimizer
            elif self.p['model'] == 'PPOPolicy':
                # Use Adam for the PPO agent
                self.optimizer = torch.optim.Adam(self.ppo_agent.parameters(), **adam_params)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_params)
        elif self.p['optimizer'] == "SGD":
            # SGD parameters
            sgd_params = {
                'lr': self.p.get('lr', 0.01),
                'momentum': self.p.get('momentum', 0.9),
                'weight_decay': self.p.get('weight_decay', 0),
                'nesterov': self.p.get('nesterov', False)
            }
            
            if self.p['model'] == 'RLPolicy':
                # Use SGD for both policy and value networks
                self.policy_optimizer = torch.optim.SGD(self.policy_net.parameters(), **sgd_params)
                self.value_optimizer = torch.optim.SGD(self.value_net.parameters(), **sgd_params)
                # Reference to policy optimizer as the main optimizer
                self.optimizer = self.policy_optimizer
            elif self.p['model'] == 'PPOPolicy':
                # Use SGD for the PPO agent
                self.optimizer = torch.optim.SGD(self.ppo_agent.parameters(), **sgd_params)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), **sgd_params)
        elif self.p['optimizer'] == "RMSPROP":
            # RMSprop parameters
            rmsprop_params = {
                'lr': self.p.get('lr', 0.001),
                'alpha': self.p.get('alpha', 0.99),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0),
                'momentum': self.p.get('momentum', 0)
            }
            
            if self.p['model'] == 'RLPolicy':
                # Use RMSprop for both policy and value networks
                self.policy_optimizer = torch.optim.RMSprop(self.policy_net.parameters(), **rmsprop_params)
                self.value_optimizer = torch.optim.RMSprop(self.value_net.parameters(), **rmsprop_params)
                # Reference to policy optimizer as the main optimizer
                self.optimizer = self.policy_optimizer
            elif self.p['model'] == 'PPOPolicy':
                # Use RMSprop for the PPO agent
                self.optimizer = torch.optim.RMSprop(self.ppo_agent.parameters(), **rmsprop_params)
            else:
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), **rmsprop_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.p['optimizer']}")

    def _nt_xent_loss(self, z_i, z_j):
        """
        Calculates NT-Xent loss for contrastive learning
        
        Args:
            z_i, z_j: Batch of projected features from different augmentations [batch_size, proj_dim]
        """
        batch_size = z_i.size(0)
        temperature = self.temperature
        
        # Concatenate representations to get all possible combinations
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, proj_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.t())  # [2*batch_size, 2*batch_size]
        
        # Normalization
        sim_i_j = torch.diag(similarity_matrix, batch_size)  # Similarity between i and j
        sim_j_i = torch.diag(similarity_matrix, -batch_size)  # Similarity between j and i
        
        # Positive samples are pairs (i,j) and (j,i)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Remove diagonals (self similarities are not considered)
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=self.device)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)  # [2*batch_size, 2*batch_size-1]
        
        # Create labels for cross-entropy loss: the positive sample is always at position 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
        
        # Concatenate positive and negative for softmax computation
        logits = torch.cat([positives.unsqueeze(1), negatives / temperature], dim=1)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def train_epoch(self, epoch):
        if self.task_type == "reinforcement_learning":
            # Reinforcement learning training loop
            
            if self.p['model'] == 'PPOPolicy':
                # PPO training
                
                # Setup optimizer for PPO based on optimizer type
                if not hasattr(self.ppo_agent, 'actor_optim') or self.ppo_agent.actor_optim is None:
                    if self.p['optimizer'] == 'ADAM':
                        optimizer_params = {
                            'lr': self.p.get('lr', 0.001),
                            'betas': (self.p.get('beta1', 0.9), self.p.get('beta2', 0.999)),
                            'eps': self.p.get('eps', 1e-8),
                            'weight_decay': self.p.get('weight_decay', 0)
                        }
                        self.ppo_agent.setup_optimizers('ADAM', optimizer_params)
                    elif self.p['optimizer'] == 'VRADAM':
                        optimizer_params = {
                            'lr': self.p.get('lr', 0.001),  # Not used directly but passed for consistency
                            'eta': self.p.get('eta', self.p.get('lr', 0.001)),
                            'beta1': self.p.get('beta1', 0.9),
                            'beta2': self.p.get('beta2', 0.999),
                            'beta3': self.p.get('beta3', 1.0),
                            'eps': self.p.get('eps', 1e-8),
                            'weight_decay': self.p.get('weight_decay', 0),
                            'power': self.p.get('power', 2),
                            'normgrad': self.p.get('normgrad', True),
                            'lr_cutoff': self.p.get('lr_cutoff', 19)
                        }
                        # VRADAM optimizers will be created in the benchmarker's train_ppo method
                
                # Sample batch of trajectories
                batch = self.rl_env.sample_batch(self.ppo_agent.pi, n_steps=2048)
                
                # Ensure all tensors are on the right device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                dones = batch['dones'].to(self.device)
                avg_reward = batch['avg_reward']
                
                # Convert for numpy processing
                states_np = states.cpu().numpy()
                actions_np = actions.cpu().numpy()
                rewards_np = rewards.cpu().numpy()
                next_states_np = next_states.cpu().numpy()
                dones_np = dones.cpu().numpy()
                
                # Compute values for the observed states
                with torch.no_grad():
                    values = self.ppo_agent.critic(states).cpu().numpy()
                    
                    # For VRADAM, we need to handle optimizer creation here
                    if self.p['optimizer'] == 'VRADAM' and (not hasattr(self.ppo_agent, 'actor_optim') or self.ppo_agent.actor_optim is None):
                        # Create VRADAM optimizers for the PPO agent's actor and critic
                        vradam_params = {
                            'beta1': self.p.get('beta1', 0.9),
                            'beta2': self.p.get('beta2', 0.999),
                            'beta3': self.p.get('beta3', 1.0),
                            'eta': self.p.get('eta', self.p.get('lr', 0.001)),
                            'eps': self.p.get('eps', 1e-8),
                            'weight_decay': self.p.get('weight_decay', 0),
                            'power': self.p.get('power', 2),
                            'normgrad': self.p.get('normgrad', True),
                            'lr_cutoff': self.p.get('lr_cutoff', 19)
                        }
                        self.ppo_agent.actor_optim = VRADAM(self.ppo_agent.pi.parameters(), **vradam_params)
                        self.ppo_agent.critic_optim = VRADAM(self.ppo_agent.critic.parameters(), **vradam_params)
                
                # Forward pass to get action logprobs
                _, action_logprob, _ = self.ppo_agent.forward(states)
                action_logprob = action_logprob.detach().cpu().numpy()
                
                # Update the PPO agent
                policy_loss, value_loss = self.ppo_agent.update(
                    states_np, actions_np, rewards_np.squeeze(), next_states_np, dones_np.squeeze(), action_logprob, values.squeeze()
                )
                
                # Track metrics and losses
                total_loss = policy_loss + value_loss
                
                print(f'Epoch {epoch} | Policy Loss: {policy_loss:.6f} | Value Loss: {value_loss:.6f} | Avg Reward: {avg_reward:.2f}')
                
                # Store the losses
                self.policy_losses.append(policy_loss)
                self.value_losses.append(value_loss)
                
                # Evaluate policy
                eval_results = self.rl_env.evaluate_policy(self.ppo_agent.pi)
                self.mean_rewards.append(eval_results['mean_reward'])
                
                print(f'Evaluation | Mean Reward: {eval_results["mean_reward"]:.2f} | Std Reward: {eval_results["std_reward"]:.2f}')
                
                return total_loss, None, None
            
            else:
                # Original RL Policy training
                
                # Sample batch of trajectories
                batch = self.rl_env.sample_batch(self.policy_net)
                
                states = batch['states']
                actions = batch['actions']
                rewards = batch['rewards']
                next_states = batch['next_states']
                dones = batch['dones']
                avg_reward = batch['avg_reward']
                
                # Compute returns and advantages
                with torch.no_grad():
                    values = self.value_net(states)
                    next_values = self.value_net(next_states)
                    
                    # Compute returns with bootstrapping
                    returns = rewards + (1 - dones) * self.p.get('gamma', 0.99) * next_values
                    
                    # Compute advantages
                    advantages = returns - values
                    
                # Update value network
                self.value_optimizer.zero_grad()
                current_values = self.value_net(states)
                value_loss = F.mse_loss(current_values, returns.detach())
                value_loss.backward()
                self.value_optimizer.step()
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                mean, std = self.policy_net(states)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
                
                # Policy gradient loss
                policy_loss = -(log_probs * advantages.detach()).mean()
                
                # Add entropy regularization
                entropy = dist.entropy().mean()
                policy_loss = policy_loss - self.p.get('entropy_coef', 0.01) * entropy
                
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # Record metrics
                total_loss = policy_loss.item() + value_loss.item()
                
                print(f'Epoch {epoch} | Policy Loss: {policy_loss.item():.6f} | Value Loss: {value_loss.item():.6f} | Avg Reward: {avg_reward:.2f}')
                
                # Store the losses
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                
                # Evaluate policy
                eval_results = self.rl_env.evaluate_policy(self.policy_net)
                self.mean_rewards.append(eval_results['mean_reward'])
                
                print(f'Evaluation | Mean Reward: {eval_results["mean_reward"]:.2f} | Std Reward: {eval_results["std_reward"]:.2f}')
                
                return total_loss, None, None
        elif self.task_type == "contrastive_learning":
            # Contrastive learning training loop
            self.model.train()
            total_loss = 0
            
            for batch_idx, (img1, img2, _) in enumerate(self.train_loader):
                img1, img2 = img1.to(self.device), img2.to(self.device)
                
                # Forward pass for both augmented views
                proj1 = self.model(img1, return_projection=True, return_embedding=False)
                proj2 = self.model(img2, return_projection=True, return_embedding=False)
                
                # Normalize projections
                proj1 = F.normalize(proj1, dim=1)
                proj2 = F.normalize(proj2, dim=1)
                
                # Calculate contrastive loss
                loss = self.contrastive_criterion(proj1, proj2)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
                    
            # Return average loss
            avg_loss = total_loss / len(self.train_loader)
            return avg_loss, None, None  # No accuracy or perplexity for contrastive learning
        elif self.task_type == "gflownet":
            # GFlowNet training loop
            self.model.train()
            total_loss = 0
            flow_match_loss_total = 0
            reward_loss_total = 0
            entropy_total = 0
            
            # Number of training iterations per epoch
            iterations = 10  # Adjust based on desired training intensity
            
            for batch_idx in range(iterations):
                # Generate batch of initial states
                batch_size = self.p['batch_size']
                
                # Sample trajectories from the model
                states, actions, terminated = self.model.sample_trajectory(batch_size)
                
                # Compute rewards for terminal states
                rewards = self.grid_env.compute_rewards(states[-1])
                
                # Apply temperature scaling to rewards
                scaled_rewards = rewards / self.reward_temp
                
                # Compute trajectory balance loss (flow matching)
                flow_match_loss = self.model.compute_trajectory_balance_loss(states, actions, scaled_rewards)
                
                # Compute reward loss (to encourage high-reward trajectories)
                # We want to maximize reward, so we minimize negative reward
                reward_loss = -scaled_rewards.mean()
                
                # Compute entropy regularization
                # Get action probabilities from forward policy
                entropy_loss = 0
                for state in states[:-1]:  # Exclude terminal state
                    logits = self.model.forward_policy(state)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                    entropy_loss -= entropy  # Negative because we want to maximize entropy
                
                # Combine losses with weights
                loss = (
                    self.flow_matching_weight * flow_match_loss +
                    self.reward_weight * reward_loss +
                    self.entropy_weight * entropy_loss
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                flow_match_loss_total += flow_match_loss.item()
                reward_loss_total += reward_loss.item()
                entropy_total += -entropy_loss.item()  # Negate back to get positive entropy
                
                if batch_idx % 2 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx}/{iterations}] '
                          f'Loss: {loss.item():.6f} | '
                          f'Flow Match: {flow_match_loss.item():.6f} | '
                          f'Reward: {rewards.mean().item():.6f} | '
                          f'Entropy: {-entropy_loss.item():.6f}')
            
            # Compute averages
            avg_loss = total_loss / iterations
            avg_flow_match_loss = flow_match_loss_total / iterations
            avg_reward_loss = reward_loss_total / iterations
            avg_entropy = entropy_total / iterations
            
            # Store metrics
            self.flow_match_losses.append(avg_flow_match_loss)
            self.reward_losses.append(avg_reward_loss)
            self.entropy.append(avg_entropy)
            
            return avg_loss, None, None  # No accuracy or perplexity for GFlowNet
        else:
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            if self.task_type == "classification" or self.task_type == "text_classification":
                # For image classification (CIFAR10) and text classification (IMDB)
                if self.p['dataset'] == 'CIFAR10':
                    # Standard PyTorch DataLoader for CIFAR10
                    for batch_idx, (data, target) in enumerate(self.train_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        
                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss.backward()
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        
                        if batch_idx % 10 == 0:
                            print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
                else:
                    # For IMDB with our custom DataLoader
                    for batch_idx, batch in enumerate(self.train_loader):
                        # Our IMDB loader returns tokenized texts, lengths, and labels
                        texts, lengths, labels = batch
                        
                        # Convert texts to indices and create tensor
                        indices = [[self.vocab[token] for token in text] for text in texts]
                        input_tensor = torch.tensor(indices).to(self.device)
                        labels = labels.to(self.device)
                        
                        self.optimizer.zero_grad()
                        output = self.model(input_tensor)
                        loss = self.criterion(output, labels)
                        loss.backward()
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1)
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
                        
                        if batch_idx % 10 == 0:
                            print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
            
                # Return average loss and accuracy
                avg_loss = total_loss / len(self.train_loader)
                accuracy = correct / total if total > 0 else 0
                return avg_loss, accuracy, None  # None for perplexity
            
            elif self.task_type == "language_modeling":
                # For WikiText2 language modeling
                total_tokens = 0
                
                for batch_idx, batch in enumerate(self.train_loader):
                    # For language modeling, batch contains input and target tensors
                    # These are already in shape [seq_len, batch_size]
                    src, targets = batch
                    src, targets = src.to(self.device), targets.to(self.device)
                    
                    if self.p['model'] == 'TransformerModel':
                        # TransformerModel expects src and target with shape [seq_len, batch_size]
                        tgt = src.clone()
                        output = self.model(src, tgt)  # output shape: [seq_len, batch_size, vocab_size]
                        
                        # Reshape for loss calculation
                        # Flatten sequence and batch dimensions
                        output_flat = output.reshape(-1, self.vocab_size)  # [seq_len*batch_size, vocab_size]
                        targets_flat = targets.reshape(-1)  # [seq_len*batch_size]
                        
                        # Calculate loss
                        loss = self.criterion(output_flat, targets_flat)
                    else:
                        # Other models only need src
                        output = self.model(src)
                        
                        # Reshape for loss calculation
                        output_flat = output.reshape(-1, self.vocab_size)
                        targets_flat = targets.reshape(-1)
                        
                        loss = self.criterion(output_flat, targets_flat)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    # Count non-padding tokens
                    non_padding_mask = targets_flat != self.vocab['<pad>']
                    num_tokens = non_padding_mask.int().sum().item()
                    
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    
                    if batch_idx % 5 == 0:
                        print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
            
                # Calculate average loss and perplexity
                avg_loss = total_loss / max(total_tokens, 1)
                perplexity = math.exp(avg_loss)
                return avg_loss, None, perplexity  # None for accuracy

    def evaluate(self, data_loader=None):
        """Evaluate the model on the provided data"""
        if self.task_type == "reinforcement_learning":
            # For RL, we evaluate the policy directly
            self.model.eval()
            
            with torch.no_grad():
                if self.p['model'] == 'PPOPolicy':
                    eval_results = self.rl_env.evaluate_policy(self.ppo_agent.pi)
                else:
                    eval_results = self.rl_env.evaluate_policy(self.policy_net)
                
            return 0.0, eval_results['mean_reward'], None  # Use mean reward as "accuracy" for RL
        elif self.task_type == "contrastive_learning":
            self.model.eval()
            total_loss = 0
            
            with torch.no_grad():
                for img1, img2, _ in data_loader:
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    
                    # Forward pass for both augmented views
                    proj1 = self.model(img1, return_projection=True, return_embedding=False)
                    proj2 = self.model(img2, return_projection=True, return_embedding=False)
                    
                    # Normalize projections
                    proj1 = F.normalize(proj1, dim=1)
                    proj2 = F.normalize(proj2, dim=1)
                    
                    # Calculate contrastive loss
                    loss = self.contrastive_criterion(proj1, proj2)
                    
                    total_loss += loss.item()
            
            # For contrastive learning, downstream classification accuracy can be used as evaluation metric
            if data_loader == self.test_loader:
                # Evaluate linear classifier on embeddings
                test_accuracy = self._evaluate_downstream_classification()
                return total_loss / len(data_loader), test_accuracy, None
            else:
                return total_loss / len(data_loader), None, None
        elif self.task_type == "gflownet":
            self.model.eval()
            
            with torch.no_grad():
                # Sample larger batch for evaluation
                eval_batch_size = min(100, self.p['batch_size'] * 2)
                
                # Sample trajectories from the model
                states, actions, terminated = self.model.sample_trajectory(eval_batch_size)
                
                # Compute rewards for terminal states
                rewards = self.grid_env.compute_rewards(states[-1])
                
                # Compute trajectory balance loss
                flow_match_loss = self.model.compute_trajectory_balance_loss(states, actions, rewards)
                
                # Evaluate trajectories
                metrics = self.grid_env.evaluate_trajectories(states)
                
                # Store diversity and reward metrics
                self.sample_diversity = metrics['diversity']
                self.reward_mean = metrics['reward_mean']
                self.reward_max = metrics['reward_max']
                
                # Return flow matching loss as the main validation metric
                return flow_match_loss.item(), metrics['reward_mean'], None
        else:
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                if self.task_type == "classification" or self.task_type == "text_classification":
                    if self.p['dataset'] == 'CIFAR10':
                        # Standard PyTorch DataLoader for CIFAR10
                        for data, target in data_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output = self.model(data)
                            loss = self.criterion(output, target)
                            total_loss += loss.item() * target.size(0)
                            
                            # Calculate accuracy
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)).sum().item()
                            total += target.size(0)
                    else:
                        # For IMDB with our custom DataLoader
                        for batch in data_loader:
                            # Our IMDB loader returns tokenized texts, lengths, and labels
                            texts, lengths, labels = batch
                            
                            # Convert texts to indices and create tensor
                            indices = [[self.vocab[token] for token in text] for text in texts]
                            input_tensor = torch.tensor(indices).to(self.device)
                            labels = labels.to(self.device)
                            
                            output = self.model(input_tensor)
                            loss = self.criterion(output, labels)
                            total_loss += loss.item() * labels.size(0)
                            
                            # Calculate accuracy
                            pred = output.argmax(dim=1)
                            correct += (pred == labels).sum().item()
                            total += labels.size(0)
                
                    avg_loss = total_loss / max(total, 1)
                    accuracy = correct / max(total, 1)
                    return avg_loss, accuracy, None  # None for perplexity
                
                elif self.task_type == "language_modeling":
                    # For WikiText2 language modeling
                    total_tokens = 0
                    
                    for batch in data_loader:
                        # For language modeling, batch contains input and target tensors
                        # These are already in shape [seq_len, batch_size]
                        src, targets = batch
                        src, targets = src.to(self.device), targets.to(self.device)
                        
                        if self.p['model'] == 'TransformerModel':
                            # TransformerModel expects src and target
                            tgt = src.clone()
                            output = self.model(src, tgt)  # output shape: [seq_len, batch_size, vocab_size]
                            
                            # Reshape for loss calculation
                            output_flat = output.reshape(-1, self.vocab_size)  # [seq_len*batch_size, vocab_size]
                            targets_flat = targets.reshape(-1)  # [seq_len*batch_size]
                            
                            # Calculate loss
                            loss = self.criterion(output_flat, targets_flat)
                        else:
                            # Other models only need src
                            output = self.model(src)
                            
                            # Reshape for loss calculation
                            output_flat = output.reshape(-1, self.vocab_size)
                            targets_flat = targets.reshape(-1)
                            
                            loss = self.criterion(output_flat, targets_flat)
                        
                        # Count non-padding tokens
                        non_padding_mask = targets_flat != self.vocab['<pad>']
                        num_tokens = non_padding_mask.int().sum().item()
                        
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                    
                    # Calculate average loss and perplexity
                    avg_loss = total_loss / max(total_tokens, 1)
                    perplexity = math.exp(avg_loss)
                    return avg_loss, None, perplexity  # None for accuracy

    def _evaluate_downstream_classification(self):
        """Evaluate embeddings with a linear classifier for downstream task evaluation"""
        # Use standard test dataset for CIFAR10
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.p['batch_size'], shuffle=False, num_workers=4)
        
        # Extract features from test samples
        self.model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                embeddings = self.model(images, return_embedding=True, return_projection=False)
                features.append(embeddings.cpu())
                labels.append(targets)
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # Train linear classifier on features
        classifier = torch.nn.Linear(features.size(1), 10)
        classifier = classifier.to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Split features for training and validation
        train_size = int(0.8 * len(features))
        train_features, val_features = features[:train_size], features[train_size:]
        train_labels, val_labels = labels[:train_size], labels[train_size:]
        
        # Train classifier for a few epochs
        classifier.train()
        for epoch in range(10):  # Quick training for evaluation
            # Train
            permutation = torch.randperm(train_size)
            train_loss = 0
            
            for i in range(0, train_size, self.p['batch_size']):
                indices = permutation[i:i + self.p['batch_size']]
                batch_features = train_features[indices].to(self.device)
                batch_labels = train_labels[indices].to(self.device)
                
                optimizer.zero_grad()
                outputs = classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
        
        # Evaluate classifier
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_features), self.p['batch_size']):
                batch_features = val_features[i:i + self.p['batch_size']].to(self.device)
                batch_labels = val_labels[i:i + self.p['batch_size']].to(self.device)
                
                outputs = classifier(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total
        return accuracy

    def run(self):
        """Run the benchmark and track all metrics"""
        self.setup_data()
        self.setup_training()
        
        start_time = time.time()
        
        for epoch in range(1, self.p['epochs'] + 1):
            # Train
            train_loss, train_acc, train_ppl = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            if train_acc is not None:
                self.train_accs.append(train_acc)
            if train_ppl is not None:
                self.train_perplexities.append(train_ppl)
            
            # Validate (only for non-RL tasks)
            if self.task_type != "reinforcement_learning":
                if self.task_type == "gflownet":
                    # For GFlowNet, we evaluate without a data loader
                    val_loss, val_acc, val_ppl = self.evaluate()
                else:
                val_loss, val_acc, val_ppl = self.evaluate(self.val_loader)
                    
                self.val_losses.append(val_loss)
                
                if val_acc is not None:
                    self.val_accs.append(val_acc)
                if val_ppl is not None:
                    self.val_perplexities.append(val_ppl)
                
                # Print epoch results
                if self.task_type == "language_modeling":
                    print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Train PPL: {train_ppl:.2f} | Val Loss: {val_loss:.6f} | Val PPL: {val_ppl:.2f}')
                elif self.task_type == "gflownet":
                    print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Flow Match Loss: {val_loss:.6f} | Mean Reward: {val_acc:.4f}')
                else:
                    print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}')
        
        # Evaluate on the training data for final metrics
        if self.task_type == "reinforcement_learning":
            # For RL, we already have the metrics from the last training epoch
            self.final_train_loss = self.train_losses[-1]
            self.final_train_acc = self.mean_rewards[-1]  # Use last mean reward as final "accuracy"
            
            # Evaluate on test (more episodes for final evaluation)
            if self.p['model'] == 'PPOPolicy':
                # Evaluate the PPO agent pi network
                self.model.eval()
                with torch.no_grad():
                    eval_results = self.rl_env.evaluate_policy(self.ppo_agent.pi, n_episodes=10)
                self.test_acc = eval_results['mean_reward']
            else:
                # Use original evaluation
                _, self.test_acc, _ = self.evaluate()
                
            self.test_loss = 0.0  # Not relevant for RL
            
            print(f'Final Mean Reward: {self.final_train_acc:.2f} | Test Mean Reward: {self.test_acc:.2f}')
        elif self.task_type == "gflownet":
            # For GFlowNet, we evaluate without a data loader
            self.final_train_loss = self.train_losses[-1]
            
            # Evaluate on test (more samples for final evaluation)
            self.test_loss, self.test_acc, _ = self.evaluate()
            
            print(f'Final Train Loss: {self.final_train_loss:.6f} | Test Flow Match Loss: {self.test_loss:.6f} | Mean Reward: {self.test_acc:.4f}')
        else:
            self.final_train_loss, self.final_train_acc, self.final_train_perplexity = self.evaluate(self.train_loader)
            
            if self.task_type == "language_modeling":
                print(f'Final Train Loss: {self.final_train_loss:.6f} | Final Train PPL: {self.final_train_perplexity:.2f}')
            else:
                print(f'Final Train Loss: {self.final_train_loss:.6f} | Final Train Acc: {self.final_train_acc:.4f}')
                    
            # Evaluate on test data
            self.test_loss, self.test_acc, self.test_perplexity = self.evaluate(self.test_loader)
            
            if self.task_type == "language_modeling":
                print(f'Test Loss: {self.test_loss:.6f} | Test PPL: {self.test_perplexity:.2f}')
            else:
                print(f'Test Loss: {self.test_loss:.6f} | Test Acc: {self.test_acc:.4f}')
        
        self.train_time = time.time() - start_time
        
        # Compile results
        self.results = {
            'params': self.p,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'final_train_loss': self.final_train_loss,
            'final_train_acc': self.final_train_acc,
            'final_train_perplexity': self.final_train_perplexity,
            'test_loss': self.test_loss,
            'test_acc': self.test_acc,
            'test_perplexity': self.test_perplexity,
            'train_time': self.train_time
        }
        
        # Add RL-specific metrics if applicable
        if self.task_type == "reinforcement_learning":
            self.results.update({
                'mean_rewards': self.mean_rewards,
                'policy_losses': self.policy_losses,
                'value_losses': self.value_losses,
            })
            
        # Add GFlowNet-specific metrics if applicable
        if self.task_type == "gflownet":
            self.results.update({
                'flow_match_losses': self.flow_match_losses,
                'reward_losses': self.reward_losses,
                'entropy': self.entropy,
                'sample_diversity': self.sample_diversity,
                'reward_mean': self.reward_mean,
                'reward_max': self.reward_max,
            })
        
        return self.results