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
from torch.optim import lr_scheduler

# Import our custom text dataset implementations
import text_datasets

from VADAM import VADAM
import architectures

# Import optimizers and schedulers
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

class Benchmarker:
    def __init__(self,p= None):
        # Add scheduler defaults
        default_params = {
            'model': 'SimpleCNN', # 'SimpleCNN', 'TransformerModel', 'MLPModel'
            'device': 'mps', # 'mps' or 'cpu'
            'dataset': 'CIFAR10', # 'CIFAR10', 'WikiText2', 'IMDB'
            'dataset_size': 'small', # 'small' or 'full'
            'optimizer': 'ADAM', # 'ADAM' or 'VADAM'
            'batch_size': 128,
            'max_seq_len': 256,  # For NLP tasks
            'embed_dim': 300,    # For MLP and Transformer models
            'hidden_dim': 512,   # For MLP model
            'lr': 0.001,         # Learning rate
            'epochs': 100,       # Number of training epochs
            # --- Scheduler params (replacing StepLR) ---
            'lr_scheduler_type': None, # e.g., 'WarmupCosineAnnealing'
            # Cosine Annealing specific
            'lr_eta_min': 1e-6,    # Minimum learning rate for cosine part
            # Warmup specific
            'lr_warmup_epochs': 5,  # Default warmup epochs
            'lr_warmup_factor': 0.0, # Default starting LR factor (as per example)
            # --- Add Seed --- 
            'seed': 0, # Set to an integer for reproducible initialization
            # Parameters needed ONLY for other schedulers (StepLR, CosineAnnealingWarmRestarts)
            'lr_step_size': 10,
            'lr_gamma': 0.1,
            'lr_T_0': 10,
            'lr_T_mult': 1,
        }

        if p is None:
            p = default_params.copy() # Use copy to avoid modifying default dict
        else:
            self.p = p

        # Ensure all required parameters (including scheduler ones) are in the dict
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
        else:
            raise ValueError(f"Unknown model: {self.p['model']}")

        # --- Optimizer Setup (AFTER model instantiation) ---
        if self.p['optimizer'] == "VADAM":
            # VADAM specific parameters
            vadam_params = {
                'beta1': self.p.get('beta1', 0.9),
                'beta2': self.p.get('beta2', 0.999),
                'beta3': self.p.get('beta3', 1.0),
                'lr': self.p.get('lr', 0.001),  # Use lr from params as base eta
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0),
                'power': self.p.get('power', 2),
                'normgrad': self.p.get('normgrad', True),
                'lr_cutoff': self.p.get('lr_cutoff', 19)
            }
            self.optimizer = VADAM(self.model.parameters(), **vadam_params)
        elif self.p['optimizer'] == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.p.get('lr', 0.001),
                'betas': (self.p.get('beta1', 0.9), self.p.get('beta2', 0.999)),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0)
            }
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **adam_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.p['optimizer']}")

        # --- Scheduler Setup (AFTER optimizer setup) ---
        self.scheduler = None
        scheduler_type = self.p.get('lr_scheduler_type')

        if scheduler_type == 'WarmupCosineAnnealing':
            warmup_epochs = self.p.get('lr_warmup_epochs', 5)
            total_epochs = self.p.get('epochs', 100)
            eta_min = self.p.get('lr_eta_min', 1e-6)
            start_factor = self.p.get('lr_warmup_factor', 0.0) # Get start factor from params

            if warmup_epochs >= total_epochs:
                 print(f"Warning: warmup_epochs ({warmup_epochs}) >= total_epochs ({total_epochs}). Disabling scheduler.")
            else:
                # 1. Linear Warmup Scheduler
                warmup_sched = LinearLR(
                    self.optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                print(f"Using Linear Warmup for {warmup_epochs} epochs, start_factor={start_factor}")

                # 2. Cosine Annealing Scheduler (for the rest of the epochs)
                cosine_sched = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs - warmup_epochs, # Correct T_max
                    eta_min=eta_min
                )
                print(f"Using CosineAnnealingLR for remaining {total_epochs - warmup_epochs} epochs, eta_min={eta_min}")

                # 3. Combine with SequentialLR
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs] # Switch after warmup_epochs
                )
        elif scheduler_type is not None:
            print(f"Warning: Unknown or removed scheduler type '{scheduler_type}'. No scheduler will be used.")

    def train_epoch(self, epoch):
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
                
                if self.p['dataset'] == 'WikiText2':
                    # Handle for TransformerModel
                    self.optimizer.zero_grad()
                    
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

        # --- Step the scheduler after training epoch and before validation ---
        if self.scheduler:
            self.scheduler.step()
            # Optional: Log current learning rate if desired
            # current_lr = self.scheduler.get_last_lr()[0]
            # print(f"Epoch {epoch}: LR set to {current_lr}")

    def evaluate(self, data_loader):
        """Evaluate the model on the provided data"""
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
            
            # Validate
            val_loss, val_acc, val_ppl = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            
            if val_acc is not None:
                self.val_accs.append(val_acc)
            if val_ppl is not None:
                self.val_perplexities.append(val_ppl)
            
            # Print epoch results
            if self.task_type == "language_modeling":
                print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Train PPL: {train_ppl:.2f} | Val Loss: {val_loss:.6f} | Val PPL: {val_ppl:.2f}')
            else:
                print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}')
        
        # Evaluate on the training data for final metrics
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
        
        return self.results