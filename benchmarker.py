import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import torchtext
from torchtext.datasets import WikiText2, IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
import os
import time
import math


from VADAM import VADAM
import architectures

class Benchmarker:
    def __init__(self,p= None):
        if p is None:
            p= {
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
                'epochs': 5          # Number of training epochs
            }
        else:
            self.p = p

        # Ensure all required parameters are in the dict
        for key, val in {
            'model': 'SimpleCNN',
            'device': 'mps',
            'dataset': 'CIFAR10',
            'dataset_size': 'small',
            'optimizer': 'ADAM',
            'batch_size': 128,
            'max_seq_len': 256,
            'embed_dim': 300,
            'hidden_dim': 512,
            'lr': 0.001,
            'epochs': 5
        }.items():
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
            # For language modeling using torchtext 0.6.0 API
            try:
                from torchtext.data import Field, BPTTIterator
                from torchtext.datasets import WikiText2 as legacy_WikiText2
                
                # Create Field for text processing with legacy API
                TEXT = Field(tokenize=lambda x: x.split(), lower=True)
                
                # Load dataset with legacy API
                try:
                    train_data, val_data, test_data = legacy_WikiText2.splits(TEXT)
                    print("Loaded WikiText2 dataset using legacy torchtext API")
                except:
                    # Create a minimal dataset manually if loading fails
                    print("Falling back to manual tiny WikiText2 dataset")
                    
                    # Create tiny data manually
                    import os
                    
                    tiny_data_dir = './tiny_wikitext'
                    os.makedirs(tiny_data_dir, exist_ok=True)
                    
                    # Create tiny train.txt, valid.txt, and test.txt files
                    train_text = """
                    The tower is 324 metres tall .
                    The metal structure weighs 7,300 tonnes .
                    The tower has three levels for visitors .
                    Tickets can be purchased to ascend by stairs or lift .
                    The tower is the most-visited paid monument in the world .
                    """
                    
                    val_text = """
                    The tower was built as the entrance to the 1889 World's Fair .
                    It was named after the engineer Gustave Eiffel .
                    The design of the tower was criticized by many artists .
                    """
                    
                    test_text = """
                    More than 250 million people have visited the tower since it was completed .
                    The tower was almost demolished in 1909 .
                    Today , it is considered a distinctive landmark of Paris .
                    """
                    
                    # Write the files
                    for filename, content in [('train.txt', train_text), 
                                             ('valid.txt', val_text), 
                                             ('test.txt', test_text)]:
                        with open(os.path.join(tiny_data_dir, filename), 'w') as f:
                            f.write(content)
                    
                    # Load the custom data
                    from torchtext.data import Dataset, Example
                    
                    def load_custom_data(path):
                        examples = []
                        with open(path, 'r') as f:
                            for line in f:
                                if line.strip():
                                    examples.append(Example.fromlist([line], fields=[('text', TEXT)]))
                        return Dataset(examples, fields=[('text', TEXT)])
                    
                    train_data = load_custom_data(os.path.join(tiny_data_dir, 'train.txt'))
                    val_data = load_custom_data(os.path.join(tiny_data_dir, 'valid.txt'))
                    test_data = load_custom_data(os.path.join(tiny_data_dir, 'test.txt'))
                
                # Build vocabulary from training data
                TEXT.build_vocab(train_data)
                
                # Store vocab information
                self.vocab = TEXT.vocab
                self.vocab_size = len(TEXT.vocab)
                
                # Create BPTTIterator for language modeling
                self.bptt = self.p['max_seq_len']
                batch_size = min(self.p['batch_size'], 20) if self.p['dataset_size'] == 'small' else self.p['batch_size']
                
                # Create iterators
                self.train_loader, self.val_loader, self.test_loader = BPTTIterator.splits(
                    (train_data, val_data, test_data),
                    batch_size=batch_size,
                    bptt_len=self.bptt,
                    device=self.device
                )
                
                # We need to adapt the training and evaluation methods for this iterator
                self.criterion = torch.nn.CrossEntropyLoss()
                self.task_type = "language_modeling"
                
            except ImportError as e:
                raise ImportError(f"Could not load torchtext 0.6.0 APIs: {e}. Please install torchtext 0.6.0 (pip install torchtext==0.6.0)")

        elif self.p['dataset'] == 'IMDB':
            # For sentiment analysis (text classification) using torchtext 0.6.0 API
            try:
                from torchtext.data import Field, TabularDataset, BucketIterator
                from torchtext.datasets import IMDB as legacy_IMDB
                
                # Define fields for legacy torchtext format
                TEXT = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
                LABEL = Field(sequential=False, use_vocab=False, 
                             preprocessing=lambda x: 1 if x == 'pos' else 0)
                
                # Load datasets using legacy API
                fields = [('text', TEXT), ('label', LABEL)]
                try:
                    # Try to load IMDB directly
                    train_data, test_data = legacy_IMDB.splits(TEXT, LABEL)
                    print("Loaded IMDB dataset using legacy torchtext API")
                except:
                    # Create a minimal test dataset manually
                    print("Falling back to manual tiny IMDB dataset")
                    # Create tiny train and test data
                    import os
                    
                    tiny_data_dir = './tiny_imdb'
                    os.makedirs(tiny_data_dir, exist_ok=True)
                    os.makedirs(os.path.join(tiny_data_dir, 'train'), exist_ok=True)
                    os.makedirs(os.path.join(tiny_data_dir, 'test'), exist_ok=True)
                    
                    # Create tiny train.csv and test.csv
                    with open(os.path.join(tiny_data_dir, 'train.csv'), 'w') as f:
                        f.write("text,label\n")
                        f.write("this movie was great,pos\n")
                        f.write("terrible waste of time,neg\n")
                        f.write("i loved this film,pos\n")
                        f.write("awful acting and direction,neg\n")
                        f.write("brilliant screenplay and effects,pos\n")
                    
                    with open(os.path.join(tiny_data_dir, 'test.csv'), 'w') as f:
                        f.write("text,label\n")
                        f.write("excellent movie experience,pos\n")
                        f.write("one of the worst films,neg\n")
                        f.write("amazing visual effects,pos\n")
                        f.write("poor screenplay and acting,neg\n")
                    
                    train_data, test_data = TabularDataset.splits(
                        path=tiny_data_dir,
                        train='train.csv',
                        test='test.csv',
                        format='csv',
                        fields=fields,
                        skip_header=True
                    )
                
                # Build vocabulary
                TEXT.build_vocab(train_data, max_size=25000)
                
                # Split train into train/val
                train_data, val_data = train_data.split(split_ratio=0.8)
                
                # Create iterators
                batch_size = self.p['batch_size']
                self.train_loader, self.val_loader, self.test_loader = BucketIterator.splits(
                    (train_data, val_data, test_data),
                    batch_size=batch_size,
                    sort_key=lambda x: len(x.text),
                    sort_within_batch=True,
                    device=self.device
                )
                
                self.vocab = TEXT.vocab
                self.vocab_size = len(TEXT.vocab)
                self.criterion = torch.nn.CrossEntropyLoss()
                self.task_type = "text_classification"
                self.num_classes = 2  # Binary classification
                
            except ImportError as e:
                raise ImportError(f"Could not load torchtext 0.6.0 APIs: {e}. Please install torchtext 0.6.0 (pip install torchtext==0.6.0)")

    def setup_training(self):
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

        # Setup optimizer
        if self.p['optimizer'] == "VADAM":
            # VADAM specific parameters
            vadam_params = {
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
            self.optimizer = VADAM(self.model.parameters(), **vadam_params)
        elif self.p['optimizer'] == "ADAM":
            # Standard Adam parameters
            adam_params = {
                'lr': self.p.get('lr', 0.001),
                'betas': (self.p.get('beta1', 0.9), self.p.get('beta2', 0.999)),
                'eps': self.p.get('eps', 1e-8),
                'weight_decay': self.p.get('weight_decay', 0)
            }
            self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.p['optimizer']}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        if self.task_type == "classification" or self.task_type == "text_classification":
            # Check if we're using torchtext 0.6.0 iterators
            if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'examples'):
                # Using torchtext 0.6.0 iterators
                for batch_idx, batch in enumerate(self.train_loader):
                    # Handle IMDB data format with include_lengths=True
                    if hasattr(batch, 'text') and isinstance(batch.text, tuple):
                        text, lengths = batch.text
                        target = batch.label
                    else:
                        # Generic format
                        text = batch.text
                        target = batch.label
                    
                    self.optimizer.zero_grad()
                    output = self.model(text)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    
                    if batch_idx % 10 == 0:
                        print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
            else:
                # Standard PyTorch DataLoader
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
            
            # Return average loss and accuracy
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct / total if total > 0 else 0
            return avg_loss, accuracy, None  # None for perplexity
        
        elif self.task_type == "language_modeling":
            # Check if we're using torchtext 0.6.0 iterators
            if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'examples'):
                # Using torchtext 0.6.0 BPTTIterator
                for batch_idx, batch in enumerate(self.train_loader):
                    text, targets = batch.text, batch.target
                    total_tokens = targets.numel()
                    
                    # Format for transformer input
                    if self.p['model'] == 'TransformerModel':
                        src = text
                        tgt = torch.zeros_like(src)
                        if len(src) > 1:  # Ensure we have enough tokens
                            tgt[:-1] = src[1:]  # Shift right to predict next token
                        
                        self.optimizer.zero_grad()
                        output = self.model(src, tgt)
                    else:
                        # Format for other models
                        self.optimizer.zero_grad()
                        output = self.model(text)
                    
                    # Reshape output if needed
                    if output.size(0) != targets.size(0):
                        output = output.view(-1, self.vocab_size)
                    
                    loss = self.criterion(output, targets)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item() * total_tokens
                    
                    if batch_idx % 5 == 0:
                        print(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.6f}')
                
                # Calculate average loss and perplexity
                total_tokens = sum(len(batch.target.flatten()) for batch in self.train_loader)
                avg_loss = total_loss / max(total_tokens, 1)
                perplexity = math.exp(avg_loss)
                return avg_loss, None, perplexity  # None for accuracy
            else:
                # Legacy method with manually batched data
                total_tokens = 0
                for i in range(0, self.train_data.size(0) - 1, self.bptt):
                    # Get batch for language modeling
                    bptt = self.bptt if np.random.random() < 0.95 else self.bptt // 2
                    data, targets = self.get_batch(self.train_data, i, bptt)
                    total_tokens += targets.size(0)
                    
                    # Generate input and target sequences
                    src = data
                    tgt = torch.zeros_like(src)
                    if len(src) > 1:  # Ensure we have enough tokens
                        tgt[:-1] = src[1:]  # Shift right to predict next token
                    
                    self.optimizer.zero_grad()
                    output = self.model(src, tgt)
                    
                    # Calculate loss
                    output = output.view(-1, self.vocab_size)
                    loss = self.criterion(output, targets)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item() * targets.size(0)
                    
                    if i % (self.bptt * 5) == 0:
                        print(f'Train Epoch: {epoch} [{i}/{self.train_data.size(0)}] Loss: {loss.item():.6f}')
                
                # Calculate average loss and perplexity
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                return avg_loss, None, perplexity  # None for accuracy

    def evaluate(self, data_source):
        """Evaluate the model on the provided data"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            if self.task_type == "classification" or self.task_type == "text_classification":
                # Check if we're using torchtext 0.6.0 iterators
                if hasattr(data_source, 'dataset') and hasattr(data_source.dataset, 'examples'):
                    # Using torchtext 0.6.0 iterators
                    for batch in data_source:
                        # Handle IMDB data format with include_lengths=True
                        if hasattr(batch, 'text') and isinstance(batch.text, tuple):
                            text, lengths = batch.text
                            target = batch.label
                        else:
                            # Generic format
                            text = batch.text
                            target = batch.label
                        
                        output = self.model(text)
                        loss = self.criterion(output, target)
                        total_loss += loss.item() * target.size(0)
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1)
                        correct += (pred == target).sum().item()
                        total += target.size(0)
                else:
                    # Standard PyTorch DataLoader
                    for data, target in data_source:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        total_loss += loss.item() * target.size(0)
                        
                        # Calculate accuracy
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                
                avg_loss = total_loss / total if total > 0 else 0
                accuracy = correct / total if total > 0 else 0
                return avg_loss, accuracy, None  # None for perplexity
            
            elif self.task_type == "language_modeling":
                # Check if we're using torchtext 0.6.0 iterators
                if hasattr(data_source, 'dataset') and hasattr(data_source.dataset, 'examples'):
                    # Using torchtext 0.6.0 BPTTIterator
                    total_tokens = 0
                    for batch in data_source:
                        text, targets = batch.text, batch.target
                        total_tokens += targets.numel()
                        
                        # Format for transformer input
                        if self.p['model'] == 'TransformerModel':
                            src = text
                            tgt = torch.zeros_like(src)
                            if len(src) > 1:
                                tgt[:-1] = src[1:]
                            
                            output = self.model(src, tgt)
                        else:
                            # Format for other models
                            output = self.model(text)
                        
                        # Reshape output if needed
                        if output.size(0) != targets.size(0):
                            output = output.view(-1, self.vocab_size)
                        
                        loss = self.criterion(output, targets)
                        total_loss += loss.item() * targets.numel()
                    
                    # Calculate average loss and perplexity
                    avg_loss = total_loss / max(total_tokens, 1)
                    perplexity = math.exp(avg_loss)
                    return avg_loss, None, perplexity  # None for accuracy
                else:
                    # Legacy method with manually batched data
                    if not isinstance(data_source, torch.Tensor):
                        raise ValueError("Expected Tensor for language modeling data")
                    
                    data_size = data_source.size(0) - 1
                    total_tokens = 0
                    
                    for i in range(0, data_size - 1, self.bptt):
                        data, targets = self.get_batch(data_source, i, self.bptt)
                        total_tokens += targets.size(0)
                        
                        src = data
                        tgt = torch.zeros_like(src)
                        if len(src) > 1:
                            tgt[:-1] = src[1:]
                        
                        output = self.model(src, tgt)
                        output = output.view(-1, self.vocab_size)
                        total_loss += self.criterion(output, targets).item() * targets.size(0)
                    
                    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
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
            if self.task_type == "language_modeling":
                if hasattr(self, 'val_data'):
                    # Using legacy data structure
                    val_loss, val_acc, val_ppl = self.evaluate(self.val_data)
                else:
                    # Using torchtext 0.6.0 iterators
                    val_loss, val_acc, val_ppl = self.evaluate(self.val_loader)
                self.val_losses.append(val_loss)
                if val_ppl is not None:
                    self.val_perplexities.append(val_ppl)
                print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Train PPL: {train_ppl:.2f} | Val Loss: {val_loss:.6f} | Val PPL: {val_ppl:.2f}')
            else:
                if hasattr(self, 'val_loader'):
                    val_loss, val_acc, _ = self.evaluate(self.val_loader)
                    self.val_losses.append(val_loss)
                    if val_acc is not None:
                        self.val_accs.append(val_acc)
                    print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}')
        
        # Evaluate on the training data for final metrics
        if self.task_type == "language_modeling":
            if hasattr(self, 'train_data'):
                # Using legacy data structure
                self.final_train_loss, _, self.final_train_perplexity = self.evaluate(self.train_data)
            else:
                # Using torchtext 0.6.0 iterators
                self.final_train_loss, _, self.final_train_perplexity = self.evaluate(self.train_loader)
            print(f'Final Train Loss: {self.final_train_loss:.6f} | Final Train PPL: {self.final_train_perplexity:.2f}')
        else:
            self.final_train_loss, self.final_train_acc, _ = self.evaluate(self.train_loader)
            print(f'Final Train Loss: {self.final_train_loss:.6f} | Final Train Acc: {self.final_train_acc:.4f}')
                
        # Evaluate on test data
        if self.task_type == "language_modeling":
            if hasattr(self, 'test_data'):
                # Using legacy data structure
                self.test_loss, _, self.test_perplexity = self.evaluate(self.test_data)
            else:
                # Using torchtext 0.6.0 iterators
                self.test_loss, _, self.test_perplexity = self.evaluate(self.test_loader)
            print(f'Test Loss: {self.test_loss:.6f} | Test PPL: {self.test_perplexity:.2f}')
        else:
            self.test_loss, self.test_acc, _ = self.evaluate(self.test_loader)
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