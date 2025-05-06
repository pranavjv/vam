import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split




from VADAM import VADAM
import architectures

class Benchmarker:
    def __init__(self,p= None):
        if p is None:
            p= {
                'model': 'SimpleCNN', # 'SimpleCNN', 
                'device': 'mps', # 'mps' or 'cpu
                'dataset': 'CIFAR10', # 'CIFAR10'
                'dataset_size': 'small', # 'small' or 'full'

                
                
            }
        else:
            self.p= p

        if self.p['device'] == 'mps':
            self.device= torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            if not torch.backends.mps.is_available():
                print("MPS is not available. Using CPU instead.")
        else:
            self.device= torch.device('cpu')

    def setup_data(self):
        if self.p['dataset'] == 'CIFAR10':
            transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
        ])

            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_set  = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
                                        ]))

            if self.p['dataset_size']== "small":
                # Dataloading
                subset_size  = int(0.002 * len(train_set))
                rest_size    = len(train_set) - subset_size
                small_train, _ = random_split(train_set, [subset_size, rest_size])

                self.train_loader = DataLoader(
                    small_train,
                    batch_size=128,
                    shuffle=True,
                    num_workers=4)
            elif self.p['dataset_size']== "full":
                self.train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
            self.test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4)
            self.criterion = torch.nn.CrossEntropyLoss()


    def setup_training(self):

        if self.p['model'] == 'SimpleCNN':
            self.model= architectures.SimpleCNN().to(self.device)
        
        if self.p['optimizer']== "VADAM":
            self.optimizer= VADAM(self.model.parameters(), lr=0.001)

        elif self.p['optimizer']== "ADAM":
            self.optimizer= torch.optim.Adam(self.model.parameters(), lr=0.001)


        

        

        

        





    def run(self):
        pass