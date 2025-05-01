
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import time
from vanillavam import VAM



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4)

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total

def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total

# -----------------------------
# 5) Benchmark Loop
# -----------------------------
def benchmark(optimizer_name, optimizer_cls, model_fn, optim_kwargs, device):
    print(f"\n=== Benchmarking {optimizer_name} ===")
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(), **optim_kwargs)

    stats = {
        'train_loss': [], 'train_acc': [],
        'test_loss':  [], 'test_acc':  []
    }

    start = time.time()
    for epoch in range(1, 11):
        tl, ta = train_one_epoch(model, device, train_loader, optimizer, criterion)
        vl, va = evaluate(model, device, test_loader,  criterion)
        stats['train_loss'].append(tl)
        stats['train_acc'].append(ta)
        stats['test_loss'].append(vl)
        stats['test_acc'].append(va)
        print(f"Epoch {epoch:2d} | "
              f"Train: loss={tl:.4f}, acc={ta:.4f} | "
              f"Test: loss={vl:.4f}, acc={va:.4f}")
    elapsed = time.time() - start
    print(f"{optimizer_name} finished in {elapsed/60:.2f} min")
    return stats

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmarks = {}

    # 1) SGD with Momentum
    benchmarks['SGD-Momentum'] = benchmark(
        "SGD with Momentum",
        optim.SGD,
        SimpleCNN,
        dict(lr=0.1, momentum=0.9, weight_decay=5e-4),
        device
    )

    # 2) SGD with Nesterov
    benchmarks['SGD-Nesterov'] = benchmark(
        "SGD with Nesterov",
        optim.SGD,
        SimpleCNN,
        dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4),
        device
    )

    # 3) VAM
    benchmarks['VAM'] = benchmark(
        "Velocity-Adaptive Momentum",
        VAM,
        SimpleCNN,
        dict(lr=0.1, momentum=0.9, m=1.0, beta=1.5, eps=1e-8, weight_decay=5e-4),
        device
    )

    # Optionally: save benchmarks for later plotting
    torch.save(benchmarks, "benchmark_results.pth")