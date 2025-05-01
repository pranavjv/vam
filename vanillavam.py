import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import time


class VAM(Optimizer):
    r"""
    Velocity-Adaptive Momentum (VAM) optimizer.
    """
    def __init__(self, params, lr, momentum=0.9, m=1.0, beta=0.1, eps=1e-8, weight_decay=0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if m <= 0.0:
            raise ValueError(f"Invalid mass parameter m: {m}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        defaults = dict(lr=lr, momentum=momentum, m=m, beta=beta,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            m_base = group['m']
            beta = group['beta']
            eps = group['eps']
            wd = group['weight_decay']

            # 1) compute total squared norm of momentum buffers
            total_sq_norm = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                buf = state.get('momentum_buffer', None)
                if buf is not None:
                    total_sq_norm += float(buf.pow(2).sum())

            # 2) adaptive learning rate
            adaptive_lr = lr / (m_base + beta * total_sq_norm + eps)

            # 3) per-parameter update
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)

                state = self.state[p]
                buf = state.get('momentum_buffer', None)
                if buf is None:
                    buf = torch.zeros_like(p)
                    state['momentum_buffer'] = buf

                buf.mul_(mu).add_(d_p, alpha=-adaptive_lr)
                p.add_(buf)

        return loss