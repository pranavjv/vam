import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import wandb
from VADAM import VADAM
from benchmarker import Benchmarker
from SimpleCNN import SimpleCNN
import architectures

class EdgeOfStabilityAnalyzer:
    """
    
    This class implements experiments to test whether VADAM can surpass
    this bound and remain stable at higher eigenvalues of the preconditioned
    Hessian.
    """
    def __init__(self, 
                 model_type='SimpleCNN', 
                 dataset='CIFAR10',
                 device=None,
                 batch_size=128,
                 seed=42,
                 use_wandb=False):
        """
        Initialize the analyzer
        
        Args:
            model_type: Type of model to use
            dataset: Dataset to use
            device: Device to use (cuda, mps, cpu)
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            use_wandb: Whether to log results to W&B
        """
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.use_wandb = use_wandb
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        # Initialize W&B if requested
        if use_wandb:
            wandb.init(
                project="vadam-edge-of-stability",
                config={
                    "model": model_type,
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "seed": seed
                }
            )
            
    def setup_data_and_model(self):
        """Set up dataset and model"""
        # Set up data
        benchmark_params = {
            'model': self.model_type,
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'device': str(self.device)
        }
        
        benchmarker = Benchmarker(benchmark_params)
        benchmarker.setup_data()
        
        self.train_loader = benchmarker.train_loader
        
        # Set up model
        if self.model_type == 'SimpleCNN':
            self.model = SimpleCNN().to(self.device)
        else:
            # Add support for other model types if needed
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def compute_max_eigenvalue(self, model, optimizer, use_preconditioned=True):
        """
        Compute the maximum eigenvalue of the Hessian
        
        Args:
            model: The model
            optimizer: The optimizer (Adam or VADAM)
            use_preconditioned: Whether to compute the eigenvalue of the preconditioned Hessian
            
        Returns:
            The maximum eigenvalue
        """
        # Get a batch of data
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            break
        
        # Compute loss and gradients
        optimizer.zero_grad()
        output = model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        # Get parameter vectors and gradients
        params = []
        grads = []
        
        for p in model.parameters():
            if p.requires_grad:
                params.append(p.data.reshape(-1))
                grads.append(p.grad.data.reshape(-1))
                
        # Concatenate parameters and gradients
        params = torch.cat(params)
        grads = torch.cat(grads)
        
        # For preconditioned Hessian, we need to apply the preconditioner
        if use_preconditioned and isinstance(optimizer, (torch.optim.Adam, VADAM)):
            # Get preconditioner from optimizer state
            preconditioned_grads = []
            
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = optimizer.state[p]
                        
                        if isinstance(optimizer, torch.optim.Adam):
                            # Adam preconditioner
                            if 'exp_avg_sq' in state:
                                beta2 = group['betas'][1]
                                exp_avg_sq = state['exp_avg_sq']
                                bias_correction2 = 1 - beta2 ** (state.get('step', 0) or 1)
                                
                                # Apply preconditioner
                                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group['eps'])
                                preconditioned_grad = p.grad.data / denom
                                preconditioned_grads.append(preconditioned_grad.reshape(-1))
                        
                        elif isinstance(optimizer, VADAM):
                            # VADAM preconditioner
                            if 'sec_momentum_buffer' in state:
                                beta2 = group['beta2']
                                buf_sec_mom = state['sec_momentum_buffer']
                                t = state.get('step', 0)
                                
                                # Apply preconditioner (similar to VADAM implementation)
                                denom = buf_sec_mom.div(1 - beta2**t).sqrt().add_(group['eps'])
                                preconditioned_grad = p.grad.data / denom
                                preconditioned_grads.append(preconditioned_grad.reshape(-1))
            
            if preconditioned_grads:
                grads = torch.cat(preconditioned_grads)
        
        # Compute the power iteration to find the max eigenvalue
        v = torch.randn_like(grads).to(self.device)
        v = v / torch.norm(v)
        
        # Power iteration
        for _ in range(10):  # Usually 10 iterations is enough
            # Compute Hv product
            Hv = self._hessian_vector_product(model, v, data, target)
            
            # Update v
            v = Hv / torch.norm(Hv)
            
        # Compute the Rayleigh quotient
        Hv = self._hessian_vector_product(model, v, data, target)
        max_eigenvalue = torch.dot(v, Hv).item()
        
        return max_eigenvalue
    
    def _hessian_vector_product(self, model, v, data, target):
        """
        Compute the Hessian-vector product using the Pearlmutter trick
        
        Args:
            model: The model
            v: The vector to multiply with the Hessian
            data: Input data
            target: Target data
            
        Returns:
            The Hessian-vector product
        """
        # Compute gradients
        output = model(data)
        loss = self.criterion(output, target)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Flatten gradients
        grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        
        # Compute grad-vector product
        gvp = torch.sum(grads * v)
        
        # Compute Hessian-vector product
        Hv = torch.autograd.grad(gvp, model.parameters(), retain_graph=True)
        Hv = torch.cat([h.reshape(-1) for h in Hv if h is not None])
        
        return Hv
    
    def run_edge_of_stability_experiment(self, 
                                         eta_values=[0.001, 0.005, 0.01, 0.05],
                                         beta1=0.9,
                                         beta2=0.999,
                                         beta3=1.0,
                                         num_steps=100,
                                         plot_results=True,
                                         output_dir='plots',
                                         **vadam_params):
        """
        Run edge of stability experiment with different learning rates
        
        Args:
            eta_values: List of learning rates to test
            beta1: beta1 parameter for optimizers
            beta2: beta2 parameter for optimizers
            beta3: beta3 parameter for VADAM
            num_steps: Number of steps to run for each configuration
            plot_results: Whether to plot results
            output_dir: Directory to save plots to
            **vadam_params: Additional parameters for VADAM
            
        Returns:
            Dictionary of results
        """
        results = {
            'adam': {eta: [] for eta in eta_values},
            'vadam': {eta: [] for eta in eta_values}
        }
        
        # Initialize data and model
        self.setup_data_and_model()
        
        # Theoretical AEoS bound for Adam with beta1=0.9 is 38/eta
        adam_theoretical_bounds = {eta: 38/eta for eta in eta_values}
        
        for eta in eta_values:
            print(f"\nTesting with eta={eta}")
            print(f"Adam theoretical AEoS bound: {adam_theoretical_bounds[eta]:.2f}")
            
            # Run experiment with Adam
            print("Running Adam experiment...")
            model_adam = SimpleCNN().to(self.device)
            optimizer_adam = torch.optim.Adam(
                model_adam.parameters(),
                lr=eta,
                betas=(beta1, beta2)
            )
            
            # Track max eigenvalues
            max_eigenvalues_adam = []
            
            for step in tqdm(range(num_steps)):
                # Training step
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer_adam.zero_grad()
                    output = model_adam(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer_adam.step()
                    break  # Only use first batch
                
                # Compute max eigenvalue
                if step % 5 == 0 or step == num_steps - 1:  # Compute every 5 steps to save time
                    max_eig = self.compute_max_eigenvalue(model_adam, optimizer_adam)
                    max_eigenvalues_adam.append(max_eig)
                    
                    if self.use_wandb:
                        wandb.log({
                            'step': step,
                            f'adam_eta_{eta}_max_eigenvalue': max_eig,
                            f'adam_eta_{eta}_theoretical_bound': adam_theoretical_bounds[eta]
                        })
            
            results['adam'][eta] = max_eigenvalues_adam
            
            # Run experiment with VADAM
            print("Running VADAM experiment...")
            model_vadam = SimpleCNN().to(self.device)
            optimizer_vadam = VADAM(
                model_vadam.parameters(),
                eta=eta,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                **vadam_params
            )
            
            # Track max eigenvalues
            max_eigenvalues_vadam = []
            
            for step in tqdm(range(num_steps)):
                # Training step
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer_vadam.zero_grad()
                    output = model_vadam(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer_vadam.step()
                    break  # Only use first batch
                
                # Compute max eigenvalue
                if step % 5 == 0 or step == num_steps - 1:  # Compute every 5 steps to save time
                    max_eig = self.compute_max_eigenvalue(model_vadam, optimizer_vadam)
                    max_eigenvalues_vadam.append(max_eig)
                    
                    if self.use_wandb:
                        wandb.log({
                            'step': step,
                            f'vadam_eta_{eta}_max_eigenvalue': max_eig,
                            f'adam_eta_{eta}_theoretical_bound': adam_theoretical_bounds[eta]
                        })
            
            results['vadam'][eta] = max_eigenvalues_vadam
        
        # Plot results if requested
        if plot_results:
            self._plot_results(results, adam_theoretical_bounds, eta_values, beta1, beta3, output_dir)
        
        return results
    
    def _plot_results(self, results, adam_theoretical_bounds, eta_values, beta1, beta3, output_dir='plots'):
        """
        Plot the results of the edge of stability experiment
        
        Args:
            results: Dictionary of results
            adam_theoretical_bounds: Dictionary of theoretical bounds for Adam
            eta_values: List of learning rates tested
            beta1: beta1 parameter used for optimizers
            beta3: beta3 parameter used for VADAM
            output_dir: Directory to save plots to
        """
        # Create plots directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot each learning rate separately
        for eta in eta_values:
            plt.figure(figsize=(10, 6))
            
            # Plot Adam results
            steps = np.arange(0, len(results['adam'][eta]) * 5, 5)
            plt.plot(steps, results['adam'][eta], label=f'Adam (eta={eta})', color='blue')
            
            # Plot VADAM results
            steps = np.arange(0, len(results['vadam'][eta]) * 5, 5)
            plt.plot(steps, results['vadam'][eta], label=f'VADAM (eta={eta}, beta3={beta3})', color='red')
            
            # Plot theoretical bound
            #plt.axhline(y=adam_theoretical_bounds[eta], linestyle='--', color='green', 
            #          label=f'Adam Theoretical Bound (38/eta = {adam_theoretical_bounds[eta]:.2f})')
            
            plt.xlabel('Training Steps')
            plt.ylabel('Max Eigenvalue of Preconditioned Hessian')
            plt.title(f'Edge of Stability Analysis (eta={eta}, beta1={beta1})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f"eos_eta_{eta}_beta1_{beta1}_beta3_{beta3}.png"))
            plt.close()
        
        # Plot all learning rates together
        plt.figure(figsize=(12, 8))
        
        for eta in eta_values:
            # Plot Adam results
            steps = np.arange(0, len(results['adam'][eta]) * 5, 5)
            plt.plot(steps, results['adam'][eta], label=f'Adam (eta={eta})', linestyle='-')
            
            # Plot VADAM results
            steps = np.arange(0, len(results['vadam'][eta]) * 5, 5)
            plt.plot(steps, results['vadam'][eta], label=f'VADAM (eta={eta})', linestyle='--')
            
            # Plot theoretical bound
            #plt.axhline(y=adam_theoretical_bounds[eta], linestyle=':', alpha=0.5,
            #          label=f'Adam Bound (eta={eta}): {adam_theoretical_bounds[eta]:.2f}')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Max Eigenvalue of Preconditioned Hessian')
        plt.title(f'Edge of Stability Analysis (beta1={beta1}, VADAM beta3={beta3})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"eos_all_etas_beta1_{beta1}_beta3_{beta3}.png"))
        plt.close()
        
        # Create VADAM to Adam ratio plot to see if VADAM can go beyond the bound
        plt.figure(figsize=(10, 6))
        
        for eta in eta_values:
            # Calculate ratio of VADAM max eigenvalue to Adam theoretical bound
            adam_bound = adam_theoretical_bounds[eta]
            vadam_values = results['vadam'][eta]
            steps = np.arange(0, len(vadam_values) * 5, 5)
            ratio = [val / adam_bound for val in vadam_values]
            
            plt.plot(steps, ratio, label=f'VADAM/Adam_Bound Ratio (eta={eta})')
        
        # Add horizontal line at ratio = 1
        plt.axhline(y=1.0, linestyle='--', color='black', label='Ratio = 1.0 (Adam bound)')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Ratio of VADAM Max Eigenvalue to Adam Theoretical Bound')
        plt.title(f'VADAM Stability Beyond Adam Bound (beta1={beta1}, beta3={beta3})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"eos_ratio_beta1_{beta1}_beta3_{beta3}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Edge of Stability Analysis for VADAM')
    parser.add_argument('--model', type=str, default='SimpleCNN', help='Model type')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eta_values', type=float, nargs='+', default=[0.001, 0.005, 0.01, 0.05],
                      help='Learning rates to test')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 parameter')
    parser.add_argument('--beta3', type=float, default=1.0, help='beta3 parameter for VADAM')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps to run')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to log results to W&B')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots to')
    parser.add_argument('--normgrad', action='store_true', help='Whether to use gradient norm for VADAM (default: True)')
    parser.add_argument('--power', type=float, default=2.0, help='Power parameter for VADAM')
    parser.add_argument('--lr_cutoff', type=float, default=19.0, help='Learning rate cutoff for VADAM')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer = EdgeOfStabilityAnalyzer(
        model_type=args.model,
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        use_wandb=args.use_wandb
    )
    
    # Pass extra VADAM parameters
    vadam_params = {}
    if hasattr(args, 'normgrad'):
        vadam_params['normgrad'] = args.normgrad
    if hasattr(args, 'power'):
        vadam_params['power'] = args.power
    if hasattr(args, 'lr_cutoff'):
        vadam_params['lr_cutoff'] = args.lr_cutoff
    
    analyzer.run_edge_of_stability_experiment(
        eta_values=args.eta_values,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        **vadam_params
    )

if __name__ == '__main__':
    main() 