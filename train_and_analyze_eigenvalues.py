import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import wandb

# Ensure VADAM.py and SimpleCNN.py (or architectures.py if SimpleCNN is there)
# are in the same directory or accessible via PYTHONPATH.
try:
    from VADAM import VADAM
except ImportError:
    print("Failed to import VADAM. Ensure VADAM.py is in the correct path.")
    exit()
try:
    # Ensure DeeperCNN is imported
    from architectures import SimpleCNN, DeeperCNN 
    print("Imported SimpleCNN and DeeperCNN from architectures.py")
except ImportError:
    print("Failed to import DeeperCNN from architectures.py. Ensure architectures.py is in the correct path.")
    exit()

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else \
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 1024  # Updated BATCH_SIZE
EPOCHS = 20  # Number of epochs to train for
LEARNING_RATES = [0.01, 0.003]
OPTIMIZERS_TO_TEST = ['ADAM', 'VADAM']
NUM_POWER_ITERATIONS = 10 # For eigenvalue calculation
PLOT_DIR = "eigenvalue_experiment_plots_deepercnn_bs1024" # Updated plot directory
MODEL_NAME = "DeeperCNN" # Updated MODEL_NAME
DATASET_NAME = "CIFAR10" 

# VADAM specific parameters for effective LR calculation
VADAM_BETA3 = 0.1 
VADAM_LR_CUTOFF = 10.0

os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Using Model: {MODEL_NAME}, Batch Size: {BATCH_SIZE}")

# --- Data Setup ---
def setup_cifar10_data(batch_size):
    """Prepares CIFAR-10 training DataLoader."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), # More common CIFAR-10 stds
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type != 'cpu' else False)
    return train_loader

# --- Eigenvalue Computation (without preconditioning) ---
def _hvp_product(model_for_hvp, v_flat, data_batch, target_batch, criterion_for_hvp):
    """Computes Hessian-vector product H*v for the given model and data."""
    model_for_hvp.zero_grad()
    output = model_for_hvp(data_batch)
    loss = criterion_for_hvp(output, target_batch)
    
    params_requiring_grad = [p for p in model_for_hvp.parameters() if p.requires_grad]
    
    grads_L_tuple = torch.autograd.grad(loss, params_requiring_grad, create_graph=True)
    flat_grads_L = torch.cat([g.reshape(-1) for g in grads_L_tuple if g is not None])
    
    if flat_grads_L.numel() != v_flat.numel():
         raise ValueError(
             f"Shape mismatch in HVP: flat_grads_L ({flat_grads_L.numel()}) vs v_flat ({v_flat.numel()}). "
             "This might happen if v_flat was sized based on all params, but some don't receive gradients."
         )
    
    gvp = torch.sum(flat_grads_L * v_flat)
    
    Hv_tuple = torch.autograd.grad(gvp, params_requiring_grad) 
    flat_Hv = torch.cat([h.reshape(-1) for h in Hv_tuple if h is not None])
    return flat_Hv

def compute_max_hessian_eigenvalue(model_to_analyze, data_loader, criterion_for_eig, device_for_eig, num_iterations=NUM_POWER_ITERATIONS):
    """Computes the max eigenvalue of the Hessian (without preconditioning) using power iteration."""
    original_training_state = model_to_analyze.training
    model_to_analyze.eval() # Set model to eval mode for consistent behavior
    
    try:
        data_batch, target_batch = next(iter(data_loader))
    except StopIteration:
        print("Warning: DataLoader is empty. Cannot compute eigenvalue.")
        return 0.0
        
    data_batch, target_batch = data_batch.to(device_for_eig), target_batch.to(device_for_eig)

    params_requiring_grad = [p for p in model_to_analyze.parameters() if p.requires_grad]
    if not params_requiring_grad:
        if original_training_state: model_to_analyze.train()
        return 0.0

    num_params = sum(p.numel() for p in params_requiring_grad)
    if num_params == 0:
        if original_training_state: model_to_analyze.train()
        return 0.0

    v = torch.randn(num_params, device=device_for_eig)
    if torch.norm(v) == 0: v = torch.ones(num_params, device=device_for_eig) # Avoid zero vector
    v = v / torch.norm(v)
    
    for _ in range(num_iterations):
        # Pass v.detach() to avoid building up computation graph across power iterations
        Hv = _hvp_product(model_to_analyze, v.detach(), data_batch, target_batch, criterion_for_eig)
        v_norm = torch.norm(Hv)
        if v_norm == 0: # Eigenvalue is 0 if Hv is 0
            if original_training_state: model_to_analyze.train()
            return 0.0 
        v = Hv / v_norm
            
    Hv_final = _hvp_product(model_to_analyze, v, data_batch, target_batch, criterion_for_eig) # Final product with converged v
    max_eigenvalue = torch.dot(v, Hv_final).item()
    
    if original_training_state: model_to_analyze.train() # Restore original training state
    return max_eigenvalue

# --- Training Function ---
def train_one_configuration(model_class_to_train, optimizer_name, learning_rate, num_epochs, device_to_use, 
                            train_loader_for_train, criterion_for_train):
    """Trains one configuration and records losses, eigenvalues, and VAdam effective LR."""
    print(f"\\n--- Training {optimizer_name} with LR={learning_rate} for {num_epochs} epochs ---")
    
    model = model_class_to_train().to(device_to_use)
    
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'VADAM':
        optimizer = VADAM(model.parameters(), eta=learning_rate, beta1=0.9, beta2=0.999, 
                          beta3=VADAM_BETA3, lr_cutoff=VADAM_LR_CUTOFF) 
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    train_losses_per_epoch = []
    eigenvalues_per_epoch = [] 
    vadam_effective_lrs_this_run = []

    for epoch in range(num_epochs): # epoch from 0 to num_epochs-1
        print(f"Epoch {epoch+1}/{num_epochs}: Computing pre-epoch max eigenvalue...")
        current_max_eig = compute_max_hessian_eigenvalue(model, train_loader_for_train, criterion_for_train, device_to_use)
        eigenvalues_per_epoch.append(current_max_eig)
        print(f"Epoch {epoch+1}/{num_epochs} - Max Eigenvalue (before this epoch's training): {current_max_eig:.4e}")

        model.train() 
        running_loss = 0.0
        num_batches_processed = 0
        
        progress_bar = tqdm(train_loader_for_train, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for data, target in progress_bar:
            data, target = data.to(device_to_use), target.to(device_to_use)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion_for_train(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches_processed += 1
            progress_bar.set_postfix(loss=loss.item())
        
        log_dict_epoch = {"epoch": epoch}

        if num_batches_processed > 0:
            epoch_loss = running_loss / num_batches_processed
            train_losses_per_epoch.append(epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {epoch_loss:.4f}")
            log_dict_epoch["training_loss"] = epoch_loss
            log_dict_epoch["max_hessian_eigenvalue_at_epoch_start"] = current_max_eig
        else:
            train_losses_per_epoch.append(float('nan')) 
            print(f"Epoch {epoch+1}/{num_epochs} - No batches processed, loss not recorded.")
            log_dict_epoch["training_loss"] = float('nan')
            log_dict_epoch["max_hessian_eigenvalue_at_epoch_start"] = current_max_eig
        
        if optimizer_name == 'VADAM':
            # Calculate VAdam effective LR after the epoch's updates
            current_eta = optimizer.param_groups[0]['eta']
            # beta3 and lr_cutoff should be in param_groups if VADAM init stores them there.
            # Or, access them via optimizer.beta3, optimizer.lr_cutoff if VADAM stores them as attributes.
            # Assuming they were passed in constructor and are available in param_groups:
            beta3_val = optimizer.param_groups[0].get('beta3', VADAM_BETA3) 
            lr_cutoff_val = optimizer.param_groups[0].get('lr_cutoff', VADAM_LR_CUTOFF)

            all_momentum_buffers_flat = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and optimizer.state.get(p) and 'momentum_buffer' in optimizer.state[p]:
                        all_momentum_buffers_flat.append(optimizer.state[p]['momentum_buffer'].detach().reshape(-1))
            
            if all_momentum_buffers_flat:
                total_momentum_vector = torch.cat(all_momentum_buffers_flat)
                total_sq_norm = torch.sum(total_momentum_vector * total_momentum_vector).item() # .item() to get scalar
                
                effective_lr_val = current_eta / (1 + min(beta3_val * total_sq_norm, lr_cutoff_val))
                vadam_effective_lrs_this_run.append(effective_lr_val)
                log_dict_epoch["vadam_effective_lr"] = effective_lr_val
            else:
                vadam_effective_lrs_this_run.append(float('nan'))
                log_dict_epoch["vadam_effective_lr"] = float('nan')
        
        wandb.log(log_dict_epoch)
            
    print("Computing final max eigenvalue (post-training)...")
    final_max_eig = compute_max_hessian_eigenvalue(model, train_loader_for_train, criterion_for_train, device_to_use)
    eigenvalues_per_epoch.append(final_max_eig) 
    print(f"Max Eigenvalue (after all training): {final_max_eig:.4e}")

    wandb.log({
        "epoch": num_epochs, 
        "max_hessian_eigenvalue_post_training": final_max_eig
    })

    return train_losses_per_epoch, eigenvalues_per_epoch, vadam_effective_lrs_this_run

# --- Main Execution ---
def main():
    """Main function to run experiments and plot results."""
    train_loader = setup_cifar10_data(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    
    all_results = {} 

    model_class_to_run = DeeperCNN # Use DeeperCNN

    for opt_name in OPTIMIZERS_TO_TEST:
        for lr in LEARNING_RATES:
            # MODEL_NAME and BATCH_SIZE are now global constants reflecting the current setup
            run_name = f"{opt_name}_LR-{lr}_Model-{MODEL_NAME}_BS-{BATCH_SIZE}"
            group_name = f"{MODEL_NAME}-{DATASET_NAME}-EigenvalueStudy_BS-{BATCH_SIZE}"
            
            wandb.init(
                project="edge-of-stability-analysis", 
                name=run_name,
                group=group_name,
                config={
                    "optimizer": opt_name,
                    "learning_rate": lr,
                    "epochs": EPOCHS,
                    "model_class": MODEL_NAME, 
                    "dataset": DATASET_NAME,
                    "batch_size": BATCH_SIZE,
                    "device": str(DEVICE),
                    "num_power_iterations": NUM_POWER_ITERATIONS,
                    "criterion": type(criterion).__name__,
                    "vadam_beta3_config": VADAM_BETA3 if opt_name == "VADAM" else "N/A",
                    "vadam_lr_cutoff_config": VADAM_LR_CUTOFF if opt_name == "VADAM" else "N/A"
                },
                reinit=True 
            )
            
            train_losses, eigenvalues, vadam_eff_lrs = train_one_configuration(
                model_class_to_run, opt_name, lr, EPOCHS, DEVICE, train_loader, criterion
            )
            all_results[(opt_name, lr)] = {
                'losses': train_losses, 
                'eigenvalues': eigenvalues,
                'vadam_effective_lr': vadam_eff_lrs if opt_name == 'VADAM' else []
            }
            
            wandb.finish()

    # Plotting
    # Plot Training Losses
    plt.figure(figsize=(12, 8))
    for (opt_name, lr), data in all_results.items():
        plt.plot(range(1, EPOCHS + 1), data['losses'], marker='o', linestyle='-', label=f'{opt_name} LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title(f'Training Loss Curves on {DATASET_NAME} ({MODEL_NAME}, BS={BATCH_SIZE}, {EPOCHS} Epochs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, EPOCHS + 1))
    plt.tight_layout()
    loss_plot_filename = f"training_loss_comparison_{MODEL_NAME.lower()}_bs{BATCH_SIZE}.png"
    plt.savefig(os.path.join(PLOT_DIR, loss_plot_filename))
    print(f"Training loss plot saved to {os.path.join(PLOT_DIR, loss_plot_filename)}")

    # Plot Max Eigenvalues
    plt.figure(figsize=(12, 8))
    for (opt_name, lr), data in all_results.items():
        num_eigenvalue_points = len(data['eigenvalues'])
        x_plot_ticks = list(range(num_eigenvalue_points))
        
        plot_labels_eigen = ['Initial'] 
        if EPOCHS > 0 :
            plot_labels_eigen += [f'Start E{i+1}' for i in range(EPOCHS -1)] + ['Start Last Epoch', 'Final']
        elif EPOCHS == 0 and num_eigenvalue_points == 1: # only initial
             plot_labels_eigen = ['Initial']
        elif EPOCHS == 0 and num_eigenvalue_points == 2: # initial and final
             plot_labels_eigen = ['Initial', 'Final']
        
        # Ensure x_plot_ticks matches the length of plot_labels_eigen for plotting
        # If eigenvalues has N+1 points (Initial + N epochs start + final), then x_plot_ticks goes from 0 to N+1
        # current data['eigenvalues'] has EPOCHS values from start of epoch + 1 initial + 1 final = EPOCHS + 1 points
        # No, eigenvalues_per_epoch appends value for start of each of N epochs = N values. Then appends 1 final. So N+1 values
        # current_max_eig = eigenvalues_per_epoch.append(current_max_eig) is called N times for epoch 0 to N-1
        # eigenvalues_per_epoch.append(final_max_eig) is called once. Total N+1 values
        # Example: EPOCHS = 2. Eigenvalues for Start E1, Start E2. Then Final. Total 3 points.
        # X-axis needs to be [0, 1, 2]. Labels: Initial/Start E1, Start E2, Final

        actual_x_ticks_for_plot = list(range(len(data['eigenvalues'])))

        plt.plot(actual_x_ticks_for_plot, data['eigenvalues'], marker='o', linestyle='-', label=f'{opt_name} LR={lr}')
    
    # Construct labels for eigenvalue plot x-axis
    xtick_labels_eigen = ['Initial']
    if EPOCHS > 0:
        for i in range(EPOCHS):
            xtick_labels_eigen.append(f'Start E{i+1}')
        xtick_labels_eigen[-1] = 'Start Last Epoch' # Override last "Start EX"
        xtick_labels_eigen.append('Final')
    elif EPOCHS == 0 and len(actual_x_ticks_for_plot) == 2: # Only Initial and Final if EPOCHS=0
        xtick_labels_eigen = ['Initial', 'Final']
    elif EPOCHS == 0 and len(actual_x_ticks_for_plot) == 1: # Only Initial if EPOCHS=0 and somehow only one point
        xtick_labels_eigen = ['Initial']


    # Only use as many labels as there are ticks
    plt.xticks(ticks=actual_x_ticks_for_plot, labels=xtick_labels_eigen[:len(actual_x_ticks_for_plot)], rotation=45, ha="right")
    plt.xlabel('Timing of Eigenvalue Computation')
    plt.ylabel('Max Hessian Eigenvalue (No Preconditioning)')
    plt.title(f'Max Hessian Eigenvalue on {DATASET_NAME} ({MODEL_NAME}, BS={BATCH_SIZE}, {EPOCHS} Epochs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    eigenvalue_plot_filename = f"max_eigenvalue_comparison_{MODEL_NAME.lower()}_bs{BATCH_SIZE}.png"
    plt.savefig(os.path.join(PLOT_DIR, eigenvalue_plot_filename))
    print(f"Max eigenvalue plot saved to {os.path.join(PLOT_DIR, eigenvalue_plot_filename)}")

    # Plot VAdam Effective Learning Rate
    plt.figure(figsize=(12, 8))
    plotted_vadam_eff_lr = False
    for (opt_name, lr_val), data in all_results.items():
        if opt_name == 'VADAM' and data.get('vadam_effective_lr'):
            # Effective LR is calculated at the end of each epoch 0 to EPOCHS-1
            # So there are EPOCHS number of points.
            epochs_axis = range(EPOCHS)
            if len(data['vadam_effective_lr']) == EPOCHS:
                 plt.plot(epochs_axis, data['vadam_effective_lr'], marker='x', linestyle='--', label=f'VADAM LR={lr_val} Eff. LR')
                 plotted_vadam_eff_lr = True
            else:
                print(f"Warning: Mismatch in VAdam effective LR data length for LR={lr_val}. Expected {EPOCHS}, got {len(data['vadam_effective_lr'])}")


    if plotted_vadam_eff_lr:
        plt.xlabel('Epoch')
        plt.ylabel('VAdam Effective Learning Rate')
        plt.title(f'VAdam Effective LR on {DATASET_NAME} ({MODEL_NAME}, BS={BATCH_SIZE}, {EPOCHS} Epochs)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if EPOCHS > 0: plt.xticks(range(EPOCHS))
        plt.tight_layout()
        eff_lr_plot_filename = f"vadam_effective_lr_{MODEL_NAME.lower()}_bs{BATCH_SIZE}.png"
        plt.savefig(os.path.join(PLOT_DIR, eff_lr_plot_filename))
        print(f"VAdam effective LR plot saved to {os.path.join(PLOT_DIR, eff_lr_plot_filename)}")
    # plt.show() 

if __name__ == '__main__':
    # Make sure to handle potential issues with running on servers without displays for plt.show()
    # Forcing a non-interactive backend for matplotlib if no display is available.
    if "DISPLAY" not in os.environ:
        print("No display found. Using Agg backend for Matplotlib.")
        plt.switch_backend('Agg')
    main() 