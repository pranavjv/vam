import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from os import environ
import argparse

# --- Font Size Configuration ---
TITLE_FONT_SIZE = 25
LEGEND_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 15
# --- End Font Size Configuration ---

# def plot_adam_vradam_comparison(vradam_eta=0.001, vradam_beta1=0.9, vradam_beta2=0.999, vradam_beta3=1.0, 
#                               vradam_power=2.0, vradam_normgrad=False, vradam_lr_cutoff=19.0, vradam_epsilon=1e-7,
#                               adam_lr=0.001, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-7,
#                               dataset="cifar10-5k", arch="fc-tanh", loss="mse", seed=0):
#     """
#     Plot comparison between Adam and VRADAM results
#     """



vradam_eta=0.002
vradam_beta1=0.9
vradam_beta2=0.999
vradam_beta3=1.0
vradam_power=2.0
vradam_normgrad=False
vradam_lr_cutoff=19.0
vradam_epsilon=1e-7

adam_lr=0.001
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-7


vradam_dir = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/vradam/lr_0.002_beta1_0.9_beta2_0.999_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"

adam_dir = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/adam/lr_0.001_beta1_0.9_beta2_0.999_eps_1e-07"

try:
    # Load VRADAM data
    vradam_train_loss = torch.load(f"{vradam_dir}/train_loss_final")
    vradam_train_acc = torch.load(f"{vradam_dir}/train_acc_final")
    
    try:
        vradam_eigs = torch.load(f"{vradam_dir}/eigs_final")
        vradam_has_eigs = True
        vradam_eig_freq = 5  # Default value, can be detected from data
    except FileNotFoundError:
        vradam_has_eigs = False
        print("No VRADAM eigenvalue data found")
    
    try:
        vradam_learning_rates = torch.load(f"{vradam_dir}/learning_rates_final")
        vradam_has_lr = True
    except FileNotFoundError:
        vradam_has_lr = False
        print("No VRADAM learning rate data found")
    
    # Load Adam data
    adam_train_loss = torch.load(f"{adam_dir}/train_loss_final")
    adam_train_acc = torch.load(f"{adam_dir}/train_acc_final")
    
    try:
        adam_eigs = torch.load(f"{adam_dir}/eigs_final")
        adam_has_eigs = True
        adam_eig_freq = 5  # Default value, can be detected from data
    except FileNotFoundError:
        adam_has_eigs = False
        print("No Adam eigenvalue data found")



    ## Parameters
    xlima = -5
    xlimb = 195
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12), dpi=100)
    
    # 1. Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(vradam_train_loss)), vradam_train_loss, label="VRAdam")
    plt.plot(np.arange(len(adam_train_loss)), adam_train_loss, label="Adam")
    plt.title("Training Loss Comparison", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Loss", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlim(xlima, xlimb)
    plt.grid(True)
    
    # 2. Plot training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(vradam_train_acc)), vradam_train_acc, label="VRAdam")
    plt.plot(np.arange(len(adam_train_acc)), adam_train_acc, label="Adam")
    plt.title("Training Accuracy Comparison", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlim(xlima, xlimb)
    plt.grid(True)
    

    # 3. Plot eigenvalues if available
    if vradam_has_eigs and adam_has_eigs:
        plt.subplot(2, 2, 3)
        vradam_eig_iterations = torch.arange(len(vradam_eigs)) * vradam_eig_freq
        adam_eig_iterations = torch.arange(len(adam_eigs)) * adam_eig_freq
        
        # Plot leading eigenvalues
        plt.scatter(vradam_eig_iterations, vradam_eigs[:, 0], s=10, label="VRAdam max eigenvalue")
        plt.scatter(adam_eig_iterations, adam_eigs[:, 0], s=10, label="Adam max eigenvalue")
        
        # # Plot thresholds
        # if vradam_has_lr:
        #     # For VRADAM: dynamic threshold based on current learning rate
        #     sampled_lr = vradam_learning_rates[vradam_eig_iterations]
        #     vradam_thresholds = 2.0 / sampled_lr
        #     plt.plot(vradam_eig_iterations, vradam_thresholds, 'r--', 
        #             label="VRADAM dynamic threshold (2/lr)")
            
        # plt.axhline(38/0.002, linestyle='dotted', color='b', 
        #             label=f"38/0.002")

        # plt.axhline(38/0.001, linestyle='dotted', color='orange', 
        #             label=f"38/0.001")
        
        # For Adam: fixed threshold based on learning rate and beta1
        #adam_threshold = (2 + 2*adam_beta1)/((1 - adam_beta1)*adam_lr)
        #plt.axhline(adam_threshold, linestyle='dotted', color='b', 
        #            label=f"Adam threshold: {adam_threshold:.1f}")
        plt.xlim(xlima, xlimb)
        plt.title("Sharpness Comparison", fontsize=TITLE_FONT_SIZE)
        plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel("Eigenvalue Magnitude", fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
        plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax = plt.gca() # Get current axes
        ax.yaxis.get_offset_text().set_fontsize(AXIS_LABEL_FONT_SIZE) # Increase exponent font size
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.grid(True)
    
    # 4. Plot learning rates (dynamic for VRADAM, fixed for Adam)
    plt.subplot(2, 2, 4)
    if vradam_has_lr:
        plt.plot(vradam_learning_rates[:-1], label="VRAdam dynamic lr")
        
        # Show min/max rates for VRADAM
        plt.axhline(vradam_eta, linestyle='dotted', color='b',
                    label=f"VRAdam max lr")
        min_lr = vradam_eta / (1 + vradam_lr_cutoff)
        plt.axhline(min_lr, linestyle='dotted', color='b',
                    label=f"VRAdam min lr")
    
    # Show Adam fixed learning rate
    plt.axhline(adam_lr, linestyle='-', color='orange', 
                label=f"Adam fixed lr")
    
    plt.xlim(xlima, xlimb)
    plt.title("Learning Rate Comparison", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Iteration", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Learning Rate", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.yticks(fontsize=AXIS_TICK_FONT_SIZE)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = plt.gca() # Get current axes
    ax.yaxis.get_offset_text().set_fontsize(AXIS_LABEL_FONT_SIZE) # Increase exponent font size
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True)
    
    plt.tight_layout()
    output_file = "adam_vradam_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved to {output_file}")
    
    # If both have eigenvalues and VRADAM has learning rates, create normalized sharpness plot
    # if vradam_has_eigs and adam_has_eigs and vradam_has_lr:
    #     plt.figure(figsize=(10, 6), dpi=100)
        
    #     # Compute normalized eigenvalues (eig / threshold)
    #     # For VRADAM: dynamic normalization
    #     vradam_norm_eigs = vradam_eigs[:, 0] / vradam_thresholds
        
    #     # For Adam: fixed normalization
    #     adam_norm_eigs = adam_eigs[:, 0] / adam_threshold
        
    #     # Plot normalized eigenvalues
    #     plt.scatter(vradam_eig_iterations, vradam_norm_eigs, s=10, label="VRADAM normalized eigenvalue")
    #     plt.scatter(adam_eig_iterations, adam_norm_eigs, s=10, label="Adam normalized eigenvalue")
        
    #     # Stability threshold is always 1.0 after normalization
    #     plt.axhline(1.0, linestyle='dotted', color='k', label="Stability threshold")
        
    #     plt.title("Normalized Sharpness Comparison")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Eigenvalue / Threshold")
    #     plt.legend()
    #     plt.grid(True)
        
    #     normalized_file = "adam_vradam_normalized_sharpness.png"
    #     plt.savefig(normalized_file)
    #     plt.close()
    #     print(f"Normalized sharpness plot saved to {normalized_file}")
    
except FileNotFoundError as e:
    print(f"Error: {e} - Required data files not found.")

