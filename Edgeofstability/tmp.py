'''

https://github.com/locuslab/edge-of-stability.git

https://github.com/locuslab/edge-of-stability.git

python src/gd.py cifar10-5k fc-tanh  mse  0.01 30 --acc_goal 0.99 --neigs 2  --eig_freq 20



export RESULTS="/home/ubuntu/EOS/googleeos/results"

export DATASETS="/home/ubuntu/EOS/googleeos/datasets"


export DATASETS="/home/ubuntu/EOS/googleeos/datasets"





export DATASETS="/Users/scat8701/Documents/Programming /vam/Edgeofstability/datasets"

export RESULTS="/Users/scat8701/Documents/Programming /vam/Edgeofstability/results"





python Edgeofstability/vradam.py cifar10-5k resnet32 mse 0.001 20000 --eig_freq 5 --neigs 1 --seed 0 --beta1 0.9 --beta2 0.999 --epsilon 1e-7 --beta3 1.0 --physical_batch_size 1000 --loss_goal 0.1 --acc_goal 0.97 --nproj 0 --iterate_freq -1 --abridged_size 5000 --save_freq -1

'''
import torch
import matplotlib.pyplot as plt
from os import environ


# gd_directory = f"{environ['RESULTS']}/cifar10-5k/fc-tanh/seed_0/mse/adam/lr_0.0002_beta1_0.9_beta2_0.995_eps_1e-07"

# gd_directory = f"{environ['RESULTS']}/cifar10-5k/fc-tanh/seed_0/mse/vradam/lr_0.0002_beta1_0.9_beta2_0.995_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"

# gd_train_loss = torch.load(f"{gd_directory}/test_loss_final")

# plt.plot(gd_train_loss)
# plt.yscale("log")
# plt.savefig("plot_tmp.png")

vradam_dir = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/vradam/lr_0.002_beta1_0.9_beta2_0.999_eps_1e-07_vbeta3_1.0_vpower_2_vnormgrad_True_vlrcutoff_19.0"

adam_dir = f"{environ['RESULTS']}/cifar10-5k/resnet32/seed_0/mse/adam/lr_0.001_beta1_0.9_beta2_0.999_eps_1e-07"

vradam_train_loss = torch.load(f"{vradam_dir}/train_loss_final")
vradam_train_acc = torch.load(f"{vradam_dir}/train_acc_final")
print(vradam_train_loss[-5:])


