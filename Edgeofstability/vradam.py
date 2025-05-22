from os import makedirs
import os

import torch
from torch.nn.utils import parameters_to_vector

import argparse
from archs import load_architecture
from utilities import get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset
from data import load_dataset, take_first, DATASETS


class VRADAM(torch.optim.Optimizer):
    """
    Velocity-Adaptive Momentum (VRADAM) optimizer with Adam-like behavior and weight decay according to ADAMW
    """
    def __init__(self, params, beta1=0.9, beta2=0.999, beta3=1, eta=0.001, eps=1e-8, weight_decay=0, power=2, normgrad=False, lr_cutoff=19):
        # eta corresponds to the maximal learning rate
        # if normgrad True the norm in the lr is is computed on the gradient, otherwise the velocity!
        # lr_cutoff controls the minimal learning rate, if = 19 minimal learning rate is eta/(19+1)
        defaults = dict(beta1=beta1, beta2=beta2, beta3=beta3, eps=eps, weight_decay=weight_decay, power=power, eta=eta, normgrad=normgrad, lr_cutoff=lr_cutoff)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            beta3 = group['beta3']
            eps = group['eps']
            wd = group['weight_decay']
            power = group['power']
            eta = group['eta']
            normgrad = group['normgrad']
            lr_cutoff = group['lr_cutoff']

            # get velocity and second moment terms
            total_sq_norm = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                state = self.state[p]
                buf_vel = state.get('momentum_buffer', None)
                if buf_vel is None:
                    buf_vel = torch.zeros_like(p)
                    state['momentum_buffer'] = buf_vel
                buf_vel.mul_(beta1).add_(d_p, alpha=1 - beta1)
                if normgrad:
                    total_sq_norm += float(d_p.abs().pow(power).sum())
                else:
                    total_sq_norm += float(buf_vel.abs().pow(power).sum())

                buf_sec_mom = state.get('sec_momentum_buffer', None)
                if buf_sec_mom is None:
                    buf_sec_mom = torch.zeros_like(p)
                    state['sec_momentum_buffer'] = buf_sec_mom
                buf_sec_mom.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)

                _t = state.get('step', None)
                if _t is None:
                    state['step'] = 0
                state['step'] += 1

            # Calculate adaptive learning rate based on gradient/velocity norm
            lr = eta/(1 + min(float(beta3 * total_sq_norm), float(lr_cutoff)))
            
            for p in group['params']:
                if p.grad is None:
                    continue
                buf_vel = self.state[p]['momentum_buffer']
                buf_sec_mom = self.state[p]['sec_momentum_buffer']
                t = self.state[p]['step']

                # Apply bias correction
                tmp1 = buf_vel.div(1 - beta1**t)
                tmp2 = buf_sec_mom.div(1 - beta2**t)

                tmp2.sqrt_().add_(eps)
                tmp1.div_(tmp2)

                # update parameter with weight decay
                p.mul_(1 - wd * lr).add_(tmp1, alpha=-lr)
        
        return loss


def get_vradam_nu(optimizer) -> torch.Tensor:
    """Get the second moment vector from VRADAM optimizer"""
    vec = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            vec.append(state['sec_momentum_buffer'].view(-1))
    return torch.cat(vec)


def main(dataset: str, arch_id: str, loss: str,
         eta: float, beta1: float, beta2: float, beta3: float, 
         power: float, normgrad: bool, lr_cutoff: float, epsilon: float,
         max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0):
    
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/vradam/eta_{eta}_beta1_{beta1}_beta2_{beta2}_beta3_{beta3}_power_{power}_normgrad_{normgrad}_lr_cutoff_{lr_cutoff}_eps_{epsilon}"
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = VRADAM(
        network.parameters(), 
        beta1=beta1, 
        beta2=beta2, 
        beta3=beta3, 
        eta=eta, 
        eps=epsilon, 
        power=power, 
        normgrad=normgrad, 
        lr_cutoff=lr_cutoff
    )

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    learning_rates = torch.zeros(max_steps)  # Track the dynamic learning rate

    for step in range(0, max_steps):
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                          physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        # Calculate effective learning rate
        if step > 0:
            # Extract the current learning rate (roughly approximated from the first parameter group)
            curr_lr = optimizer.param_groups[0]['eta'] / (1 + min(beta3 * sum(p.grad.abs().pow(power).sum().item() 
                                                          for p in network.parameters() if p.grad is not None), 
                                                      lr_cutoff))
            learning_rates[step] = curr_lr
        
        # At step = 0, optimizer has no state, so don't record eigs then
        if step > 0 and eig_freq != -1 and step % eig_freq == 0:
            nu = get_vradam_nu(optimizer)
            P = torch.ones_like(nu)  # Just use preconditioner of 1 for computing eigenvalues as baseline
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                             physical_batch_size=physical_batch_size, P=P)
            print("eigenvalues: ", eigs[step//eig_freq, :])

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                 ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                 ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step]),
                                 ("learning_rates", learning_rates[:step])])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal is not None and train_loss[step] < loss_goal) or (acc_goal is not None and train_acc[step] > acc_goal):
            break

        optimizer.zero_grad()
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss.backward()
        optimizer.step()

    save_files_final(directory,
                   [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                    ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                    ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1]),
                    ("learning_rates", learning_rates[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using VRADAM optimizer.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("eta", type=float, help="the maximum learning rate (eta parameter)")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                       default=0)
    parser.add_argument("--beta1", type=float, help="VRADAM beta1 parameter (momentum)", default=0.9)
    parser.add_argument("--beta2", type=float, help="VRADAM beta2 parameter (second moment)", default=0.999)
    parser.add_argument("--beta3", type=float, help="VRADAM beta3 parameter (scaling factor for norm)", default=1.0)
    parser.add_argument("--power", type=float, help="Power for the norm computation in VRADAM", default=2.0)
    parser.add_argument("--normgrad", action="store_true", help="Use gradient norm instead of velocity norm")
    parser.add_argument("--lr_cutoff", type=float, help="Learning rate cutoff parameter", default=19.0)
    parser.add_argument("--epsilon", type=float, help="VRADAM epsilon parameter", default=1e-7)
    parser.add_argument("--physical_batch_size", type=int,
                       help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                       help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                       help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                       help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                       help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                       help="the frequency at which we save resuls")
    parser.add_argument("--save_model", action="store_true", default=False,
                       help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(
        dataset=args.dataset, 
        arch_id=args.arch_id, 
        loss=args.loss, 
        eta=args.eta,
        max_steps=args.max_steps,
        neigs=args.neigs, 
        physical_batch_size=args.physical_batch_size, 
        eig_freq=args.eig_freq,
        iterate_freq=args.iterate_freq, 
        save_freq=args.save_freq, 
        save_model=args.save_model, 
        beta1=args.beta1,
        beta2=args.beta2, 
        beta3=args.beta3, 
        epsilon=args.epsilon, 
        nproj=args.nproj, 
        loss_goal=args.loss_goal,
        acc_goal=args.acc_goal, 
        abridged_size=args.abridged_size, 
        seed=args.seed,
        power=args.power,
        normgrad=args.normgrad,
        lr_cutoff=args.lr_cutoff
    ) 