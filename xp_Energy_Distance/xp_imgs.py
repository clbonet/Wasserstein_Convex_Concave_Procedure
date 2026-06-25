import jax
import sys
import argparse

import numpy as np

from datasets import get_dataset

sys.path.append("../")
from lib_jax.gd_images import wasserstein_gradient_descent_save
from lib_jax.cccp import cccp_save
from lib_jax.mmd import target_value_and_grad_riesz_kernel, mmd_riesz
from lib_jax.mmd_riesz_dc import target_value_and_grad_mmd_riesz1, target_value_and_grad_mmd_riesz2


@jax.jit
def compute_mmd(L_particles, X_tgt, r=1):
    func_compute_mmd = lambda x: mmd_riesz(x, X_tgt, r=r)
    return jax.vmap(func_compute_mmd)(L_particles)


def main_xp(rng, method, n_epochs, n_data_by_class=100,
            path_data="~/torch_datasets", tgt_dataset="MNIST",
            save_interval=1000, n_try=1):
    
    keys_tries = jax.random.split(rng, num=n_try)

    L = []

    for rng in keys_tries:
        master_key, key_src, key_tgt, key_wgd = jax.random.split(rng, num=4)

        X_data_tgt, _, _, _ = get_dataset(key_tgt, tgt_dataset, n_data_by_class, path_data)

        if tgt_dataset == "MNIST":
            X_data_src = jax.random.normal(key_src, (10*n_data_by_class, 28*28))
            X_data_tgt = X_data_tgt.reshape(-1, 784)
        elif tgt_dataset == "CIFAR10":
            X_data_src = jax.random.normal(key_src, (10*n_data_by_class, 32*32*3))
            X_data_tgt = X_data_tgt.reshape(-1, 32*32*3)

        target_func = lambda x: compute_mmd(x[None], X_data_tgt, r=1)[0]

        if method == "gd":
            target_grad = lambda x, y, key: target_value_and_grad_riesz_kernel(x, y, key)
            
            _, L_particles, L_mmd = wasserstein_gradient_descent_save(
                X_data_src, X_data_tgt, target_grad, key_wgd,
                lr=1, m=0, n_epochs=n_epochs, save_interval=save_interval,
                target_func=target_func
            )

        elif method == "cccp":
            def target_value_and_grad_cccp1(x, y, rng, r=1, n_sample_batch=None):
                return target_value_and_grad_mmd_riesz1(x, y, rng, r=r, n_sample_batch=n_sample_batch)

            def target_value_and_grad_cccp2(x, y, rng, r=1, n_sample_batch=None):
                return target_value_and_grad_mmd_riesz2(x, y, rng, r=r, n_sample_batch=n_sample_batch)

            lr = 1
            _, L_particles, L_mmd = cccp_save(
                key_wgd, target_value_and_grad_cccp1, target_value_and_grad_cccp2,
                n_epochs, lr, x0=X_data_src, x_tgt=X_data_tgt, n_inner_epochs=20, # 50,
                m=0, save_interval=save_interval, target_func=target_func
            )

        L.append(L_mmd)

    return L_particles, L   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MNIST images with Energy Distance')
    parser.add_argument('--n_epochs', type=int, default=10_000, help='Number of epochs')
    parser.add_argument('--n_data_by_class', type=int, default=200, help='Number of samples by class')
    parser.add_argument("--method", type=str, default="cccp", help="Method to use: cccp or gd")
    parser.add_argument("--path_data", type=str, default="~/torch_datasets", help="Path to the dataset")
    parser.add_argument("--path_results", type=str, default="./results", help="Path to save results")
    parser.add_argument("--tgt_dataset", type=str, default="MNIST", help="Target dataset")
    parser.add_argument('--save_interval', type=int, default=1000, help='Interval to save results')
    parser.add_argument('--n_try', type=int, default=5, help='Number of tries')
    args = parser.parse_args()

    rng = jax.random.PRNGKey(42)

    L_particles, L_mmd = main_xp(
        rng, args.method, args.n_epochs, args.n_data_by_class,
        args.path_data, tgt_dataset=args.tgt_dataset,
        save_interval=args.save_interval, n_try=args.n_try
    )

    np.save(args.path_results + f"/particles_{args.tgt_dataset}_{args.method}", np.array(L_particles))

    np.savetxt(args.path_results + f"/convergence_{args.tgt_dataset}_{args.method}.csv", np.array(L_mmd),
               delimiter=",")
