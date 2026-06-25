import jax
import argparse
import sys

import numpy as np

sys.path.append("../")

from lib_jax.gd import gradient_descent, gradient_descent_nosave
from lib_jax.cccp import cccp, cccp_nosave
from lib_jax.fb import forward_backward, forward_backward_nosave

from lib_jax.mmd import mmd_riesz, target_value_and_grad_riesz_kernel
from lib_jax.mmd_riesz_dc import target_value_and_grad_mmd_riesz1, target_value_and_grad_mmd_riesz2

from lib_jax.utils_data import generate_data


@jax.jit
def compute_mmd(L_particles, X_tgt, r=1):
    func_compute_mmd = lambda x: mmd_riesz(x, X_tgt, r=r)
    return jax.vmap(func_compute_mmd)(L_particles)


def main_xp(rng, method, n_epochs, n_particles, d, n_try=1, lr=1, r=1, target="gaussian"):
    master_key, key = jax.random.split(rng, num=2)
    keys = jax.random.split(key, num=n_try)

    if target == "heart":
        target = "img"
        path_img = "../lib_jax/img/heart.png"
    elif target == "disk":
        target = "img"
        path_img = "../lib_jax/img/disk.png"
    elif target == "cat":
        target = "img"
        path_img = "../lib_jax/img/cat.png"
    elif target == "spiral":
        target = "img"
        path_img = "../lib_jax/img/spiral3d.jpg"
    else:
        path_img = None

    def target_value_and_grad_gd(x, y, rng, r=1, n_sample_batch=None):
        return target_value_and_grad_riesz_kernel(x, y, rng, r=r, n_sample_batch=n_sample_batch)
        
    def target_value_and_grad_cccp1(x, y, rng, r=1, n_sample_batch=None):
        return target_value_and_grad_mmd_riesz1(x, y, rng, r=r, n_sample_batch=n_sample_batch)

    def target_value_and_grad_cccp2(x, y, rng, r=1, n_sample_batch=None):
        return target_value_and_grad_mmd_riesz2(x, y, rng, r=r, n_sample_batch=n_sample_batch)


    L = []

    for key in keys:
        key_tgt, key_batchs = jax.random.split(key, num=2)

        X_tgt, X0 = generate_data(
            key_tgt, n_samples=n_particles, d=d, target=target, path_img=path_img
        )

        target_func = lambda x: compute_mmd(x[None], X_tgt, r=1)[0]

        if method == "gd":
            L_loss, L_mmd = gradient_descent_nosave(
                key_batchs, target_value_and_grad_gd, n_epochs=n_epochs,
                lr=lr, x0=X0, x_tgt=X_tgt, target_func=target_func
            )
        elif method == "cccp":
            lr = 0.1
            L_loss, L_mmd = cccp_nosave(
                key_batchs, target_value_and_grad_cccp1, target_value_and_grad_cccp2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=1000, m=0.9,
                target_func=target_func
            )
        elif method == "fb":
            tau = 1
            lr = 0.1
            L_loss, L_mmd = forward_backward_nosave(
                key_batchs, target_value_and_grad_cccp1, target_value_and_grad_cccp2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=1000, m=0.9,
                tau=tau, target_func=target_func
            )

        L.append(L_mmd)

    if method == "gd":
        _, L_particles = gradient_descent(
            key_batchs, target_value_and_grad_gd, n_epochs=n_epochs,
            lr=lr, x0=X0, x_tgt=X_tgt
        )
    elif method == "cccp":
        lr = 0.1
        _, L_particles = cccp(
            key_batchs, target_value_and_grad_cccp1, target_value_and_grad_cccp2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=1000, m=0.9
        )
    elif method == "fb":
        tau = 1
        lr = 0.1
        _, L_particles = forward_backward(
            key_batchs, target_value_and_grad_cccp1, target_value_and_grad_cccp2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=1000, m=0.9,
            tau=tau
        )

    return L, L_particles


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the MMD Riesz experiment')
    parser.add_argument('--n_try', type=int, default=100, help='Number of tries (to average results)')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1., help='Learning rate')
    parser.add_argument('--n_particles', type=int, default=500, help='Number of particles')
    parser.add_argument('--d', type=int, default=2, help='Dimension')
    parser.add_argument("--method", type=str, default="cccp", help="Method to use: cccp, or gd")
    parser.add_argument("--target", type=str, default="cat", help="Target distribution")
    parser.add_argument("--path_results", type=str, default="./results", help="Path to save results")
    args = parser.parse_args()


    rng = jax.random.PRNGKey(42)
    L, L_particles = main_xp(
        rng, method=args.method, n_epochs=args.n_epochs, n_particles=args.n_particles, d=args.d,
        n_try=args.n_try, lr=args.lr, r=1, target=args.target
    )

    np.savetxt(args.path_results + f"/convergence_{args.target}_{args.method}.csv", np.array(L),
               delimiter=",")
    
    np.save(args.path_results + f"/particles_{args.target}_{args.method}", np.array(L_particles))
