import sys
import argparse
import jax

import numpy as np


sys.path.append("../")
from lib_jax.gd import gradient_descent, gradient_descent_nosave, gradient_descent_noise_nosave, gradient_descent_noise
from lib_jax.cccp import cccp, cccp_implicit, cccp_nosave, cccp_implicit_nosave
from lib_jax.fb import forward_backward, forward_backward_nosave

from lib_jax.mmd import target_value_and_grad_gaussian_kernel, mmd, gaussian_kernel
from lib_jax.mmd_gaussian_dc import target_value_and_grad_positive_gaussian_kernel, target_value_and_grad_negative_gaussian_kernel
from lib_jax.mmd_gaussian_dc import target_value_and_grad_positive_jordan_gaussian_kernel, target_value_and_grad_negative_jordan_gaussian_kernel
from lib_jax.mmd_gaussian_dc import target_value_and_grad_positive_gaussian_kernel_clipped, target_value_and_grad_negative_gaussian_kernel_clipped
from lib_jax.mmd_dc_luu import target_grad_positive_part_mmd_luu, target_grad_negative_part_mmd_luu

from lib_jax.utils_data import generate_data


@jax.jit
def compute_mmd(L_particles, X_tgt, bandwidth=10):
    kernel = lambda x, y: gaussian_kernel(x, y, h=bandwidth)
    func_compute_mmd = lambda x: mmd(x, X_tgt, kernel)
    return jax.vmap(func_compute_mmd)(L_particles)


def main_xp(rng, method, n_epochs, n_particles, d, n_try=1, lr=1, bandwidth=10, target="gaussian"):
    master_key, key = jax.random.split(rng, num=2)
    keys = jax.random.split(key, num=n_try)

    def target_value_and_grad(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_gaussian_kernel(x, y, rng, h=h, n_sample_batch=n_sample_batch)
    
    def target_value_and_grad1(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_positive_gaussian_kernel(
            x, y, rng, h=h, n_sample_batch=n_sample_batch
        )

    def target_value_and_grad2(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_negative_gaussian_kernel(
            x, y, rng, h=h, n_sample_batch=n_sample_batch
        )
    
    def target_value_and_grad1_clipped(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_positive_gaussian_kernel_clipped(
            x, y, rng, h=h, n_sample_batch=n_sample_batch, clip_value=25
        )

    def target_value_and_grad2_clipped(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_negative_gaussian_kernel_clipped(
            x, y, rng, h=h, n_sample_batch=n_sample_batch, clip_value=25
        )
    
    def target_value_and_grad_signed1(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_positive_jordan_gaussian_kernel(
            x, y, rng, h=h, n_sample_batch=n_sample_batch
        )

    def target_value_and_grad_signed2(x, y, rng, h=bandwidth, n_sample_batch=n_particles):
        return target_value_and_grad_negative_jordan_gaussian_kernel(
            x, y, rng, h=h, n_sample_batch=n_sample_batch
        )
    
    def target_luu_value_and_grad1(x, y, rng, h=bandwidth, alpha=1/bandwidth, n_sample_batch=None):
        kernel = lambda k, l: gaussian_kernel(k, l, h=h)
        return target_grad_positive_part_mmd_luu(x, y, kernel, rng, alpha=alpha, n_sample_batch=n_sample_batch)

    def target_luu_value_and_grad2(x, y, rng, h=bandwidth, alpha=1/bandwidth, n_sample_batch=None):
        kernel = lambda k, l: gaussian_kernel(k, l, h=h)
        return target_grad_negative_part_mmd_luu(x, y, kernel, rng, alpha=alpha, n_sample_batch=n_sample_batch)

    
    L = []

    for key in keys:
        key_tgt, key_batchs = jax.random.split(key, num=2)

        X_tgt, X0 = generate_data(key_tgt, n_samples=n_particles, d=d, target=target)

        target_func = lambda x: compute_mmd(x[None], X_tgt, bandwidth=bandwidth)[0]

        if method == "gd":
            L_loss, L_mmd = gradient_descent_nosave(
                key_batchs, target_value_and_grad, n_epochs=n_epochs,
                lr=lr, x0=X0, x_tgt=X_tgt, target_func=target_func
            )
        elif method == "gd_noise":
            L_loss, L_mmd = gradient_descent_noise_nosave(
                key_batchs, target_value_and_grad, n_epochs=n_epochs,
                lr=lr, x0=X0, x_tgt=X_tgt, target_func=target_func
            )
        elif method == "cccp":
            lr = 0.0005
            L_loss, L_mmd = cccp_nosave(
                key_batchs, target_value_and_grad1, target_value_and_grad2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=250, m=0.9,
                target_func=target_func
            )
        elif method == "cccp_jordan_decomposition":
            lr = 0.1
            L_loss, L_mmd = cccp_nosave(
                key_batchs, target_value_and_grad_signed1, target_value_and_grad_signed2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=500, m=0,
                target_func=target_func
            )
        elif method == "implicit_cccp":
            L_loss, L_mmd = cccp_implicit_nosave(
                key_batchs, target_value_and_grad1, target_value_and_grad2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, eps=1e-5, target_func=target_func
            )
        elif method == "cccp_luu":
            L_loss, L_mmd = cccp_nosave(
                key_batchs, target_luu_value_and_grad1, target_luu_value_and_grad2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=100, m=0, target_func=target_func
            )
        elif method == "implicit_cccp_luu":
            L_loss, L_mmd = cccp_implicit_nosave(
                key_batchs, target_luu_value_and_grad1, target_luu_value_and_grad2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, eps=1e-5, target_func=target_func
            )
        elif method == "cccp_clipped":
            lr = 0.001
            L_loss, L_mmd = cccp_nosave(
                key_batchs, target_value_and_grad1_clipped, target_value_and_grad2_clipped,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=100, m=0.9,
                target_func=target_func
            )
        elif method == "fb":
            lr = 0.0005
            tau = 1.
            L_loss, L_mmd = forward_backward_nosave(
                key_batchs, target_value_and_grad1, target_value_and_grad2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=250, m=0.9,
                tau=tau, target_func=target_func
            )
        elif method == "fb_luu":
            tau = 1.
            L_loss, L_mmd = forward_backward_nosave(
                key_batchs, target_luu_value_and_grad1, target_luu_value_and_grad2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=100, m=0,
                tau=tau, target_func=target_func
            )

        elif method == "fb_jordan_decomposition":
            tau = 0.1
            L_loss, L_mmd = forward_backward_nosave(
                key_batchs, target_value_and_grad_signed1, target_value_and_grad_signed2,
                n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=500, m=0,
                tau=tau, target_func=target_func
            )

        L.append(L_mmd)


    # Get also the particles for the last iteration
    if method == "gd":
        _, L_particles = gradient_descent(
            key_batchs, target_value_and_grad, n_epochs=n_epochs,
            lr=lr, x0=X0, x_tgt=X_tgt
        )
    if method == "gd_noise":
        _, L_particles = gradient_descent_noise(
            key_batchs, target_value_and_grad, n_epochs=n_epochs,
            lr=lr, x0=X0, x_tgt=X_tgt
        )
    elif method == "cccp":
        lr = 0.0005
        _, L_particles = cccp(
            key_batchs, target_value_and_grad1, target_value_and_grad2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=250, m=0.9
        )
    elif method == "implicit_cccp":
        _, L_particles = cccp_implicit(
            key_batchs, target_value_and_grad1, target_value_and_grad2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, eps=1e-5
        )
    elif method == "cccp_jordan_decomposition":
        lr = 0.1
        _, L_particles = cccp(
            key_batchs, target_value_and_grad_signed1, target_value_and_grad_signed2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=500, m=0
        )
    elif method == "cccp_luu":
        _, L_particles = cccp(
            key_batchs, target_luu_value_and_grad1, target_luu_value_and_grad2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=100, m=0
        )
    elif method == "implicit_cccp_luu":
        _, L_particles = cccp_implicit(
            key_batchs, target_luu_value_and_grad1, target_luu_value_and_grad2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, eps=1e-5
        )
    elif method == "cccp_clipped":
        lr = 0.001
        _, L_particles = cccp(
            key_batchs, target_value_and_grad1_clipped, target_value_and_grad2_clipped,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=100, m=0.9
        )
    elif method == "fb":
        lr = 0.0005
        tau = 1.
        _, L_particles = forward_backward(
            key_batchs, target_value_and_grad1, target_value_and_grad2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=250, m=0.9, tau=tau
        )
    elif method == "fb_luu":
        tau = 1.
        _, L_particles = forward_backward(
            key_batchs, target_luu_value_and_grad1, target_luu_value_and_grad2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=100, m=0, tau=tau
        )
    elif method == "fb_jordan_decomposition":
        tau = 0.1
        _, L_particles = forward_backward(
            key_batchs, target_value_and_grad_signed1, target_value_and_grad_signed2,
            n_epochs, lr, x0=X0, x_tgt=X_tgt, n_inner_epochs=500, m=0, tau=tau
        )

    return L, L_particles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Gaussian experiment')
    parser.add_argument('--n_try', type=int, default=100, help='Number of tries (to average results)')
    parser.add_argument('--n_epochs', type=int, default=100_000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1., help='Learning rate')
    parser.add_argument('--n_particles', type=int, default=100, help='Number of particles')
    parser.add_argument('--bandwidth', type=float, default=10, help='Bandwidth for the Gaussian kernel')
    parser.add_argument('--d', type=int, default=2, help='Dimension')
    parser.add_argument("--method", type=str, default="cccp", help="Method to use: cccp, implicit cccp or gd")
    parser.add_argument("--path_results", type=str, default="./results", help="Path to save results")
    parser.add_argument("--target", type=str, default="gaussian", help="Target distribution: gaussian or gaussian_mixture")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(42)

    L, L_particles = main_xp(rng, args.method, args.n_epochs, args.n_particles, args.d,
                             n_try=args.n_try, lr=args.lr, bandwidth=args.bandwidth,
                             target=args.target)
    
    np.savetxt(args.path_results + f"/convergence_{args.target}_{args.method}_h{args.bandwidth}.csv", np.array(L),
               delimiter=",")
    
    print(np.array(L_particles).shape)

    np.save(args.path_results + f"/particles_{args.target}_{args.method}_h{args.bandwidth}", np.array(L_particles))
