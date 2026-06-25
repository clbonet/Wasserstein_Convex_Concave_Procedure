import jax

import jax.numpy as jnp

from jax_tqdm import scan_tqdm
from functools import partial


@partial(jax.jit, static_argnums=[2, 4])
def inner_step_fb(rng, xk, target_value_and_grad_plus, grad_g_xk, n_epochs=50, lr=1, x_tgt=None, m=0, tau=1.):
    """
        Solve the inner optimization step of the Forward-Backward procedure procedure
    """
    x0 = xk
    
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)

        value_f, grad_f = target_value_and_grad_plus(xk, x_tgt, subkey)
        vk = grad_f - grad_g_xk + (xk-x0) / tau + m * vk
        
        xk = xk - lr * vk
        return (xk, vk, master_key), (xk, value_f)

    v0 = jnp.zeros_like(xk)
    (xk, _, _), _ = jax.lax.scan(step, (xk, v0, rng), jnp.arange(n_epochs))
    return xk


@partial(jax.jit, static_argnums=[1, 2, 3, 8])
def forward_backward(
    rng, target_value_and_grad_plus, target_value_and_grad_minus, n_epochs=100, lr=1, n_particles=500,
    d=2, x0=None, n_inner_epochs=50, x_tgt=None, m=0, tau=1
    ):
    """
        Forward-Backward Procedure on an objectif of the form F-G as proposed in [1].

        [1] Luu, H. P. H., Yu, H., Williams, B., Mikkola, P., Hartmann, M., Puolamäki, K., & Klami, A. (2024).
        Non-geodesically-convex optimization in the Wasserstein space. Advances in Neural Information Processing Systems, 37, 16772-16809.

        Inputs:
        - rng: Jax random key
        - target_value_and_grad_plus: function returning value and the (Wasserstein) gradient of F
        - target_value_and_grad_minus: function returning value and the (Wasserstein) gradient of G
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)

        Outputs:
        - List of losses
        - List of particles at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, key_g, key_inner, key_f = jax.random.split(key, num=4)
        value_g_xk, grad_g_xk = target_value_and_grad_minus(xk, x_tgt, key_g)

        # If it is random, should we take the same batchs?
        xk = inner_step_fb(
            key_inner,
            xk,
            target_value_and_grad_plus,
            grad_g_xk, 
            n_epochs=n_inner_epochs,
            lr=lr,
            x_tgt=x_tgt,
            m=m,
            tau=tau
        )

        value_f, _ = target_value_and_grad_plus(xk, x_tgt, key_f)
        return (xk, master_key), (xk, value_f - value_g_xk)

    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    # Use `lax.scan` to loop over epochs
    _, L = jax.lax.scan(step, (x0, rng), jnp.arange(n_epochs))

    L_particles, L_loss = L
    return L_loss, jnp.insert(L_particles, 0, x0, axis=0)


@partial(jax.jit, static_argnums=[1, 2, 3, 8, 12])
def forward_backward_nosave(
    rng, target_value_and_grad_plus, target_value_and_grad_minus, n_epochs=100, lr=1, n_particles=500,
    d=2, x0=None, n_inner_epochs=50, x_tgt=None, m=0, tau=1, target_func=lambda x: 0
    ):
    """
        Forward-Backward Procedure on an objectif of the form F-G as proposed in [1].

        [1] Luu, H. P. H., Yu, H., Williams, B., Mikkola, P., Hartmann, M., Puolamäki, K., & Klami, A. (2024).
        Non-geodesically-convex optimization in the Wasserstein space. Advances in Neural Information Processing Systems, 37, 16772-16809.

        Inputs:
        - rng: Jax random key
        - target_value_and_grad_plus: function returning value and the (Wasserstein) gradient of F
        - target_value_and_grad_minus: function returning value and the (Wasserstein) gradient of G
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)

        Outputs:
        - List of losses
        - List of particles at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, key_g, key_inner, key_f = jax.random.split(key, num=4)
        value_g_xk, grad_g_xk = target_value_and_grad_minus(xk, x_tgt, key_g)

        # If it is random, should we take the same batchs?
        xk = inner_step_fb(
            key_inner,
            xk,
            target_value_and_grad_plus,
            grad_g_xk, 
            n_epochs=n_inner_epochs,
            lr=lr,
            x_tgt=x_tgt,
            m=m,
            tau=tau
        )

        value_f, _ = target_value_and_grad_plus(xk, x_tgt, key_f)
        return (xk, master_key), (value_f - value_g_xk, target_func(xk))

    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    # Use `lax.scan` to loop over epochs
    _, L = jax.lax.scan(step, (x0, rng), jnp.arange(n_epochs))

    L_loss, L_target_func = L
    return L_loss, jnp.insert(L_target_func, 0, target_func(x0), axis=0)
