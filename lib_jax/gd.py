import jax

import jax.numpy as jnp

from jax_tqdm import scan_tqdm
from functools import partial


@partial(jax.jit, static_argnums=[1, 2, 6])
def gradient_descent(
    rng, target_value_and_grad, n_epochs=100, lr=1, n_particles=500, d=2,
    preconditioner=lambda x: x, x0=None, x_tgt=None, scale_grad=1.
    ):
    """
        (Preconditioned) gradient descent

        Inputs:
        - rng: Jax random key
        - target_value_and_grad: function returning the value and the (Wasserstein) gradient of the target
                                 (takes as input (xk, x_tgt, key))
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - preconditioner: function returning the preconditioner on the gradient
        - x0: initial particles (default None, randomly generated)
        - x_tgt: target particles to evaluate the divergence (default None, if no target)

        Outputs:
        - List loss
        - List of particles at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, subkey = jax.random.split(key)
        value_loss, grad_x = target_value_and_grad(xk, x_tgt, subkey)
        xk = xk - lr * preconditioner(scale_grad * grad_x)
        return (xk, master_key), (xk, value_loss)

    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    # Use `lax.scan` to loop over epochs
    _, L = jax.lax.scan(step, (x0, rng), jnp.arange(n_epochs))
    L_particles, L_loss = L

    return L_loss, jnp.insert(L_particles, 0, x0, axis=0)


@partial(jax.jit, static_argnums=[1, 2, 6, 10])
def gradient_descent_nosave(
    rng, target_value_and_grad, n_epochs=100, lr=1, n_particles=500, d=2,
    preconditioner=lambda x: x, x0=None, x_tgt=None, scale_grad=1., 
    target_func=lambda x: 0
    ):
    """
        (Preconditioned) gradient descent without saving the particles at each step, only the loss.

        Inputs:
        - rng: Jax random key
        - target_value_and_grad: function returning the value and the (Wasserstein) gradient of the target
                                 (takes as input (xk, x_tgt, key))
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - preconditioner: function returning the preconditioner on the gradient
        - x0: initial particles (default None, randomly generated)
        - x_tgt: target particles to evaluate the divergence (default None, if no target)
        - target_func: function to evaluate at each step

        Outputs:
        - List loss
        - List target function value at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, subkey = jax.random.split(key)
        value_loss, grad_x = target_value_and_grad(xk, x_tgt, subkey)
        xk = xk - lr * preconditioner(scale_grad * grad_x)
        return (xk, master_key), (value_loss, target_func(xk))

    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    # Use `lax.scan` to loop over epochs
    _, L = jax.lax.scan(step, (x0, rng), jnp.arange(n_epochs))

    L_loss, L_target_func = L
    return L_loss, jnp.insert(L_target_func, 0, target_func(x0), axis=0)



@partial(jax.jit, static_argnums=[1, 2, 6])
def gradient_descent_noise(
    rng, target_value_and_grad, n_epochs=100, lr=1, n_particles=500, d=2,
    preconditioner=lambda x: x, x0=None, x_tgt=None, scale_grad=1., noise_scale=1., iter_stop=4000
    ):
    """
        (Preconditioned) gradient descent

        Inputs:
        - rng: Jax random key
        - target_value_and_grad: function returning the value and the (Wasserstein) gradient of the target
                                 (takes as input (xk, x_tgt, key))
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - preconditioner: function returning the preconditioner on the gradient
        - x0: initial particles (default None, randomly generated)
        - x_tgt: target particles to evaluate the divergence (default None, if no target)

        Outputs:
        - List loss
        - List of particles at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, subkey, key_noise = jax.random.split(key, 3)
        noise = jnp.where(iter_num < iter_stop, noise_scale * jax.random.normal(key_noise, shape=xk.shape), 0.)
        value_loss, grad_x = target_value_and_grad(xk + noise, x_tgt, subkey)
        xk = xk - lr * preconditioner(scale_grad * grad_x)
        return (xk, master_key), (xk, value_loss)

    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    # Use `lax.scan` to loop over epochs
    _, L = jax.lax.scan(step, (x0, rng), jnp.arange(n_epochs))
    L_particles, L_loss = L

    return L_loss, jnp.insert(L_particles, 0, x0, axis=0)


@partial(jax.jit, static_argnums=[1, 2, 6, 10])
def gradient_descent_noise_nosave(
    rng, target_value_and_grad, n_epochs=100, lr=1, n_particles=500, d=2,
    preconditioner=lambda x: x, x0=None, x_tgt=None, scale_grad=1., 
    target_func=lambda x: 0, noise_scale=10., iter_stop=4000
    ):
    """
        (Preconditioned) gradient descent without saving the particles at each step, only the loss.

        Inputs:
        - rng: Jax random key
        - target_value_and_grad: function returning the value and the (Wasserstein) gradient of the target
                                 (takes as input (xk, x_tgt, key))
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - preconditioner: function returning the preconditioner on the gradient
        - x0: initial particles (default None, randomly generated)
        - x_tgt: target particles to evaluate the divergence (default None, if no target)
        - target_func: function to evaluate at each step

        Outputs:
        - List loss
        - List target function value at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, subkey, key_noise = jax.random.split(key, 3)
        noise = jnp.where(iter_num < iter_stop, noise_scale * jax.random.normal(key_noise, shape=xk.shape), 0.)
        value_loss, grad_x = target_value_and_grad(xk + noise, x_tgt, subkey)
        xk = xk - lr * preconditioner(scale_grad * grad_x)
        return (xk, master_key), (value_loss, target_func(xk))

    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    # Use `lax.scan` to loop over epochs
    _, L = jax.lax.scan(step, (x0, rng), jnp.arange(n_epochs))

    L_loss, L_target_func = L
    return L_loss, jnp.insert(L_target_func, 0, target_func(x0), axis=0)
