import jax
import jaxopt

import jax.numpy as jnp

from jax_tqdm import scan_tqdm
from functools import partial
from .utils_newton import newton_solver


@partial(jax.jit, static_argnums=[2, 4])
def inner_step_cccp(rng, xk, target_value_and_grad_plus, grad_g_xk, n_epochs=50, lr=1, x_tgt=None, m=0):
    """
        Solve the inner optimization step of the CCCP procedure, i.e.
        T^{l+1} = T^l - tau (nabla F(T^l# mu_k) circ T^l - nabla G(mu_k))
    """
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)

        value_f, grad_f = target_value_and_grad_plus(xk, x_tgt, subkey)
        vk = grad_f - grad_g_xk + m * vk
        
        xk = xk - lr * vk
        return (xk, vk, master_key), (xk, value_f)

    v0 = jnp.zeros_like(xk)
    (xk, _, _), _ = jax.lax.scan(step, (xk, v0, rng), jnp.arange(n_epochs))
    return xk


@partial(jax.jit, static_argnums=[1, 2, 3, 8])
def cccp(rng, target_value_and_grad_plus, target_value_and_grad_minus, n_epochs=100, lr=1, n_particles=500,
         d=2, x0=None, n_inner_epochs=50, x_tgt=None, m=0):
    """
        Convex Concave Procedure on an objectif of the form F-G.

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
        xk = inner_step_cccp(
            key_inner,
            xk,
            target_value_and_grad_plus,
            grad_g_xk, 
            n_epochs=n_inner_epochs,
            lr=lr,
            x_tgt=x_tgt,
            m=m
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


@partial(jax.jit, static_argnums=[1, 2, 3])
def cccp_implicit(
    rng, target_value_and_grad_plus, target_value_and_grad_minus,
    n_epochs=100, lr=1, n_particles=500, d=2, x0=None, n_inner_epochs=50,
    x_tgt=None, eps=1e-5
    ):
    """
        Convex Concave Procedure on an objectif of the form F-G.
        Inner steps are solved through a Newton solver, i.e. solving
        nabla F(T^l# mu_k) corc T^l - nabla G(mu_k) = 0

        Inputs:
        - rng: Jax random key
        - target_value_and_grad_plus: function returning value and the (Wasserstein) gradient of F
        - target_value_and_grad_minus: function returning value and the (Wasserstein) gradient of G
        - n_epochs: number of epochs
        - lr: step size (for Newton iterations)
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)
        - eps: regularization inverse in Newton

        Outputs:
        - List of losses
        - List of particles at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, key_g, key_inner, key_f = jax.random.split(key, num=4)
        value_g_xk, grad_g_xk = target_value_and_grad_minus(xk, x_tgt, key_g)

        @jax.jit
        def f(y):
            _, grad_f_y = target_value_and_grad_plus(
                y, x_tgt, key_inner
            )
            return grad_f_y - grad_g_xk

        xk = newton_solver(f, xk, eps=eps, lr=lr)
        jax.clear_caches()

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


@partial(jax.jit, static_argnums=[1, 2, 3, 8, 11])
def cccp_nosave(rng, target_value_and_grad_plus, target_value_and_grad_minus, n_epochs=100,
                lr=1, n_particles=500, d=2, x0=None, n_inner_epochs=50, x_tgt=None, m=0,
                target_func=lambda x: 0):
    """
        Convex Concave Procedure on an objectif of the form F-G, without saving the particles at each step, only the loss.

        Inputs:
        - rng: Jax random key
        - target_value_and_grad_plus: function returning value and the (Wasserstein) gradient of F
        - target_value_and_grad_minus: function returning value and the (Wasserstein) gradient of G
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)
        - target_func: function to evaluate at each step

        Outputs:
        - List of losses
        - List target function value at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, key_g, key_inner, key_f = jax.random.split(key, num=4)
        value_g_xk, grad_g_xk = target_value_and_grad_minus(xk, x_tgt, key_g)

        # If it is random, should we take the same batchs?
        xk = inner_step_cccp(
            key_inner,
            xk,
            target_value_and_grad_plus,
            grad_g_xk, 
            n_epochs=n_inner_epochs,
            lr=lr,
            x_tgt=x_tgt,
            m=m
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



@partial(jax.jit, static_argnums=[1, 2, 3, 11])
def cccp_implicit_nosave(rng, target_value_and_grad_plus, target_value_and_grad_minus,
                         n_epochs=100, lr=1, n_particles=500, d=2, x0=None, n_inner_epochs=50,
                         x_tgt=None, eps=1e-5, target_func=lambda x: 0):
    """
        Convex Concave Procedure on an objectif of the form F-G.
        Inner steps are solved through a Newton solver, i.e. solving
        nabla F(T^l# mu_k) corc T^l - nabla G(mu_k) = 0 (without saving the particles at each step, only the loss).

        Inputs:
        - rng: Jax random key
        - target_value_and_grad_plus: function returning value and the (Wasserstein) gradient of F
        - target_value_and_grad_minus: function returning value and the (Wasserstein) gradient of G
        - n_epochs: number of epochs
        - lr: step size (for Newton iterations)
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)
        - eps: regularization inverse in Newton
        - target_func: function to evaluate at each step

        Outputs:
        - List of losses
        - List target function value at each step
    """

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key = carry
        master_key, key_g, key_inner, key_f = jax.random.split(key, num=4)
        value_g_xk, grad_g_xk = target_value_and_grad_minus(xk, x_tgt, key_g)

        @jax.jit
        def f(y):
            _, grad_f_y = target_value_and_grad_plus(
                y, x_tgt, key_inner
            )
            return grad_f_y - grad_g_xk

        xk = newton_solver(f, xk, eps=eps, lr=lr)
        jax.clear_caches()

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


@partial(jax.jit, static_argnums=[1, 2, 3, 8, 11, 12])
def cccp_save(
    rng, target_value_and_grad_plus, target_value_and_grad_minus,
    n_epochs=100, lr=1, n_particles=500, d=2, x0=None,
    n_inner_epochs=50, x_tgt=None, m=0, save_interval=1,
    target_func=lambda x: 0
    ):
    """
        Convex Concave Procedure on an objectif of the form F-G.
        Save samples every save_interval.

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
    # Initial state
    if x0 is None:
        rng, key = jax.random.split(rng, num=2)
        x0 = jax.normal(key, shape=(n_particles, d))

    num_saves = (n_epochs // save_interval) + 1
    L_particles = jnp.zeros((num_saves,) + x0.shape)

    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, key, L_particles = carry
        master_key, key_g, key_inner, key_f = jax.random.split(key, num=4)
        value_g_xk, grad_g_xk = target_value_and_grad_minus(xk, x_tgt, key_g)

        # If it is random, should we take the same batchs?
        xk = inner_step_cccp(
            key_inner,
            xk,
            target_value_and_grad_plus,
            grad_g_xk, 
            n_epochs=n_inner_epochs,
            lr=lr,
            x_tgt=x_tgt,
            m=m
        )

        save_particle = (iter_num % save_interval == 0)
        particle_idx = (iter_num // save_interval)
        L_particles = L_particles.at[particle_idx].set(jax.device_get(xk)) * save_particle + L_particles * (1 - save_particle)

        value_f, _ = target_value_and_grad_plus(xk, x_tgt, key_f)
        return (xk, master_key, L_particles), (value_f - value_g_xk,  target_func(xk))

    # Use `lax.scan` to loop over epochs
    (xk, _, L_particles), L = jax.lax.scan(step, (x0, rng, L_particles), jnp.arange(n_epochs))
    L_loss, L_target_func = L
    L_particles = L_particles.at[0].set(jax.device_get(x0))

    return L_loss, L_particles, jnp.insert(L_target_func, 0, target_func(x0), axis=0)
