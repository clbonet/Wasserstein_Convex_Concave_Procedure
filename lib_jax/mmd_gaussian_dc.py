import jax
import ott

import jax.numpy as jnp

from functools import partial

from .mmd_radial_dc import target_grad_signed_mmd


euc_cost = jax.jit(ott.geometry.costs.Euclidean())


## Decomposition of the kernel using the Taylor serie
@partial(jax.jit, static_argnums=[1])
def truncated_exp_even(z, order):
    """
        Computes sum_{k even} z^k/k!
    """
    def body(carry, n):
        term, acc = carry
        term = term * z**2 / (n * (n-1))
        acc = acc + term
        return (term, acc), None

    init = (jnp.ones_like(z), jnp.ones_like(z))  # term_0 = 1
    (final_term, result), _ = jax.lax.scan(body, init, jnp.arange(2, order + 1, step=2))

    return result


@partial(jax.jit, static_argnums=[1])
def truncated_exp_odd(z, order):
    """
        Computes sum_{k odd} z^k/k!
    """
    def body(carry, n):
        term, acc = carry
        term = term * z**2 / (n * (n-1))
        acc = acc + term
        return (term, acc), None

    init = (z, z)  # term_1 = _z
    (final_term, result), _ = jax.lax.scan(body, init, jnp.arange(3, order + 1, step=2))

    return result


@jax.jit
def cosh_gaussian_kernel(x, y, h=1):
    z = jnp.sum(jnp.square(x-y), axis=-1)/(2*h)
    return jnp.cosh(z)


@jax.jit
def sinh_gaussian_kernel(x, y, h=1):
    z = jnp.sum(jnp.square(x-y), axis=-1)/(2*h)
    return jnp.sinh(z)


@jax.jit
def positive_gaussian_kernel(x, y, h=1):
    squared_dist = jnp.sum(jnp.square(x-y), axis=-1)/(2*h)
    return jnp.exp(-squared_dist) + squared_dist


@jax.jit
def negative_gaussian_kernel(x, y, h=1):
    squared_dist = jnp.sum(jnp.square(x-y), axis=-1)/(2*h)
    return squared_dist


def target_value_and_grad_positive_gaussian_kernel(
        x, y, rng, x_weights=None, h=1, n_sample_batch=None
    ):
    """
        Convex part of the MMD with Gaussian kernel decomposed using the cosh/sinh.
        It contains the positive part of the interaction term, and the negative part of the potential term.
    """
    master_key, key = jax.random.split(rng, num=2)
    kernel_positive = lambda k, l: cosh_gaussian_kernel(k, l, h=h)
    kernel_negative = lambda k, l: sinh_gaussian_kernel(k, l, h=h)
    l, grad = target_grad_signed_mmd(
        x, y, kernel_positive, kernel_negative, master_key,
        x_weights=x_weights, n_sample_batch=n_sample_batch
    )
    return l, grad


def target_value_and_grad_negative_gaussian_kernel(
        x, y, rng, x_weights=None, h=1, n_sample_batch=None
    ):
    """
        Concave part of the MMD with Gaussian kernel decomposed using the cosh/sinh.
        It contains the negative part of the interaction term, and the positive part of the potential term
    """
    master_key, key = jax.random.split(rng, num=2)
    kernel_positive = lambda k, l: sinh_gaussian_kernel(k, l, h=h)
    kernel_negative = lambda k, l: cosh_gaussian_kernel(k, l, h=h)
    l, grad = target_grad_signed_mmd(
        x, y, kernel_positive, kernel_negative, master_key,
        x_weights=x_weights, n_sample_batch=n_sample_batch
    )
    return l, grad


def target_value_and_grad_positive_jordan_gaussian_kernel(
        x, y, rng, x_weights=None, h=1, n_sample_batch=None
    ):
    """
        Convex part of the MMD with Gaussian kernel decomposed using the Jordan decomposition.
        It contains the positive part of the interaction term, and the negative part of the potential term.
    """
    master_key, key = jax.random.split(rng, num=2)
    kernel_positive = lambda k, l: positive_gaussian_kernel(k, l, h=h)
    kernel_negative = lambda k, l: negative_gaussian_kernel(k, l, h=h)
    l, grad = target_grad_signed_mmd(
        x, y, kernel_positive, kernel_negative, master_key,
        x_weights=x_weights, n_sample_batch=n_sample_batch
    )
    return l, grad


def target_value_and_grad_negative_jordan_gaussian_kernel(
        x, y, rng, x_weights=None, h=1, n_sample_batch=None
    ):
    """
        Concave part of the MMD with Gaussian kernel decomposed using the Jordan decomposition.
        It contains the negative part of the interaction term, and the positive part of the potential term
    """
    master_key, key = jax.random.split(rng, num=2)
    kernel_positive = lambda k, l: negative_gaussian_kernel(k, l, h=h)
    kernel_negative = lambda k, l: positive_gaussian_kernel(k, l, h=h)
    l, grad = target_grad_signed_mmd(
        x, y, kernel_positive, kernel_negative, master_key,
        x_weights=x_weights, n_sample_batch=n_sample_batch
    )
    return l, grad


def clip_grad_norm(grads, max_norm):
    grad_norm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
    )

    clip_coef = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))

    clipped_grads = jax.tree_util.tree_map(
        lambda g: g * clip_coef, grads
    )

    return clipped_grads, grad_norm


def target_value_and_grad_positive_gaussian_kernel_clipped(
        x, y, rng, x_weights=None, h=1, n_sample_batch=None, order=20, clip_value=25
    ):
    """
        Convex part of the MMD with Gaussian kernel decomposed using the cosh/sinh.
        It contains the positive part of the interaction term, and the negative part of the potential term.

        The gradient is clipped to avoid NaN values.
    """
    master_key, key = jax.random.split(rng, num=2)
    kernel_positive = lambda k, l: cosh_gaussian_kernel(k, l, h=h, order=order)
    kernel_negative = lambda k, l: sinh_gaussian_kernel(k, l, h=h, order=order)
    l, grad = target_grad_signed_mmd(
        x, y, kernel_positive, kernel_negative, master_key,
        x_weights=x_weights, n_sample_batch=n_sample_batch
    )

    # Clip grad
    grad, grad_norm = clip_grad_norm(grad, clip_value)
    return l, grad


def target_value_and_grad_negative_gaussian_kernel_clipped(
        x, y, rng, x_weights=None, h=1, n_sample_batch=None, order=20, clip_value=25
    ):
    """
        Concave part of the MMD with Gaussian kernel decomposed using the cosh/sinh.
        It contains the negative part of the interaction term, and the positive part of the potential term
    """
    master_key, key = jax.random.split(rng, num=2)
    kernel_positive = lambda k, l: sinh_gaussian_kernel(k, l, h=h, order=order)
    kernel_negative = lambda k, l: cosh_gaussian_kernel(k, l, h=h, order=order)
    l, grad = target_grad_signed_mmd(
        x, y, kernel_positive, kernel_negative, master_key,
        x_weights=x_weights, n_sample_batch=n_sample_batch
    )

    # Clip grad
    grad, grad_norm = clip_grad_norm(grad, clip_value)
    return l, grad