import jax
import ott
import jax.numpy as jnp

from ott.math.utils import norm


## kernels
@jax.jit
def gaussian_kernel(x, y, h=1):
    return jnp.exp(-jnp.sum(jnp.square(x-y), axis=-1)/ (2*h))

@jax.jit
def imq_kernel(x, y, c=1):
    return 1 / jnp.sqrt(c + jnp.sum(jnp.square(x-y), axis=-1))

@jax.jit
def laplace_kernel(x, y, h):
    return jnp.exp(-jnp.sum(jnp.abs(x-y), axis=-1)/ h)
    
@jax.jit
def riesz_kernel(x, y, r=1):
    return -norm(x-y, axis=-1)**r


## MMD
def mmd(x, y, kernel, x_weights=None, y_weights=None):
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))
    
    Kxx = pairwise_kernel(x, x)
    Kyy = pairwise_kernel(y, y)
    Kxy = pairwise_kernel(x, y)

    n = x.shape[0]
    m = y.shape[0]

    if x_weights is None:
        x_weights = jnp.ones(n) / n
    if y_weights is None:
        y_weights = jnp.ones(m) / m

    cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
    cpt2 = jnp.einsum("n, nm, m", y_weights, Kyy, y_weights)
    cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)

    return (cpt1+cpt2-2*cpt3)/2


def target_grad_mmd(x, y, kernel, rng, x_weights=None, y_weights=None, n_sample_batch=None):
    """
        Use autodifferentiation.
        
        Parameters
        ----------
        x: array of size (n_samples, d)
        y: array of size (m_samples, d)
        kernel: function taking x,y as input
        rng: key
        x_weights: array of size (n_samples,)
        y_weights: array of size (m_samples,)
        n_sample_batch: number of particles to sample from y, default None
    """
    n, _ = x.shape
    m, _ = y.shape

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        master_key, key = jax.random.split(rng, num=2)
        y_tgt = jax.random.choice(key, y, (n_sample_batch,), replace=False)

    out, grad = jax.value_and_grad(lambda z: mmd(z, y_tgt, kernel, x_weights, y_weights))(x)
    return out, n * grad # Wasserstein gradient (Euclidean gradient rescaled by the number of samples)


def target_value_and_grad_gaussian_kernel(x, y, rng, x_weights=None, h=0.1, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: gaussian_kernel(k, l, h=h)
    l, grad = target_grad_mmd(x, y, kernel, master_key, x_weights=x_weights, n_sample_batch=n_sample_batch)
    return l, grad


def target_value_and_grad_laplace_kernel(x, y, rng, x_weights=None, h=1, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: laplace_kernel(k, l, h=h)
    l, grad = target_grad_mmd(x, y, kernel, master_key, x_weights=x_weights, n_sample_batch=n_sample_batch)
    return l, grad


def target_value_and_grad_riesz_kernel(x, y, rng, x_weights=None, r=1, n_sample_batch=None):
    master_key, key = jax.random.split(rng, num=2)
    kernel = lambda k, l: riesz_kernel(k, l, r=r)
    l, grad = target_grad_mmd(x, y, kernel, master_key, x_weights=x_weights, n_sample_batch=n_sample_batch)
    return l, grad


def grad_mmd(x, x_tgt, rng, sum_kernel_grad):
    n = x.shape[0]

    grad_x = sum_kernel_grad(x, x, rng)
    grad_tgt = sum_kernel_grad(x, x_tgt, rng)
    nabla_mmd = (grad_x - grad_tgt) / n

    return nabla_mmd


## MMD Riesz with OTT JAX
def mmd_riesz(
    source: jnp.ndarray, target: jnp.ndarray, r=1,
    *args, **kwargs # if rng is passed
) -> float:
    euc_cost = ott.geometry.costs.Euclidean()

    dist_xx = euc_cost.all_pairs(source, source) ** r
    dist_xy = euc_cost.all_pairs(source, target) ** r
    dist_yy = euc_cost.all_pairs(target, target) ** r
    
    pairwise = jnp.mean(dist_xy)
    single = jnp.mean(dist_xx) + jnp.mean(dist_yy)
    return pairwise - single/2


def target_value_and_grad_mmd_riesz(x, y, rng, r=1, n_sample_batch=None):
    n, m = len(x), len(y)
    
    master_key, key_samples = jax.random.split(rng)

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        y_tgt = jax.random.choice(key_samples, y, (n_sample_batch,), replace=False)

    value_and_grad_mmd = jax.value_and_grad(lambda z: mmd_riesz(z, y_tgt, r=r))
    val, grad = value_and_grad_mmd(x)
    
    return val, n * grad
