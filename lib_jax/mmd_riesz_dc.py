import jax
import ott
import jax.numpy as jnp


## MMD Riesz with OTT JAX, and as a DC functional
def mmd_riesz1(
    source: jnp.ndarray, target: jnp.ndarray, r=1,
    *args, **kwargs # if rng is passed
) -> float:
    """
        Convex part: F(mu) = int -V dmu
    """
    euc_cost = ott.geometry.costs.Euclidean()

    dist_xy = euc_cost.all_pairs(source, target) ** r
    
    pairwise = jnp.mean(dist_xy)
    return pairwise


def mmd_riesz2(
    source: jnp.ndarray, target: jnp.ndarray, r=1,
    *args, **kwargs # if rng is passed
) -> float:
    """
        Concave part: G(mu) = - frac12 int int k(x,y) dmu(x) dmu(y) - frac12 int int k(x,y) dnu(x) dnu(y)
        where k(x,y)=-|x-y|^r
    """
    euc_cost = ott.geometry.costs.Euclidean()

    dist_xx = euc_cost.all_pairs(source, source) ** r
    dist_yy = euc_cost.all_pairs(target, target) ** r
    
    single = jnp.mean(dist_xx) + jnp.mean(dist_yy)
    return single/2
    

def target_value_and_grad_mmd_riesz1(x, y, rng, r=1, n_sample_batch=None):
    n, m = len(x), len(y)
    
    master_key, key_samples = jax.random.split(rng)

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        y_tgt = jax.random.choice(key_samples, y, (n_sample_batch,), replace=False)

    value_and_grad_mmd = jax.value_and_grad(lambda z: mmd_riesz1(z, y_tgt, r=r))
    val, grad = value_and_grad_mmd(x)
    
    return val, n * grad


def target_value_and_grad_mmd_riesz2(x, y, rng, r=1, n_sample_batch=None):
    n, m = len(x), len(y)
    
    master_key, key_samples = jax.random.split(rng)

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        y_tgt = jax.random.choice(key_samples, y, (n_sample_batch,), replace=False)

    value_and_grad_mmd = jax.value_and_grad(lambda z: mmd_riesz2(z, y_tgt, r=r))
    val, grad = value_and_grad_mmd(x)
    
    return val, n * grad


def mmd_smoothed_riesz1(
    source: jnp.ndarray, target: jnp.ndarray, eps=1e-8,
    *args, **kwargs # if rng is passed
) -> float:
    """
        Convex part: F(mu) = int -V dmu
    """
    euc_cost = ott.geometry.costs.SqEuclidean()

    squared_dist_xy = euc_cost.all_pairs(source, target)
    dist_xy = jnp.power(squared_dist_xy + eps, 0.5)

    
    pairwise = jnp.mean(dist_xy)
    return pairwise


def mmd_smoothed_riesz2(
    source: jnp.ndarray, target: jnp.ndarray, eps=1e-8,
    *args, **kwargs # if rng is passed
) -> float:
    """
        Concave part: G(mu) = - frac12 int int k(x,y) dmu(x) dmu(y) - frac12 int int k(x,y) dnu(x) dnu(y)
        where k(x,y)=-|x-y|^r
    """
    euc_cost = ott.geometry.costs.SqEuclidean()

    squared_dist_xx = euc_cost.all_pairs(source, source)
    squared_dist_yy = euc_cost.all_pairs(target, target)

    dist_xx = jnp.power(squared_dist_xx + eps, 0.5)
    dist_yy = jnp.power(squared_dist_yy + eps, 0.5)

    single = jnp.mean(dist_xx) + jnp.mean(dist_yy)
    return single/2


def target_value_and_grad_mmd_smoothed_riesz1(x, y, rng, eps=1e-8, n_sample_batch=None):
    n, m = len(x), len(y)
    
    master_key, key_samples = jax.random.split(rng)

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        y_tgt = jax.random.choice(key_samples, y, (n_sample_batch,), replace=False)

    value_and_grad_mmd = jax.value_and_grad(lambda z: mmd_smoothed_riesz1(z, y_tgt, eps=eps))
    val, grad = value_and_grad_mmd(x)
    
    return val, n * grad


def target_value_and_grad_mmd_smoothed_riesz2(x, y, rng, eps=1e-8, n_sample_batch=None):
    n, m = len(x), len(y)
    
    master_key, key_samples = jax.random.split(rng)

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        y_tgt = jax.random.choice(key_samples, y, (n_sample_batch,), replace=False)

    value_and_grad_mmd = jax.value_and_grad(lambda z: mmd_smoothed_riesz2(z, y_tgt, eps=eps))
    val, grad = value_and_grad_mmd(x)
    
    return val, n * grad