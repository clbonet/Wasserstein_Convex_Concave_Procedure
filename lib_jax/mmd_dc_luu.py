import jax
import jax.numpy as jnp


def positive_part_mmd_luu(x, y, kernel, alpha=0, x_weights=None, y_weights=None):
    """
        Positive part (using decomposition from [1]), i.e. 
        F_+(mu) = (1/2) iint (alpha |x|^2 + alpha |y|^2 + k(x, y)) dmu(x) dmu(y)

        [1] Luu, H. P. H., Yu, H., Williams, B., Mikkola, P., Hartmann, M., Puolamäki, K., & Klami, A. (2024). 
        Non-geodesically-convex optimization in the Wasserstein space. NeurIPS, 37, 16772-16809.
    """
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))

    Kxx = pairwise_kernel(x, x)
    # Kxy = -pairwise_kernel_negative(x, y)

    n = x.shape[0]
    if x_weights is None:
        x_weights = jnp.ones(n) / n

    cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
    cpt2 = 2 * alpha * jnp.mean(jnp.sum(jnp.square(x), axis=-1))

    return (cpt1 + cpt2)/2


def negative_part_mmd_luu(x, y, kernel, alpha=0, x_weights=None, y_weights=None):
    """
        Negative part (using decomposition from [1]), i.e.
        F_-(mu) = int (alpha |x|^2 + int k(x,y) dnu(y) ) dmu(x)

        [1] Luu, H. P. H., Yu, H., Williams, B., Mikkola, P., Hartmann, M., Puolamäki, K., & Klami, A. (2024).
        Non-geodesically-convex optimization in the Wasserstein space. NeurIPS, 37, 16772-16809.
    """
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))

    Kxy = pairwise_kernel(x, y)

    n = x.shape[0]
    m = y.shape[0]

    if x_weights is None:
        x_weights = jnp.ones(n) / n
    if y_weights is None:
        y_weights = jnp.ones(m) / m

    cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)
    cpt2 = alpha * jnp.mean(jnp.sum(jnp.square(x), axis=-1))

    return cpt2 + cpt3


def target_grad_positive_part_mmd_luu(x, y, kernel, rng, x_weights=None, y_weights=None, n_sample_batch=None, alpha=0):
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
        alpha: float
    """
    n, _ = x.shape
    m, _ = y.shape

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        master_key, key = jax.random.split(rng, num=2)
        y_tgt = jax.random.choice(key, y, (n_sample_batch,), replace=False)

    out, grad = jax.value_and_grad(lambda z: positive_part_mmd_luu(z, y_tgt, kernel, alpha, x_weights, y_weights))(x)
    return out, n * grad # Wasserstein gradient (Euclidean gradient rescaled by the number of samples)


def target_grad_negative_part_mmd_luu(x, y, kernel, rng, x_weights=None, y_weights=None, n_sample_batch=None, alpha=0):
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
        alpha: float
    """
    n, _ = x.shape
    m, _ = y.shape

    if n_sample_batch is None or n_sample_batch==m:
        y_tgt = y
    else:
        master_key, key = jax.random.split(rng, num=2)
        y_tgt = jax.random.choice(key, y, (n_sample_batch,), replace=False)

    out, grad = jax.value_and_grad(lambda z: negative_part_mmd_luu(z, y_tgt, kernel, alpha, x_weights, y_weights))(x)
    return out, n * grad # Wasserstein gradient (Euclidean gradient rescaled by the number of samples)
