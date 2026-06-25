import jax
import jax.numpy as jnp


def signed_mmd(x, y, kernel_positive, kernel_negative, x_weights=None, y_weights=None):
    """
        Positive part (for radial kernel decomposed as DC)
    """
    vmapped_kernel_positive = jax.vmap(kernel_positive, in_axes=(0, None))
    pairwise_kernel_positive = jax.vmap(vmapped_kernel_positive, in_axes=(None, 0))

    vmapped_kernel_negative = jax.vmap(kernel_negative, in_axes=(0, None))
    pairwise_kernel_negative = jax.vmap(vmapped_kernel_negative, in_axes=(None, 0))

    Kxx = pairwise_kernel_positive(x, x)
    Kyy = pairwise_kernel_positive(y, y)
    Kxy = -pairwise_kernel_negative(x, y)

    n = x.shape[0]
    m = y.shape[0]

    if x_weights is None:
        x_weights = jnp.ones(n) / n
    if y_weights is None:
        y_weights = jnp.ones(m) / m

    cpt1 = jnp.einsum("n, nm, m", x_weights, Kxx, x_weights)
    cpt2 = jnp.einsum("n, nm, m", y_weights, Kyy, y_weights)
    cpt3 = jnp.einsum("n, nm, m", y_weights, Kxy, x_weights)

    return (cpt1 + cpt2 - 2*cpt3)/2


def target_grad_signed_mmd(x, y, kernel_positive, kernel_negative, rng, x_weights=None, y_weights=None, n_sample_batch=None):
    """
        Use autodifferentiation.
        
        Parameters
        ----------
        x: array of size (n_samples, d)
        y: array of size (m_samples, d)
        kernel_positive: function taking x,y as input
        kernel_negative: function taking x,y as input
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

    out, grad = jax.value_and_grad(lambda z: signed_mmd(z, y_tgt, kernel_positive, kernel_negative, x_weights, y_weights))(x)
    return out, n * grad # Wasserstein gradient (Euclidean gradient rescaled by the number of samples)
