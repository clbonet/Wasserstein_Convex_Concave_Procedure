import jax

import numpy as np
import jax.numpy as jnp

from functools import partial
from PIL import Image


@jax.jit
def sqrtm(a):
    """
        From POT https://github.com/PythonOT/POT/blob/master/ot/backend.py
    """

    L, V = jnp.linalg.eigh(a)
    L = jnp.sqrt(L)
    # Q[...] = V[...] @ diag(L[...])
    Q = jnp.einsum('...jk,...k->...jk', V, L)
    # R[...] = Q[...] @ V[...].T
    return jnp.einsum('...jk,...kl->...jl', Q, jnp.swapaxes(V, -1, -2))


@partial(jax.jit, static_argnums=[4])
def sample_gmm_mv(key, weights, means, covs, n_samples):
    """
    weights: (K,)
    means:   (K, d)
    covs:    (K, d, d)
    """

    key_comp, key_gauss = jax.random.split(key)

    K, d = means.shape

    comps = jax.random.categorical(key_comp, jnp.log(weights), shape=(n_samples,))
    eps = jax.random.normal(key_gauss, shape=(n_samples, d))

    means_sel = means[comps]
    covs_sel = covs[comps]

    L = jnp.linalg.cholesky(covs_sel)

    return means_sel + jnp.einsum("nij,nj->ni", L, eps)


def generate_three_ring_and_gaussian(key, n_samples):
    _, key_perm, key_gauss = jax.random.split(key, 3)
    r, _delta = 0.3, 0.5
    Nx = n_samples // 3
    
    X = jnp.c_[r * jnp.cos(jnp.linspace(0, 2 * jnp.pi, Nx + 1)), r * jnp.sin(jnp.linspace(0, 2 * jnp.pi, Nx + 1))][:-1]  # noqa
    for i in [1, 2]:
        X = jnp.r_[X, X[:Nx, :]-i*jnp.array([0, (2 + _delta) * r])]

    X = jax.random.permutation(key_perm, X)
    Y = jax.random.normal(key_gauss, (n_samples, 2)) / 100 - jnp.array([0, r])
    return X, Y


def load_img(rng, fn='img/heart.png', size=200, max_samples=None):
    r"""Returns x,y of black pixels (between -1 and 1)
    """
    pic = np.array(Image.open(fn).resize((size,size)).convert('L'))
    y_inv, x = np.nonzero(pic<=128)
    y = size - y_inv - 1

    if max_samples and x.size > max_samples:
        ixsel = jax.random.choice(rng, x.size, shape=(max_samples,), replace=False)
        x, y = x[ixsel], y[ixsel]
    return np.stack((x, y), 1) / size * 2 - 1



def generate_data(rng, n_samples=500, d=2, target="gaussian", path_img="img/cat.png"):
    """
        Generate data from a centered Gaussian distribution, a Gaussian Mixture or three rings.

        Example from [1] (for Gaussian and Mixture of Gaussian).

        [1] Gladin, E., Dvurechenskii, P., Mielke, A., & Zhu, J. J. (2024). Interaction-force transport gradient flows. Advances in Neural Information Processing Systems, 37, 14484-14508.
    """
    _, key_X, key_x0 = jax.random.split(rng, num=3)

    if target == "gaussian":
        X_tgt = jax.random.normal(key_X, (n_samples, d)) @ sqrtm(jnp.array([[1, 0.5], [0.5, 1]]))
        X0 = 5 + jax.random.normal(key_x0, (n_samples, d))
    elif target == "gaussian_mixture":
        weights = jnp.array([1/3,1/3,1/3])
        means = jnp.array([[0,0],[3,-1],[1,4]])
        covs = jnp.array([[[1,1/2],[1/2,2]], [[1,0],[0,1]], [[3,1/2],[1/2,1]]])
        X_tgt = sample_gmm_mv(key_X, weights, means, covs, n_samples)
        X0 = 5 + jax.random.normal(key_x0, (n_samples, d))
    elif target == "rings":
        X_tgt, X0 = generate_three_ring_and_gaussian(key_X, n_samples)
    elif target == "img":
        X_tgt = load_img(key_X, path_img, size=200, max_samples=n_samples)
        X0 = jax.random.normal(key_x0, (n_samples, 2)) # + 5

    return X_tgt, X0
