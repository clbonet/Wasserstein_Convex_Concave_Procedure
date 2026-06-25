import jax
import jaxopt

import jax.numpy as jnp


def fwd_solver(f, z_init):
    """
        From https://implicit-layers-tutorial.org/implicit_functions/
    """
    input_dtype = z_init.dtype

    def cond_fun(carry):
        z_prev, z = carry
        return jnp.linalg.norm(z_prev - z) > 1e-5

    def body_fun(carry):
        _, z = carry
        fz = f(z)
        # Ensure dtype consistency for while_loop
        fz = fz.astype(input_dtype)
        return z, fz

    init_carry = (z_init, f(z_init).astype(input_dtype))
    # _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    _, z_star = jaxopt.loop.while_loop(cond_fun, body_fun, init_carry,
                                       maxiter=100, jit=True)
    return z_star


def newton_solver(f, z_init, eps=0, lr=1.0):
    """
        Newton solver
    """
    n, d = z_init.shape
    input_dtype = z_init.dtype

    grad_f = jax.jit(jax.jacobian(f))
    Id_eps = eps * jnp.eye(n*d, n*d, dtype=input_dtype)

    def g(z):
        grad = grad_f(z).reshape(n*d, n*d).astype(input_dtype) + Id_eps
        fz = f(z).reshape((n*d,)).astype(input_dtype)
        newton_step = jnp.linalg.solve(grad, fz).reshape(n, d).astype(input_dtype)
        result = z - lr * newton_step
        # Ensure dtype consistency
        return result.astype(input_dtype)

    return fwd_solver(g, z_init)
