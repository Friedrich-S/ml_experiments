import equinox as eqx
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
import jax


# Implementation based on https://github.com/andylolu2/jax-diffusion/tree/main
class Diffuser(eqx.Module):
    betas: jax.Array
    alphas: jax.Array
    alpha_bars: jax.Array

    def __init__(
        self, n_steps: int, beta_1: float = 1e-4, beta_t: float = 0.02, *, key
    ):
        self.betas = jnp.linspace(beta_1, beta_t, n_steps, dtype=jnp.float32)
        self.alphas = 1 - self.betas
        self.alpha_bars = jnp.cumprod(self.alphas)

    def forward(self, x_0, t, key):
        alpha_t_bar = jax.lax.stop_gradient(self.alpha_bars[t])

        eps = jax.random.normal(key, shape=x_0.shape, dtype=x_0.dtype)
        x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps
        return x_t, eps

    def ddim_backward_step(self, x_t, t, t_next, eps):
        alpha_t = jax.lax.stop_gradient(self.alpha_bars[t])
        alpha_t_next = jax.lax.stop_gradient(self.alpha_bars[t_next])

        x_0 = (x_t - (1 - alpha_t) ** 0.5 * eps) / alpha_t**0.5
        x_t_direction = (1 - alpha_t_next) ** 0.5 * eps
        x_t_next = alpha_t_next**0.5 * x_0 + x_t_direction

        return x_t_next
