import equinox as eqx
from equinox import nn
from jax import nn as jnn
import jax


class LowRankLinear(eqx.Module):
    """
    A linear layer that uses a low-rank instead of a full-rank matrix.
    Internally implemented as two linear layers which down-project to a
    rank-dimensional vector.
    """

    a: nn.Linear
    b: nn.Linear

    def __init__(self, d_in: int, d_out: int, rank: int, *, key, bias: bool = True):
        keys = jax.random.split(key, 2)
        self.a = nn.Linear(d_in, rank, use_bias=False, key=keys[0])
        self.b = nn.Linear(rank, d_out, use_bias=bias, key=keys[1])

    def __call__(self, x):
        return self.b(self.a(x))


class SelectGate(eqx.Module):
    proj: nn.Linear
    gate: LowRankLinear

    def __init__(self, d_in: int, d_out: int, rank: int, *, key):
        keys = jax.random.split(key, 2)
        self.proj = nn.Linear(d_in, d_out, key=keys[0])
        self.gate = LowRankLinear(d_in, d_in, rank, key=keys[1])

    def __call__(self, a, b):
        return self.proj(a * jnn.sigmoid(self.gate(b)))
