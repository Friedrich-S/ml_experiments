import equinox as eqx
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
import jax
from ..common.utils import RngKey
from . import SeqModelTimeAxisProj, BaseSeqModel


# Based on:
#  - Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De.
#    Resurrecting Recurrent Neural Networks for Long Sequences. https://arxiv.org/abs/2303.06349
#  - Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, Caglar Gulcehre
#    Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models. https://arxiv.org/abs/2402.19427
class RealGatedLRU(eqx.Module):
    pre_a: jax.Array
    r_proj: nn.Linear
    i_proj: nn.Linear
    x_proj: nn.Linear
    out_proj: nn.Linear

    def __init__(self, d_in: int, d_out: int, d_hidden: int, *, key):
        key = RngKey(key)

        C = 8
        min_a = 0.99 ** (1.0 / C)
        max_a = 0.999 ** (1.0 / C)
        self.pre_a = (
            jax.random.uniform(key.next(), (d_hidden,)) * (max_a - min_a) + min_a
        )
        self.r_proj = nn.Linear(d_in, d_hidden, key=key.next())
        self.i_proj = nn.Linear(d_in, d_hidden, key=key.next())
        self.x_proj = nn.Linear(d_in, d_hidden, key=key.next())
        self.out_proj = nn.Linear(d_hidden, d_out, key=key.next())

    def __call__(self, inputs):
        C = 8

        @jax.vmap
        def scan_op(q_i, q_j):
            A_i, b_i = q_i
            A_j, b_j = q_j
            return A_j * A_i, A_j * b_i + b_j

        r_t = jnn.sigmoid(jax.vmap(self.r_proj)(inputs))
        i_t = jnn.sigmoid(jax.vmap(self.i_proj)(inputs))
        a_t = jnn.softplus(self.pre_a) * r_t
        a_t = jnp.exp(-C * a_t)

        x = jax.vmap(self.x_proj)(inputs)
        gated_x = jnp.sqrt(1.0 - jnp.square(a_t)) * (i_t * x)

        _, hidden_states = jax.lax.associative_scan(scan_op, (a_t, gated_x))
        outputs = jax.vmap(self.out_proj)(hidden_states)

        return outputs


class RealGatedLruTimeAxisProj(eqx.Module, SeqModelTimeAxisProj):
    lru: RealGatedLRU

    def __init__(self, d_in: int, d_out: int, d_hidden: int, *, key):
        key = RngKey(key)
        self.lru = RealGatedLRU(d_in, d_out, d_hidden, key=key.next())

    def seq_forward(
        self, x: jax.Array, pad_mask: jax.Array | None, *, key
    ) -> jax.Array:
        return self.lru(x)


class RealGatedLruSeqModel(BaseSeqModel):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_ff: int,
        n_layers: int,
        *,
        key,
        dropout: float = 0.1,
    ):
        key = RngKey(key)
        proj = RealGatedLruTimeAxisProj(d_model, d_model, d_hidden, key=key.next())
        super().__init__(proj, d_model, d_ff, n_layers, dropout=dropout, key=key.next())
