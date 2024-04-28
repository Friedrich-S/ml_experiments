import equinox as eqx
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
import jax
from jaxtyping import ArrayLike, PRNGKeyArray
from ..common import split_key, RngKey
from abc import ABC, abstractmethod


class SeqModelTimeAxisProj(ABC):
    @abstractmethod
    def seq_forward(
        self, x: jax.Array, pad_mask: jax.Array | None, *, key
    ) -> jax.Array:
        raise NotImplementedError()


class BaseSeqModelFeedForward(eqx.Module):
    inner_proj: nn.Linear
    outer_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, d_model: int, d_ff: int, dropout: float, *, key):
        keys = jax.random.split(key, 2)
        self.inner_proj = nn.Linear(d_model, d_ff, key=keys[0])
        self.outer_proj = nn.Linear(d_ff, d_model, key=keys[1])
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: ArrayLike, key: PRNGKeyArray | None = None):
        x = self.inner_proj(x)
        x = jnn.gelu(x)
        x = self.dropout(x, key=key)
        x = self.outer_proj(x)
        return x


class BaseSeqModelLayer(eqx.Module):
    time_axis_proj: SeqModelTimeAxisProj
    ffn: BaseSeqModelFeedForward
    norm_1: nn.RMSNorm
    norm_2: nn.RMSNorm
    dropout: nn.Dropout

    def __init__(
        self,
        time_axis_proj: SeqModelTimeAxisProj,
        d_model: int,
        d_ff: int,
        dropout: float,
        *,
        key,
    ):
        key = RngKey(key)
        self.time_axis_proj = time_axis_proj
        self.ffn = BaseSeqModelFeedForward(d_model, d_ff, dropout, key=key.next())
        self.norm_1 = nn.RMSNorm(d_model)
        self.norm_2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @eqx.filter_jit
    def __call__(
        self, x: ArrayLike, pad_mask: ArrayLike | None, key: PRNGKeyArray | None = None
    ):
        key = RngKey(key)
        seq_len = x.shape[-2]

        residual = x
        residual = self.time_axis_proj.seq_forward(residual, pad_mask, key=key.next())
        residual = jax.vmap(self.dropout)(residual, key=key.next(seq_len))
        x = jax.vmap(self.norm_1)(x + residual)

        residual = x
        residual = jax.vmap(self.ffn)(residual, key=key.next(seq_len))
        residual = jax.vmap(self.dropout)(residual, key=key.next(seq_len))
        x = jax.vmap(self.norm_2)(x + residual)

        return x


class BaseSeqModel(eqx.Module):
    layers: list[BaseSeqModelLayer]

    def __init__(
        self,
        time_axis_proj: SeqModelTimeAxisProj,
        d_model: int,
        d_ff: int,
        n_layers: int,
        *,
        key,
        dropout: float = 0.1,
    ):
        keys = jax.random.split(key, n_layers)
        self.layers = [
            BaseSeqModelLayer(time_axis_proj, d_model, d_ff, dropout, key=k)
            for k in keys
        ]

    def __call__(
        self, x: ArrayLike, pad_mask: ArrayLike | None, key: PRNGKeyArray | None = None
    ):
        if pad_mask is not None:
            pad_mask = jnp.repeat(pad_mask[None, :], pad_mask.shape[0], axis=0)

        keys = split_key(key, len(self.layers))
        for k, l in zip(keys, self.layers):
            x = l(x, pad_mask, key=k)

        return x


class AttentionTimeAxisProj(eqx.Module, SeqModelTimeAxisProj):
    attn: nn.MultiheadAttention

    def __init__(self, n_heads: int, d_model: int, dropout: float, *, key):
        key = RngKey(key)

        self.attn = nn.MultiheadAttention(
            n_heads, d_model, dropout_p=dropout, key=key.next()
        )

    def seq_forward(
        self, x: jax.Array, pad_mask: jax.Array | None, *, key
    ) -> jax.Array:
        key = RngKey(key)

        if pad_mask is not None:
            pad_mask = jnp.repeat(pad_mask[None, :], pad_mask.shape[0], axis=0)

        return self.attn(x, x, x, mask=pad_mask, key=key.next())


class TransformerEncoder(BaseSeqModel):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        *,
        key,
        dropout: float = 0.1,
    ):
        key = RngKey(key)
        proj = AttentionTimeAxisProj(n_heads, d_model, dropout, key=key.next())
        super().__init__(proj, d_model, d_ff, n_layers, dropout=dropout, key=key.next())
