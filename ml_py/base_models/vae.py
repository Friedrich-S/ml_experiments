from typing import Tuple
import equinox as eqx
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
import jax
from ..common.model import BCESequenceOutput
from .misc import SelectGate
from .seq_model import TransformerEncoder
from jaxtyping import ArrayLike, PRNGKeyArray


class SeqVae(eqx.Module):
    seq_len: int
    enc_tf: TransformerEncoder
    dec_tf: TransformerEncoder
    enc_pos_emb: nn.Embedding
    dec_pos_emb: nn.Embedding
    token_emb: nn.Embedding
    dec_scale: jax.Array
    gate_1: SelectGate
    gate_2: SelectGate
    output: nn.Linear

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        seq_len: int,
        n_heads: int,
        n_layers: int,
        *,
        key
    ):
        self.seq_len = seq_len

        keys = jax.random.split(key, 9)
        self.enc_tf = TransformerEncoder(
            d_model, d_model * 4, n_heads, n_layers, key=keys[0]
        )
        self.dec_tf = TransformerEncoder(
            d_model, d_model * 4, n_heads, n_layers, key=keys[1]
        )
        self.enc_pos_emb = nn.Embedding(seq_len, d_model, key=keys[2])
        self.dec_pos_emb = nn.Embedding(seq_len, d_model, key=keys[3])
        self.token_emb = nn.Embedding(vocab_size, d_model, key=keys[4])
        self.dec_scale = jax.random.normal(keys[5], (seq_len, d_model))
        self.gate_1 = SelectGate(d_model, d_model, d_model // 8, key=keys[6])
        self.gate_2 = SelectGate(d_model, d_model, d_model // 8, key=keys[7])
        self.output = nn.Linear(d_model, vocab_size, key=keys[8])

    def encode(
        self, x: ArrayLike, pad_mask: ArrayLike, key: PRNGKeyArray | None = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        seq_pos = jnp.arange(0, x.shape[-1])
        pos_emb = jax.vmap(self.enc_pos_emb)(seq_pos)
        tok_emb = jax.vmap(self.token_emb)(x)
        emb = (pos_emb + tok_emb) / 2
        x = self.enc_tf(emb, pad_mask, key=key)

        mean = jax.vmap(self.gate_1)(x, x).mean(-2)
        logvar = jax.vmap(self.gate_2)(x, x).mean(-2)

        return mean, logvar

    def decode(self, x: ArrayLike, key: PRNGKeyArray | None = None) -> ArrayLike:
        x = jnp.repeat(x[None, :], self.seq_len, axis=-2)
        x = x * jnn.sigmoid(self.dec_scale)

        seq_pos = jnp.arange(0, x.shape[-2])
        pos_emb = jax.vmap(self.dec_pos_emb)(seq_pos)
        emb = (pos_emb + x) / 2

        x = self.dec_tf(emb, None, key=key)
        x = jax.vmap(self.output)(x)

        return x

    def reparameterize(self, mean: ArrayLike, logvar: ArrayLike, *, key) -> ArrayLike:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, logvar.shape)
        return mean + eps * std

    def loss(
        self,
        mean: ArrayLike,
        logvar: ArrayLike,
        result: ArrayLike,
        target: ArrayLike,
        pad_token: int,
    ) -> ArrayLike:
        @jax.vmap
        def kl_divergence(mean, logvar):
            return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

        loss = BCESequenceOutput.create(result, target, pad_token).get_loss()
        latent_loss = kl_divergence(mean, logvar).mean()
        return loss + latent_loss
