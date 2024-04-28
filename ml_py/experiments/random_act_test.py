from typing import Any, NamedTuple
from einops import rearrange
from ..common import *
import optax
import jax
from ..base_models import RealGatedLRU
from equinox import nn
from jax import nn as jnn


class ModelConfig(NamedTuple):
    vocab_size: int
    d_model: int
    d_hidden: int
    d_ff: int
    n_layers: int
    dropout: float
    pad_token: int


class FeedForward(eqx.Module):
    inner_proj: nn.Linear
    outer_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, d_model: int, d_ff: int, dropout: float, *, key):
        key = RngKey(key)
        self.inner_proj = nn.Linear(d_model, d_ff, key=key.next())
        self.outer_proj = nn.Linear(d_ff, d_model, key=key.next())
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: ArrayLike, key):
        key = RngKey(key)

        x = self.inner_proj(x)
        x = jnn.gelu(x)
        x = self.dropout(x, key=key.next())
        x = self.outer_proj(x)

        return x


class Layer(eqx.Module):
    lru: RealGatedLRU
    ffn: FeedForward
    act_proj_1: nn.Linear
    act_proj_2: nn.Linear
    norm_1: nn.RMSNorm
    norm_2: nn.RMSNorm
    dropout: nn.Dropout

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        key,
    ):
        key = RngKey(key)
        self.lru = RealGatedLRU(cfg.d_model, cfg.d_model, cfg.d_hidden, key=key.next())
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout, key=key.next())
        self.act_proj_1 = nn.Linear(cfg.d_model, 1, key=key.next())
        self.act_proj_2 = nn.Linear(cfg.d_model, 1, key=key.next())
        self.norm_1 = nn.RMSNorm(cfg.d_model)
        self.norm_2 = nn.RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x: ArrayLike, key):
        key = RngKey(key)
        seq_len = x.shape[-2]

        residual = x
        residual = self.lru(residual)
        residual = jax.vmap(self.dropout)(residual, key=key.next(seq_len))

        act_prob = jnn.sigmoid(jax.vmap(self.act_proj_1)(residual))
        act = jax.random.uniform(key.next(), act_prob.shape) < act_prob
        residual = jnp.where(act, residual, 0)
        x = jax.vmap(self.norm_1)(x + residual)

        residual = x
        residual = jax.vmap(self.ffn)(residual, key.next(seq_len))
        residual = jax.vmap(self.dropout)(residual, key=key.next(seq_len))

        act_prob = jnn.sigmoid(jax.vmap(self.act_proj_2)(residual))
        act = jax.random.uniform(key.next(), act_prob.shape) < act_prob
        residual = jnp.where(act, residual, 0)
        x = jax.vmap(self.norm_2)(x + residual)

        return x


class Model(TrainableModel):
    cfg: ModelConfig
    layers: list[Layer]
    tok_emb: nn.Embedding

    def __init__(self, cfg: ModelConfig, *, key):
        self.cfg = cfg

        key = RngKey(key)

        self.layers = [Layer(cfg, key=k) for k in key.next(cfg.n_layers)]
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, key=key.next())

    def forward(self, input: TextBatch, key, state: Any | None) -> TrainableModelOutput:
        key = RngKey(key)

        x = jnp.concatenate([input.inputs, input.outputs], axis=0)
        x = jax.vmap(self.tok_emb)(x)
        for k, l in zip(key.next(self.cfg.n_layers), self.layers):
            x = l(x, k)

        x = x[(input.inputs.shape[0] - 1) : -1]
        out = BCESequenceOutput.create(x, input.outputs, self.cfg.pad_token)

        return out

    def forward_test(
        self, input: TextBatch, key, state: Any | None
    ) -> TrainableModelOutput:
        return self.forward(input, key, state)


def run(data: ExperimentData):
    max_seq_len = 128
    seed = 1234
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4

    tokenizer = ByteTokenizer(ByteTokenizerConfig())
    enc_cfg = EncodeConfig(max_length=max_seq_len, pad=True)

    ds = SimpleCsvDataset.load("./datasets/def_words/main.csv").shuffled(seed)
    ds = ds.subset(50_000).rename_columns({"text": "in_ids", "target": "out_ids"})
    ds = ds.tokenize(tokenizer, enc_cfg).jax()
    train_dl, test_dl = ds.train_test_split_loaders(seed, batch_size, 0.1, TextBatch)

    rng, model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size(),
        d_model=512,
        d_hidden=256,
        d_ff=2048,
        n_layers=4,
        dropout=0.1,
        pad_token=tokenizer.pad_token(),
    )
    model = Model(cfg, key=model_key)
    optim = optax.adam(learning_rate)
    optim = optax.chain(optax.clip(1.0), optim)
    opt_state = optim.init(model)

    model = model.fit(rng, num_epochs, optim, opt_state, train_dl, test_dl)
