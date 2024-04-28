from typing import Any, NamedTuple
from einops import rearrange
from ..common import *
import optax
import jax
from ..base_models import SeqVae, Diffuser, LstmGate, Rnn, RnnGate
from equinox import nn
from jaxtyping import PRNGKeyArray
from jax import nn as jnn


class ModelConfig(NamedTuple):
    vocab_size: int
    max_seq_len: int
    d_model: int
    vae_chunk_size: int
    ddpm_steps: int
    ddim_steps: int
    pad_token: int


# ToDo: the model could be a simple next-token predictor with teacher forcing,
# with time-axis communication using something like an LRU. Each linear layer
# would be equipped with a next-input predictor, which is trained to predict the
# next input into that layer. The input is mixed with the predicted next-input.
#
# ToDo: Try implementing an architecture like MoE, where there are multiple
# experts, but each expert also has a next-input prediction module. The expert
# weighting is determined through another projection like MoE and the next-input
# prediction is simply fed into the experts weighted by some learnable factor.
# Here, the gradients will flow through the input prediction model to the gating
# projection for the experts, but in the actual model, the input prediction
# model should probably be cut off from other gradients and simply serve as a
# routing mechanism (but also influence the activations of the nodes).
class Model(TrainableModel):
    cfg: ModelConfig

    def __init__(self, cfg: ModelConfig, *, key):
        self.cfg = cfg

        keys = jax.random.split(key, 11)

    def forward(self, input: TextBatch, key, state: Any | None) -> TrainableModelOutput:
        keys = split_key(key, 4)

        # ToDo
        pass

    def forward_test(
        self, input: TextBatch, key, state: Any | None
    ) -> TrainableModelOutput:
        keys = split_key(key, 4)

        # ToDo
        pass


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
        max_seq_len=max_seq_len,
        d_model=512,
        vae_chunk_size=64,
        ddpm_steps=1000,
        ddim_steps=20,
        pad_token=tokenizer.pad_token(),
    )
    model, state = nn.make_with_state(Model)(cfg, key=model_key)
    optim = optax.adam(learning_rate)
    optim = optax.chain(optax.clip(1.0), optim)
    opt_state = optim.init(model)

    model = model.fit(rng, num_epochs, optim, opt_state, train_dl, test_dl, state=state)
