from typing import Any, NamedTuple
from einops import rearrange
from ..common import *
import optax
import jax
from ..base_models import SeqVae, Diffuser, LstmGate, Rnn
from equinox import nn
from jaxtyping import PRNGKeyArray


class ModelConfig(NamedTuple):
    vocab_size: int
    max_seq_len: int
    d_model: int
    vae_chunk_size: int
    ddpm_steps: int
    ddim_steps: int
    pad_token: int


class Model(TrainableModel):
    cfg: ModelConfig
    vae: SeqVae
    diffuser: Diffuser
    rnn_1: Rnn
    in_state_proj_1: nn.Linear
    in_state_proj_2: nn.Linear
    rnn_2: Rnn
    rnn_3: Rnn
    rnn_4: Rnn
    guide_proj_1: nn.Linear
    guide_proj_2: nn.Linear
    eps_proj: nn.Linear
    t_remap: jax.Array

    def __init__(self, cfg: ModelConfig, *, key):
        self.cfg = cfg

        keys = jax.random.split(key, 11)
        self.vae = SeqVae(
            cfg.d_model, cfg.vocab_size, cfg.vae_chunk_size, 8, 8, key=keys[0]
        )
        self.diffuser = Diffuser(cfg.ddpm_steps, key=keys[1])
        self.rnn_1 = Rnn(LstmGate(cfg.d_model, cfg.d_model, key=keys[2]))
        self.in_state_proj_1 = nn.Linear(cfg.d_model, cfg.d_model, key=keys[3])
        self.in_state_proj_2 = nn.Linear(cfg.d_model, cfg.d_model, key=keys[4])
        self.rnn_2 = Rnn(LstmGate(cfg.d_model * 2, cfg.d_model, key=keys[5]))
        self.rnn_3 = Rnn(LstmGate(cfg.d_model * 2, cfg.d_model, key=keys[6]))
        self.rnn_4 = Rnn(LstmGate(cfg.d_model * 2, cfg.d_model, key=keys[7]))
        self.guide_proj_1 = nn.Linear(cfg.d_model, cfg.d_model, key=keys[8])
        self.guide_proj_2 = nn.Linear(cfg.d_model, cfg.d_model, key=keys[9])
        self.eps_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, key=keys[10])

        max_step = cfg.ddim_steps - 1
        t_remap = [int((i / max_step) * cfg.ddpm_steps) for i in range(cfg.ddim_steps)]
        self.t_remap = jnp.asarray(t_remap)

    def pred_eps(self, input, noisy_input, *, key):
        key = RngKey(key)

        (_, in_state), _ = self.rnn_1.forward(input, key=key.next())
        in_state = jnp.repeat(
            self.in_state_proj_1(in_state)[None, :], noisy_input.shape[0], axis=0
        )
        (_, guide_state), _ = self.rnn_2.forward(
            jnp.concatenate([in_state, noisy_input], axis=-1), key=key.next()
        )
        guide_1 = jnp.repeat(
            self.guide_proj_1(guide_state)[None, :], input.shape[0], axis=0
        )
        guide_2 = self.guide_proj_2(guide_state)
        (_, in_state), _ = self.rnn_3.forward(
            jnp.concatenate([guide_1, input], axis=-1), key=key.next()
        )
        in_state = self.in_state_proj_2(in_state)
        guide = jnp.repeat((guide_2 + in_state)[None, :], noisy_input.shape[0], axis=0)
        _, eps = self.rnn_4.forward(
            jnp.concatenate([guide, noisy_input], axis=-1), key=key.next()
        )
        eps = jax.vmap(self.eps_proj)(eps)

        return eps

    def vae_encode(self, seq, pad_mask, key: PRNGKeyArray | None = None):
        keys = split_key(key, 1)

        seq_chunks = rearrange(seq, "(s c) -> s c", c=self.cfg.vae_chunk_size)
        pad_mask_chunks = rearrange(pad_mask, "(s c) -> s c", c=self.cfg.vae_chunk_size)

        vae_keys = split_key(keys[0], seq_chunks.shape[0])
        vae_mean, vae_logvar = jax.vmap(self.vae.encode)(
            seq_chunks, pad_mask_chunks, key=vae_keys
        )

        return vae_mean, vae_logvar

    def vae_encode_train(
        self, seq, pad_mask, pad_token, key: PRNGKeyArray | None = None
    ):
        keys = split_key(key, 3)

        vae_mean, vae_logvar = self.vae_encode(seq, pad_mask, keys[0])
        vae_keys = split_key(keys[1], vae_mean.shape[0])
        latents = jax.vmap(self.vae.reparameterize)(vae_mean, vae_logvar, key=vae_keys)
        vae_keys = split_key(keys[2], latents.shape[0])
        vae_reconstruction = jax.vmap(self.vae.decode)(latents, key=vae_keys)

        seq_chunks = rearrange(seq, "(s c) -> s c", c=self.cfg.vae_chunk_size)
        vae_loss = jax.vmap(partial(self.vae.loss, pad_token=pad_token))(
            vae_mean, vae_logvar, vae_reconstruction, seq_chunks
        ).mean()

        return vae_mean, vae_loss

    def forward(self, input: TextBatch, key, state: Any | None) -> TrainableModelOutput:
        key = RngKey(key)

        pad_token = self.cfg.pad_token
        in_latent_seq, in_vae_loss = self.vae_encode_train(
            input.inputs, input.inputs != pad_token, pad_token, key=key.next()
        )
        out_latent_seq, out_vae_loss = self.vae_encode_train(
            input.outputs, input.outputs != pad_token, pad_token, key=key.next()
        )

        t = jax.random.randint(key.next(), (1,), 0, self.cfg.ddpm_steps)
        x_t, eps = jax.vmap(lambda x, k: self.diffuser.forward(x, t, k))(
            out_latent_seq, key.next(out_latent_seq.shape[0])
        )
        eps_pred = self.pred_eps(in_latent_seq, x_t, key=key.next())

        out = (
            L2SequenceOutput.create(eps_pred, eps)
            .add_loss(in_vae_loss)
            .add_loss(out_vae_loss)
        )

        return out

    def forward_test(
        self, input: TextBatch, key, state: Any | None
    ) -> TrainableModelOutput:
        key = RngKey(key)

        latent_seq, _ = self.vae_encode(
            input.inputs, input.inputs != self.cfg.pad_token, key=key.next()
        )

        out_chunks = input.outputs.shape[-1] // self.cfg.vae_chunk_size
        seq = jax.random.normal(key.next(), (out_chunks, latent_seq.shape[-1]))

        # ToDo: use scan
        for t in reversed(range(1, self.cfg.ddim_steps)):
            eps = self.pred_eps(latent_seq, seq, key=key.next())
            ddim_t = self.t_remap[t]
            ddim_next_t = self.t_remap[t - 1]
            seq = jax.vmap(
                lambda x, e: self.diffuser.ddim_backward_step(x, ddim_t, ddim_next_t, e)
            )(seq, eps)

        seq = jax.vmap(self.vae.decode)(seq, key=key.next(seq.shape[0]))
        seq = rearrange(seq, "c s d -> (c s) d")

        return BCESequenceOutput.create(seq, input.outputs, self.cfg.pad_token)


def run(data: ExperimentData):
    from jax import config

    # config.update("jax_debug_nans", True)

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
    model = Model(cfg, key=model_key)
    optim = optax.adam(learning_rate)
    optim = optax.chain(optax.clip(1.0), optim)
    opt_state = optim.init(model)

    model = model.fit(rng, num_epochs, optim, opt_state, train_dl, test_dl)
