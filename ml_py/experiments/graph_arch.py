from typing import Any, NamedTuple
from einops import rearrange
import jax.test_util
from ..common import *
import optax
import jax
from ..base_models import SeqVae, Diffuser, LstmGate, Rnn, RnnGate
from equinox import nn
from jaxtyping import PRNGKeyArray
from jax import nn as jnn
from jax.experimental import pallas as pl
import numpy as np

# ToDo: this is still unrealistic to run even with the custom kernel. Maybe
# implement this in Rust (Burn + custom WGPU kernels) to make it realistic to
# run.


def sparse_matmul_kernel(indices_ref, vecs_ref, mats_ref, o_ref):
    """
    Inputs:
        - indices_ref: (num_indices,)
        - vecs_ref: (num_elems, d_model)
        - mats_ref: (num_elems, d_model, d_model)
        - o_ref: (num_indices, d_model)
    """

    idx = pl.program_id(0)
    row_idx = pl.program_id(1)

    # Index of the target element in vecs/mats/out_vecs in [0, num_elems]
    elem_idx = indices_ref[idx]

    # (d_model,)
    vec = vecs_ref[elem_idx, :]
    # (d_model,)
    mat_row = mats_ref[elem_idx, row_idx, :]

    # scalar (0-D array)
    res = (vec * mat_row).sum()
    o_ref[idx, row_idx] = res


def sparse_matmul_kernel_bwd(
    indices_ref, vecs_ref, mats_ref, grad_ref, mats_grad_ref, in_grad_ref
):
    """
    Inputs:
        - indices_ref: (num_indices,)
        - vecs_ref: (num_elems, d_model)
        - mats_ref: (num_elems, d_model, d_model)
        - grad_ref: (num_indices, d_model)
        - mats_grad_ref: (num_indices, d_model, d_model)
        - in_grad_ref: (num_indices, d_model)
    """

    idx = pl.program_id(0)
    col_idx = pl.program_id(1)

    # Index of the target element in vecs/mats/out_vecs in [0, num_elems]
    elem_idx = indices_ref[idx]

    # (d_model,)
    vec = vecs_ref[elem_idx, :]
    # (d_model,)
    mat_row = mats_ref[elem_idx, col_idx, :]
    # (d_model,)
    grad = grad_ref[idx, :]

    # dL/dW = X^T @ dL/dY
    res = vec * grad[col_idx]
    mats_grad_ref[idx, col_idx, :] = res

    # scalar (0-D array)
    res = (grad * mat_row).sum()
    in_grad_ref[idx, col_idx] = res


@jax.custom_vjp
def sparse_matmul(indices: jax.Array, vecs: jax.Array, mats: jax.Array):
    out_shape = (indices.shape[0], vecs.shape[1])
    return pl.pallas_call(
        sparse_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, vecs.dtype),
        grid=out_shape,
    )(indices, vecs, mats)


def sparse_matmul_fwd(indices: jax.Array, vecs: jax.Array, mats: jax.Array):
    return sparse_matmul(indices, vecs, mats), (indices, vecs, mats)


def sparse_matmul_bwd(res, g: jax.Array):
    indices, vecs, mats = res

    out_shape = (indices.shape[0], vecs.shape[1])
    mats_grad, in_grad = pl.pallas_call(
        sparse_matmul_kernel_bwd,
        out_shape=(
            jax.ShapeDtypeStruct(out_shape + (vecs.shape[1],), vecs.dtype),
            jax.ShapeDtypeStruct(out_shape, vecs.dtype),
        ),
        grid=out_shape,
        interpret=True,
    )(indices, vecs, jnp.matrix_transpose(mats), g)

    full_mats_grad = jnp.zeros_like(mats).at[indices].set(mats_grad)
    full_in_grad = jnp.zeros_like(vecs).at[indices].set(in_grad)

    return jnp.zeros_like(indices), full_in_grad, full_mats_grad


sparse_matmul.defvjp(sparse_matmul_fwd, sparse_matmul_bwd)


def test_sparse_matmul():
    print("Testing sparse_matmul...")

    key = RngKey(jax.random.PRNGKey(0))
    indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 21, 15, 13, 14, 11, 16, 17])
    vecs = jax.random.normal(key.next(), (64, 32))
    mats = jax.random.normal(key.next(), (64, 32, 32))

    def loss(params, indices, fn):
        vecs, mats = params
        return optax.l2_loss(fn(indices, vecs, mats)).mean()

    target_fn = lambda i, v, m: jax.vmap(lambda v, m: m @ v)(v, m)[i, :]

    sparse_res, sparse_grad = jax.value_and_grad(partial(loss, fn=sparse_matmul))(
        (vecs, mats), indices
    )
    target_res, target_grad = jax.value_and_grad(partial(loss, fn=target_fn))(
        (vecs, mats), indices
    )

    np.testing.assert_allclose(sparse_res, target_res)
    jax.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b), sparse_grad, target_grad
    )


class ModelConfig(NamedTuple):
    vocab_size: int
    d_model: int
    n_nodes: int
    n_connections: int
    max_n_act: int
    pad_token: int


# ToDo: make connectivity sparse
class NodeGraph(eqx.Module, RnnGate):
    d_model: int
    n_nodes: int
    n_connections: int
    max_n_act: int
    state_decay_alpha: float
    in_idx: int
    out_idx: int
    nodes: jax.Array
    pred_nodes: jax.Array
    nodes_act: jax.Array
    pred_scale: jax.Array
    pred_offset: jax.Array
    connections: nn.StateIndex

    def __init__(
        self,
        d_model: int,
        n_nodes: int,
        n_connections: int,
        max_n_act: int,
        in_idx: int,
        out_idx: int,
        *,
        state_decay_alpha: float = 0.98,
        key,
    ):
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.n_connections = n_connections
        self.max_n_act = max_n_act
        self.state_decay_alpha = state_decay_alpha
        self.in_idx = in_idx
        self.out_idx = out_idx
        key = RngKey(key)

        self.nodes = jax.random.normal(key.next(), (n_nodes, d_model, d_model))
        self.pred_nodes = jax.random.normal(key.next(), (n_nodes, d_model, d_model))
        self.nodes_act = jax.random.normal(key.next(), (n_nodes, d_model, 1))
        self.pred_scale = jax.random.normal(key.next(), (n_nodes, d_model))
        self.pred_offset = jax.random.normal(key.next(), (n_nodes, d_model))

        connections = jax.random.normal(key.next(), (n_nodes, n_nodes))
        self.connections = nn.StateIndex(connections)

    # ToDo: this consumes way too much memory since XLA doesn't seem to try to
    # minimize the memory footprint of intermediate arrays. Maybe reimplement in
    # PyTorch (Rust would be faster for misc computations, but Python would make
    # the code more compact. Additionally PyTorch has more features compared to
    # LibTorch, especially things like JIT)
    def __call__(
        self, carry: Tuple[jax.Array, jax.Array, jax.Array], state: nn.State, key
    ):
        node_state, node_exhaustion, input_pred = carry

        # (n_nodes, 1)
        act_weight = jnn.sigmoid(
            jax.vmap(lambda s, w: s @ w)(node_state, self.nodes_act)
        )
        # (n_nodes,)
        act_weight = jnp.squeeze(act_weight, axis=-1)

        # (n_nodes,)
        act_mask = act_weight > jax.random.uniform(key, act_weight.shape)
        # (max_n_act,) in [0, n_nodes]
        act_indices = jnp.nonzero(act_mask, size=self.max_n_act, fill_value=-1)[0]
        null_act_indices = act_indices == -1
        act_indices = jnp.maximum(act_indices, 0)

        # ToDo: ^^^^ include node exhaustion

        # (max_n_act, d_model)
        act_vals = sparse_matmul(act_indices, node_state, self.nodes)
        # Handle null indices
        # act_vals = jnp.where(null_act_indices, 0, act_vals)

        # Clear the values of the nodes that just activated
        node_state = node_state.at[act_indices].set(0)

        def propagate_values(act_vals, null_act_indices, node_state):
            # (max_n_act, n_nodes)
            connections = state.get(self.connections)[act_indices]
            connections = jnn.softmax(connections, axis=-1)
            # ToDo: maybe use jax.lax.approx_max_k here for speedup, since the
            # precision top_k is not that relevant
            # (max_n_act, n_connections) in [0, n_connections]
            top_weights, top_indices = jax.lax.top_k(connections, self.n_connections)

            # (max_n_act, n_connections, d_model)
            x = jax.vmap(lambda w, v: w[:, None] * v[None, :])(top_weights, act_vals)
            null_x = jnp.zeros_like(x)
            null_idx = jnp.broadcast_to(null_act_indices[:, None, None], x.shape)
            # (max_n_act, n_connections, d_model)
            x = jax.lax.select(null_idx, null_x, x)

            # ToDo: validate this
            indices = rearrange(top_indices, "n m -> (n m)")
            x = rearrange(x, "n m d -> (n m) d")
            # ToDo: could maybe use jax.ops.segment_sum here?
            new_state = node_state.at[indices].add(x)

            return new_state

        # (n_nodes, d_model)
        prev_node_state = node_state
        node_state = propagate_values(act_vals, null_act_indices, node_state)
        node_inputs = node_state - prev_node_state

        pred_loss = optax.l2_loss(input_pred, node_inputs).mean()

        # (n_nodes, d_model)
        next_input_pred = jax.vmap(lambda x, w: x @ w)(
            # jax.lax.stop_gradient(node_inputs),
            # self.pred_nodes,
            node_inputs,
            self.pred_nodes,
        )

        node_state = node_state + (next_input_pred * self.pred_scale + self.pred_offset)

        # Apply value decay to prevent the state from becoming too large
        node_state = node_state * self.state_decay_alpha

        carry = node_state, node_exhaustion, next_input_pred

        return (carry, pred_loss), state

    def empty_state(self, *, key, state: nn.State | None):
        node_state = jnp.zeros((self.n_nodes, self.d_model))
        node_exhaustion = jnp.zeros((self.n_nodes,))
        input_pred = jnp.zeros((self.n_nodes, self.d_model))
        return (node_state, node_exhaustion, input_pred), state

    def forward(
        self,
        x: jax.Array | None,
        carry: Tuple[jax.Array, jax.Array, jax.Array],
        *,
        key,
        state: nn.State | None,
    ):
        node_state, node_exhaustion, input_pred = carry

        node_state = node_state.at[self.out_idx].add(x)

        carry = node_state, node_exhaustion, input_pred
        return self.__call__(carry, state, key=key)

    def state_output(
        self, carry: Tuple[jax.Array, jax.Array, jax.Array], state: nn.State | None
    ):
        node_state, node_exhaustion, input_pred = carry

        return node_state[self.out_idx], state

    @property
    def has_aux(self) -> bool:
        return True


class Model(TrainableModel):
    cfg: ModelConfig
    graph: Rnn

    def __init__(self, cfg: ModelConfig, *, key):
        self.cfg = cfg

        key = RngKey(key)
        self.graph = Rnn(
            NodeGraph(
                cfg.d_model,
                cfg.n_nodes,
                cfg.n_connections,
                cfg.max_n_act,
                0,
                1,
                key=key.next(),
            )
        )

    def forward(self, input: TextBatch, key, state: Any | None) -> TrainableModelOutput:
        key = RngKey(key)

        (carry, aux, _), state = self.graph.forward(
            input.inputs, key=key.next(), state=state
        )
        pred_loss = jnp.mean(aux)

        (_, aux, res), state = self.graph.forward(
            None,
            key=key.next(),
            state=state,
            initial_carry=carry,
            length=input.outputs.shape[0],
        )
        pred_loss = pred_loss + jnp.mean(aux)

        out = BCESequenceOutput.create(res, input.outputs, self.cfg.pad_token).add_loss(
            pred_loss
        )

        return out, state

    def forward_test(
        self, input: TextBatch, key, state: Any | None
    ) -> TrainableModelOutput:
        return self.forward(input, key, state)


def test_impl():
    print("Testing implementation...")
    test_sparse_matmul()


def run(data: ExperimentData):
    test_impl()

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
        d_model=128,
        n_nodes=256,
        n_connections=64,
        max_n_act=32,
        pad_token=tokenizer.pad_token(),
    )
    model, state = nn.make_with_state(Model)(cfg, key=model_key)
    optim = optax.adam(learning_rate)
    optim = optax.chain(optax.clip(1.0), optim)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    model = model.fit(rng, num_epochs, optim, opt_state, train_dl, test_dl, state=state)
