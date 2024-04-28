from einops import rearrange
from ..common import *
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    ByT5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import CausalLMOutput
import torch
from torch import nn, Tensor
import triton
import triton.language as tl

@triton.jit
def sparse_matmul_kernel(indices, vecs, mats, out_vecs, D_MODEL: tl.constexpr):
    """
    Inputs:
        - indices: (num_indices,)
        - vecs: (num_elems, d_model)
        - mats: (num_elems, d_model, d_model)
        - out_vecs: (num_elems, d_model)
    """

    idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_offsets = tl.arange(0, D_MODEL)

    # Index of the target element in vecs/mats/out_vecs in [0, num_elems]
    elem_idx = tl.load(indices + idx)
    
    # Load the entire vector from `idx`
    vec_start = vecs + elem_idx * D_MODEL
    vec = tl.load(vec_start + col_offsets)
    
    # Load the row at `row_idx` from the corresponding matrix at `idx`
    mat_stride = D_MODEL * D_MODEL
    mat_row_start = mats + elem_idx * mat_stride + row_idx * D_MODEL
    mat_row = tl.load(mat_row_start + col_offsets)

    res = tl.dot(vec, mat_row)
    out_vec_start = out_vecs + elem_idx * D_MODEL
    tl.store(out_vec_start + row_idx, res)


class ModelConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_nodes: int,
        n_connections: int,
        max_n_act: int,
        pad_token: int,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.n_connections = n_connections
        self.max_n_act = max_n_act
        self.pad_token = pad_token
        super().__init__(**kwargs)


class NodeGraph(nn.Module):
    def __init__(self, cfg: ModelConfig):
        self.nodes = nn.Parameter(
            torch.empty(cfg.n_nodes, cfg.d_model, cfg.d_model).normal_()
        )
        self.pred_nodes = nn.Parameter(
            torch.empty(cfg.n_nodes, cfg.d_model, cfg.d_model).normal_()
        )

        self.nodes_act_weight = nn.Parameter(
            torch.empty(cfg.n_nodes, cfg.d_model, 1).normal_()
        )
        self.pred_scale = nn.Parameter(torch.empty(cfg.n_nodes, cfg.d_model).normal_())
        self.pred_offset = nn.Parameter(torch.empty(cfg.n_nodes, cfg.d_model).normal_())
        self.connections = nn.Parameter(
            torch.empty(cfg.n_nodes, cfg.n_nodes).normal_(), requires_grad=False
        )

    def forward(
        self, state: tuple[Tensor, Tensor, Tensor], input_ids: Tensor, labels: Tensor
    ):
        node_state, node_exhaustion, input_pred = state

        act_weight = torch.sigmoid(node_state @ self.nodes_act_weight)
        act_weight = act_weight.squeeze(-1)

        act_mask = act_weight > torch.empty_like(act_weight).uniform_()

        # ToDo: Group the act mask along the batch axes to the closest 2^n of True
        # values => 1, 2, 4, 8, 16, .. up to batch size

        # ToDo
        logits = None
        loss = None

        return CausalLMOutput(loss=loss, logits=logits)


# ToDo: each node tries to predict the next input. The incoming connections are
# weighted by the accuracy of the prediction model for the respective input
# values. This ensures that each node knows its own confidence and receives
# inputs primarily from the nodes it has the highest confidence in. This should
# naturally result in clustering. Third, each node's output projection is
# trained to minimize the loss of the input prediction parts of all nodes that
# this node has outgoing connections to, weighted by the connection weight. This
# ensures that the output projection of the node eventually ends up at being
# something useful.
#
# ToDo: the input prediction loss should have a tradeoff between maximizing
# prediction accuracy and minimizing the loss of the output projection.
#
# ToDo: for training, the output node would be treated as an identity projection
# and the prediction target of its next input prediction model would be
# overwritten to not be the actual next input, but instead the desired output of
# the entire model. This would force the inputs into that node to adjust their
# output projection, such that the input prediction of the output can more
# closely match the desired output. This should invoke a chain reaction and
# cause the entire model to train.
class Model(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.graph = NodeGraph(cfg)

    def forward(self, input_ids: Tensor, labels: Tensor):
        # ToDo
        logits = None
        loss = None

        return CausalLMOutput(loss=loss, logits=logits)


def run(data: ExperimentData):
    max_seq_len = 128
    seed = 1234

    torch.manual_seed(seed)

    tokenizer = ByT5Tokenizer(extra_ids=0)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    enc_cfg = EncodeConfig(max_length=max_seq_len, pad=True)
    ds = SimpleCsvDataset.load("./datasets/def_words/main.csv").shuffled(seed)
    ds = ds.subset(50_000).rename_columns({"text": "input_ids", "target": "labels"})
    ds = ds.tokenize(tokenizer, enc_cfg).get().train_test_split(0.1)

    cfg = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=256,
        n_nodes=2048,
        n_connections=128,
        max_n_act=128,
        pad_token=tokenizer.pad_token_id,
    )
    model = Model(cfg)

    train_args = TrainingArguments(
        output_dir=str(data.subdir("training").resolve()),
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.001,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
