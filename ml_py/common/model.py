from abc import ABC, abstractmethod
import collections
from functools import partial
from typing import Any, NamedTuple
import equinox as eqx
from jax import numpy as jnp
from jax.typing import ArrayLike
import optax
from torch.utils.data import DataLoader
import jax
from tqdm import tqdm


class ModelInput(ABC):
    @staticmethod
    @abstractmethod
    def from_dict(dict: dict):
        raise NotImplementedError()


class __TextBatchData(NamedTuple):
    inputs: ArrayLike
    outputs: ArrayLike


class TextBatch(__TextBatchData, ModelInput):
    @staticmethod
    def from_dict(dict: dict):
        return TextBatch(dict["in_ids"], dict["out_ids"])


class ModelOutput(ABC):
    @abstractmethod
    def get_output(self):
        raise NotImplementedError()

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError()


class OutputAccuracyMetric(ABC):
    @abstractmethod
    def get_accuracy(self) -> ArrayLike:
        raise NotImplementedError()


class __BCESequenceOutputData(NamedTuple):
    loss: ArrayLike
    output: ArrayLike
    targets: ArrayLike
    pad_token: int


class BCESequenceOutput(__BCESequenceOutputData, ModelOutput, OutputAccuracyMetric):
    @staticmethod
    def create(logits: ArrayLike, labels: ArrayLike, pad_token: int):
        normal_loss = lambda: optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean(where=labels != pad_token)
        fallback_loss = lambda: jnp.array(0.0)

        loss = jax.lax.cond(jnp.all(labels == pad_token), fallback_loss, normal_loss)
        return BCESequenceOutput(loss, logits, labels, pad_token)

    def add_loss(self, loss: ArrayLike):
        return BCESequenceOutput(
            self.loss + loss, self.output, self.targets, self.pad_token
        )

    def get_output(self):
        return self.output

    def get_loss(self):
        return self.loss

    def get_accuracy(self) -> ArrayLike:
        @partial(jnp.vectorize, signature="(c),()->()")
        def compute_accuracy(logits, label):
            return jnp.argmax(logits) == label

        acc = compute_accuracy(self.output, self.targets)
        return jnp.mean(acc, where=self.targets != self.pad_token[:, None])


class __L2SequenceOutputData(NamedTuple):
    loss: ArrayLike
    output: ArrayLike
    targets: ArrayLike


class L2SequenceOutput(__L2SequenceOutputData, ModelOutput):
    @staticmethod
    def create(outputs: ArrayLike, targets: ArrayLike):
        loss = optax.l2_loss(outputs, targets).mean()
        return L2SequenceOutput(loss, outputs, targets)

    def add_loss(self, loss: ArrayLike):
        return L2SequenceOutput(self.loss + loss, self.output, self.targets)

    def get_output(self):
        return self.output

    def get_loss(self):
        return self.loss


TrainableModelOutput = ModelOutput | tuple[ModelOutput]


class TrainableModel(eqx.Module, ABC):
    auto_batch: bool = True

    @abstractmethod
    def forward(self, input, key, state: Any | None) -> TrainableModelOutput:
        raise NotImplementedError()

    @abstractmethod
    def forward_test(self, input, key, state: Any | None) -> TrainableModelOutput:
        raise NotImplementedError()

    def num_parameters(self) -> int:
        tree = eqx.filter(self, eqx.is_array)
        return sum((x.size if x is not None else 0) for x in jax.tree_leaves(tree))

    def make_step_funcs(self, optim):
        return self.make_train_step(optim), self.make_valid_step()

    def make_train_step(self, optim):
        @partial(eqx.filter_value_and_grad, has_aux=True)
        def compute_loss(model: TrainableModel, item, key, state: Any | None):
            forward = (
                jax.vmap(partial(model.forward, state=state))
                if self.auto_batch
                else model.forward
            )
            if self.auto_batch:
                batch_size = jax.tree_leaves(
                    jax.tree_map(lambda x: x.shape[0], eqx.filter(item, eqx.is_array))
                )
                batch_size = set(batch_size)
                assert len(batch_size) == 1
                key = jax.random.split(key, batch_size.pop())

            loss_proj = (lambda l: l.mean()) if self.auto_batch else (lambda l: l)
            output = forward(item, key)
            if state is not None:
                output, new_state = output
                return loss_proj(output.get_loss()), new_state
            else:
                return loss_proj(output.get_loss()), None

        @eqx.filter_jit
        def step_fn(
            model: TrainableModel, item, key, opt_state, state: Any | None = None
        ):
            (loss, new_state), grads = compute_loss(model, item, key, state)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state, new_state

        return step_fn

    def make_valid_step(self):
        @eqx.filter_jit
        def step_fn(model: TrainableModel, item, key, state: Any | None = None):
            forward_test = (
                jax.vmap(partial(model.forward_test, state=state))
                if self.auto_batch
                else model.forward_test
            )
            if self.auto_batch:
                batch_size = jax.tree_leaves(
                    jax.tree_map(lambda x: x.shape[0], eqx.filter(item, eqx.is_array))
                )
                batch_size = set(batch_size)
                assert len(batch_size) == 1
                key = jax.random.split(key, batch_size.pop())
            output = forward_test(item, key)
            if state is not None:
                output, _ = output

            return output

        return step_fn

    def fit(
        self,
        key: jax.random.PRNGKey,
        epochs: int,
        optim,
        optim_state: optax.OptState,
        train_dl: DataLoader,
        test_dl: DataLoader,
        state: Any | None = None,
    ):
        model = self

        print("JAX running on", jax.devices()[0].platform.upper())
        print(f"[*] Number of trainable parameters: {self.num_parameters()}")

        train_step, test_step = self.make_step_funcs(optim)
        train_rng, test_rng = jax.random.split(key)

        for epoch in range(epochs):
            print(f"[*] Starting Training Epoch {epoch + 1}...")

            pbar = tqdm(train_dl)
            batch_losses = []
            losses = collections.deque(maxlen=10)
            for batch in pbar:
                train_rng, rng = jax.random.split(train_rng, 2)
                loss, model, optim_state, state = train_step(
                    model, batch, rng, optim_state, state=state
                )
                batch_losses.append(loss)
                losses.append(loss)
                pbar.set_description(f"loss={jnp.mean(jnp.array(losses)):.4f}")

            train_loss = jnp.mean(jnp.array(batch_losses))

            print(f"[*] Running Epoch {epoch + 1} Test...")

            infer_model = eqx.nn.inference_mode(model)
            val_losses, val_accs = [], []
            for batch in tqdm(test_dl):
                test_rng, rng = jax.random.split(test_rng, 2)
                output = test_step(infer_model, batch, rng, state=state)
                val_losses.append(output.get_loss())
                if isinstance(output, OutputAccuracyMetric):
                    val_accs.append(output.get_accuracy())
                else:
                    val_accs.append(0)

            val_loss = jnp.mean(jnp.array(val_losses))
            val_acc = jnp.mean(jnp.array(val_accs))

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f}\n"
                f"\tTest Accuracy: {val_acc:.4f}"
            )

        return model
