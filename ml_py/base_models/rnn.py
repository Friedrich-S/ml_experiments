from typing import Any
import equinox as eqx
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
from abc import ABC, abstractmethod
from jax.typing import ArrayLike
import jax
from .misc import LowRankLinear
from ..common.utils import RngKey


class RnnGate(ABC):
    @abstractmethod
    def empty_state(self, *, key, state: nn.State | None):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: jax.Array, carry, *, key, state: nn.State | None):
        raise NotImplementedError()

    @abstractmethod
    def state_output(self, carry, state: nn.State | None) -> jax.Array:
        raise NotImplementedError()

    @property
    @abstractmethod
    def has_aux(self) -> bool:
        return False


class Rnn(eqx.Module):
    gate: RnnGate

    def forward(
        self,
        x: jax.Array | None,
        *,
        key,
        state: nn.State | None = None,
        initial_carry: Any | None = None,
        length: int | None = None
    ):
        if x is None:
            assert length is not None, "if x is None, then length must be specified"

        key = RngKey(key)

        if state is None:
            apply_fn = lambda x, carry, key, state: (
                self.gate.forward(x, carry, key=key, state=state),
                None,
            )
            out_fn = lambda carry, state: (
                self.gate.state_output(carry, state=None),
                state,
            )
            carry = (
                (self.gate.empty_state(key=key.next(), state=None), state)
                if initial_carry is None
                else (initial_carry, state)
            )
        else:
            apply_fn = lambda x, carry, key, state: self.gate.forward(
                x, carry, key=key, state=state
            )
            out_fn = lambda carry, state: self.gate.state_output(carry, state=state)
            carry = (
                self.gate.empty_state(key=key.next(), state=state)
                if initial_carry is None
                else (initial_carry, state)
            )

        if self.gate.has_aux:

            def scan_fn(carry, x):
                carry, state = carry
                x, key = x
                (carry, aux), state = apply_fn(x, carry, key, state)
                (out, state) = out_fn(carry, state)
                return (carry, state), (out, aux)

        else:

            def scan_fn(carry, x):
                carry, state = carry
                x, key = x
                carry, state = apply_fn(x, carry, key, state)
                (out, state) = out_fn(carry, state)
                return (carry, state), out

        keys = key.next(x.shape[0]) if length is None else key.next(length)
        if self.gate.has_aux:
            (carry, state), (res, aux) = jax.lax.scan(
                scan_fn, carry, (x, keys), length=length
            )
            ret = carry, aux, res
        else:
            (carry, state), res = jax.lax.scan(scan_fn, carry, (x, keys), length=length)
            ret = carry, res

        if state is not None:
            return ret, state
        else:
            return ret


class LstmGate(eqx.Module, RnnGate):
    d_in: int
    d_hidden: int
    long_gate: LowRankLinear
    diff_proj: nn.Linear
    diff_gate: LowRankLinear
    short_proj: nn.Linear
    short_gate: LowRankLinear

    def __init__(self, d_in: int, d_hidden: int, *, key, rank: int | None = None):
        self.d_in = d_in
        self.d_hidden = d_hidden

        keys = jax.random.split(key, 5)
        self.long_gate = LowRankLinear(
            d_in * 2, d_hidden, (d_hidden // 8) if rank is None else rank, key=keys[0]
        )
        self.diff_proj = nn.Linear(d_in * 2, d_hidden, key=keys[1])
        self.diff_gate = LowRankLinear(
            d_in * 2, d_hidden, (d_hidden // 8) if rank is None else rank, key=keys[2]
        )
        self.short_proj = nn.Linear(d_hidden, d_in, key=keys[3])
        self.short_gate = LowRankLinear(
            d_in * 2, d_in, (d_in // 8) if rank is None else rank, key=keys[4]
        )

    def empty_state(self, *, key, state: nn.State | None):
        short_state = jnp.zeros((self.d_in,))
        long_state = jnp.zeros((self.d_hidden,))
        return short_state, long_state

    def forward(self, x, carry, *, key, state: nn.State | None):
        short_state, long_state = carry

        x = jnp.concatenate([short_state, x], axis=-1)
        long_state = long_state * jnn.sigmoid(self.long_gate(x))
        diff = jnn.tanh(self.diff_proj(x)) * jnn.sigmoid(self.diff_gate(x))
        long_state = long_state + diff
        short_state = jnn.tanh(self.short_proj(long_state)) * jnn.sigmoid(
            self.short_gate(x)
        )

        return short_state, long_state

    def state_output(self, carry, state: nn.State | None):
        short_state, long_state = carry
        return short_state
