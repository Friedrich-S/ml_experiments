import jax
from jaxtyping import PRNGKeyArray


class RngKey:
    def __init__(self, key: PRNGKeyArray | None):
        self.key = key

    def next(self, n: int = 1) -> PRNGKeyArray | None:
        assert n > 0, "n has to be greater than zero"

        if self.key is None:
            if n == 1:
                return None
            else:
                return [None] * n

        if n == 1:
            key, res = jax.random.split(self.key, n + 1)
            self.key = key
        else:
            keys = jax.random.split(self.key, n + 1)
            self.key = keys[0]
            res = keys[1:]
        return res

    def split(self):
        return RngKey(self.next())


def split_key(key: PRNGKeyArray | None, num: int) -> list[PRNGKeyArray | None]:
    return [None for _ in range(num)] if key is None else jax.random.split(key, num)
