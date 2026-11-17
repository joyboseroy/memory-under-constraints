"""A stream of queries about keys whose true answers can change over time.

Each key owns an independent schedule of changes, determined only by
(seed, key). The schedule does not depend on when or how often anything
looks at the key, so every policy sees the same world for a given seed.
"""

import math
import random


class _Key:
    """Current value of one key plus the time of its next change."""

    def __init__(self, seed: int, key: int, drift: float, n_values: int):
        self._rng = random.Random(f"{seed}:{key}")
        self._drift = drift
        self._n = n_values
        self.value = self._rng.randrange(n_values)
        self._next = self._gap()

    def _gap(self) -> float:
        d = self._drift
        if d <= 0.0:
            return math.inf
        if d >= 1.0:
            return 1
        # Geometric gap, at least 1 step.
        return 1 + int(math.log(1.0 - self._rng.random()) / math.log(1.0 - d))

    def at(self, t: int) -> int:
        # Times only move forward, so the current value is all we keep.
        while self._next <= t:
            self.value = (self.value + 1 + self._rng.randrange(self._n - 1)) % self._n
            self._next += self._gap()
        return self.value


class World:
    """Query stream plus ground truth.

    recurrence: probability a query repeats a recent key rather than a new one
    drift: per-step probability that a key's answer changes
    window: repeats are drawn uniformly from the last `window` new keys
    """

    def __init__(
        self,
        recurrence: float = 0.5,
        drift: float = 0.0,
        n_values: int = 1000,
        window: int = 32,
        seed: int = 0,
    ):
        if not 0.0 <= recurrence <= 1.0:
            raise ValueError("recurrence must be in [0, 1]")
        if n_values < 2:
            raise ValueError("n_values must be at least 2")
        self.recurrence = recurrence
        self.drift = drift
        self.n_values = n_values
        self.window = window
        self.seed = seed
        self.t = 0
        self._stream = random.Random(f"stream:{seed}")
        self._keys: dict[int, _Key] = {}
        self._n_keys = 0

    def next_query(self) -> int:
        self.t += 1
        if self._n_keys and self._stream.random() < self.recurrence:
            lo = max(0, self._n_keys - self.window)
            return self._stream.randrange(lo, self._n_keys)
        key = self._n_keys
        self._n_keys += 1
        return key

    def truth(self, key: int) -> int:
        state = self._keys.get(key)
        if state is None:
            state = self._keys[key] = _Key(self.seed, key, self.drift, self.n_values)
        return state.at(self.t)


def stationary(recurrence: float = 0.5, seed: int = 0, **kw) -> World:
    return World(recurrence=recurrence, drift=0.0, seed=seed, **kw)


def drifting(recurrence: float = 0.5, drift: float = 0.01, seed: int = 0, **kw) -> World:
    return World(recurrence=recurrence, drift=drift, seed=seed, **kw)
