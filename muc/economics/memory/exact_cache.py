import random
from collections import OrderedDict

from .base import Entry, Memory


class ExactCache(Memory):
    """Exact-key cache with optional LRU capacity and read noise.

    capacity: maximum number of entries (None for unbounded); least recently
        read or written entry is evicted first
    noise: probability that a hit returns a wrong value
    """

    def __init__(self, capacity: int | None = None, noise: float = 0.0, n_values: int = 1000, seed: int = 0):
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.noise = noise
        self.n_values = n_values
        self._rng = random.Random(f"cache:{seed}")
        self._data: OrderedDict[int, tuple[int, int]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def read(self, key: int) -> Entry | None:
        item = self._data.get(key)
        if item is None:
            return None
        self._data.move_to_end(key)
        value, written_at = item
        if self.noise > 0.0 and self._rng.random() < self.noise:
            value = (value + 1 + self._rng.randrange(self.n_values - 1)) % self.n_values
        return Entry(value, written_at)

    def write(self, key: int, value: int, now: int) -> None:
        self._data[key] = (value, now)
        self._data.move_to_end(key)
        if self.capacity is not None and len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def forget(self, key: int) -> None:
        self._data.pop(key, None)
