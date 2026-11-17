from .base import Entry, Memory


class OracleMemory(Memory):
    """Remembers which keys it was given and always returns their current truth.

    It is the floor for retrieval cost: reliable and never stale. The truth
    function is supplied by the caller, usually `world.truth`.
    """

    def __init__(self, truth):
        self._truth = truth
        self._known: set[int] = set()

    def read(self, key: int) -> Entry | None:
        if key not in self._known:
            return None
        return Entry(self._truth(key), None)

    def write(self, key: int, value: int, now: int) -> None:
        self._known.add(key)

    def forget(self, key: int) -> None:
        self._known.discard(key)
