"""Memory interface.

A memory stores and returns values. It knows nothing about costs: the
simulator counts reads and writes around these calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    value: int
    written_at: int | None = None  # None when the mechanism cannot tell


class Memory(ABC):
    @abstractmethod
    def read(self, key: int) -> Entry | None:
        """Return an entry on a hit, None on a miss. May be wrong or stale."""

    @abstractmethod
    def write(self, key: int, value: int, now: int) -> None:
        ...

    @abstractmethod
    def forget(self, key: int) -> None:
        ...
