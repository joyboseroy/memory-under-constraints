"""Decision policies: what to do when a query arrives.

A policy gets a key and a Context and returns an answer. The context counts
every operation, so a policy cannot spend anything it does not report.
"""

from abc import ABC, abstractmethod


class Policy(ABC):
    name = "policy"

    @abstractmethod
    def answer(self, key: int, ctx) -> int:
        ...


class Recompute(Policy):
    """Always compute. Never touches memory."""

    name = "recompute"

    def answer(self, key, ctx):
        return ctx.compute(key)


class Retrieve(Policy):
    """Read memory. On a miss compute and write; on a hit trust the value."""

    name = "retrieve"

    def answer(self, key, ctx):
        entry = ctx.read(key)
        if entry is not None:
            return entry.value
        value = ctx.compute(key)
        ctx.write(key, value)
        return value


class Verify(Policy):
    """Read memory and pay to check a hit. A stale or wrong hit is recomputed and rewritten."""

    name = "verify"

    def answer(self, key, ctx):
        entry = ctx.read(key)
        if entry is not None and ctx.verify(key, entry.value):
            return entry.value
        value = ctx.compute(key)
        ctx.write(key, value)
        return value


class Refresh(Policy):
    """Read memory. A hit older than `ttl` steps is recomputed and rewritten.

    An entry whose age is unknown (written_at is None) is treated as expired.
    """

    def __init__(self, ttl: int = 20):
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
        self.ttl = ttl
        self.name = "refresh"

    def answer(self, key, ctx):
        entry = ctx.read(key)
        if entry is not None and entry.written_at is not None and ctx.now - entry.written_at <= self.ttl:
            return entry.value
        value = ctx.compute(key)
        ctx.write(key, value)
        return value
