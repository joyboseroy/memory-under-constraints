"""The simulator: one policy, one memory, one world, one stream of queries."""

from dataclasses import dataclass

from .cost import CostModel, Ledger


class Context:
    """What a policy may do. Every call is counted in the ledger."""

    def __init__(self, memory, world, ledger: Ledger):
        self.memory = memory
        self.world = world
        self.ledger = ledger
        self.hits = 0

    @property
    def now(self) -> int:
        return self.world.t

    def read(self, key):
        self.ledger.reads += 1
        entry = self.memory.read(key)
        if entry is not None:
            self.hits += 1
        return entry

    def write(self, key, value):
        self.ledger.writes += 1
        self.memory.write(key, value, self.world.t)

    def forget(self, key):
        # Free in v0.1. A forgetting policy would need a real price for this.
        self.memory.forget(key)

    def compute(self, key):
        self.ledger.computes += 1
        return self.world.truth(key)

    def verify(self, key, value) -> bool:
        self.ledger.verifies += 1
        return value == self.world.truth(key)


@dataclass
class SimResult:
    ledger: Ledger
    steps: int
    hits: int

    def cost(self, model: CostModel) -> float:
        """Mean cost per query under the given cost model."""
        return model.total(self.ledger) / self.steps

    @property
    def error_rate(self) -> float:
        return self.ledger.errors / self.steps

    @property
    def hit_rate(self) -> float:
        return self.hits / self.steps


def simulate(policy, memory, world, steps: int) -> SimResult:
    ledger = Ledger()
    ctx = Context(memory, world, ledger)
    for _ in range(steps):
        key = world.next_query()
        if policy.answer(key, ctx) != world.truth(key):
            ledger.errors += 1
    return SimResult(ledger, steps, ctx.hits)
