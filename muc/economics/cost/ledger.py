"""Operation counts and the cost models that turn them into numbers.

The ledger stores counts, not costs. A CostModel assigns a weight to each kind
of operation, so one simulation can be scored under many models afterwards.
"""

from dataclasses import dataclass, fields


@dataclass
class Ledger:
    computes: int = 0
    reads: int = 0
    writes: int = 0
    verifies: int = 0
    errors: int = 0

    def __add__(self, other: "Ledger") -> "Ledger":
        return Ledger(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))


@dataclass(frozen=True)
class CostModel:
    compute: float = 50.0
    read: float = 1.0
    write: float = 1.0
    verify: float = 5.0
    error: float = 100.0

    def total(self, ledger: Ledger) -> float:
        return (
            self.compute * ledger.computes
            + self.read * ledger.reads
            + self.write * ledger.writes
            + self.verify * ledger.verifies
            + self.error * ledger.errors
        )

    def replace(self, **changes) -> "CostModel":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(changes)
        return CostModel(**values)
