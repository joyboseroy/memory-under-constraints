"""Stationary analytic model of the retrieve and recompute policies.

This is the control experiment. With recurrence rho (probability a query hits a
key already stored), retrieval reliability q (probability a hit is correct) and
a cost model, the expected cost per query is:

    recompute:  compute
    retrieve:   read + rho (1 - q) error + (1 - rho)(compute + write)

It assumes an unbounded memory, a window of recent keys that fits in memory,
and a world that does not change. Anything it fails to predict is something
the simulator is showing beyond arithmetic.
"""

from .cost import CostModel


def recompute_cost(model: CostModel) -> float:
    return model.compute


def retrieve_cost(recurrence: float, q: float, model: CostModel) -> float:
    return (
        model.read
        + recurrence * (1.0 - q) * model.error
        + (1.0 - recurrence) * (model.compute + model.write)
    )


def retrieve_variance(recurrence: float, q: float, model: CostModel) -> float:
    """Variance of the cost of one query under the retrieve policy.

    A miss costs read + compute + write. A hit costs read plus the error cost
    with probability 1 - q. Rare costly errors make the mean slow to settle,
    so a fixed percentage tolerance is the wrong yardstick for a simulation check.
    """
    p_err = 1.0 - q
    miss = model.read + model.compute + model.write
    mean = retrieve_cost(recurrence, q, model)
    second = (1.0 - recurrence) * miss**2 + recurrence * (
        model.read**2 + 2.0 * model.read * model.error * p_err + model.error**2 * p_err
    )
    return second - mean**2


def retrieval_wins(recurrence: float, q: float, model: CostModel) -> bool:
    return retrieve_cost(recurrence, q, model) < recompute_cost(model)
