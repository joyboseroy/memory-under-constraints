"""Regime grid over (recurrence, drift) for each policy, scored under several cost models."""

from ..cost import CostModel, Ledger
from ..environment import drifting
from ..memory import ExactCache
from ..policy import Recompute, Refresh, Retrieve, Verify
from ..sim import simulate

RECURRENCES = (0.2, 0.5, 0.9)
DRIFTS = (0.0, 0.002, 0.01, 0.05, 0.2)

COST_MODELS = {
    "default": CostModel(),
    "verify-dear": CostModel().replace(verify=40.0),
    "error-cheap": CostModel().replace(error=10.0),
}

LETTER = {"recompute": "C", "retrieve": "R", "verify": "V", "refresh": "F"}
TIE = 0.02


def make_policies():
    return [Recompute(), Retrieve(), Verify(), Refresh(ttl=20)]


def grid_ledgers(steps=2000, seeds=(0, 1, 2), recurrences=RECURRENCES, drifts=DRIFTS):
    """Run every policy in every cell. Returns {(rho, drift): {policy: (Ledger, steps)}}.

    Counts are kept, not costs, so the grid can be scored under any cost model.
    """
    grid = {}
    for rho in recurrences:
        for d in drifts:
            cell = {}
            for policy in make_policies():
                total, n = Ledger(), 0
                for seed in seeds:
                    world = drifting(recurrence=rho, drift=d, seed=seed)
                    memory = ExactCache(seed=seed)
                    result = simulate(policy, memory, world, steps)
                    total, n = total + result.ledger, n + result.steps
                cell[policy.name] = (total, n)
            grid[(rho, d)] = cell
    return grid


def scores(cell, model):
    return {name: model.total(ledger) / n for name, (ledger, n) in cell.items()}


def best(cell, model):
    """Cheapest policy and whether the runner-up is within TIE of it."""
    ranked = sorted(scores(cell, model).items(), key=lambda kv: kv[1])
    (name, top), (_, second) = ranked[0], ranked[1]
    return name, (second - top) <= TIE * top


def check_predictions(grid):
    """Evaluate the predictions written in hypotheses.md. Returns {label: (passed, detail)}."""

    def winner(rho, d, model):
        return best(grid[(rho, d)], model)[0]

    default, dear = COST_MODELS["default"], COST_MODELS["verify-dear"]
    rhos = sorted({k[0] for k in grid})
    out = {}

    def describe(d, want):
        bad = {}
        for r in rhos:
            name, tie = best(grid[(r, d)], default)
            if name != want:
                s = scores(grid[(r, d)], default)
                bad[r] = f"{name}{' (tie)' if tie else ''} {s[name]:.2f} vs {want} {s[want]:.2f}"
        return not bad, (str(bad) if bad else "all recurrences")

    out["R-P1 retrieve wins at drift 0"] = describe(0.0, "retrieve")
    out["R-P2 recompute wins at drift 0.2"] = describe(0.2, "recompute")

    got = {r: winner(r, 0.01, default) for r in (0.5, 0.9) if (r, 0.01) in grid}
    out["R-P3 verify wins at drift 0.01, recurrence 0.5 and 0.9"] = (all(v == "verify" for v in got.values()), str(got))

    wins = [k for k in grid if best(grid[k], default)[0] == "refresh"]
    out["R-P4 refresh wins nowhere under default costs"] = (not wins, f"refresh wins at {wins}" if wins else "none")

    wins = [k for k in grid if best(grid[k], dear)[0] == "refresh"]
    out["R-P5 refresh wins somewhere when verify is dear"] = (bool(wins), f"refresh wins at {wins}" if wins else "none")
    return out


def main(grid=None):
    grid = grid or grid_ledgers()
    for label, model in COST_MODELS.items():
        print(f"\ncost model: {label}  {model}")
        print(f"{'recurrence':>10} {'drift':>7}  " + "  ".join(f"{n:>9}" for n in ("recompute", "retrieve", "verify", "refresh")) + "  best")
        for (rho, d), cell in sorted(grid.items()):
            s = scores(cell, model)
            name, tie = best(cell, model)
            row = "  ".join(f"{s[n]:>9.2f}" for n in ("recompute", "retrieve", "verify", "refresh"))
            print(f"{rho:>10.1f} {d:>7.3f}  {row}  {name}{' (tie)' if tie else ''}")
    print("\npredictions from hypotheses.md")
    for label, (ok, detail) in check_predictions(grid).items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {detail}")
    return grid


if __name__ == "__main__":
    main()
