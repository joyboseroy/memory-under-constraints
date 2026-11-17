"""ASCII regime maps: cheapest policy per (recurrence, drift) cell, per cost model.

C recompute, R retrieve, V verify, F refresh. A star marks a near tie.
"""

from .drift import COST_MODELS, DRIFTS, LETTER, RECURRENCES, best, grid_ledgers


def render(grid, model):
    lines = ["recurrence \\ drift  " + "  ".join(f"{d:>6.3f}" for d in DRIFTS)]
    for rho in sorted(RECURRENCES, reverse=True):
        cells = []
        for d in DRIFTS:
            name, tie = best(grid[(rho, d)], model)
            cells.append(f"{LETTER[name] + ('*' if tie else ' '):>6} ")
        lines.append(f"{rho:>18.1f}  " + " ".join(c.rstrip().rjust(6) for c in cells))
    return "\n".join(lines)


def disagreements(grid, model_a, model_b):
    """Cells where the cheapest policy changes between two cost models."""
    return [k for k in grid if best(grid[k], model_a)[0] != best(grid[k], model_b)[0]]


def main(grid=None):
    grid = grid or grid_ledgers()
    names = list(COST_MODELS)
    for label in names:
        print(f"\ncost model: {label}")
        print(render(grid, COST_MODELS[label]))
    base = COST_MODELS[names[0]]
    print()
    for label in names[1:]:
        moved = disagreements(grid, base, COST_MODELS[label])
        print(f"{label}: best policy differs from default in {len(moved)} of {len(grid)} cells")
    return grid


if __name__ == "__main__":
    main()
