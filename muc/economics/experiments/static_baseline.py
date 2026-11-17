"""R-H1: does the simulator reproduce the stationary analytic model?

Agreement is judged by a z-score against the analytic standard error of the
mean, plus whether the cheaper policy comes out the same. A fixed percentage
tolerance was tried first and rejected: with an error cost of 100 and an error
rate of a few percent, the mean cost of 12000 queries moves by several percent
from sampling noise alone.
"""

from math import sqrt

from ..analytic import recompute_cost, retrieve_cost, retrieve_variance
from ..cost import CostModel
from ..environment import stationary
from ..memory import ExactCache
from ..policy import Recompute, Retrieve
from ..sim import simulate

Z_LIMIT = 3.0
DECIDED = 0.05  # boundary is only checked where the predicted gap exceeds 5 percent of recompute cost


def _mean_cost(policy_factory, steps, seeds, recurrence, noise, model):
    total = 0.0
    for seed in seeds:
        world = stationary(recurrence=recurrence, seed=seed)
        memory = ExactCache(noise=noise, seed=seed)
        total += simulate(policy_factory(), memory, world, steps).cost(model)
    return total / len(seeds)


def run(steps=4000, seeds=(0, 1, 2), recurrence=0.6, computes=(2, 5, 20, 100), noises=(0.0, 0.1, 0.3, 0.6)):
    rows = []
    n = steps * len(seeds)
    for compute in computes:
        model = CostModel().replace(compute=float(compute))
        for noise in noises:
            q = 1.0 - noise
            sim_ret = _mean_cost(Retrieve, steps, seeds, recurrence, noise, model)
            sim_rec = _mean_cost(Recompute, steps, seeds, recurrence, noise, model)
            pred_ret = retrieve_cost(recurrence, q, model)
            pred_rec = recompute_cost(model)
            se = sqrt(retrieve_variance(recurrence, q, model) / n)
            z = (sim_ret - pred_ret) / se
            gap = pred_ret - pred_rec
            decided = abs(gap) > DECIDED * pred_rec
            sign_ok = ((sim_ret - sim_rec) < 0) == (gap < 0) if decided else None
            rows.append(
                dict(
                    compute=compute, noise=noise,
                    sim_retrieve=sim_ret, pred_retrieve=pred_ret,
                    sim_recompute=sim_rec, pred_recompute=pred_rec,
                    rel_err=abs(sim_ret - pred_ret) / pred_ret, z=z, sign_ok=sign_ok,
                )
            )
    return rows


def passes(rows) -> bool:
    return all(abs(r["z"]) <= Z_LIMIT for r in rows) and not any(r["sign_ok"] is False for r in rows)


def main():
    rows = run()
    print("R-H1: simulated vs analytic cost per query (retrieve policy, recurrence 0.6)")
    print(f"{'compute':>8} {'noise':>6} {'sim':>8} {'pred':>8} {'recompute':>10} {'rel.err':>8} {'z':>6}  boundary")
    for r in rows:
        mark = {True: "agrees", False: "DISAGREES", None: "too close to call"}[r["sign_ok"]]
        print(
            f"{r['compute']:>8} {r['noise']:>6.2f} {r['sim_retrieve']:>8.2f} {r['pred_retrieve']:>8.2f} "
            f"{r['pred_recompute']:>10.2f} {r['rel_err']:>8.3f} {r['z']:>6.2f}  {mark}"
        )
    worst = max(abs(r["z"]) for r in rows)
    bad = sum(1 for r in rows if r["sign_ok"] is False)
    print(f"largest |z| {worst:.2f} (limit {Z_LIMIT}), boundary disagreements {bad}: R-H1 {'holds' if passes(rows) else 'FAILS'}")
    return rows


if __name__ == "__main__":
    main()
