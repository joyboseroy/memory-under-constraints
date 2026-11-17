"""Equal-budget random search. Every (model, tier) gets the same number of trials, the
same tuning seeds, the same objective, on automata disjoint from evaluation."""
from __future__ import annotations

import numpy as np

from .harness import evaluate
from .tasks import TaskConfig


def sample_space(space: dict, rng) -> dict:
    out = {}
    for k, s in space.items():
        t = s["type"]
        if t == "choice":
            out[k] = s["values"][int(rng.integers(len(s["values"])))]
        elif t == "uniform":
            out[k] = float(rng.uniform(s["low"], s["high"]))
        elif t == "loguniform":
            out[k] = float(np.exp(rng.uniform(np.log(s["low"]), np.log(s["high"]))))
        elif t == "intuniform":
            out[k] = int(rng.integers(s["low"], s["high"] + 1))
        else:
            raise ValueError(f"unknown space type {t}")
    return out


def random_search(base_spec: dict, space: dict, task_cfg: TaskConfig, n_trials: int,
                  tune_seeds, search_seed: int = 0, log=print):
    """Minimise mean KL(P||Q) over the whole drifting stream (no probes)."""
    rng = np.random.default_rng([search_seed, 7])
    best, best_score, trials = {}, float("inf"), []
    for i in range(n_trials):
        hp = sample_space(space, rng)
        spec = {**base_spec, **hp}
        rows = evaluate(spec, task_cfg, tune_seeds, do_probes=False)
        score = float(np.mean([r["kl_mean"] for r in rows]))
        trials.append({"trial": i, "hp": hp, "score": score})
        log(f"    trial {i + 1}/{n_trials} kl={score:.4f} {hp}")
        if score < best_score:
            best, best_score = hp, score
    return best, best_score, trials
