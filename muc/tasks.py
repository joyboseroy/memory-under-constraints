"""Task generator: variable-order sources with a computable true conditional distribution.

The source is a context-tree (variable-order Markov) process over `n_main` symbols.
For a history, the next-symbol distribution is given by the longest stored suffix rule.
Optional corruptions, all included in the *exact* conditional distribution:

* noise: with prob `noise` the main symbol is replaced by a uniform draw
* distractors: with prob `distract_prob` a token from a separate alphabet is emitted;
  distractors never enter the rule context, so a good model must ignore them

Drift is a schedule of regimes, e.g. phases=(0, 1, 0). Regime 0 is the base rule set.
Regime i >= 1 resamples a fraction `drift_frac` of the rules. Returning to regime 0
restores the exact original rules, so the third phase tests re-learning.

Every stream carries probs[t] = P(x_t | x_<t), the true distribution that x_t was
sampled from. Log-loss, KL, calibration and adaptation metrics all derive from it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    n_main: int = 8
    n_distract: int = 4
    max_order: int = 4
    n_rules: int = 24  # rules of order >= 2 added on top of orders 0 and 1
    concentration: float = 0.25  # Dirichlet alpha; small = peaky = predictable
    noise: float = 0.05
    distract_prob: float = 0.0
    phases: tuple = (0, 1, 0)
    phase_len: int = 2000
    drift_frac: float = 0.5

    @property
    def vocab(self) -> int:
        return self.n_main + self.n_distract

    def replace(self, **kw) -> "TaskConfig":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["phases"] = list(self.phases)
        return d

    @staticmethod
    def from_dict(d: dict) -> "TaskConfig":
        d = dict(d)
        if "phases" in d:
            d["phases"] = tuple(d["phases"])
        return TaskConfig(**d)


class Automaton:
    """A context-tree source. Rules map a tuple of the last k main tokens to a
    distribution over main tokens. The empty tuple is the order-0 rule."""

    def __init__(self, rules: dict, cfg: TaskConfig):
        self.rules = rules
        self.cfg = cfg

    def main_dist(self, hist_main) -> np.ndarray:
        h = list(hist_main)
        for k in range(min(self.cfg.max_order, len(h)), -1, -1):
            key = tuple(h[len(h) - k:]) if k > 0 else ()
            r = self.rules.get(key)
            if r is not None:
                return r
        raise KeyError("order-0 rule missing")  # unreachable: () always exists

    def full_dist(self, hist_main) -> np.ndarray:
        c = self.cfg
        pm = self.main_dist(hist_main)
        pm = (1.0 - c.noise) * pm + c.noise / c.n_main
        out = np.zeros(c.vocab)
        out[: c.n_main] = (1.0 - c.distract_prob) * pm
        if c.n_distract > 0 and c.distract_prob > 0:
            out[c.n_main:] = c.distract_prob / c.n_distract
        elif c.distract_prob == 0:
            pass
        return out / out.sum()


def _dirichlet(rng, n, alpha):
    p = rng.dirichlet(np.full(n, alpha))
    return 0.98 * p + 0.02 / n  # floor keeps the true KL finite and comparable


def build_base_rules(cfg: TaskConfig, rng) -> dict:
    rules = {(): _dirichlet(rng, cfg.n_main, cfg.concentration)}
    for s in range(cfg.n_main):
        rules[(s,)] = _dirichlet(rng, cfg.n_main, cfg.concentration)
    attempts = 0
    added = 0
    while added < cfg.n_rules and attempts < 50 * max(cfg.n_rules, 1):
        attempts += 1
        parents = [k for k in rules if 1 <= len(k) < cfg.max_order]
        if not parents:
            break
        parent = parents[rng.integers(len(parents))]
        new = (int(rng.integers(cfg.n_main)),) + parent
        if new in rules:
            continue
        rules[new] = _dirichlet(rng, cfg.n_main, cfg.concentration)
        added += 1
    return rules


def build_variants(cfg: TaskConfig, seed: int) -> dict:
    """Return {regime_id: Automaton} for all regimes in cfg.phases.
    Regime 0 is the base. Regime i >= 1 resamples drift_frac of the rules."""
    base = build_base_rules(cfg, np.random.default_rng([seed, 0]))
    out = {0: Automaton(base, cfg)}
    for r in sorted(set(cfg.phases) - {0}):
        rng = np.random.default_rng([seed, 10 + r])
        keys = sorted(base.keys(), key=lambda k: (len(k), k))
        n_change = max(1, int(round(cfg.drift_frac * len(keys))))
        chosen = rng.choice(len(keys), size=n_change, replace=False)
        rules = dict(base)
        for i in chosen:
            rules[keys[i]] = _dirichlet(rng, cfg.n_main, cfg.concentration)
        out[r] = Automaton(rules, cfg)
    return out


@dataclass
class Stream:
    cfg: TaskConfig
    seed: int
    tokens: np.ndarray  # (T,) int64
    probs: np.ndarray  # (T, V) true P(x_t | x_<t)
    phase: np.ndarray  # (T,) phase index
    regime: np.ndarray  # (T,) regime id
    starts: list  # phase start indices
    phases: tuple

    @property
    def T(self) -> int:
        return len(self.tokens)


def generate_stream(cfg: TaskConfig, seed: int, stream_tag: int = 0,
                    phases: tuple | None = None, phase_len: int | None = None) -> Stream:
    """The automaton depends on `seed`; the sampled tokens on (`seed`, `stream_tag`)."""
    phases = tuple(cfg.phases if phases is None else phases)
    plen = cfg.phase_len if phase_len is None else phase_len
    variants = build_variants(cfg.replace(phases=tuple(sorted(set(cfg.phases) | set(phases)))), seed)
    rng = np.random.default_rng([seed, 1, stream_tag])
    T = plen * len(phases)
    tokens = np.zeros(T, dtype=np.int64)
    probs = np.zeros((T, cfg.vocab))
    phase = np.zeros(T, dtype=np.int64)
    regime = np.zeros(T, dtype=np.int64)
    starts = []
    hist: list = []
    t = 0
    for pi, rg in enumerate(phases):
        starts.append(t)
        auto = variants[rg]
        for _ in range(plen):
            p = auto.full_dist(hist)
            x = int(rng.choice(cfg.vocab, p=p))
            tokens[t], probs[t], phase[t], regime[t] = x, p, pi, rg
            if x < cfg.n_main:
                hist.append(x)
                if len(hist) > cfg.max_order:
                    hist.pop(0)
            t += 1
    return Stream(cfg, seed, tokens, probs, phase, regime, starts, phases)
