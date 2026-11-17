"""Steerable context-tree MDP (E05).

The pilot's context-tree source, with an agent that can push it. State s is the last m
main tokens (m = max_order). The agent picks an action a in {0..n_main-1}. The next token is

    P(x' | s, a) = c * 1[x' = a] + (1 - c) * rule(s)[x']

where rule(s) is the source's own next-token distribution (noise included). c = 0 is the
pilot's source, uncontrolled. c = 1 makes the next token whatever the agent says.
Reward is R[s, x'] in {0, 1}, sparse (reward_density of the entries are 1), hidden.

Everything is exact: transition tensor, value iteration, the optimal policy's long-run
reward rate and a random policy's reward rate, per regime. A regime is a pair
(dynamics id, reward id); returning to (0, 0) restores the original MDP exactly.

Drift kinds, applied at phase boundaries (phases default (0, 1, 0)):
    dynamics  a drift_frac of the source rules resampled (as in the pilot)
    reward    a drift_frac of the rewarding entries relocated to random non-rewarding entries
    both      both of the above
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..tasks import TaskConfig, build_variants


@dataclass
class SteerConfig:
    source: TaskConfig = field(default_factory=lambda: TaskConfig(n_main=4, n_distract=0, max_order=3, n_rules=10,
                                                                   concentration=0.25, noise=0.05, distract_prob=0.0))
    c: float = 0.6
    reward_density: float = 0.1
    gamma: float = 0.9
    drift_kind: str = "dynamics"
    drift_frac: float = 0.5
    phases: tuple = (0, 1, 0)
    phase_len: int = 3000

    def replace(self, **kw) -> "SteerConfig":
        return dataclasses.replace(self, **kw)

    @property
    def n(self) -> int:
        return self.source.n_main

    @property
    def m(self) -> int:
        return self.source.max_order

    @property
    def n_states(self) -> int:
        return self.n ** self.m

    def to_dict(self) -> dict:
        d = {k: v for k, v in dataclasses.asdict(self).items() if k != "source"}
        d["phases"] = list(self.phases)
        d["source"] = self.source.to_dict()
        return d


class SteerableMDP:
    """All regimes of one seed, with exact solutions. Build once, query often."""

    def __init__(self, cfg: SteerConfig, seed: int):
        if cfg.drift_kind not in ("dynamics", "reward", "both"):
            raise ValueError("drift_kind must be dynamics, reward or both")
        if not 0.0 <= cfg.c <= 1.0:
            raise ValueError("c must be in [0, 1]")
        self.cfg, self.seed = cfg, seed
        self.n, self.m, self.S = cfg.n, cfg.m, cfg.n_states
        n, S = self.n, self.S
        ids = sorted(set(cfg.phases))
        src = cfg.source.replace(phases=tuple(sorted(set(ids) | {0})), drift_frac=cfg.drift_frac, distract_prob=0.0,
                                 n_distract=0)
        autos = build_variants(src, seed)
        self.ctx = np.array([self._decode(s) for s in range(S)], dtype=np.int64)  # (S, m)
        self.ns = (np.arange(S)[:, None] * n + np.arange(n)[None, :]) % S  # next state, (S, n)
        self._rule = {}
        for r, auto in autos.items():
            self._rule[r] = np.array([auto.full_dist(list(self.ctx[s]))[:n] for s in range(S)])
        base_rng = np.random.default_rng([seed, 500])
        self._rew = {0: base_rng.random((S, n)) < cfg.reward_density}
        for r in ids:
            if r == 0:
                continue
            rng = np.random.default_rng([seed, 510 + r])
            R = self._rew[0].copy()
            ones = np.flatnonzero(R.ravel())
            zeros = np.flatnonzero(~R.ravel())
            k = min(len(ones), len(zeros), max(1, int(round(cfg.drift_frac * len(ones)))))
            drop = rng.choice(ones, size=k, replace=False)
            add = rng.choice(zeros, size=k, replace=False)
            flat = R.ravel().copy()
            flat[drop], flat[add] = False, True
            self._rew[r] = flat.reshape(S, n)
        self._solved = {}

    # ---- encoding ----
    def _decode(self, s):
        out = []
        for _ in range(self.m):
            out.append(s % self.n)
            s //= self.n
        return out[::-1]

    def encode(self, ctx) -> int:
        s = 0
        for x in ctx:
            s = s * self.n + int(x)
        return s

    # ---- regimes ----
    def regime_of_phase(self, phase_index: int):
        r = self.cfg.phases[phase_index]
        k = self.cfg.drift_kind
        return (r if k in ("dynamics", "both") else 0, r if k in ("reward", "both") else 0)

    def rule(self, regime) -> np.ndarray:
        return self._rule[regime[0]]

    def reward(self, regime) -> np.ndarray:
        return self._rew[regime[1]].astype(float)

    def transition(self, regime) -> np.ndarray:
        """P[s, a, x'] for the regime."""
        c, n = self.cfg.c, self.n
        P = (1.0 - c) * self.rule(regime)[:, None, :] * np.ones((1, n, 1))
        P[:, np.arange(n), np.arange(n)] += c
        return P

    # ---- exact solution ----
    def solve(self, regime) -> dict:
        if regime in self._solved:
            return self._solved[regime]
        P, R, g = self.transition(regime), self.reward(regime), self.cfg.gamma
        V = np.zeros(self.S)
        for _ in range(2000):
            Q = np.einsum("sax,sx->sa", P, R) + g * np.einsum("sax,sx->sa", P, V[self.ns])
            V2 = Q.max(axis=1)
            done = np.max(np.abs(V2 - V)) < 1e-10
            V = V2
            if done:
                break
        Q = np.einsum("sax,sx->sa", P, R) + g * np.einsum("sax,sx->sa", P, V[self.ns])
        pi = Q.argmax(axis=1)
        out = dict(Q=Q, V=V, pi=pi,
                   rate_opt=self.reward_rate(regime, lambda s: np.eye(self.n)[pi[s]]),
                   rate_rand=self.reward_rate(regime, lambda s: np.full(self.n, 1.0 / self.n)))
        self._solved[regime] = out
        return out

    def reward_rate(self, regime, policy_probs, iters=400) -> float:
        """Long-run reward per step of a stochastic policy, from the stationary state distribution."""
        P, R = self.transition(regime), self.reward(regime)
        pol = np.array([policy_probs(s) for s in range(self.S)])  # (S, n)
        Px = np.einsum("sa,sax->sx", pol, P)  # (S, n) next-token probs under the policy
        d = np.full(self.S, 1.0 / self.S)
        for _ in range(iters):
            nd = np.zeros(self.S)
            for x in range(self.n):
                np.add.at(nd, self.ns[:, x], d * Px[:, x])
            d = nd
        return float((d[:, None] * Px * R).sum())

    # ---- simulation ----
    def start_state(self, rng) -> int:
        return int(rng.integers(self.S))

    def step(self, s, a, regime, rng):
        p = (1.0 - self.cfg.c) * self.rule(regime)[s]
        p = p.copy()
        p[a] += self.cfg.c
        x = int(rng.choice(self.n, p=p))
        return x, float(self.reward(regime)[s, x]), int(self.ns[s, x])
