"""Action-conditioned dynamics memories for E05 (pure NumPy; the neural ones are not built yet).

Protocol (an extension of the pilot's SeqModel, see docs/e05_dyna_spec.md):

    update(x_context, a, x_next)   learn one real transition (the only learning call)
    predict_at(x_context, a)       read-only distribution over the n observation tokens
    budget()                       param_bits, state_bits, approx_flops_per_event, flops_per_query

`x_context` is the list of the last m observation tokens. The action is a conditioning input
at the query point. It is not interleaved into the history.

`predict_at` must not change anything the model will later predict. `check_action_adapter`
tests that, plus normalisation and determinism.
"""
from __future__ import annotations

import copy
import math
from collections import OrderedDict

import numpy as np

from .base import bits_per_token

COUNT_BITS = 32
TIME_BITS = 32


class ActionCountModel:
    """Decayed variable-order counts keyed on (observation suffix, action).

    Back-off chain: action-free root, then (empty suffix, a), then longer suffixes with a.
    `use_action = False` drops `a` from every key; it is the degenerate control arm.
    """
    name = "count"

    def __init__(self, n, max_order=3, decay=0.999, wb_scale=1.0, max_entries=1024, use_action=True, seed=0, **_):
        self.n, self.m = int(n), int(max_order)
        self.decay, self.wb_scale, self.max_entries = float(decay), float(wb_scale), int(max_entries)
        self.use_action = bool(use_action)
        self.table: "OrderedDict[tuple, list]" = OrderedDict()
        self.t = 0

    def _key(self, ctx, a, k):
        suffix = tuple(int(x) for x in ctx[len(ctx) - k:]) if k > 0 else ()
        return (suffix, int(a) if self.use_action else None)

    def predict_at(self, x_context, a):
        p = np.full(self.n, 1.0 / self.n)
        root = self.table.get(((), None))
        if root is not None:
            p = self._blend(p, root)
        for k in range(0, min(self.m, len(x_context)) + 1):
            e = self.table.get(self._key(x_context, a, k))
            if e is None:
                break
            p = self._blend(p, e)
        return p

    def _blend(self, p, e):
        c = e[0] * (self.decay ** (self.t - e[1]))
        lam = self.wb_scale * max(int(e[2].sum()), 1)
        return (c + lam * p) / (c.sum() + lam)

    def update(self, x_context, a, x_next):
        keys = [((), None)] + [self._key(x_context, a, k) for k in range(0, min(self.m, len(x_context)) + 1)]
        for key in dict.fromkeys(keys):
            e = self.table.get(key)
            if e is None:
                e = [np.zeros(self.n), self.t, np.zeros(self.n, dtype=bool)]
                self.table[key] = e
            else:
                e[0] *= self.decay ** (self.t - e[1])
                self.table.move_to_end(key)
            e[0][int(x_next)] += 1.0
            e[1] = self.t
            e[2][int(x_next)] = True
            while len(self.table) > self.max_entries:
                self.table.popitem(last=False)
        self.t += 1

    def _entry_bits(self, k):
        return (k + (1 if self.use_action else 0)) * bits_per_token(self.n) + TIME_BITS + self.n * COUNT_BITS

    def budget(self):
        used = sum(self._entry_bits(len(k[0])) for k in self.table)
        return {"param_bits": int(self.max_entries * self._entry_bits(self.m)),
                "param_bits_used": int(used),
                "state_bits": int((self.m + 1) * bits_per_token(self.n)),
                "approx_flops_per_event": int(4 * self.n * (self.m + 2) * 2),
                "flops_per_query": int(4 * self.n * (self.m + 2))}


class ActionSDMModel:
    """Sparse distributed memory. Address = codes of the m context tokens, then the action code."""
    name = "sdm"

    def __init__(self, n, max_order=3, n_locations=256, code_bits=8, active_frac=0.1, decay=0.999, prior=1.0,
                 use_action=True, seed=0, **_):
        self.n, self.k = int(n), int(max_order)
        self.L, self.code_bits = int(n_locations), int(code_bits)
        self.use_action = bool(use_action)
        self.A = (self.k + (1 if use_action else 0)) * self.code_bits
        self.n_active = max(1, min(self.L, int(round(active_frac * self.L))))
        self.decay, self.prior = float(decay), float(prior)
        rng = np.random.default_rng([int(seed), 4242])
        codes, seen = [], set()
        while len(codes) < 2 * self.n:  # n observation codes, n action codes
            c = tuple(int(b) for b in rng.integers(0, 2, size=self.code_bits))
            if c not in seen:
                seen.add(c)
                codes.append(c)
        self.codes = np.array(codes, dtype=np.uint8)
        self.hard = rng.integers(0, 2, size=(self.L, self.A), dtype=np.uint8)
        self.counts = np.zeros((self.L, self.n))

    def _active(self, x_context, a):
        parts = [self.codes[int(x)] for x in x_context[-self.k:]]
        if self.use_action:
            parts.append(self.codes[self.n + int(a)])
        addr = np.concatenate(parts)
        dist = (self.hard != addr).sum(axis=1)
        return np.argsort(dist, kind="stable")[: self.n_active]

    def predict_at(self, x_context, a):
        s = self.counts[self._active(x_context, a)].sum(axis=0)
        return (s + self.prior / self.n) / (s.sum() + self.prior)

    def update(self, x_context, a, x_next):
        idx = self._active(x_context, a)
        self.counts *= self.decay
        self.counts[idx, int(x_next)] += 1.0

    def budget(self):
        loc_bits = self.A + self.n * 32
        used = int((self.counts.sum(axis=1) > 0).sum())
        L, A, n, a = self.L, self.A, self.n, self.n_active
        read = L * A + L * max(1, math.ceil(math.log2(L))) + a * n
        return {"param_bits": int(L * loc_bits), "param_bits_used": int(used * loc_bits),
                "state_bits": int((self.k + 1) * bits_per_token(self.n)),
                "approx_flops_per_event": int(read + a * n + L * n), "flops_per_query": int(read)}


class OracleDynamics:
    """Upper bound: the true dynamics of the current regime, known instantly after a drift."""
    name = "oracle"

    def __init__(self, mdp):
        self.mdp, self.P, self._regime = mdp, None, None

    def sync(self, regime):
        if regime != self._regime:
            self.P, self._regime = self.mdp.transition(regime), regime

    def predict_at(self, x_context, a):
        return self.P[self.mdp.encode(x_context), int(a)]

    def update(self, x_context, a, x_next):
        pass

    def budget(self):
        return {"param_bits": 0, "param_bits_used": 0, "state_bits": 0, "approx_flops_per_event": 0,
                "flops_per_query": 0}


def check_action_adapter(factory, n=4, order=3, steps=300, seed=0):
    """Contract for action-conditioned memories. Raises AssertionError."""
    rng = np.random.default_rng(seed)
    m = factory()
    ctx = [int(x) for x in rng.integers(0, n, size=order)]
    for _ in range(steps):
        a = int(rng.integers(n))
        q = np.asarray(m.predict_at(ctx, a), dtype=float)
        assert q.shape == (n,), f"predict_at must return shape ({n},), got {q.shape}"
        assert np.all(np.isfinite(q)) and q.min() >= 0 and abs(q.sum() - 1) < 1e-6, "must be a distribution"
        x = int(rng.integers(n))
        m.update(ctx, a, x)
        ctx = ctx[1:] + [x]
    probes = [([int(v) for v in rng.integers(0, n, size=order)], int(rng.integers(n))) for _ in range(20)]
    before = copy.deepcopy(m)
    first = [np.asarray(m.predict_at(c, a)).copy() for c, a in probes]
    second = [np.asarray(m.predict_at(c, a)).copy() for c, a in probes]
    assert np.allclose(first, second), "predict_at must be deterministic"
    after = [np.asarray(m.predict_at(c, a)) for c, a in probes]
    ref = [np.asarray(before.predict_at(c, a)) for c, a in probes]
    assert np.allclose(after, ref), "predict_at must not change the model"
    b = m.budget()
    for k in ("param_bits", "state_bits", "approx_flops_per_event", "flops_per_query"):
        assert k in b, f"budget() must include {k}"
    return True
