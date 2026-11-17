"""Exponentially decayed variable-order count model (PPM / Witten-Bell style).

This is the baseline most likely to be strong on this task family, so it is built to be
a genuine competitor rather than a straw-man n-gram:

* contexts of order 0..max_order, counts discounted by `decay` per learned event (lazily)
* interpolated back-off: P_k(s) = (c_k(s) + lam_k * P_{k-1}(s)) / (n_k + lam_k),
  lam_k = wb_scale * (number of distinct symbols seen in that context)
* a hard cap on stored contexts (`max_entries`), least recently updated contexts evicted
* it sees raw tokens, including distractors; it is not told which tokens to ignore

`param_bits` is the capacity of the table (max_entries contexts of maximal key length).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .base import SeqModel, bits_per_token

COUNT_BITS = 32
TIME_BITS = 32


class DecayedVarOrderCounts(SeqModel):
    name = "count"

    def __init__(self, vocab, max_order=4, decay=0.999, wb_scale=1.0,
                 max_entries=2048, seed=0, **_):
        self.V = vocab
        self.max_order = int(max_order)
        self.decay = float(decay)
        self.wb_scale = float(wb_scale)
        self.max_entries = int(max_entries)
        self.table: "OrderedDict[tuple, list]" = OrderedDict()  # key -> [counts, last_t, types]
        self.t = 0
        self.hist: list = []
        self.learning = True

    def reset_context(self):
        self.hist = []

    def _key(self, k):
        return tuple(self.hist[len(self.hist) - k:]) if k > 0 else ()

    def predict(self):
        p = np.full(self.V, 1.0 / self.V)
        for k in range(0, min(self.max_order, len(self.hist)) + 1):
            e = self.table.get(self._key(k))
            if e is None:
                break
            c = e[0] * (self.decay ** (self.t - e[1]))
            n = c.sum()
            lam = self.wb_scale * max(int(e[2].sum()), 1)
            p = (c + lam * p) / (n + lam)
        return p

    def update(self, token):
        token = int(token)
        if self.learning:
            for k in range(0, min(self.max_order, len(self.hist)) + 1):
                key = self._key(k)
                e = self.table.get(key)
                if e is None:
                    e = [np.zeros(self.V), self.t, np.zeros(self.V, dtype=bool)]
                    self.table[key] = e
                else:
                    e[0] *= self.decay ** (self.t - e[1])
                    self.table.move_to_end(key)
                e[0][token] += 1.0
                e[1] = self.t
                e[2][token] = True
                while len(self.table) > self.max_entries:
                    self.table.popitem(last=False)
            self.t += 1
        self.hist.append(token)
        if len(self.hist) > self.max_order:
            self.hist.pop(0)

    def _ctx_bits(self, k):
        return k * bits_per_token(self.V) + TIME_BITS + self.V * COUNT_BITS

    def budget(self):
        used = sum(self._ctx_bits(len(k)) for k in self.table)
        return {
            "param_bits": int(self.max_entries * self._ctx_bits(self.max_order)),
            "param_bits_used": int(used),
            "state_bits": int(self.max_order * bits_per_token(self.V)),
            "approx_flops_per_event": int(4 * self.V * (self.max_order + 1) * 2),
        }
