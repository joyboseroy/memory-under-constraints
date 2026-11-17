"""Generic Sparse Distributed Memory sequence predictor (Kanerva-style, counters over tokens).

This is a placeholder baseline for the SDM family, deliberately plain, so that the
`sdm_snn` slot stays free for a rank-order or spiking model. It is not that model.

Mechanism
* address: the last `k_ctx` tokens, each mapped to a fixed random binary code of `code_bits`
  bits (index V is a padding code for contexts shorter than k_ctx), concatenated
* store: `n_locations` hard locations with fixed random binary addresses; each holds a
  counter vector over the vocabulary. The counters are the learned memory
* read: the `n_active` locations nearest to the address in Hamming distance (ties broken by
  location index) are activated and their counters summed; a prior of total mass `prior`
  spread uniformly is added, then normalised
* write: on `update(token)` with `learning=True`, all counters decay by `decay`, then the
  token's counter is incremented at the locations that were active for the context the
  token followed

Differences from the count baseline that matter for interpretation: contexts are not stored
under exact keys, so similar contexts share locations (graceful degradation, and also
interference once more contexts than locations are in use); there is no variable-order
back-off, only the single order `k_ctx`.

Budget accounting (docs/budget_protocol.md):
  param_bits  n_locations * (address_bits + vocab * 32), counting the fixed random hard
              addresses as the stub docstring suggests; the token codes are derived from the
              seed and are not counted, as for a fixed random encoder
  state_bits  k_ctx * bits_per_token(vocab), the token buffer the address is rebuilt from
  flops       Hamming comparison, selection, counter sum, decay and write, analytic
"""
from __future__ import annotations

import math

import numpy as np

from .base import SeqModel, bits_per_token

COUNTER_BITS = 32


class SDMModel(SeqModel):
    name = "sdm"

    def __init__(self, vocab, n_locations=256, k_ctx=3, code_bits=8, active_frac=0.1,
                 decay=0.999, prior=1.0, seed=0, **_):
        if k_ctx < 1 or n_locations < 1:
            raise ValueError("k_ctx and n_locations must be at least 1")
        self.V = int(vocab)
        self.L = int(n_locations)
        self.k = int(k_ctx)
        self.code_bits = int(code_bits)
        self.A = self.k * self.code_bits
        self.n_active = max(1, min(self.L, int(round(active_frac * self.L))))
        self.decay = float(decay)
        self.prior = float(prior)
        rng = np.random.default_rng([int(seed), 31337])
        self.codes = self._distinct_codes(rng, self.V + 1, self.code_bits)
        self.hard = rng.integers(0, 2, size=(self.L, self.A), dtype=np.uint8)
        self.counts = np.zeros((self.L, self.V))
        self.hist: list = []
        self._active_idx = None
        self.learning = True

    @staticmethod
    def _distinct_codes(rng, n, bits):
        if n > 2 ** bits:
            raise ValueError("code_bits too small for the vocabulary")
        seen, out = set(), []
        while len(out) < n:
            c = tuple(int(b) for b in rng.integers(0, 2, size=bits))
            if c not in seen:
                seen.add(c)
                out.append(c)
        return np.array(out, dtype=np.uint8)

    def reset_context(self):
        self.hist = []
        self._active_idx = None

    def _address(self):
        pad = self.V
        toks = ([pad] * self.k + self.hist)[-self.k:]
        return np.concatenate([self.codes[t] for t in toks])

    def _active(self):
        if self._active_idx is None:
            dist = (self.hard != self._address()).sum(axis=1)
            self._active_idx = np.argsort(dist, kind="stable")[: self.n_active]
        return self._active_idx

    def predict(self):
        s = self.counts[self._active()].sum(axis=0)
        return (s + self.prior / self.V) / (s.sum() + self.prior)

    def update(self, token):
        token = int(token)
        if self.learning:
            idx = self._active()
            self.counts *= self.decay
            self.counts[idx, token] += 1.0
        self.hist.append(token)
        if len(self.hist) > self.k:
            self.hist.pop(0)
        self._active_idx = None

    def _location_bits(self):
        return self.A + self.V * COUNTER_BITS

    def budget(self):
        used = int((self.counts.sum(axis=1) > 0).sum())
        L, A, V, a = self.L, self.A, self.V, self.n_active
        read = L * A + L * max(1, math.ceil(math.log2(L))) + a * V
        write = a * V + L * V  # increment plus the decay sweep
        return {
            "param_bits": int(L * self._location_bits()),
            "param_bits_used": int(used * self._location_bits()),
            "state_bits": int(self.k * bits_per_token(V)),
            "approx_flops_per_event": int(read + write),
        }
