"""Model protocol. Every model is an online sequence predictor.

The harness drives it event by event:

    q = model.predict()      # distribution over the next token, shape (vocab,)
    model.update(token)      # observe the true token; learn (if model.learning) and advance context

`learning` is switched off on a copy of the model for frozen probes.
`reset_context()` clears recent-event context but keeps everything learned.

Budget accounting (see docs/budget_protocol.md):
    param_bits   capacity of the slowly learned store (weights, count tables, synapses)
    state_bits   fast context carried between events (token buffers, hidden state)
    approx_flops_per_event   analytic, approximate, includes backward pass if learning
"""
from __future__ import annotations

import math

import numpy as np


def bits_per_token(vocab: int) -> int:
    return max(1, math.ceil(math.log2(vocab + 1)))


class SeqModel:
    name = "base"
    learning = True
    pretrain_tokens = 0  # experience consumed before the stream starts

    def predict(self) -> np.ndarray:
        raise NotImplementedError

    def update(self, token: int) -> None:
        raise NotImplementedError

    def reset_context(self) -> None:
        raise NotImplementedError

    def clear_cache(self) -> None:
        pass

    def budget(self) -> dict:
        raise NotImplementedError


def check_adapter(factory, vocab=12, n=60, seed=0):
    """Contract check for new model adapters (HTM, SDM/SNN, ...). Raises AssertionError."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, vocab, size=n)
    m = factory()
    for x in xs:
        q = np.asarray(m.predict(), dtype=float)
        assert q.shape == (vocab,), f"predict() must return shape ({vocab},), got {q.shape}"
        assert np.all(np.isfinite(q)) and q.min() >= 0, "predict() must be finite and non-negative"
        assert abs(q.sum() - 1.0) < 1e-4, "predict() must sum to 1"
        m.update(int(x))
    b = m.budget()
    for k in ("param_bits", "state_bits", "approx_flops_per_event"):
        assert k in b, f"budget() must include {k}"
    import copy
    m.clear_cache()
    runs = []
    for _ in range(2):
        c = copy.deepcopy(m)
        c.learning = False
        c.reset_context()
        outs = []
        for x in xs[:20]:
            outs.append(np.asarray(c.predict()).copy())
            c.update(int(x))
        runs.append(outs)
    assert np.allclose(runs[0], runs[1], atol=1e-6), "frozen probes must be deterministic"
    return True
