"""HTM adapter stub. Not implemented: bring your own, then run `check_adapter`.

Intended design (keep it this simple, so the comparison stays about the mechanism):

* encoder: one fixed random SDR per token id (n bits, w active), no learning
* spatial pooler: optional; first version can skip it and feed the SDR to the TM directly
* temporal memory: htm.core `TemporalMemory` (or your own minimal TM)
* decoder: maps the TM's predicted cells to a distribution over next tokens. This is the
  part that must not cheat. Use a simple online count classifier over (predicted-cell
  set -> next token) and COUNT ITS PARAMETERS in `param_bits`.
* learning: `update(token)` calls `tm.compute(sdr, learn=self.learning)`, then trains the
  decoder only if `self.learning`.

Budget accounting:
* param_bits: synapses (permanence + target) across all segments at capacity
  (max_segments_per_cell * max_synapses_per_segment * cells * bits per synapse) + decoder
* state_bits: active + predictive cell bitmaps carried between events
* approx_flops_per_event: segments evaluated per event * synapses per segment

Tuning knobs worth putting in the search space: permanence increment/decrement,
activation threshold, learning threshold, cells per column, max new synapses.

After implementing, add a tier table to experiments/configs/*.yaml under `htm:` and
verify with tests/test_adapters.py (it auto-runs once this class stops raising).
"""
from __future__ import annotations

from .base import SeqModel


class HTMModel(SeqModel):
    name = "htm"

    def __init__(self, vocab, **kwargs):
        raise NotImplementedError(
            "HTM adapter is a stub. Implement muc/models/htm.py following its docstring, "
            "then run tests/test_adapters.py.")
