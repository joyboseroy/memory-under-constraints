"""SDM / SNN sequence-memory adapter stub. Not implemented: plug in your own model.

Suggested shape for a sparse-distributed-memory sequence model:

* address encoder: map the last k tokens (or a rank-order / spike-time code of them) to a
  sparse address; k and the code are hyperparameters
* store: `n_locations` hard locations, each with a counter vector over the vocabulary
  (that is the learned memory; count it in `param_bits`)
* read: activate locations within a Hamming radius (or top-k by similarity), sum counters,
  normalise to a distribution (add a small smoothing prior)
* write: on `update(token)` with `learning=True`, increment counters at the locations that
  were active for the previous context
* optional decay of counters if you want a forgetting mechanism comparable to the
  count baseline's `decay`

If you build a spiking version, keep the readout honest in the same way as for HTM: any
trained readout counts toward `param_bits`.

Budget accounting: param_bits = n_locations * (address bits + vocab * counter bits);
state_bits = whatever recent-event code you carry (spike traces, buffers).
Verify with `muc.models.base.check_adapter` and tests/test_adapters.py.
"""
from __future__ import annotations

from .base import SeqModel


class SDMSNNModel(SeqModel):
    name = "sdm_snn"

    def __init__(self, vocab, **kwargs):
        raise NotImplementedError(
            "SDM/SNN adapter is a stub. Implement muc/models/sdm_snn.py following its "
            "docstring, then run tests/test_adapters.py.")
