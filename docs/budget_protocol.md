# Budget protocol

Perfect equation of FLOPs across radically different architectures is a rabbit hole, so
this protocol reports separate axes and plots against each. A conclusion should hold on the
axes it is claimed for.

## Axes

| axis | meaning | how it is counted |
|---|---|---|
| `param_bits` | capacity of the slowly learned store, shared across the stream | neural: 32 x number of parameters. count: `max_entries` contexts x (key bits at max order + 32-bit timestamp + vocab x 32-bit counts). HTM/SDM: see the stub docstrings |
| `param_bits_used` | what was actually occupied at the end of the run (count only differs) | measured |
| `state_bits` | fast context carried between events | windowed models: window x ceil(log2(vocab + 1)) (the token buffer, from which activations can be recomputed); count: max_order tokens |
| `approx_flops_per_event` | analytic estimate, forward + 2x backward if learning | formulas in `muc/models/neural.py`; count: rough operation count |
| `experience_events` | stream events + pretraining tokens | measured |
| `wall_ms_per_event` | measured latency (machine dependent) | measured, excludes probes |

## Caveats you must keep in mind

* Online learners blur the line between "parameters" and "state": adapting weights stores
  information about the recent past. The two axes are a convention, not a law.
* Counting the token buffer as the Transformer's state ignores the activation cache; if you
  prefer the KV cache, the Transformer's `state_bits` is larger. Report both if it matters.
* Optimizer state (Adam moments) is not counted in `param_bits`.
* A model with a trained readout (HTM decoder, SDM output layer) must count the readout
  parameters, or the comparison is not fair.
* Tier sizes are matched on `param_bits` only. Other axes differ and that is informative.
* `sdm`: `param_bits` is n_locations x (address bits + vocab x 32). The fixed random hard
  addresses are counted, the token codes (derived from the seed) are not. The address length
  follows the tuned `k_ctx`, so measured `param_bits` moves by up to about 4 percent within a
  tier. The decay is a dense sweep over every counter, which dominates its FLOPs (roughly 80
  times the count model at the large tier); a lazy decay as in the count model would cut that.
  Report both axes.

## Fairness checklist before reading any result

1. Same task streams, same eval seeds, same probes for every model.
2. Same hyperparameter-search budget and objective; tuning sources disjoint from evaluation.
3. Equal effort in reading each model's hyperparameter space (no model tuned harder by hand).
4. `count` baseline included and tuned like the others.
5. Results reported against more than one budget axis.
