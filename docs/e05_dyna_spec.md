# E05 spec: Dyna agent with a swappable memory as its world model

Status: **proposed, gated on the pilot.** This supersedes the E04b sketch in `e04_spec.md`.
Build it only if E01 and E02 leave at least two memory models worth comparing; the pilot
decides which memories are worth putting inside an agent.

## Question

Does the choice of memory mechanism change planning quality under drift and resource limits,
once the same memory is used as an agent's learned world model?

Everything except the dynamics memory is held fixed, so any difference in return is
attributable to the memory. That is the whole point of the experiment, and it is why this is
one new environment plus one agent, not a framework.

What it can show: whether a learned model helps over model-free learning and over plain replay
in some drift regimes, and whether one-step predictive accuracy (or calibration) of the memory
translates into planning quality. It cannot show that evolution followed Bennett's sequence.
(The "simulating" link to Bennett motivates the design; the experiment stands without it.)

## Environment: steerable context-tree MDP

Reuse the pilot's source. State s = the last m main tokens (m = max_order, so at most
n_main^m = 4096 states). At each step the agent picks an action a in {0..n_main-1}. The next
token x' is the agent's action with probability c (control strength), otherwise it is drawn
from the source's rule for the current context. The new state is (s, x') truncated to m.
Reward is R[s, x'] in {0, 1} from a sparse hidden table (about 10% of pairs rewarding).

* Exact MDP: P(x' | s, a) = c * 1[x' = a] + (1 - c) * rule(s)[x']. Value iteration over at most
  4096 x 8 state-actions gives V*, Q* and the optimal return for the current regime. All
  results are reported as a fraction of the optimal return, with regret as the complement.
* Drift, three kinds, applied at phase boundaries as in the pilot (phases 0, 1, 0):
  dynamics drift (a fraction of rules resampled), reward drift (the reward table resampled or
  reversed), or both. Returning to regime 0 restores the exact original MDP.
* Distractors are off in v1.

## Agent: Dyna with expected backups

```text
real step:  observe s, choose a (epsilon-greedy on Q), receive x', r
            Q-learning update from the real transition
            model.update(context, a, x')          <- the memory under test
            reward_table.update(s, x', r)          <- identical in every arm
plan:       repeat n_plan times:
              pick (s, a) from visited pairs (recency-weighted)
              p = model.predict_at(context(s), a)  # distribution over x', read-only
              Q(s, a) <- sum_x' p[x'] * (R_hat[s, x'] + gamma * max_a' Q(s', a'))
```

Expected backups use the model's full distribution, so there is no sampling noise from the
model and the arms differ only in what the model predicts. The tabular Q-learner, exploration
schedule, reward model and planning schedule are the same in all arms. The tabular Q table is
a fixed cost shared by all arms and is reported separately from the model's budget.

### Swappable dynamics memory

Action-conditioning is done by interleaving: the model sees the history a1 x1 a2 x2 ... and
is asked for the token that follows an action. Actions use token ids n_main..2*n_main-1, the
returned distribution is restricted to the observation tokens and renormalised. Windows and
orders are counted in raw tokens, so they double relative to the pilot; tier tables must
state that.

Arms:

| arm | what is swapped |
|---|---|
| `q_learning` | no model, n_plan = 0 |
| `replay` | no model; replay stored real transitions n_plan times (the control that separates "modeling" from "more updates") |
| `dyna_count` | decayed variable-order count model (the baseline to beat) |
| `dyna_lstm`, `dyna_transformer` | pilot neural models, online |
| `dyna_sdm` | your SDM model through the adapter |
| `dyna_oracle` | true dynamics (upper bound; shows the headroom a perfect model gives) |

HTM is likely **excluded**: its temporal state cannot be queried from an arbitrary context
without disturbing it. State this in the write-up as a property of the mechanism, not omit it
silently.

## Required interface extension

Add one method to the model protocol: `predict_at(context: list[int]) -> np.ndarray`,
read-only, no state change, no learning. Count: look up the given suffix. Windowed neural:
forward pass on the given window under no_grad. SDM: read at the address derived from the
given context. Extend `check_adapter` with a test that `predict_at` leaves the model's
predictions unchanged. This is the main implementation cost and the main risk; models that
cannot support it are reported as unsupported.

## Grid

* control strength c in {0.3, 0.6, 0.9}
* drift kind in {dynamics, reward, both}; drift_frac in {0.1, 0.5, 1.0}
* n_plan in {0, 5, 20, 50}
* budget tier in {small, medium} for the model (same param_bits tiers as E01)
* seeds >= 5; every seed is a new MDP; identical tuning budget for every arm; tuned once on the
  base condition and evaluated across the grid without retuning

## Metrics

* return as a fraction of the optimal return, overall and per phase; regret rate
* adaptation time after each drift and retention on return to regime 0, defined on the return
  series with the window and threshold fixed in advance (pick the threshold from a smoke test
  before any sweep, as was done for the pilot)
* sample efficiency: real steps to reach 50% of optimal return
* compute: approx FLOPs per real step including planning queries; memory: param_bits and
  state_bits of the model, replay buffer size for `replay`
* diagnostic: KL between the model's predicted dynamics and the true dynamics over visited
  contexts, and the model's calibration (ECE). This decouples model quality from planning use.

## Hypotheses (fix before any serious run)

* **E5-H1.** Under reward drift with unchanged dynamics, any Dyna arm with a reasonable model
  beats `q_learning` and `replay` at equal real samples, because the model can re-propagate
  the new reward without new experience.
* **E5-H2.** Under dynamics drift, the advantage depends on how fast the memory adapts. Models
  that adapt quickly (decayed counts, SDM) lose less than slowly adapting gradient models at
  small budgets.
* **E5-H3.** Return tracks one-step KL across memories, but with exceptions: overconfident
  models (high ECE) plan worse than their KL predicts.
* **E5-H4.** The Dyna advantage over `replay` appears only at large n_plan; at small n_plan
  replay matches it. Report gain per FLOP.
* **E5-H5 (null).** `dyna_count` ties every other memory in every cell, in which case the
  choice of memory does not matter for planning in this environment.

## Stop rule

Stop and record a negative result if `replay` matches the best Dyna arm across essentially the
whole grid, if `dyna_count` ties the other memories everywhere, if results flip under
hyperparameter changes, or if planning compute cannot be accounted for defensibly. A tie of
`dyna_count` with everything is a legitimate and publishable null.

## Build plan (about one to two weeks if the pilot clears)

1. `muc/envs/steerable.py` with exact transitions, value iteration and tests: c = 0 reduces to
   the pilot's source; value iteration agrees with Monte Carlo rollouts of the optimal policy.
2. `predict_at` on count, LSTM, Transformer and your SDM adapter, with the contract test.
3. The interleaving wrapper and tests (restriction to observation tokens, normalisation).
4. `muc/agents/dyna.py`, `q_learning`, `replay`, `dyna_oracle`; agent contract test and a
   reproducibility test.
5. Decision-stream harness and `experiments/configs/e05_dyna.yaml`; figures for return vs
   n_plan, return vs KL, adaptation time by drift kind, compute vs return.
6. Record config, seeds and deviations in `docs/hypotheses.md` before the first sweep.
