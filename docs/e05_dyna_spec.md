# E05 spec: Dyna agent with a swappable memory as its world model

Status: **first pass built and run on a reduced grid, count and SDM memories only** (see `docs/e05_results_v0_1.md`). The pilot gate (E01 and E02 with the neural models) has not been cleared, so this is an early look, not the experiment as specified. Built: the exact environment, the action-conditioned count and SDM memories, all arms below, tests. Not built: LSTM and Transformer `predict_at`, the full grid, per-arm tuning. Original status: proposed, gated on the pilot. Revised: action enters as a query input rather than an interleaved token; the replay arms are specified so reward drift cannot trivially favour the model. This supersedes the E04b sketch in `e04_spec.md`.
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

**Replay is specified so that it does not lose by construction.** The buffer holds
(s, a, x') transitions, not rewards. Every replayed backup reads the reward from the current
`R_hat[s, x']`, which is updated from real steps exactly as in the Dyna arms. If replay reused
the rewards stored at collection time it would be unable to react to reward drift and E5-H1
would be trivially true. Buffer capacity is matched to the model's budget tier by counting
bits per stored transition. Replay samples uniformly from the buffer; Dyna samples visited
(s, a) pairs with the same recency weighting. State both and use identical sampling in
`replay_sa` and `dyna_*`. The difference that remains is the intended one: replay updates
only transitions it has seen, while a model also backs up the transitions it predicts.

Expected backups use the model's full distribution, so there is no sampling noise from the
model and the arms differ only in what the model predicts. The tabular Q-learner, exploration
schedule, reward model and planning schedule are the same in all arms. The tabular Q table is
a fixed cost shared by all arms and is reported separately from the model's budget.

### Swappable dynamics memory

The memory models the observation stream x1 x2 x3 ..., and the action enters as a conditioning
input at the query point only. It is not interleaved into the history. Interleaving would
split count statistics across action contexts at every position, halve the effective window
for every model, and penalise high-order count models unfairly. Instead the query is
`predict_at(x_context, a)`: the distribution of the next observation token given the last
observation tokens `x_context` and the action `a` about to be taken. Each model uses `a`
as follows:

* **count:** keys are (observation suffix, a), with the same variable-order backoff and decay
  as the pilot. Backoff drops observation tokens first and keeps `a` until last, then
  backs off to an a-marginal.
* **LSTM and Transformer:** the window of observation tokens plus one extra input for `a`
  (a learned embedding added at the final position). Training pairs are
  (window, a) -> next token. State the extra parameters in the tier table.
* **SDM:** the address is the observation-context code concatenated or bound with a fixed
  random code for `a`. Report the address width and its effect on param_bits.

Because the source ignores the action except through the control mixture, a memory that
cannot condition on `a` at all (a plain `predict_at(x_context)`) is an informative degenerate
arm: it learns the marginal dynamics and should plan badly when c is large. Include it as
`dyna_count_noaction` to show that the action input is doing work.

Arms:

| arm | what is swapped |
|---|---|
| `q_learning` | no model, n_plan = 0 |
| `replay` | no model; Q-learning updates replayed on stored real transitions, n_plan per real step (the control that separates "modeling" from "more updates") |
| `replay_sa` | as `replay`, but each backup is an expected backup over all observation tokens ever seen after (s, a), weighted by their empirical counts (an uncompressed table, not a memory model; the strong non-generalising baseline) |
| `dyna_count` | decayed variable-order count model (the baseline to beat) |
| `dyna_lstm`, `dyna_transformer` | pilot neural models, online |
| `dyna_sdm` | your SDM model through the adapter |
| `dyna_oracle` | true dynamics (upper bound; shows the headroom a perfect model gives) |

HTM is likely **excluded**: its temporal state cannot be queried from an arbitrary context
without disturbing it. State this in the write-up as a property of the mechanism, not omit it
silently.

## Required interface extension

Add one method to the model protocol:
`predict_at(x_context: list[int], a: int) -> np.ndarray`, read-only, no state change, no
learning, returning a distribution over the n_main observation tokens. Count: look up the
given (suffix, a). Windowed neural: forward pass on the given window and action under
no_grad. SDM: read at the address derived from the context and the action. Extend `check_adapter` with three tests: `predict_at` leaves the model's own predictions
unchanged, `predict_at` called twice returns the same array, and the output is a normalised
distribution over the n_main observation tokens. This is the main implementation cost and the main risk; models that
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
  the new reward through transitions the agent has not recently visited. Compare against
  `replay` with current rewards (as specified above) and against `replay_sa`. The claim is
  weaker and more interesting than "models beat replay": the gain should shrink toward zero
  as visitation coverage of (s, a) grows, and it should be largest when the state space is
  poorly covered.
* **E5-H2.** Under dynamics drift, the advantage depends on how fast the memory adapts. Models
  that adapt quickly (decayed counts, SDM) lose less than slowly adapting gradient models at
  small budgets.
* **E5-H3.** Return tracks one-step KL across memories, but with exceptions: overconfident
  models (high ECE) plan worse than their KL predicts.
* **E5-H4.** The Dyna advantage over `replay` appears only at large n_plan; at small n_plan
  replay matches it. Report gain per FLOP.
* **E5-H4b.** `dyna_count_noaction` loses to `dyna_count` increasingly as c rises, confirming
  that the action input is the part of the model doing planning work.
* **E5-H5 (null).** `dyna_count` ties every other memory in every cell, in which case the
  choice of memory does not matter for planning in this environment.

## Stop rule

Stop and record a negative result if `replay` matches the best Dyna arm across essentially the
whole grid, if `dyna_count` ties the other memories everywhere, if results flip under
hyperparameter changes, or if planning compute cannot be accounted for defensibly. A tie of
`dyna_count` with everything is a legitimate and publishable null.

## Changes made when it was built (v0.1)

Recorded here and in `docs/hypotheses.md` so the spec and the code do not drift apart.

* **Arm added: `dyna_oracle_full`.** The spec's planning samples only visited (s, a) pairs. A single-seed diagnostic
  before any sweep showed that about half the Q table is never visited in 1500 steps, so a perfect dynamics model
  restricted to visited pairs is not an upper bound. `dyna_oracle_full` plans over all pairs uniformly and separates
  model-accuracy headroom from coverage headroom. `dyna_oracle` is kept as specified.
* **Score is normalised.** Return is reported as (reward rate - random policy rate) / (optimal rate - random policy
  rate), computed exactly per regime, because with about 10% rewarding pairs a random policy already earns about a quarter
  of the optimal raw rate. Fraction of optimal is also in the CSV (`frac_opt`). Adaptation time uses a rolling
  normalised score of 0.5 (window 200), not 50% of raw optimal return.
* **No per-arm tuning.** Every arm uses the same fixed Q-learning settings (alpha 0.3, epsilon 0.1) and each memory its
  default hyperparameters. The spec's "identical tuning budget" was not met because there was no tuning at all. This
  favours no arm deliberately but may favour whichever default happens to suit the task.
* **Reduced grid.** 64 states (4 tokens, order 3), c in {0.3, 0.6, 0.9}, drift_frac 0.5 for all three drift kinds plus
  drift_frac 1.0 at c = 0.6 (3 seeds), n_plan in {5, 20, 50} (and 0 for `q_learning`), tier small, plus a small
  versus medium check at c = 0.6 (count, SDM, replay), 5 seeds (3 for severity). The spec's grid has 4096 states.
* **Reward drift** relocates a drift_frac of the rewarding entries to random non-rewarding entries, so density stays
  fixed and drift_frac 1.0 moves every reward.
* **Context size.** `x_context` is always the full last-m window (the state). Models back off inside it.
* **Models not built.** LSTM, Transformer and HTM `predict_at`. The LSTM and Transformer are the main remaining cost.

## Build plan (about one to two weeks if the pilot clears)

1. (done) `muc/envs/steerable.py` with exact transitions, value iteration and tests: c = 0 reduces to
   the pilot's source; value iteration agrees with Monte Carlo rollouts of the optimal policy.
2. (count and SDM done; LSTM and Transformer not) `predict_at(x_context, a)` on count, LSTM, Transformer and your SDM adapter, with the
   contract tests.
3. (done for the environment and the count model; the all-memories check is not) Tests that action conditioning is real: with c = 0 the action has no effect and all
   memories should agree with their `predict_at` without `a`; with c = 1 the next token equals
   `a` and the oracle memory must return a point mass.
4. (done) `muc/agents/dyna.py`, `q_learning`, `replay`, `dyna_oracle`; agent contract test and a
   reproducibility test.
5. (done, `experiments/run_e05.py`, `analysis/e05_summary.py`) Decision-stream harness and `experiments/configs/e05_dyna.yaml`; figures for return vs
   n_plan, return vs KL, adaptation time by drift kind, compute vs return.
6. (done) Record config, seeds and deviations in `docs/hypotheses.md` before the first sweep.
