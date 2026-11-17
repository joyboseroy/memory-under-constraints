# E04 spec: does a learned model buy anything over direct association?

Status: **proposed, gated on the pilot.** Do not build this until E01 and E02 have produced a
figure worth following (a crossover, a surprising failure mode, or a clear adaptation-retention
tradeoff). If the pilot says stop, this spec is shelved with it.

## Why this experiment

The theory-level claim borrowed from Bennett's *A Brief History of Intelligence* is that adding
an internal model that can be run forward ("simulating") lets an agent adapt to a changed
environment better than agents that only cache values or retrieve experience. The pilot
already has a drifting source and a strong cheap competitor. E04 asks the same question in
decision form, so the variable is how an agent uses what it has learned, not only how well it
predicts.

What this can and cannot show: whether lookahead through a learned one-step model beats direct
k-step association in some regimes, at some budgets. It says nothing about whether evolution
happened that way, and nothing about selfhood or consciousness.

## E04a: prediction for delayed reward (build this first)

Same generator, same drift schedule, no distractors in v1. At every step t the agent chooses a
guess a_t for the token **k steps ahead**. It gets reward 1 at step t+k if a_t == x_{t+k}.
Actions do not change the source, so this is lookahead, not control. That limitation is
deliberate: it keeps the oracle exact and isolates the question "model rollout versus direct
k-step association". It is not planning in the full sense, and the write-up must say so.

### Agents (all online, same budget accounting as the pilot)

| agent | what it learns | how it chooses a_t |
|---|---|---|
| `direct_value` (model-free) | Q(state, a) for a hashed window of recent tokens, updated when the delayed reward arrives | argmax Q, with an exploration schedule that is part of the tuned space |
| `direct_memory` (memory-based) | stored (context -> token seen k steps later) episodes; variants: decayed variable-order counts keyed on the k-step transition, your SDM model, Transformer retrieval | argmax of the retrieved k-step distribution |
| `rollout` (model-based) | a one-step predictor from the pilot (count, LSTM, Transformer, SDM) | simulate n rollouts of depth k from the current context, pick the most frequent token at depth k |

`rollout` with the count model is the main contender. `direct_memory` with the count model keyed
on k-step transitions is the baseline it must beat; if the two tie everywhere, simulation adds
nothing in this task and that is the result.

### Oracle (exact)

The source is a context tree, so the true k-step predictive distribution is computable by
dynamic programming over context states (at most n_main^max_order = 4096 states, each with a
known next-token distribution). Reward rate is reported as a fraction of the oracle's expected
reward, and regret is oracle minus agent. A test must show that for k = 1 the oracle reduces to
the pilot's argmax of the true conditional.

### Grid

* horizon k in {1, 2, 4, 8}
* drift_frac in {0.1, 0.5, 1.0} (few rules changed vs. everything changed)
* budget tier in {small, medium} (same tier definitions as E01)
* rollouts n in {1, 8, 32} for `rollout`; the number of rollouts counts toward compute
* seeds >= 5; each seed is a new random source; tuning budget identical across agents

### Metrics

* expected-reward fraction of oracle, over the whole stream and per phase
* adaptation time and retention: same definitions as `docs/hypotheses.md`, applied to reward
  rate instead of p (state the window and threshold in advance)
* sample efficiency: events to reach 50% of the oracle's reward rate
* compute per decision (approx FLOPs including rollouts) and persistent memory (param_bits,
  state_bits) as in the budget protocol
* compounding error: reward fraction as a function of k for `rollout` versus `direct_memory`

### Hypotheses (fix before running)

* **E4-H1.** When few rules change at a drift (drift_frac 0.1), `rollout` re-adapts faster than
  `direct_value` and `direct_memory`, because the one-step model retains most structure while
  direct k-step statistics must be re-estimated.
* **E4-H2.** `rollout` degrades with k at small budgets because one-step errors compound; there
  is a crossover in k beyond which `direct_memory` wins.
* **E4-H3.** At k = 1 the memory-based agent matches or beats the rest at lower compute.
* **E4-H4.** Rollout count n has diminishing returns beyond a small n; report the compute
  cost of buying accuracy this way.
* **E4-H5 (null).** `direct_memory` with a strong count model ties `rollout` in every regime.

### Stop rule

Stop, and write it up as a negative result, if `direct_memory` (count, tuned equally) ties or
beats `rollout` in essentially every cell of the grid, if results flip under hyperparameter
changes, or if budget accounting for the rollouts cannot be defended. Continue to E04b only if
`rollout` shows a robust advantage in a regime predicted by E4-H1.

## E04b: steerable source (only if E04a shows a signal; design not frozen)

Actions influence the source: with probability c (control strength) the emitted token equals the
agent's action; otherwise it is drawn from the rules given the history, which now includes the
agent's tokens. Reward comes from reaching a target token pattern within k steps. This gives
genuine planning (actions change future states), at the cost of an oracle that needs value
iteration over the same context-state space. Freeze the reward definition and write the oracle
test before any agent code. A reward-reversal condition (the target pattern changes at a drift)
is the natural adaptation test. Do not add self-models, other-agent models or language to this
repository; they need separate hypotheses and their own gates.

## Build plan (about a week of work if the pilot clears)

1. `muc/kstep_oracle.py`: DP oracle plus tests (k = 1 reduction, agreement with Monte Carlo).
2. `muc/agents/`: the three agents behind a small `act()/observe()/reward()` interface, with an
   adapter contract check like `check_adapter`.
3. `muc/harness.py`: a decision-stream runner that delivers delayed rewards and scores against
   the oracle; reuse `summarize` for adaptation and retention on the reward-rate series.
4. `experiments/configs/e04_lookahead.yaml` and figures: reward fraction vs k per agent,
   adaptation time vs drift_frac, compute vs reward.
5. Record the config, seeds and any deviations in `docs/hypotheses.md` before the first sweep.
