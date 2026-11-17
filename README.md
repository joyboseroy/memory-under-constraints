# memory-under-constraints

A small, strict test bench for one question:

> When memory, compute and learning experience are limited, and the world keeps changing,
> which memory mechanism predicts best, and what does it cost?

You give the bench a model. It runs your model against count tables, an LSTM, a Transformer
and a sparse distributed memory on the same drifting stream, under the same memory budget,
and scores every prediction against the **exact** answer rather than against noisy samples.
It also ships a second, dependency-free study of when a memory should be trusted, checked,
refreshed or bypassed, and a tool that re-ranks results under different prices for compute.

Everything is seeded, every hypothesis was written down before the runs, and the repository
has a place for results that came out boring (`docs/negative_results.md`).

**Status, stated plainly.** The harness, the count and SDM models, the economics study and
the test suite are finished and tested. The full headline run (E01 to E03 with the LSTM and
Transformer) has **not** been run end to end yet. See [Status](#status) for exactly what has and has not been run.

---

## Contents

1. [Why this exists](#why-this-exists)
2. [What you get if you clone it](#what-you-get-if-you-clone-it)
3. [Status](#status)
4. [The experiment](#the-experiment)
5. [Models](#models)
6. [How budgets are matched](#how-budgets-are-matched)
7. [Metrics](#metrics)
8. [Quickstart](#quickstart)
9. [Bring your own model](#bring-your-own-model)
10. [Rescoring: what if compute were cheaper?](#rescoring-what-if-compute-were-cheaper)
11. [The remember-versus-recompute study](#the-remember-versus-recompute-study)
12. [Pre-registration and stop rules](#pre-registration-and-stop-rules)
13. [Repository layout](#repository-layout)
14. [Limitations](#limitations-read-before-interpreting-anything)
15. [Related work](#related-work)
16. [Authorship, citation, license](#authorship-citation-license)

---

## Why this exists

Claims about memory mechanisms are usually made one paper at a time. A sparse distributed
memory is shown to beat a Hopfield network on capacity. A Transformer is shown to learn a
Markov source in context. A new associative memory is shown to forget less on a continual
learning benchmark. Each paper picks its own task, its own definition of "memory size", its
own tuning effort and its own baseline, so the results cannot be laid next to each other.

Three things go wrong in practice:

* **"Memory size" means different things.** Counting weights for a network, rows for a table
  and hard locations for an SDM gives numbers that are not comparable. A model can look
  efficient only because its budget was counted generously.
* **Tuning effort is uneven.** The proposed model is tuned carefully and the baselines get
  defaults. Gains reported against lightly tuned baselines can shrink or disappear when the baselines get the same tuning effort, which is why this repository fixes the effort.
* **The score is noisy.** Accuracy on sampled data mixes the model's error with the
  randomness of the data. With a random source you cannot tell a model that is close to the
  best possible predictor from one that is far from it.

This repository removes each of those three problems for one family of tasks.

* The data source is a variable-order context tree whose conditional probabilities are
  **known exactly**, so every model is scored by its distance to the truth. A score of 1
  means "as good as the oracle" and 0 means "no better than uniform".
* Models are compared at **matched `param_bits`** (the slowly learned store) with `state_bits`
  (fast context) and FLOPs reported separately, so nobody gets to pick the axis that flatters
  them. The accounting rules are written down and the conventions are called conventions.
* Every model and tier gets the **same random-search budget**, on the same tuning seeds, on
  sources disjoint from the evaluation sources.
* The source **drifts**: a phase pattern `(0, 1, 0)` changes a fraction of its rules and then
  restores them. That measures learning, adaptation and what remains of the old regime.

The practical stake is any setting where memory and compute are scarce and the data keeps
changing: on-device and streaming models, continual learning, and caches that sit in front of
expensive computation. In those settings the useful question is rarely "what is the best
model" and usually "what is the best model for this budget, and how fast does it recover when
the world shifts?"

The question is deliberately narrow. Whether a given memory mechanism is better in general is
not something one benchmark can settle. Whether it is better *here*, at this budget, with this
amount of drift, under equal effort, is something you can measure and argue about.

## What you get if you clone it

If you work on associative memories, SDM, Hopfield variants, HTM or spiking sequence models,
you can plug your model in through a four-method interface and get an equal-effort comparison
against count, LSTM and Transformer baselines, with an oracle-normalised score, in roughly an
hour or two of CPU time (an estimate, see Status). You do not have to write the tuning loop, the budget accounting, the drift
schedule or the metrics.

If you review such papers or write them, you get a reference for what "equal tuning, matched
budget, exact oracle" looks like in code, including the places where the accounting is a
convention (`docs/budget_protocol.md`).

If you teach or learn this area, you get small readable implementations (a decayed
variable-order count model with Witten-Bell back-off, a sparse distributed memory, a
cost-ledger simulator) that run in seconds and have tests that pin down their behaviour.

If you only want a sanity check on an idea ("does my memory beat a plain count table on a
drifting source?"), the count baseline alone is a harder opponent than it sounds, and the
harness will tell you quickly.

Concretely, the repository provides:

| piece | what it gives you |
|---|---|
| `muc/tasks.py` | A drifting context-tree source that returns the true P(next token \| past) for every event. Each seed is a new random source. |
| `muc/metrics.py` | KL to truth, performance fraction, calibration, adaptation time, retention of a past regime. |
| `muc/harness.py`, `muc/hpsearch.py` | The evaluation loop and an equal-budget random search with common random numbers. |
| `muc/models/` | Count, generic SDM, LSTM, Transformer (online and in-context), plus stubs for HTM and a rank-order or spiking SDM. |
| `check_adapter` | A contract test your model must pass before it is run, so a broken adapter fails fast. |
| `muc/rescore.py` | Re-ranks finished results under any price of compute, with the Pareto frontier. |
| `muc/economics/` | A pure-Python cost-ledger study of retrieve, verify, recompute and refresh policies under drift. |
| `docs/` | Pre-registered hypotheses, budget protocol, a deviations log and a negative-results file. |
| `tests/` | About 80 tests. Count, SDM, metrics, economics and rescoring run without PyTorch. |

## Status

| item | state |
|---|---|
| Harness, source, metrics, tuning, adapter contract | done, tested |
| `count` baseline | done, tested |
| generic `sdm` (this repo's own plain SDM baseline) | done, tested, budget-matched to `count` within about 5 percent per tier |
| `lstm_online`, `transformer_online`, `transformer_incontext` | implemented, contract-tested, **not yet run at full E01 scale** |
| `htm`, `sdm_snn` | **stubs** that raise `NotImplementedError`; the runner skips them. Their docstrings describe the intended design and budget accounting. |
| `muc/economics` (remember versus recompute) | done, 36 tests, results in `docs/economics/results_v0_1.md` |
| `muc/rescore.py` | done, tested |
| Full E01, E02, E03 with all implemented models | **not run.** Estimated at one to two hours on one CPU core; the estimate has not been checked end to end. |

One partial observation exists, and it is not a result because the comparison set is
incomplete (no neural models, because PyTorch was unavailable where it was run). Five
evaluation seeds, standard tuning budget, mean performance fraction p by tier, `count` / `sdm`:

| tier | count | sdm |
|---|---|---|
| tiny | 0.52 | 0.55 |
| small | 0.76 | 0.64 |
| medium | 0.82 | 0.68 |
| large | 0.84 | 0.69 |

The tiny-tier gap is about one standard error, so it does not pass the two-SE rule in
`docs/hypotheses.md`. At the large tier the generic SDM uses about 80 times the FLOPs of
`count` per event because of dense counter decay. Treat this as a smoke test of the harness.
Whether the gradient models change the picture is exactly what the full run is for.

## The experiment

A stream is sampled from a variable-order context-tree source. The next token depends on the
longest matching suffix rule, with order 0 to 4. The source can add noise and irrelevant
distractor tokens (which never enter the rule context, so a good model has to learn to ignore
them). It drifts: phases `(0, 1, 0)` use regime 0, then a regime in which a fraction of the
rules is resampled, then regime 0 again.

Because the source is known, the generator hands back the true conditional distribution for
every event. Everything is scored against it. Each evaluation seed draws a new random source,
not just a new sample from the same one, so the spread across seeds includes source
difficulty.

| experiment | what varies | what it is for |
|---|---|---|
| **E01** | memory budget (four tiers) | accuracy versus budget, the headline figure |
| **E02** | drift severity | adaptation speed and retention under different amounts of change |
| **E03** | distractor and noise levels | how each mechanism degrades under interference |
| **E04** (spec only) | agent chooses a guess k steps ahead | lookahead through a learned model versus direct association; gated on E01 and E02, see `docs/e04_spec.md` |
| **E05** (spec only) | Dyna agent with a swappable memory as its world model, gated on E01 and E02 ||

E02 and E03 tune once on the base task and evaluate every condition without retuning, so a
model that only works at its tuned operating point shows it.

## Models

| name | what it is |
|---|---|
| `count` | Exponentially decayed variable-order counts with Witten-Bell back-off and a capped table. The baseline to beat. |
| `sdm` | Generic Sparse Distributed Memory: fixed random hard addresses, Hamming-nearest activation, per-location token counters, one tuned context length. A plain reference for the SDM family. It is **not** the rank-order or spiking model and must not be reported as such. |
| `lstm_online` | Small LSTM over a token window, one online gradient step per event. |
| `transformer_online` | Small causal Transformer over a token window, one online step per event. |
| `transformer_incontext` | The same Transformer, pretrained on other random sources and then frozen. Pretraining tokens are counted as experience. |
| `htm` | Stub. See `muc/models/htm.py`. |
| `sdm_snn` | Stub. See `muc/models/sdm_snn.py`. This is the slot for a rank-order or spiking SDM. |

## How budgets are matched

Two memory axes are reported separately, because collapsing them into one number is where
most unfair comparisons start.

* `param_bits`: capacity of the slowly learned store (weights, count table, synapses, hard
  addresses and counters).
* `state_bits`: fast context carried between events (token buffer, hidden state).

Beside them: `approx_flops_per_event` (analytic and rough, includes the backward pass when a
model learns) and `experience_events` (stream events plus any pretraining tokens).

Tiers are chosen so that `param_bits` are comparable across models. Check it yourself with
`python experiments/budget_table.py experiments/configs/e01_memory_budget.yaml` (the neural
rows need PyTorch). Figures plot
the **measured** values, not the tier names. For the default context length, the count and SDM
tiers come out as:

| tier | count `param_bits` | sdm `param_bits` |
|---|---|---|
| tiny | 6,784 | about 6.4k to 6.7k |
| small | 27,392 | about 25.6k to 26.6k |
| medium | 110,592 | about 102k to 106k |
| large | 442,368 | about 410k to 426k |

The rules, and the places where a modelling choice affects the number (for example, whether
fixed random token codes count toward SDM capacity), are in `docs/budget_protocol.md`. A
conclusion should be stated for the axes on which it holds.

## Metrics

All metrics derive from the true distribution P of each event.

* **`p`, performance fraction**: `1 - mean KL(P || Q) / mean KL(P || uniform)`. The oracle
  scores 1 and a uniform predictor scores 0. This is the headline number.
* `kl_mean`, `excess_logloss`: distance to the truth in nats, and log-loss above the oracle's.
* `acc_norm`, `agreement`, `ece`: top-1 accuracy relative to the oracle, how often the argmax
  matches the oracle's, and calibration error.
* `adapt_time_i`: events from the start of phase i until the rolling-window p first reaches a
  threshold. Censored at the phase length, and flagged when censored. A self-relative version
  uses 90 percent of the model's own end-of-phase score.
* `retention`, `forgetting`: how well a **frozen copy** does on held-out sequences from the
  old regime after the drift phase, relative to before it.

Full definitions, and the reasoning behind each threshold, are in `docs/hypotheses.md`.

## Quickstart

```bash
git clone https://github.com/joyboseroy/memory-under-constraints
cd memory-under-constraints

# Core install: enough for count, sdm, the economics study, rescoring and most tests
pip install numpy pandas pyyaml matplotlib pytest
pytest -q                                   # about 6 s; the PyTorch-only tests are skipped

# Smoke run of the harness with no PyTorch (under 1 s)
python experiments/run.py experiments/configs/e01_memory_budget.yaml --quick --models count,sdm --tiers tiny --out /tmp/smoke
python analysis/make_figures.py /tmp/smoke

# The economics study (pure Python, a few seconds)
python -m muc.economics all
```

To run the neural models as well:

```bash
pip install torch                           # CPU is enough, no GPU needed
pytest -q                                   # now includes the LSTM and Transformer tests
python experiments/run.py experiments/configs/e01_memory_budget.yaml        # the real E01
python analysis/make_figures.py results/E01_memory_budget
```

`--models`, `--tiers` (both comma-separated) and `--resume` allow partial and interrupted
runs. `--quick` produces a tiny smoke version: **numbers from `--quick` are for checking that
things run, not results.** Measured time per event on one CPU core: count about 0.03 ms, LSTM
1.2 to 1.5 ms, Transformer 2.1 to 2.4 ms. Full E01 is 20 tuning trials by 2 seeds by 3000
events, then 5 seeds by 6000 events, per model and tier.

Figures and tables land beside the results: a performance-versus-budget summary, an
adaptation-versus-retention plot, and `crossovers.txt` listing where one model overtakes
another.

## Bring your own model

Any online sequence predictor can be tested by implementing four methods. This is a complete,
working example (an order-0 model with decayed frequencies):

```python
import numpy as np
from muc.models.base import SeqModel, check_adapter


class DecayedFrequency(SeqModel):
    name = "decayed_frequency"

    def __init__(self, vocab, decay=0.99, seed=0):
        self.V, self.decay = vocab, decay
        self.counts = np.ones(vocab)

    def predict(self):                       # distribution over the next token
        return self.counts / self.counts.sum()

    def update(self, token):                 # observe the truth, then learn
        if self.learning:                    # frozen probes switch this off
            self.counts *= self.decay
            self.counts[token] += 1.0

    def reset_context(self):                 # forget recent context, keep what was learned
        pass

    def budget(self):                        # declare what you use; see budget_protocol.md
        return dict(param_bits=self.V * 32, state_bits=0, approx_flops_per_event=3 * self.V)


check_adapter(lambda: DecayedFrequency(vocab=12), vocab=12)    # returns True or raises
```

`check_adapter` verifies the output shape, that probabilities are finite and sum to one, that
`budget()` reports the three required fields, and that frozen probes are deterministic. Then:

1. Add a branch for your `kind` in `build_model` (`muc/models/__init__.py`), or replace one of
   the `htm` or `sdm_snn` stubs.
2. Add a block to the experiment configs with a hyperparameter search space and four tiers
   whose `param_bits` match the others. Run `experiments/budget_table.py` to check them.
3. Run the experiment. Your model is tuned with the same number of trials, on the same tuning
   seeds, as everything else.

The `sdm` adapter (`muc/models/sdm.py`) is a longer worked example, including how it declares
hard addresses and counters in `param_bits` and why dense decay dominates its FLOPs.

## Rescoring: what if compute were cheaper?

A finished run records, for each model and tier, how far it is from the truth and how much
work it does. Which model you should prefer depends on how expensive compute is, and that is
a property of your deployment, not of the experiment. `muc/rescore.py` turns it into one
explicit parameter, `lam`: the nats of KL error you would give up to save 1000 FLOPs per
event. The preferred model at each `lam` minimises `kl_mean + lam * kFLOPs`.

```bash
python experiments/rescore.py results/E01_memory_budget/results.csv
python experiments/rescore.py results/E01_memory_budget/results.csv --lams 0 0.001 0.01
```

The output has the mean error and compute per (model, tier), the Pareto frontier (nothing on
it is both more accurate and cheaper than another entry), and the preferred model as compute
gets dearer. Run on the partial `count` and `sdm` results above, `count` at the large tier
is preferred at every price up to 0.1 nats per kFLOP, and only at a price of 1 does the
cheaper small tier win. The generic SDM is never on the frontier. Whether that survives the
neural models is open.

FLOPs here are analytic estimates, so a crossover is a place to look, not a conclusion.
Confirm it with wall-clock time before relying on it.

## The remember-versus-recompute study

`muc/economics/` is a separate and much smaller study, written in plain Python with no
dependencies. It asks a different question from the harness: given a world that changes, when
should a system trust a stored answer, check it, refresh it, or recompute it from scratch?

A `Ledger` counts operations (computes, reads, writes, verifies, errors). A `CostModel` assigns
prices to those counts afterwards, so one simulation can be scored under many cost models. Four
policies are compared on a world with a recurrence rate (how often a recent key is asked again)
and a drift rate (how often a key's true answer changes):

* **recompute**: always compute, never touch memory
* **retrieve**: trust the cache; compute and store on a miss
* **verify**: read the cache and pay to check each hit
* **refresh**: trust a hit only if it is younger than a time-to-live

```bash
python -m muc.economics static    # does the simulator agree with the closed-form model?
python -m muc.economics drift     # policy costs across drift and recurrence, plus the written predictions
python -m muc.economics phase     # ASCII regime maps under three cost models
```

What it found (`docs/economics/results_v0_1.md`), with its own pre-registered predictions
R-P1 to R-P5:

* The simulator reproduces the stationary arithmetic (R-H1 holds, largest |z| 2.39). This is
  a control, not a contribution.
* Under default costs the cheapest policy moves from retrieve (no drift) through verify
  (moderate drift) to recompute (heavy drift, low recurrence).
* The map is **not stable across cost models.** The cheapest policy changes in 8 of 15 cells
  when verification costs 40 instead of 5, and in 10 of 15 when an error costs 10 instead of
  100. Any claim about "the right policy" has to name its cost model.
* One written prediction failed (R-P2): at heavy drift and high recurrence, verify ties
  recompute instead of losing to it. It is recorded as failed.

Its limits are stated in the same file: one memory mechanism (an exact cache), a fixed
untuned TTL, and no price for forgetting. It is the first half of the larger question, and
`muc/rescore.py` carries its central idea (store counts, apply prices later) over to the
harness.

## Pre-registration and stop rules

`docs/hypotheses.md` fixes, before serious runs, the hypotheses, the metric definitions, the
seed ranges and the definition of "beats": higher mean p over at least five seeds, by a
margin above two standard errors, after the same tuning budget, **and** surviving the `count`
baseline. Changes after that go in a dated deviations log at the bottom, never as silent
edits. Three deviations are already recorded there, including the one in which the adaptation
threshold was lowered from 0.9 to 0.5 because 0.9 turned out to be unreachable.

The five hypotheses, briefly: **H1** gradient models match or win at large budgets. **H2**
explicit associative or temporal memories win somewhere at low budget or low experience (the
interesting one). **H3** local online updates adapt faster under drift. **H4** distractors
and noise degrade the mechanisms in qualitatively different ways. **H5** no universal winner.

**What would make us stop.** Continue only if at least one of these holds after equal tuning
and against `count`: a crossover between architectures, a surprising scaling relationship, a
qualitatively different failure mode, a clear adaptation-retention tradeoff, or a win where
conventional wisdom predicts a loss. Stop if the gradient models or `count` dominate in
essentially every regime, results are unstable to hyperparameters, budget matching cannot be
defended, differences vanish under proper tuning, or the only interesting result is an
implementation artifact. A negative result is recorded in `docs/negative_results.md` and
counts as a result.

## Repository layout

```
muc/
  tasks.py            drifting context-tree source with exact conditional probabilities
  metrics.py          KL, performance fraction, calibration, adaptation, retention
  harness.py          the evaluation loop (stream, frozen probes, budgets)
  hpsearch.py         equal-budget random search with common random numbers
  rescore.py          re-rank results under any price of compute
  models/             base contract + count, sdm, neural, htm (stub), sdm_snn (stub)
  economics/          remember-versus-recompute study (pure Python)
experiments/
  run.py              run an experiment config
  budget_table.py     print measured budgets for every model and tier
  rescore.py          command-line wrapper for muc/rescore.py
  configs/            e01_memory_budget, e02_drift, e03_interference
analysis/make_figures.py
docs/                 hypotheses, budget protocol, negative results, e04 spec, economics/
tests/                about 80 tests; neural ones skip when PyTorch is missing
results/              output of runs (quick runs are git-ignored)
```

## Limitations (read before interpreting anything)

* **One task family.** The source is a variable-order context tree with exact repeats, and
  that structure favours count-based models by construction. A win for `count` here says
  little about tasks where similarity between contexts matters. A second task family with
  similarity structure is the most useful extension and is not built.
* **The in-context Transformer baseline is weak as shipped.** Its p stayed near 0.4 when
  pretraining was raised tenfold in a smoke check. With a 16-token window there is little
  in-context evidence to estimate a rule. That may be a real finding about the state budget
  or an under-trained baseline. Do not interpret it before checking with a longer window and a
  held-out in-context learning curve.
* **The generic SDM is a plain baseline.** It is one reasonable SDM, tuned like the others. A
  better-designed associative memory could do better, which is the purpose of the `sdm_snn`
  slot.
* **Budget accounting is a convention.** Report conclusions for both memory axes, not the one
  that flatters a model. FLOPs are rough analytic estimates.
* **`count` sees raw tokens, distractors included**, and is not told which to ignore. So does
  every other model.
* **Not run end to end.** The full E01 to E03 with the neural models has not been executed,
  and the time estimate is an extrapolation.
* **No claim about minds.** Nothing here supports statements about human cognition, biology
  or general intelligence. The theory-flavoured framing in `docs/e04_spec.md` is motivation
  for a planned experiment, not a finding.

## Related work

This repository does not introduce new algorithms, and none of its ingredients are new.
What is offered is the protocol that puts them side by side. Links go to the paper or its
DOI. Year and venue are as given on those pages.

**Variable-order sources as a testbed**

* Willems, Shtarkov and Tjalkens. The context-tree weighting method: basic properties.
  *IEEE Transactions on Information Theory*, 1995.
* Grau-Moya, Genewein, Hutter, Orseau, Deletang, Catt, Ruoss, Wenliang, Mattern, Aitchison
  and Veness. Learning Universal Predictors. *ICML 2024*.
  [arXiv:2401.14953](https://arxiv.org/abs/2401.14953). Trains LSTMs and Transformers on
  variable-order Markov sources and compares them with CTW.
* Zhou, Tian and Diggavi. An Information-Theoretic Approach to Understanding Transformers'
  In-Context Learning of Variable-Order Markov Chains. *AISTATS 2026*.
  [PMLR v300](https://proceedings.mlr.press/v300/zhou26b.html).

**Associative and sparse memories**

* Kanerva. *Sparse Distributed Memory*. MIT Press, 1988.
* Alonso and Krichmar. A sparse quantized Hopfield network for online-continual memory.
  *Nature Communications* 15, 3722, 2024.
  [doi:10.1038/s41467-024-46976-4](https://doi.org/10.1038/s41467-024-46976-4).
  Code: [nalonso2/SQHN](https://github.com/nalonso2/SQHN).
* Pourcel, Vu and French. Online Task-free Continual Learning with Dynamic Sparse
  Distributed Memory. *ECCV 2022*.
  Code: [Julien-pour/Dynamic-Sparse-Distributed-Memory](https://github.com/Julien-pour/Dynamic-Sparse-Distributed-Memory).
* Rank-order sparse distributed memory: [arXiv:2607.02967](https://arxiv.org/abs/2607.02967).

**Smoothing used by the count baseline**

* Witten and Bell. The zero-frequency problem: estimating the probabilities of novel events
  in adaptive text compression. *IEEE Transactions on Information Theory*, 1991.

**Trading memory against computation (for the economics study)**

* Keramati, Dezfouli and Piray. Speed/accuracy trade-off between the habitual and the
  goal-directed processes. *PLoS Computational Biology* 7(5): e1002055, 2011.
  [doi:10.1371/journal.pcbi.1002055](https://doi.org/10.1371/journal.pcbi.1002055).
  Arbitration between a cached habit and slower computation, which is the same trade-off as
  retrieve versus recompute.
* Lieder and Griffiths. Resource-rational analysis: understanding human cognition as the
  optimal use of limited computational resources. *Behavioral and Brain Sciences* 43, e1, 2020.
  [doi:10.1017/S0140525X1900061X](https://doi.org/10.1017/S0140525X1900061X).
* Semantic caching for language-model serving is the closest engineering practice to the
  verify and refresh policies. It is mentioned for context and is not cited to a single source.

**What was searched for and not found.** A few quick searches turned up no existing harness
that puts count, LSTM, Transformer, HTM and SDM models behind one adapter contract on a
drifting exact-oracle stream with matched parameter budgets and pre-registered stop rules.
That is a search, not a systematic survey. If you know of one, please open an issue.

## Authorship, citation, license

Written by Joy Bose. The code was developed with assistance from Claude. Treat it like any research code: run the tests, read
the model implementations before relying on them, and check the budget accounting against
your own judgment.

If you use the harness or its results, please cite it using `CITATION.cff`.

MIT license, see `LICENSE`.
