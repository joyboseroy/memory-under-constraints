# Hypotheses and metric definitions (pre-registration)

Commit this file before running any serious sweep. Changes after that go in the deviations
log at the bottom, with a date and the reason, never as silent edits.

## Question

When memory capacity and learning experience are constrained, which mechanisms are most
effective for continual sequence prediction under distribution drift?

## Hypotheses

* **H1 scale.** At large budgets, gradient-trained sequence models match or beat `count`,
  HTM and SDM/SNN on log-loss.
* **H2 low-resource crossover.** At small `param_bits` or small `experience_events`, an
  explicit temporal/associative memory model (count, HTM, SDM/SNN) beats the Transformer and
  LSTM on at least some regime. This is the interesting one.
* **H3 drift.** Under drift, models with local online memory updates adapt in fewer events
  than gradient models, at equal tuning.
* **H4 interference.** Distractors and noise degrade the memory mechanisms in qualitatively
  different ways (the order of degradation differs, not only its size).
* **H5 no universal winner.** Rankings depend on regime and budget. (The expected outcome.)

"Beats" means higher mean p over >= 5 seeds with a margin above 2 standard errors, after the
same hyperparameter-search budget, and surviving the `count` baseline.

## Metrics

All metrics derive from the true conditional distribution P_t of each event.

* `kl_mean`: mean KL(P_t || Q_t). `excess_logloss`: model log-loss minus oracle log-loss.
* `p` (performance fraction) = 1 - mean KL / mean KL(P || uniform). Oracle = 1, uniform = 0.
* `acc_norm`: top-1 accuracy divided by the oracle's top-1 accuracy. `agreement`: how often
  the model's argmax equals the oracle's argmax. `ece`: 10-bin calibration error of top-1.
* `adapt_time_i`: events from the start of phase i until the rolling-window p (window w)
  first reaches p_star = 0.5. `adapt_rel_time_i`: same with threshold 0.9 x the model's own
  end-of-phase p (only if that p exceeds 0.1). Both are censored at the phase length;
  `adapt_censored_i` flags it. `adapt_time_0` is learning from scratch.
* `retention`: p of a frozen copy on held-out regime-0 sequences after phase 1, divided by
  the same after phase 0. `forgetting` is the difference. `relearn_ratio`: adapt time in the
  returning phase over adapt time in phase 0.
* **Adaptation-retention tradeoff**: the pair (`adapt_rel_time_drift`, `retention`) per model
  and tier, plotted in `fig_adapt_retention.png`.

## Protocol

* Seeds: eval 1000+, tune 2000+, pretrain 3000+. Each eval seed is a new random source.
* Tuning: random search, identical number of trials, tuning seeds and objective (mean KL over
  the drifting stream) for every (model, tier). Spaces are in the config files. Where two
  models share a space they receive the same hyperparameter draws (common random numbers).
* E02 and E03 tune once on the base task and evaluate on every condition without retuning.
* >= 5 eval seeds; report mean and sd and the per-tier winner with its margin in SE.

## Deviations log

* Before any sweep: `p_star` changed from 0.9 to 0.5. A smoke test showed 0.9 is unreachable
  in 2000-event phases (even a very large count model on 30k stationary events reaches only
  p = 0.88), so every run was censored. Added the self-relative `adapt_rel_time` metric.
* 2026-11-17, before any serious sweep: added a generic `sdm` model (`muc/models/sdm.py`) and
  an `sdm` block to the E01 to E03 configs. Tier = number of hard locations, chosen so
  `param_bits` match the count tiers within about 5 percent; `k_ctx` (2, 3 or 4) is tuned like
  the other hyperparameters. The H1 to H5 wording above is unchanged. "SDM/SNN" there still
  means the `sdm_snn` slot, which remains a stub for the rank-order or spiking model. The
  generic `sdm` is reported as its own row and must not be presented as the SDM/SNN result.
* 2026-11-17, before any serious sweep: merged the separate `memory-regimes` study into this
  repository as `muc/economics/` (docs in `docs/economics/`) and added `muc/rescore.py`. The
  H1 to H5 wording above is unchanged. The merged study keeps its own pre-registered
  hypotheses, renamed R-H1 to R-H4 and predictions R-P1 to R-P5 so they cannot be confused
  with these. `muc/rescore.py` only re-reads finished results under different prices for
  compute; it does not change any metric, tuning rule or budget tier.
