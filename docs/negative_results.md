# Negative and null results

Record here anything that did not work or came out boring, with the config and commit it
came from. A clean null (for example "the gradient models dominate everywhere") is a result.

| date | experiment | what happened | config / commit |
|---|---|---|---|
| (none yet) | | | |

## Observations from the scaffold smoke test (not experimental results)

* `p_star = 0.9` was unreachable; see the deviations log in `hypotheses.md`.
* Increasing pretraining of `transformer_incontext` by 10x did not move p off about 0.4 on one
  seed at one tier. Unexplained; see Known limitations in the README.
* Partial E01 with only `count` and the generic `sdm` (torch unavailable where this was run, so
  no neural models; 5 eval seeds, standard tuning budget). Mean p by tier, count / sdm:
  tiny 0.52 / 0.55, small 0.76 / 0.64, medium 0.82 / 0.68, large 0.84 / 0.69. The tiny-tier gap
  is about 1 standard error, so not a win by the 2 SE rule in `hypotheses.md`. `k_ctx` = 2 was
  selected at every tier. SDM FLOPs per event are about 80 times count at the large tier
  (dense decay). Not an experimental result: the comparison set is incomplete.

## E05 first pass

* **A discarded SDM run.** The first E05 sweep gave the SDM 64 and 512 hard locations, about half the `param_bits` of the
  count model (about 10k and 80k bits against 22k and 176k). That violated the repository's own budget matching. The
  SDM rows were discarded and re-run at 137 and 1100 locations, which match the count model within 0.5 percent. The
  overall count-minus-SDM gap was +0.077 +/- 0.022 with half the budget and +0.074 +/- 0.022 once matched, so the
  budget error did not drive that comparison, but the table could not have been defended with it. A test
  (`test_tiers_match_param_bits_across_model_arms`) now pins the match.
* **A perfect model that did not help.** With the spec's visited-only planning, an oracle dynamics model reached only
  about 0.56 normalised return, because many (s, a) pairs are never visited and so never backed up. This was found on one
  seed before the sweep and led to the `dyna_oracle_full` arm. It is a property of the planning rule, and any E05 result
  that compares memories under that rule is bounded by it.
* **E5-H4 was contradicted.** The expectation that a Dyna advantage over replay shows only at large n_plan did not hold:
  the advantage was largest at n_plan 5 (+0.117) and shrank to +0.068 at n_plan 50 as replay caught up.
* **Action-blind planning is worse than no planning.** `dyna_count_noaction` scored 0.18 against 0.35 for Q-learning and
  degraded as n_plan grew. Not a surprise in hindsight, but it shows that a dynamics memory with the wrong interface is
  harmful rather than merely useless.
