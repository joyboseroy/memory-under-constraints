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
