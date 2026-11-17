# Documentation index

Read these in this order if you want to judge the results rather than run them.

| file | what it settles |
|---|---|
| `hypotheses.md` | The question, H1 to H5, metric definitions, tuning protocol, and a dated log of every deviation. Fixed before serious runs. |
| `budget_protocol.md` | What `param_bits`, `state_bits` and FLOPs mean for each model, and where the accounting is a convention. |
| `negative_results.md` | Anything that failed, tied or came out boring, plus smoke-test observations that are not results. |
| `e04_spec.md` | A planned experiment (lookahead through a learned model versus direct association). Not run. |
| `e05_dyna_spec.md` | A planned experiment (a Dyna agent whose learned world model is the swappable memory). Supersedes the E04b sketch. Gated on E01 and E02. Not run. |
| `e05_results_v0_1.md` | First-pass E05 results (count and SDM memories, reduced grid), with the deviations and what the pass does not show. |
| `economics/` | A separate, fully pure-Python study of when a memory should retrieve, verify, recompute or refresh. Start with `economics/question.md`. |

`economics/` has its own hypotheses (R-H1 to R-H4), predictions (R-P1 to R-P5), stop rules
(`kill_criteria.md`) and results (`results_v0_1.md`). It shares the cost-ledger idea with
`muc/rescore.py`, which applies prices to the harness results.
