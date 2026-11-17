# Question

When the world changes while an online memory is running, when should the
system retrieve, verify, recompute, refresh or forget?

This is an empirical study of memory management under environmental change.
It is not a theory of memory and not a cognitive architecture.

## Why this question

Remembering versus recomputing is old (caching, memoisation), and for a
stationary world the answer is arithmetic: given how reliable a retrieval is
and what each operation costs, the decision boundary follows directly. That
part is the control experiment, not the contribution.

What the arithmetic does not settle is what happens when stored answers go
stale, when writes interfere with older memories, and when retrieval is by
similarity rather than exact key. Semantic caching for LLM systems deals with
this in practice, and associative-memory papers compare capacity and noise
robustness, but the combination (online updates, drift, an explicit choice
among retrieve, verify, recompute, refresh and forget, and several memory
mechanisms under one cost ledger) is what this part of the repository isolates.

## Variables

- drift: how quickly the true answer for a key changes
- recurrence: how often a recently seen key is asked again
- noise: how often a retrieval returns a wrong value
- load: how many items compete for the memory (needs a bounded memory)
- interference: damage to old entries caused by new writes (associative memories only)
- costs: compute, read, write, verify, error, kept as counts so they can be reweighted afterwards

## Secondary question

Does a conclusion about which policy or memory is best survive a change in the
cost model? Every run stores operation counts, so the same run can be scored
under several weightings at no extra cost.

## Scope for v0.1

Exact cache, LRU capacity, an oracle memory, and the four policies. No neural
memories yet. The Hopfield, SDM and rank-order SDM memories come only after the
simulator reproduces the analytic baseline (R-H1).
