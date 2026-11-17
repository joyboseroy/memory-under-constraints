# Hypotheses

Written before any experiment was run. Predictions below were fixed first and
the results file compares against them.

## R-H1, static validity (sanity check, not a finding)

In a stationary world, the simulated per-query cost of the retrieve policy and
of the recompute policy matches the analytic model within 5 percent, and the
sign of the difference (which policy is cheaper) matches wherever the analytic
difference is larger than 5 percent of the recompute cost.

Analytic model, per query, with recurrence rho, retrieval reliability q,
and costs c:

    recompute:  c_compute
    retrieve:   c_read + rho (1 - q) c_error + (1 - rho)(c_compute + c_write)

Retrieval wins when the second expression is smaller than the first.

Failure of R-H1 means the simulator is wrong, and nothing else counts until it is
fixed.

Amendment, made after the first run: the fixed 5 percent tolerance failed in
two cells (6.3 and 5.3 percent off) even though the simulator was fine. With an
error cost of 100 and about 6 percent errors, the mean of 9000 queries moves by
several percent from sampling noise alone. The criterion is now a z-score
against the analytic standard error (|z| at most 3 in every cell) plus the
boundary sign check. To confirm the simulator is unbiased and the standard
error is right, the check was repeated over 80 independent seed sets at two
noise levels: mean z was -0.1 and 0.24 with standard deviation about 1.0.

## R-H2, dynamic deviation

Under drift, the stationary analytic model mispredicts the cost of the retrieve
policy by more than 5 percent in some region of (drift, recurrence).

Caveat recorded in advance: for an exact cache this is true by construction,
because the stationary model ignores staleness. R-H2 is only informative against
a drift-aware model that knows the drift rate and the age of each entry.
Building that model for the exact cache is open work, and R-H2 should not be
reported as a result until it exists.

## R-H3, architecture-dependent dynamics

Different memory mechanisms have statistically distinguishable optimal policies
under at least one combination of drift, interference, recurrence and load.
Not testable in v0.1, since only exact caches exist. This is the claim the
paper depends on.

## R-H4, value of verification and refresh

Verification and refresh policies beat retrieve-only and recompute-only in a
non-trivial region of the dynamic world.

## Predictions for the v0.1 drift experiment

Default costs: compute 50, read 1, write 1, verify 5, error 100.
Recurrence in {0.2, 0.5, 0.9}, per-step per-key drift in {0, 0.002, 0.01, 0.05, 0.2},
exact cache with no noise and no capacity limit, repeats drawn from the last 32 new keys.

R-P1. With drift 0, retrieve is the cheapest policy at every recurrence, including 0.2.
R-P2. With drift 0.2, recompute is the cheapest policy at every recurrence.
R-P3. In between (around 0.01), verify is cheapest at recurrence 0.5 and 0.9.
R-P4. TTL refresh is the cheapest policy nowhere under the default costs, because
    verification (5) is much cheaper than recomputation (50) and dominates it.
R-P5. Under a cost model with verify 40 (close to compute), refresh wins in some
    middle-drift cell. If it also fails to win there, R-H4 for these policies
    is weak.

A prediction that fails is reported as failed.
