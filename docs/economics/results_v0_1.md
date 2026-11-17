# Results, v0.1

Exact cache only. Costs: compute 50, read 1, write 1, verify 5, error 100.
3 seeds, 2000 steps per seed per cell. Reproduce with `python -m muc.economics all`.

## R-H1, static validity: holds

Sixteen cells (compute cost 2, 5, 20, 100; read noise 0, 0.1, 0.3, 0.6; recurrence 0.6):
largest |z| 2.39, no boundary disagreements. A repeat over 80 independent seed
sets gave mean z of -0.1 and 0.24 (sd about 1.0), so no bias. The first version
of the check used a fixed 5 percent tolerance and failed on sampling noise; see
the amendment in hypotheses.md. This only says the simulator does what the
arithmetic says.

## Drift grid

Cheapest policy, `*` marks a runner-up within 2 percent.

    default costs
    recurrence \ drift   0.000   0.002   0.010   0.050   0.200
               0.9       R      V      V      V     V*
               0.5       R      R      V      V      C
               0.2       R     R*      V      V      C

    verify 40 instead of 5
               0.9       R      R      F      F      C
               0.5       R      R      R      F      C
               0.2       R      R      R     C*      C

    error 10 instead of 100
               all 15 cells: R

R retrieve, V verify, F refresh (TTL 20), C recompute.

## Predictions written beforehand

- R-P1 retrieve wins at drift 0: held in all three recurrences.
- R-P2 recompute wins at drift 0.2: failed at recurrence 0.9, where verify ties it
  (49.98 against 50.00). I had assumed almost every hit would be stale. About
  16 percent are still current, because a key can be asked again a few steps
  after it was written, and that is enough to pay for the verification fee.
  A tie, not a reversal, but the prediction is recorded as failed.
- R-P3 verify wins at drift 0.01 for recurrence 0.5 and 0.9: held.
- R-P4 refresh wins nowhere under default costs: held.
- R-P5 refresh wins somewhere when verify costs 40: held, in three cells.

## What this does and does not show

- Verification dominates TTL refresh while it is cheap relative to compute, and
  loses to it when it is not. That is close to arithmetic, not a discovery.
- The regime map is not stable across cost models. With errors costing 10 there
  is no structure at all; retrieve wins everywhere. Best policy changes in 8 of
  15 cells when verify goes from 5 to 40, and in 10 of 15 when error goes from
  100 to 10. Any claim about "the right policy" has to name its cost model.
- R-H2 is true here by construction, as noted in hypotheses.md. It is not
  reported as a finding.
- R-H3 and R-H4 beyond this are untested. There is one memory mechanism in this
  experiment (an exact cache). The associative SDM in `muc/models/sdm.py` is not yet
  plugged into it.

## Limits of this version

- The TTL is fixed at 20 and was not tuned. A tuned TTL could take cells from
  verify under default costs, which would weaken R-P4.
- Repeat intervals follow from drawing uniformly from the last 32 keys, which
  gives a heavy short-interval tail. The tail is what decides the R-P2 tie. A
  different recurrence model could move the boundaries.
- Forgetting is a memory operation with no policy and no price.
- The unbounded cache means load and capacity are not exercised.
