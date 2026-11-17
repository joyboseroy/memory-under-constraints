Runs: 1218 arm-runs over 54 (cell, seed) MDPs in the main and severity grids.

### Normalised return by arm and drift kind (n_plan = 20, mean over c, seeds and phases)

| arm | dynamics | reward | both | all |
|---|---|---|---|---|
| q_learning | 0.387 | 0.332 | 0.331 | 0.350 |
| replay | 0.454 | 0.416 | 0.402 | 0.424 |
| replay_sa | 0.500 | 0.487 | 0.479 | 0.489 |
| dyna_count | 0.529 | 0.505 | 0.519 | 0.518 |
| dyna_count_noaction | 0.175 | 0.174 | 0.180 | 0.176 |
| dyna_sdm | 0.456 | 0.444 | 0.432 | 0.444 |
| dyna_oracle | 0.568 | 0.552 | 0.554 | 0.558 |
| dyna_oracle_full | 0.769 | 0.722 | 0.729 | 0.740 |

### Normalised return by control strength c (n_plan = 20, all drift kinds)

| arm | c = 0.3 | c = 0.6 | c = 0.9 |
|---|---|---|---|
| q_learning | 0.256 | 0.378 | 0.416 |
| replay | 0.324 | 0.458 | 0.489 |
| replay_sa | 0.437 | 0.482 | 0.546 |
| dyna_count | 0.473 | 0.530 | 0.550 |
| dyna_count_noaction | 0.092 | 0.200 | 0.237 |
| dyna_sdm | 0.385 | 0.482 | 0.465 |
| dyna_oracle | 0.597 | 0.543 | 0.535 |
| dyna_oracle_full | 0.761 | 0.769 | 0.691 |

### Normalised return versus planning steps (main grid, all c and kinds)

| arm | n_plan 5 | n_plan 20 | n_plan 50 |
|---|---|---|---|
| q_learning | 0.350 | 0.350 | 0.350 |
| replay | 0.385 | 0.424 | 0.453 |
| replay_sa | 0.492 | 0.489 | 0.504 |
| dyna_count | 0.503 | 0.518 | 0.521 |
| dyna_count_noaction | 0.263 | 0.176 | 0.096 |
| dyna_sdm | 0.428 | 0.444 | 0.487 |
| dyna_oracle | 0.521 | 0.558 | 0.562 |
| dyna_oracle_full | 0.692 | 0.740 | 0.743 |

### Steps to reach a rolling normalised score of 0.5 after each phase starts (n_plan = 20, mean over main grid; capped at 2000, so a value near 2000 means it often did not get there)

| arm | phase 0 (initial) | phase 1 (after drift) | phase 2 (back to regime 0) |
|---|---|---|---|
| q_learning | 1259 | 1312 | 650 |
| replay | 1284 | 859 | 360 |
| replay_sa | 1225 | 609 | 286 |
| dyna_count | 1100 | 462 | 280 |
| dyna_count_noaction | 1884 | 1564 | 1552 |
| dyna_sdm | 1230 | 628 | 390 |
| dyna_oracle | 1086 | 407 | 271 |
| dyna_oracle_full | 312 | 249 | 227 |

### Hypothesis checks (paired differences within the same MDP, mean ± 1.96 SE)

| hypothesis | difference | score | mean ± 1.96 SE | units |
|---|---|---|---|---|
| E5-H1 reward drift | dyna_count − q_learning | 1 | +0.287 ± 0.081 | 15 |
| E5-H1 reward drift | dyna_count − replay | 1 | +0.120 ± 0.059 | 15 |
| E5-H1 reward drift | dyna_count − replay_sa | 1 | +0.034 ± 0.049 | 15 |
| E5-H1 reward drift | dyna_oracle_full − replay_sa | 1 | +0.234 ± 0.065 | 15 |
| E5-H2 dynamics drift | dyna_count − dyna_sdm | 1 | +0.069 ± 0.062 | 15 |
| E5-H2 dynamics drift | dyna_count − q_learning | 1 | +0.183 ± 0.055 | 15 |
| E5-H2 dynamics drift | dyna_count − replay | 1 | +0.104 ± 0.045 | 15 |
| E5-H4 small n_plan (5) | dyna_count − replay | norm | +0.117 ± 0.027 | 45 |
| E5-H4 n_plan 20 | dyna_count − replay | norm | +0.094 ± 0.025 | 45 |
| E5-H4 large n_plan (50) | dyna_count − replay | norm | +0.068 ± 0.029 | 45 |
| E5-H4b c = 0.3 | dyna_count − dyna_count_noaction | norm | +0.381 ± 0.032 | 15 |
| E5-H4b c = 0.6 | dyna_count − dyna_count_noaction | norm | +0.331 ± 0.029 | 15 |
| E5-H4b c = 0.9 | dyna_count − dyna_count_noaction | norm | +0.313 ± 0.033 | 15 |
| E5-H5 null, all drift | dyna_count − dyna_sdm | norm | +0.074 ± 0.022 | 45 |
| E5-H5 null, all drift | dyna_count − replay_sa | norm | +0.029 ± 0.019 | 45 |
| E5-H5 null, all drift | dyna_count − replay | norm | +0.094 ± 0.025 | 45 |
| E5-H5 null, all drift | dyna_sdm − replay | norm | +0.020 ± 0.032 | 45 |
| coverage headroom | dyna_oracle_full − dyna_oracle | norm | +0.182 ± 0.021 | 45 |
| model headroom | dyna_oracle − dyna_count | norm | +0.040 ± 0.025 | 45 |

### Model quality versus planning quality (one point per model-arm run; phase 0 to 2 pooled)

| arm | mean KL(true, model) | mean ECE | mean score | Spearman(KL, score) within arm |
|---|---|---|---|---|
| dyna_count | 0.177 | 0.069 | 0.524 | -0.30 |
| dyna_count_noaction | 0.610 | 0.039 | 0.178 | +0.66 |
| dyna_sdm | 0.560 | 0.195 | 0.444 | +0.21 |
| replay_sa | 0.507 | n/a | 0.489 | -0.62 |

Across all four model arms pooled, Spearman(KL, score) = -0.28.

### Cost: approximate FLOPs per real step, memory, and gain per FLOP (main grid, n_plan = 20)

| arm | FLOPs / real step | model param_bits | score | score gain per 1000 FLOPs |
|---|---|---|---|---|
| q_learning | 16 | 0 | 0.350 | n/a |
| replay | 336 | 22005 | 0.424 | 0.231 |
| replay_sa | 336 | 32768 | 0.489 | 0.433 |
| dyna_count | 2096 | 22016 | 0.518 | 0.081 |
| dyna_count_noaction | 2096 | 21632 | 0.176 | -0.083 |
| dyna_sdm | 117196 | 21920 | 0.444 | 0.001 |
| dyna_oracle | 336 | 0 | 0.558 | 0.650 |
| dyna_oracle_full | 336 | 0 | 0.740 | 1.220 |

### Capacity tier (c = 0.6, n_plan = 20): small versus medium

| arm | score small | score medium | param_bits small | param_bits medium |
|---|---|---|---|---|
| replay | 0.459 | 0.486 | 22005 | 176115 |
| dyna_count | 0.527 | 0.546 | 22016 | 176128 |
| dyna_sdm | 0.486 | 0.501 | 21920 | 176000 |

### Drift severity at c = 0.6 (n_plan = 20): drift_frac 0.5 versus 1.0

| arm | phase-1 score, drift_frac 0.5 | phase-1 score, drift_frac 1.0 |
|---|---|---|
| q_learning | 0.285 | 0.155 |
| replay | 0.455 | 0.496 |
| replay_sa | 0.483 | 0.564 |
| dyna_count | 0.576 | 0.566 |
| dyna_count_noaction | 0.210 | 0.215 |
| dyna_sdm | 0.503 | 0.583 |
| dyna_oracle | 0.572 | 0.632 |
| dyna_oracle_full | 0.773 | 0.759 |
