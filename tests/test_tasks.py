import numpy as np
from muc.tasks import TaskConfig, generate_stream, build_variants

CFG = TaskConfig(phase_len=1500, distract_prob=0.15, noise=0.1)


def test_rows_are_distributions():
    s = generate_stream(CFG, seed=1)
    assert s.probs.shape == (s.T, CFG.vocab)
    assert np.allclose(s.probs.sum(1), 1.0)
    assert (s.probs >= 0).all()


def test_sampled_nll_matches_true_entropy():
    # law of large numbers: mean -log P[x_t] ~ mean H(P_t), so P really is the sampling law
    s = generate_stream(CFG.replace(phase_len=4000), seed=2)
    nll = -np.log(s.probs[np.arange(s.T), s.tokens]).mean()
    H = -(s.probs * np.log(s.probs + 1e-300)).sum(1).mean()
    assert abs(nll - H) < 0.03


def test_deterministic_for_seed_and_tag():
    a, b = generate_stream(CFG, 3), generate_stream(CFG, 3)
    assert (a.tokens == b.tokens).all() and np.allclose(a.probs, b.probs)
    c = generate_stream(CFG, 3, stream_tag=1)
    assert not (a.tokens == c.tokens).all()
    d = generate_stream(CFG, 4)
    assert not np.allclose(a.probs[:50], d.probs[:50])


def test_drift_changes_rules_and_return_restores_them():
    v = build_variants(CFG.replace(drift_frac=0.5), 5)
    base, drifted = v[0].rules, v[1].rules
    assert base.keys() == drifted.keys()
    changed = sum(not np.allclose(base[k], drifted[k]) for k in base)
    assert 0.3 * len(base) < changed <= len(base)
    s = generate_stream(CFG, 5)
    # phases are (0,1,0): same regime id in phase 0 and 2
    assert s.regime[0] == s.regime[-1] == 0 and s.regime[CFG.phase_len] == 1


def test_distractors_only_in_distractor_alphabet_and_ignored_by_rules():
    s = generate_stream(CFG.replace(distract_prob=0.3, noise=0.0), 6)
    frac = (s.tokens >= CFG.n_main).mean()
    assert 0.25 < frac < 0.35
    # main-token mass is scaled by exactly (1 - distract_prob) in every row
    assert np.allclose(s.probs[:, : CFG.n_main].sum(1), 0.7)


def test_no_distractors_gives_zero_mass_on_distractor_symbols():
    s = generate_stream(CFG.replace(distract_prob=0.0), 7)
    assert np.allclose(s.probs[:, CFG.n_main:], 0.0)
