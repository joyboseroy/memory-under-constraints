import copy

import numpy as np
import pytest

from muc.harness import evaluate
from muc.metrics import perf_fraction
from muc.models import build_model, check_adapter
from muc.models.sdm import COUNTER_BITS, SDMModel
from muc.tasks import TaskConfig, generate_stream

CFG = TaskConfig(phase_len=300, n_rules=10)
V = CFG.vocab


def spec(**kw):
    base = dict(kind="sdm", n_locations=64, k_ctx=3, active_frac=0.1, decay=0.999, prior=1.0)
    base.update(kw)
    return base


def run_p(model, seed=1000, T=3000, tail=1000):
    s = generate_stream(CFG.replace(phase_len=T, phases=(0,)), seed)
    q = np.zeros_like(s.probs)
    for t in range(s.T):
        q[t] = model.predict()
        model.update(int(s.tokens[t]))
    return perf_fraction(s.probs[-tail:], q[-tail:])


def test_adapter_contract():
    assert check_adapter(lambda: build_model(spec(), CFG, 0), vocab=V)


def test_predictions_are_distributions_from_the_start():
    m = build_model(spec(), CFG, 0)
    q = m.predict()
    assert np.allclose(q, 1.0 / V)  # empty memory is uniform
    assert abs(q.sum() - 1.0) < 1e-9


def test_learns_far_above_uniform():
    assert run_p(build_model(spec(n_locations=256, k_ctx=2, active_frac=0.05), CFG, 0)) > 0.4


def test_more_locations_help_when_contexts_outnumber_locations():
    small = np.mean([run_p(build_model(spec(n_locations=16, k_ctx=2, active_frac=0.25), CFG, s), seed=1000 + s) for s in range(3)])
    big = np.mean([run_p(build_model(spec(n_locations=256, k_ctx=2, active_frac=0.05), CFG, s), seed=1000 + s) for s in range(3)])
    assert big > small + 0.05


def test_frozen_copy_does_not_change_counters_or_original():
    m = build_model(spec(), CFG, 0)
    for x in [0, 1, 2, 0, 1, 2, 3, 3]:
        m.predict()
        m.update(x)
    before = m.counts.copy()
    c = copy.deepcopy(m)
    c.learning = False
    for x in [5, 5, 5, 5, 5]:
        c.predict()
        c.update(x)
    assert np.array_equal(c.counts, before)
    assert np.array_equal(m.counts, before)


def test_write_goes_to_locations_active_for_the_previous_context():
    m = build_model(spec(), CFG, 0)
    m.reset_context()
    m.predict()
    idx = m._active().copy()
    m.update(4)
    touched = np.flatnonzero(m.counts[:, 4])
    assert set(touched) == set(idx.tolist())


def test_decay_applies_to_every_counter_before_the_write():
    m = build_model(spec(decay=0.5), CFG, 0)
    m.counts[:] = 1.0
    idx = m._active().copy()
    m.update(0)
    others = np.setdiff1d(np.arange(m.L), idx)
    assert np.allclose(m.counts[:, 1:], 0.5)
    assert np.allclose(m.counts[idx, 0], 1.5)
    assert np.allclose(m.counts[others, 0], 0.5)


def test_shared_recent_tokens_give_closer_addresses():
    m = SDMModel(V, k_ctx=3, seed=0)
    rng = np.random.default_rng(0)

    def addr(h):
        m.hist = list(h)
        return m._address()

    near, far = [], []
    for _ in range(200):
        a = rng.integers(0, 8, 3)
        b_near = np.array([rng.integers(0, 8), a[1], a[2]])  # same two newest tokens
        b_far = rng.integers(0, 8, 3)
        near.append((addr(a) != addr(b_near)).sum())
        far.append((addr(a) != addr(b_far)).sum())
    assert np.mean(near) < np.mean(far)


def test_budget_matches_formula():
    m = build_model(spec(n_locations=64, k_ctx=3), CFG, 0)
    b = m.budget()
    assert b["param_bits"] == 64 * (3 * 8 + V * COUNTER_BITS)
    assert b["state_bits"] == 3 * 4  # bits_per_token(12) = 4
    assert b["param_bits_used"] == 0
    m.predict(); m.update(1)
    assert m.budget()["param_bits_used"] > 0


def test_reproducible_for_a_seed_and_different_across_seeds():
    a = evaluate(spec(), CFG, (1000,), window=50, probe_seqs=2, probe_len=100)
    b = evaluate(spec(), CFG, (1000,), window=50, probe_seqs=2, probe_len=100)
    for k, v in a[0].items():
        if k.startswith("wall"):
            continue
        assert np.isclose(v, b[0][k], equal_nan=True) if isinstance(v, float) else v == b[0][k], k
    m0, m1 = build_model(spec(), CFG, 0), build_model(spec(), CFG, 1)
    assert not np.array_equal(m0.hard, m1.hard)


def test_bad_arguments():
    with pytest.raises(ValueError):
        SDMModel(V, k_ctx=0)
    with pytest.raises(ValueError):
        SDMModel(V, n_locations=0)
    with pytest.raises(ValueError):
        SDMModel(V, code_bits=2)  # cannot give 13 distinct codes
