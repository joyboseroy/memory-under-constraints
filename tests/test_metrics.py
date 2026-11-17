import numpy as np
from muc.metrics import kl_rows, uniform_kl_rows, ece, rolling_perf, perf_fraction, summarize
from muc.tasks import TaskConfig, generate_stream

CFG = TaskConfig(phase_len=600)


def test_oracle_and_uniform_reference_points():
    s = generate_stream(CFG, 1)
    assert abs(perf_fraction(s.probs, s.probs) - 1.0) < 1e-6
    U = np.full_like(s.probs, 1.0 / CFG.vocab)
    assert abs(perf_fraction(s.probs, U)) < 1e-6


def test_summary_of_oracle():
    s = generate_stream(CFG, 2)
    out = summarize(s, s.probs.copy(), window=100)
    assert out["kl_mean"] < 1e-6 and abs(out["excess_logloss"]) < 1e-6
    assert abs(out["acc_norm"] - 1.0) < 1e-9 and out["agreement"] == 1.0
    assert out["adapt_time_0"] <= 100 and not out["adapt_censored_0"]
    assert out["adapt_rel_time_0"] <= 100


def test_uniform_predictor_never_adapts():
    s = generate_stream(CFG, 3)
    out = summarize(s, np.full_like(s.probs, 1.0 / CFG.vocab), window=100)
    assert out["adapt_censored_0"] and out["adapt_time_0"] == CFG.phase_len


def test_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, 20000)
    correct = rng.random(20000) < conf
    assert ece(conf, correct) < 0.02
    assert ece(np.full(20000, 0.99), rng.random(20000) < 0.5) > 0.4


def test_rolling_perf_matches_direct():
    s = generate_stream(CFG, 4)
    Q = 0.5 * s.probs + 0.5 / CFG.vocab
    kl, klu = kl_rows(s.probs, Q), uniform_kl_rows(s.probs)
    r = rolling_perf(kl, klu, 50)
    direct = 1 - kl[100:150].mean() / klu[100:150].mean()
    assert abs(r[149] - direct) < 1e-9 and np.isnan(r[10])
