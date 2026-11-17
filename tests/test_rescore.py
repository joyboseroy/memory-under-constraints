import pandas as pd
import pytest

from muc.rescore import pareto, preferred, report, summarise, sweep


def frame():
    # cheap-and-rough, mid, expensive-and-accurate, and one that is dominated
    rows = []
    for seed in (0, 1, 2):
        rows += [
            dict(model="a", tier="t", kl_mean=0.50 + 0.01 * seed, budget_approx_flops_per_event=1_000, seed=seed),
            dict(model="b", tier="t", kl_mean=0.30 + 0.01 * seed, budget_approx_flops_per_event=10_000, seed=seed),
            dict(model="c", tier="t", kl_mean=0.10 + 0.01 * seed, budget_approx_flops_per_event=100_000, seed=seed),
            dict(model="d", tier="t", kl_mean=0.60 + 0.01 * seed, budget_approx_flops_per_event=50_000, seed=seed),
        ]
    return pd.DataFrame(rows)


def test_summarise_means_and_units():
    s = summarise(frame()).set_index("model")
    assert s.loc["a", "kl_mean"] == pytest.approx(0.51)
    assert s.loc["a", "kflops"] == pytest.approx(1.0)
    assert s.loc["a", "seeds"] == 3
    assert s.loc["a", "kl_sem"] == pytest.approx(0.01 / 3 ** 0.5)


def test_pareto_drops_dominated_row():
    names = list(pareto(summarise(frame())).model)
    assert names == ["a", "b", "c"]


def test_preferred_follows_price_of_compute():
    s = summarise(frame())
    assert preferred(s, 0.0).model == "c"        # compute is free: most accurate wins
    assert preferred(s, 1.0).model == "a"        # compute is very dear: cheapest wins
    assert preferred(s, 0.003).model == "b"      # in between


def test_sweep_never_prefers_a_dominated_row():
    chosen = set(sweep(summarise(frame()), [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]).model)
    assert "d" not in chosen


def test_negative_price_rejected():
    with pytest.raises(ValueError):
        preferred(summarise(frame()), -1.0)


def test_missing_columns_named():
    with pytest.raises(ValueError, match="kl_mean"):
        summarise(pd.DataFrame(dict(model=["a"], tier=["t"], budget_approx_flops_per_event=[1])))


def test_report_mentions_caveat_and_models():
    text = report(frame())
    assert "analytic estimates" in text
    for m in "abc":
        assert f"{m} / t" in text
