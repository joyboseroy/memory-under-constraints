import numpy as np
from muc.tasks import TaskConfig
from muc.harness import evaluate

CFG = TaskConfig(phase_len=200, n_rules=8)


def _run(spec, seeds=(1000, 1001)):
    return evaluate(spec, CFG, seeds, window=50, probe_seqs=2, probe_len=100)


def _same(a, b):
    for ra, rb in zip(a, b):
        for k, v in ra.items():
            if k.startswith("wall"):
                continue
            if isinstance(v, float):
                assert np.isclose(v, rb[k], equal_nan=True), k
            else:
                assert v == rb[k], k


def test_count_reproducible():
    spec = dict(kind="count", max_order=3, decay=0.99, max_entries=128)
    _same(_run(spec), _run(spec))


def test_transformer_reproducible():
    spec = dict(kind="transformer", d_model=8, n_heads=2, window=8, lr=0.05)
    _same(_run(spec, (1000,)), _run(spec, (1000,)))


def test_lstm_reproducible():
    spec = dict(kind="lstm", emb=8, hidden=8, window=8, lr=0.05)
    _same(_run(spec, (1000,)), _run(spec, (1000,)))


def test_seed_changes_result():
    spec = dict(kind="count", max_order=3, decay=0.99, max_entries=128)
    a, b = _run(spec, (1000,)), _run(spec, (1001,))
    assert a[0]["kl_mean"] != b[0]["kl_mean"]
