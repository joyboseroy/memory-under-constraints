import numpy as np
import pytest
from muc.tasks import TaskConfig, generate_stream
from muc.models import build_model, check_adapter
from muc.harness import evaluate

CFG = TaskConfig(phase_len=300, n_rules=10)
V = CFG.vocab

SPECS = {
    "count": dict(kind="count", max_order=3, decay=0.995, max_entries=256),
    "lstm": dict(kind="lstm", emb=8, hidden=8, window=8, lr=0.05),
    "transformer": dict(kind="transformer", d_model=8, n_heads=2, n_layers=1, window=8, lr=0.05),
}


@pytest.mark.parametrize("name", list(SPECS))
def test_adapter_contract(name):
    assert check_adapter(lambda: build_model(SPECS[name], CFG, 0), vocab=V)


def test_count_model_learns_and_respects_entry_cap():
    s = generate_stream(CFG.replace(phase_len=1500, phases=(0,)), 1)
    m = build_model(dict(SPECS["count"], max_entries=64, decay=0.9999), CFG, 0)
    q = np.zeros_like(s.probs)
    for t in range(s.T):
        q[t] = m.predict()
        m.update(int(s.tokens[t]))
    assert len(m.table) <= 64
    from muc.metrics import perf_fraction
    assert perf_fraction(s.probs[-500:], q[-500:]) > 0.5  # far above uniform


def test_frozen_copy_does_not_change_original():
    m = build_model(SPECS["count"], CFG, 0)
    for x in [0, 1, 2, 0, 1, 2]:
        m.predict(); m.update(x)
    before = {k: v[0].copy() for k, v in m.table.items()}
    import copy
    c = copy.deepcopy(m); c.learning = False
    for x in [3, 3, 3, 3]:
        c.predict(); c.update(x)
    assert all((before[k] == m.table[k][0]).all() for k in before) and len(before) == len(m.table)
    assert len(c.table) == len(m.table)  # frozen copy does not grow its table either


def test_budget_fields_present():
    for name, spec in SPECS.items():
        b = build_model(spec, CFG, 0).budget()
        assert b["param_bits"] > 0 and b["state_bits"] > 0 and b["approx_flops_per_event"] > 0
