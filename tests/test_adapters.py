"""Runs the adapter contract on the HTM and SDM/SNN adapters once they are implemented.
While they are stubs the tests are skipped, not passed."""
import pytest
from muc.tasks import TaskConfig
from muc.models import build_model, check_adapter

CFG = TaskConfig()


@pytest.mark.parametrize("kind", ["htm", "sdm_snn"])
def test_contract_if_implemented(kind):
    try:
        build_model(dict(kind=kind), CFG, 0)
    except NotImplementedError:
        pytest.skip(f"{kind} adapter is still a stub")
    assert check_adapter(lambda: build_model(dict(kind=kind), CFG, 0), vocab=CFG.vocab)
