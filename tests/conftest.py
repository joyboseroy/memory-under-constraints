"""Skip the tests that need PyTorch when it is not installed.

The count, SDM and economics code is pure NumPy or pure Python, so most of the
suite runs anywhere. Only the LSTM and Transformer adapters need torch.
Everything is still collected, so a missing optional dependency shows up as
"skipped" rather than silently shrinking the suite.
"""
import importlib.util

import pytest

HAVE_TORCH = importlib.util.find_spec("torch") is not None

# Test ids that build a neural model. Matched as substrings of the pytest node id.
NEEDS_TORCH = (
    "test_adapter_contract[lstm]",
    "test_adapter_contract[transformer]",
    "test_budget_fields_present",
    "test_transformer_reproducible",
    "test_lstm_reproducible",
)


def pytest_collection_modifyitems(config, items):
    if HAVE_TORCH:
        return
    skip = pytest.mark.skip(reason="PyTorch not installed (pip install torch to run the neural-model tests)")
    for item in items:
        if any(key in item.nodeid for key in NEEDS_TORCH):
            item.add_marker(skip)
