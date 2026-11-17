from __future__ import annotations

from ..tasks import TaskConfig
from .base import SeqModel, check_adapter  # noqa: F401


def build_model(spec: dict, task_cfg: TaskConfig, seed: int) -> SeqModel:
    """spec['kind'] in {count, sdm, lstm, transformer, htm, sdm_snn}; remaining keys are the
    architecture (from the tier) merged with the tuned hyperparameters."""
    kind = spec["kind"]
    V = task_cfg.vocab
    if kind == "sdm":
        from .sdm import SDMModel
        return SDMModel(V, seed=seed, **{k: v for k, v in spec.items() if k != "kind"})
    if kind == "count":
        from .count import DecayedVarOrderCounts
        return DecayedVarOrderCounts(V, seed=seed, **{k: v for k, v in spec.items() if k != "kind"})
    if kind in ("lstm", "transformer"):
        from .neural import build_neural
        return build_neural(kind, V, task_cfg, spec, seed)
    if kind == "htm":
        from .htm import HTMModel
        return HTMModel(V, seed=seed, **{k: v for k, v in spec.items() if k != "kind"})
    if kind == "sdm_snn":
        from .sdm_snn import SDMSNNModel
        return SDMSNNModel(V, seed=seed, **{k: v for k, v in spec.items() if k != "kind"})
    raise ValueError(f"unknown model kind {kind!r}")
