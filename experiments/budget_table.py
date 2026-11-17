"""Print measured budgets for every model/tier in a config, to check they are comparable.

    python experiments/budget_table.py experiments/configs/e01_memory_budget.yaml
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muc.models import build_model  # noqa: E402
from muc.tasks import TaskConfig  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
task = TaskConfig.from_dict(cfg["task"])
print(f"{'model':24s}{'tier':8s}{'param_bits':>12s}{'state_bits':>12s}{'flops/event':>14s}")
for name, m in cfg["models"].items():
    for tier, arch in m["tiers"].items():
        spec = {"kind": m["kind"], **({"mode": m["mode"]} if "mode" in m else {}), **m.get("fixed", {}), **arch}
        if m.get("mode") == "incontext":
            spec.update(pre_steps=1, pre_streams=2, pre_events=60)  # budget does not depend on pretraining
        try:
            b = build_model(spec, task, 0).budget()
        except NotImplementedError:
            print(f"{name:24s}{tier:8s}{'(adapter not implemented)':>38s}")
            continue
        print(f"{name:24s}{tier:8s}{b['param_bits']:>12,d}{b['state_bits']:>12,d}{b['approx_flops_per_event']:>14,d}")
