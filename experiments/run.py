"""Run one experiment config.

    python experiments/run.py experiments/configs/e01_memory_budget.yaml
    python experiments/run.py experiments/configs/e01_memory_budget.yaml --quick      # smoke test
    python experiments/run.py CONFIG --models count,lstm_online --tiers tiny,small --resume

Writes results/<name>/{results.csv, tuning.json, config_resolved.yaml, meta.json}.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muc.harness import evaluate  # noqa: E402
from muc.hpsearch import random_search  # noqa: E402
from muc.tasks import TaskConfig  # noqa: E402


def apply_quick(cfg):
    cfg.pop("tiers_only", None)
    cfg["task"]["phase_len"] = 250
    cfg["eval"].update(n_seeds=2, probe_seqs=2, probe_len=100, window=60)
    cfg["tune"].update(n_trials=3, n_seeds=1, phase_len=200)
    for m in cfg["models"].values():
        f = m.setdefault("fixed", {})
        if "pre_steps" in f:
            f.update(pre_steps=100, pre_streams=16, pre_events=300)
        keep = list(m["tiers"])[:2]
        m["tiers"] = {k: m["tiers"][k] for k in keep}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test version")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--tiers", default=None, help="comma-separated subset")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict", action="store_true", help="fail on unimplemented adapters")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.quick:
        cfg = apply_quick(cfg)
    name = cfg["name"] + ("_quick" if args.quick else "")
    out = Path(args.out) if args.out else ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config_resolved.yaml", "w"), sort_keys=False)
    json.dump({"python": platform.python_version(), "numpy": np.__version__,
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out / "meta.json", "w"))

    base_task = TaskConfig.from_dict(cfg["task"])
    grid = cfg.get("grid") or {}
    keys = list(grid)
    points = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])] or [{}]
    ev = cfg["eval"]
    tu = cfg["tune"]
    eval_seeds = [ev["seed_base"] + i for i in range(ev["n_seeds"])]
    tune_seeds = [tu["seed_base"] + i for i in range(tu["n_seeds"])]
    only_models = args.models.split(",") if args.models else None
    only_tiers = args.tiers.split(",") if args.tiers else cfg.get("tiers_only")

    csv = out / "results.csv"
    done = set()
    rows_all = []
    if args.resume and csv.exists():
        old = pd.read_csv(csv)
        rows_all = old.to_dict("records")
        done = set(zip(old["model"], old["tier"], old["grid"]))
    tuning_log = json.load(open(out / "tuning.json")) if (args.resume and (out / "tuning.json").exists()) else {}
    tuned_cache = {k: v["best"] for k, v in tuning_log.items()}

    t_start = time.time()
    for mname, m in cfg["models"].items():
        if only_models and mname not in only_models:
            continue
        for tier, arch in m["tiers"].items():
            if only_tiers and tier not in only_tiers:
                continue
            base_spec = {"kind": m["kind"], **({"mode": m["mode"]} if "mode" in m else {}),
                         **m.get("fixed", {}), **arch}
            for point in points:
                gkey = json.dumps(point, sort_keys=True)
                if (mname, tier, gkey) in done:
                    continue
                task = base_task.replace(**point)
                tune_task = (task if tu.get("on") == "each" else base_task).replace(phase_len=tu["phase_len"])
                tkey = f"{mname}|{tier}|" + (gkey if tu.get("on") == "each" else "default")
                tag = f"{mname}/{tier} {gkey}"
                try:
                    if tkey not in tuned_cache:
                        print(f"[tune] {tag} ({tu['n_trials']} trials)", flush=True)
                        if m.get("space"):
                            best, score, trials = random_search(base_spec, m["space"], tune_task,
                                                                tu["n_trials"], tune_seeds)
                        else:
                            best, score, trials = {}, float("nan"), []
                        tuned_cache[tkey] = best
                        tuning_log[tkey] = {"best": best, "score": score, "trials": trials}
                        json.dump(tuning_log, open(out / "tuning.json", "w"), indent=1)
                    spec = {**base_spec, **tuned_cache[tkey]}
                    print(f"[eval] {tag} hp={tuned_cache[tkey]}", flush=True)
                    res = evaluate(spec, task, eval_seeds, window=ev["window"], p_star=ev["p_star"], rel_star=ev.get("rel_star", 0.9),
                                   probe_seqs=ev["probe_seqs"], probe_len=ev["probe_len"])
                except NotImplementedError as e:
                    if args.strict:
                        raise
                    print(f"[skip] {mname}: {e}", flush=True)
                    break
                for r in res:
                    r.update(experiment=name, model=mname, tier=tier, grid=gkey,
                             hp=json.dumps(tuned_cache[tkey]), **{f"g_{k}": v for k, v in point.items()})
                rows_all.extend(res)
                pd.DataFrame(rows_all).to_csv(csv, index=False)
                med = np.median([r["p_overall"] for r in res])
                print(f"       p_overall median={med:.3f}  elapsed {time.time() - t_start:.0f}s", flush=True)
            else:
                continue
            break  # stub model: skip its remaining tiers
    print(f"done -> {csv}")


if __name__ == "__main__":
    main()
