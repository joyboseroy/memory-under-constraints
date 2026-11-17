"""E05 runner. Pure NumPy, no torch. Writes one CSV row per (run, phase), resumable.

    python experiments/run_e05.py --grid main --out results/e05 --workers 2
    python experiments/run_e05.py --grid smoke --out /tmp/e05_smoke
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from muc.agents import ARMS, make_agent, run_agent  # noqa: E402
from muc.envs import SteerableMDP, SteerConfig  # noqa: E402
from muc.tasks import TaskConfig  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(__file__), "configs", "e05_dyna.yaml")
FIELDS = ["grid", "c", "drift_kind", "drift_frac", "tier", "seed", "arm", "n_plan", "phase", "regime", "rate_opt",
          "rate_rand", "reward", "frac_opt", "norm", "norm_last_quarter", "adapt_steps", "censored", "kl", "ece",
          "flops_per_step", "param_bits", "q_table_bits"]


def env_cfg(conf, c, kind, frac):
    e = conf["env"]
    src = TaskConfig(n_main=e["n_main"], n_distract=0, max_order=e["max_order"], n_rules=e["n_rules"],
                     concentration=e["concentration"], noise=e["noise"], distract_prob=0.0)
    return SteerConfig(source=src, c=c, reward_density=e["reward_density"], gamma=e["gamma"], drift_kind=kind,
                       drift_frac=frac, phases=tuple(e["phases"]), phase_len=e["phase_len"])


def jobs(conf, grid_name, only_arm=None):
    g = conf["grid"][grid_name]
    tier = "medium" if grid_name == "tier" else "small"
    arms = g.get("arms", [a for a in ARMS if a != "q_learning"])
    out = []
    for c, kind, frac in itertools.product(g["c"], g["drift_kind"], g["drift_frac"]):
        for i in range(g.get("n_seeds", conf["eval"]["n_seeds"])):
            seed = conf["eval"]["seed_base"] + i
            runs = []
            if grid_name != "tier":
                runs.append(("q_learning", 0))
            runs += [(a, n) for n in g["n_plan"] for a in arms]
            if only_arm:
                runs = [(a, n) for a, n in runs if a == only_arm]
            out.append(dict(conf=conf, grid=grid_name, c=c, kind=kind, frac=frac, tier=tier, seed=seed, runs=runs))
    return out


def run_job(job):
    conf = job["conf"]
    mdp = SteerableMDP(env_cfg(conf, job["c"], job["kind"], job["frac"]), job["seed"])
    rows = []
    for arm, n_plan in job["runs"]:
        ag = make_agent(arm, mdp, n_plan, job["tier"], seed=job["seed"], **conf["agent"])
        res = run_agent(ag, mdp, job["seed"], window=conf["eval"]["window"], theta=conf["eval"]["theta"])
        for p in res["phases"]:
            rows.append(dict(grid=job["grid"], c=job["c"], drift_kind=job["kind"], drift_frac=job["frac"],
                             tier=job["tier"], seed=job["seed"], arm=arm, n_plan=n_plan,
                             flops_per_step=res["flops_per_step"], param_bits=res["param_bits"],
                             q_table_bits=res["q_table_bits"], **{k: p[k] for k in p if k in FIELDS}))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main", choices=["main", "severity", "tier", "smoke"])
    ap.add_argument("--out", default="results/e05")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-arm", default=None, help="run a single arm; results go to results_<grid>_<arm>.csv")
    a = ap.parse_args(argv)
    conf = yaml.safe_load(open(CFG_PATH))
    if a.grid == "smoke":
        conf["env"]["phase_len"] = 600
        conf["eval"]["n_seeds"] = 1
        conf["grid"]["smoke"] = dict(c=[0.6], drift_kind=["reward"], drift_frac=[0.5], n_plan=[5])
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_{a.only_arm}" if a.only_arm else ""
    path = os.path.join(a.out, f"results_{a.grid}{suffix}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((float(r["c"]), r["drift_kind"], float(r["drift_frac"]), int(r["seed"])))
    todo = [j for j in jobs(conf, a.grid, a.only_arm) if (j["c"], j["kind"], j["frac"], j["seed"]) not in done]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f, ProcessPoolExecutor(max_workers=a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, rows in enumerate(ex.map(run_job, todo), 1):
            w.writerows(rows)
            f.flush()
            print(f"[{a.grid}] {i}/{len(todo)} jobs written", flush=True)


if __name__ == "__main__":
    main()
