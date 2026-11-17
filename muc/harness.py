"""Run a model on a stream, with frozen probes at phase ends, and summarise.

Seed namespaces (disjoint, so tuning never sees evaluation automata):
    eval 1000+  tune 2000+  pretrain 3000+  probes use the eval/tune seed with stream_tag >= 100
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .metrics import perf_fraction, summarize
from .models import build_model
from .tasks import TaskConfig, generate_stream

PROBE_BURN = 20


def probe_p(model, probe_streams) -> float:
    """p of a frozen copy of `model` on held-out sequences (context reset per sequence)."""
    model.clear_cache()
    m = copy.deepcopy(model)
    m.learning = False
    Ps, Qs = [], []
    for ps in probe_streams:
        m.reset_context()
        q = np.zeros_like(ps.probs)
        for t in range(ps.T):
            q[t] = m.predict()
            m.update(int(ps.tokens[t]))
        Ps.append(ps.probs[PROBE_BURN:])
        Qs.append(q[PROBE_BURN:])
    return perf_fraction(np.concatenate(Ps), np.concatenate(Qs))


def run_stream(model, stream, probe_seqs=4, probe_len=300, do_probes=True):
    cfg = stream.cfg
    T, V = stream.T, cfg.vocab
    Q = np.zeros((T, V))
    ends = list(stream.starts[1:]) + [T]
    probes_raw = {}
    probe_sets = {}
    if do_probes:
        for rg in sorted(set(stream.phases)):
            probe_sets[rg] = [generate_stream(cfg, stream.seed, stream_tag=100 + i,
                                              phases=(rg,), phase_len=probe_len)
                              for i in range(probe_seqs)]
    t0 = time.perf_counter()
    boundary = {e - 1: i for i, e in enumerate(ends)}
    wall_probe = 0.0
    for t in range(T):
        Q[t] = model.predict()
        model.update(int(stream.tokens[t]))
        if do_probes and t in boundary:
            tp = time.perf_counter()
            for rg, ps in probe_sets.items():
                probes_raw[(boundary[t], rg)] = probe_p(model, ps)
            wall_probe += time.perf_counter() - tp
    wall = time.perf_counter() - t0 - wall_probe
    return Q, probes_raw, wall


def evaluate(spec, task_cfg: TaskConfig, seeds, window=200, p_star=0.5, rel_star=0.9,
             probe_seqs=4, probe_len=300, do_probes=True):
    """Run `spec` on one stream per seed. Returns a list of flat result dicts."""
    rows = []
    for seed in seeds:
        stream = generate_stream(task_cfg, seed)
        model = build_model(spec, task_cfg, seed)
        Q, probes, wall = run_stream(model, stream, probe_seqs, probe_len, do_probes)
        row = summarize(stream, Q, probes, window=window, p_star=p_star, rel_star=rel_star)
        b = model.budget()
        row.update({f"budget_{k}": v for k, v in b.items()})
        row["experience_events"] = int(stream.T + model.pretrain_tokens)
        row["pretrain_tokens"] = int(model.pretrain_tokens)
        row["wall_ms_per_event"] = 1000.0 * wall / stream.T
        row["seed"] = int(seed)
        rows.append(row)
    return rows
