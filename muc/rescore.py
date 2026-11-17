"""Rescore finished runs under different prices for compute.

A run of the harness records, per model and tier, how far the model's predictions
are from the exact oracle (`kl_mean`, nats per event) and an analytic estimate of
the work it did (`budget_approx_flops_per_event`). Neither number says which model
to prefer, because that depends on how expensive compute is relative to error.

This module turns that into one explicit parameter, `lam`: the number of nats of
KL error you would give up to save 1000 FLOPs per event. For each `lam` the
preferred (model, tier) minimises

    kl_mean + lam * flops_per_event / 1000

Sweeping `lam` shows where the preferred model changes, without rerunning
anything. It is the same idea as the cost ledger in `muc.economics`: store the
counts, apply prices afterwards.

Caveat that matters: FLOPs here are the harness's rough analytic estimates (see
docs/budget_protocol.md), not measurements. Treat crossovers as an indication of
where to look, and check them with wall-clock time before drawing conclusions.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED = ("model", "tier", "kl_mean", "budget_approx_flops_per_event")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean error and mean FLOPs per (model, tier), plus the seed count."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"results are missing columns: {missing}")
    g = df.groupby(["model", "tier"], sort=False)
    out = g.agg(
        kl_mean=("kl_mean", "mean"),
        kl_sem=("kl_mean", lambda s: s.std(ddof=1) / len(s) ** 0.5 if len(s) > 1 else float("nan")),
        kflops=("budget_approx_flops_per_event", lambda s: s.mean() / 1000.0),
        seeds=("kl_mean", "size"),
    ).reset_index()
    return out


def pareto(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows that no other row beats on both error and compute (lower is better for both)."""
    rows = summary.sort_values(["kflops", "kl_mean"]).reset_index(drop=True)
    keep, best = [], float("inf")
    for i, r in rows.iterrows():
        if r.kl_mean < best:
            keep.append(i)
            best = r.kl_mean
    return rows.loc[keep].reset_index(drop=True)


def preferred(summary: pd.DataFrame, lam: float) -> pd.Series:
    """The (model, tier) row with the lowest kl_mean + lam * kflops."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    score = summary.kl_mean + lam * summary.kflops
    return summary.loc[score.idxmin()]


def sweep(summary: pd.DataFrame, lams: Iterable[float]) -> pd.DataFrame:
    """Preferred (model, tier) at each price of compute."""
    rows = []
    for lam in lams:
        r = preferred(summary, lam)
        rows.append(dict(lam=lam, model=r.model, tier=r.tier, kl_mean=r.kl_mean, kflops=r.kflops))
    return pd.DataFrame(rows)


def default_lams() -> list[float]:
    return [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def report(df: pd.DataFrame, lams: Iterable[float] | None = None) -> str:
    summary = summarise(df)
    lams = list(lams) if lams is not None else default_lams()
    lines = ["Error versus compute (mean over seeds)", ""]
    lines.append(f"{'model':<24}{'tier':<8}{'KL (nats)':>10}{'+/- SE':>9}{'kFLOPs/event':>14}{'seeds':>7}")
    for _, r in summary.iterrows():
        lines.append(f"{r.model:<24}{r.tier:<8}{r.kl_mean:>10.3f}{r.kl_sem:>9.3f}{r.kflops:>14.2f}{int(r.seeds):>7}")
    front = pareto(summary)
    lines += ["", "Pareto frontier (nothing is both more accurate and cheaper):"]
    for _, r in front.iterrows():
        lines.append(f"  {r.model} / {r.tier}: KL {r.kl_mean:.3f} at {r.kflops:.2f} kFLOPs/event")
    lines += ["", "Preferred model as compute gets dearer (lam = nats of KL traded per kFLOP):"]
    lines.append(f"{'lam':>10}  {'model / tier':<32}{'KL':>8}{'kFLOPs':>10}")
    for _, r in sweep(summary, lams).iterrows():
        lines.append(f"{r.lam:>10g}  {r.model + ' / ' + r.tier:<32}{r.kl_mean:>8.3f}{r.kflops:>10.2f}")
    lines += ["", "FLOPs are analytic estimates, not measurements. See docs/budget_protocol.md."]
    return "\n".join(lines)
