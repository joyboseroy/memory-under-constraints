"""Figures and summary tables from results/<name>/results.csv.

    python analysis/make_figures.py results/E01_memory_budget

Outputs (in the results folder): summary.csv, crossovers.txt, fig_budget.png,
fig_adapt_retention.png, fig_grid_*.png (when the config had grid axes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["p_overall", "kl_mean", "excess_logloss", "acc_norm", "ece", "agreement",
           "adapt_time_drift", "adapt_rel_time_drift", "retention", "forgetting", "relearn_ratio"]


def load(folder):
    df = pd.read_csv(Path(folder) / "results.csv")
    gcols = [c for c in df.columns if c.startswith("g_")]
    return df, gcols


def summarize(df, gcols):
    keys = ["model", "tier"] + gcols
    agg = {m: ["mean", "std"] for m in METRICS if m in df}
    for c in ["budget_param_bits", "budget_state_bits", "budget_approx_flops_per_event",
              "experience_events", "wall_ms_per_event"]:
        if c in df:
            agg[c] = ["mean"]
    s = df.groupby(keys, dropna=False).agg(agg)
    s.columns = ["_".join(c) if c[1] == "std" else c[0] for c in s.columns]
    s["n_seeds"] = df.groupby(keys, dropna=False).size()
    return s.reset_index()


def fig_budget(s, folder):
    if s["tier"].nunique() < 2:
        return
    base = s
    for g in [c for c in s.columns if c.startswith("g_")]:
        base = base[base[g] == base[g].iloc[0]]
    axes_def = [("budget_param_bits", "learned store (param bits)"),
                ("budget_state_bits", "context state (bits)"),
                ("budget_approx_flops_per_event", "approx FLOPs per event")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (xc, xl) in zip(axs, axes_def):
        for model, g in base.groupby("model"):
            g = g.sort_values(xc)
            ax.errorbar(g[xc], g["p_overall"], yerr=g.get("p_overall_std"), marker="o", capsize=3, label=model)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("performance fraction p (1 = oracle, 0 = uniform)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Performance under drift versus resource budget (mean +/- sd over seeds)")
    fig.tight_layout()
    fig.savefig(Path(folder) / "fig_budget.png", dpi=150)
    plt.close(fig)


def fig_adapt_retention(s, folder):
    need = {"adapt_rel_time_drift", "retention"}
    if not need <= set(s.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, g in s.groupby("model"):
        ax.scatter(g["adapt_rel_time_drift"], g["retention"], label=model, s=40)
        for _, r in g.iterrows():
            ax.annotate(str(r["tier"]), (r["adapt_rel_time_drift"], r["retention"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("adaptation time after drift (events, lower = faster)")
    ax.set_ylabel("retention of old regime (higher = less forgetting)")
    ax.set_title("Adaptation-retention tradeoff")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(folder) / "fig_adapt_retention.png", dpi=150)
    plt.close(fig)


def fig_grid(s, gcols, folder):
    if not gcols:
        return
    x = gcols[0]
    facet = gcols[1] if len(gcols) > 1 else None
    tiers = sorted(s["tier"].unique())
    fvals = sorted(s[facet].unique()) if facet else [None]
    fig, axs = plt.subplots(len(tiers), len(fvals), figsize=(4.5 * len(fvals), 3.4 * len(tiers)),
                            squeeze=False, sharey=True)
    for i, tier in enumerate(tiers):
        for j, fv in enumerate(fvals):
            ax = axs[i][j]
            sub = s[s["tier"] == tier]
            if facet:
                sub = sub[sub[facet] == fv]
            for model, g in sub.groupby("model"):
                g = g.sort_values(x)
                ax.errorbar(g[x], g["p_overall"], yerr=g.get("p_overall_std"), marker="o", capsize=3, label=model)
            ax.set_title(f"tier={tier}" + (f", {facet}={fv}" if facet else ""), fontsize=9)
            ax.set_xlabel(x)
            ax.grid(alpha=0.3)
        axs[i][0].set_ylabel("p")
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(folder) / f"fig_grid_{x}.png", dpi=150)
    plt.close(fig)


def crossovers(s, gcols, folder):
    """Winner per tier by mean p, with margin in standard errors. A 'crossover' is a change
    of winner between tiers where both margins exceed 2 SE. Descriptive aid for the go/no-go
    rule, not a significance test."""
    lines = []
    for gv, sub in (s.groupby(gcols) if gcols else [(None, s)]):
        lines.append(f"== grid {dict(zip(gcols, gv if isinstance(gv, tuple) else (gv,))) if gcols else '{}'}")
        winners = []
        for tier, g in sub.groupby("tier"):
            g = g.sort_values("p_overall", ascending=False)
            if len(g) < 2:
                continue
            a, b = g.iloc[0], g.iloc[1]
            n = max(a["n_seeds"], 1)
            se = np.sqrt((a["p_overall_std"] ** 2 + b["p_overall_std"] ** 2) / n)
            z = (a["p_overall"] - b["p_overall"]) / se if se > 0 else float("inf")
            winners.append((float(a["budget_param_bits"]), tier, a["model"], b["model"], z))
            lines.append(f"  tier {tier:7s} winner {a['model']:22s} over {b['model']:22s} margin {z:5.1f} SE")
        winners.sort()
        for (_, t1, w1, _, z1), (_, t2, w2, _, z2) in zip(winners, winners[1:]):
            if w1 != w2 and z1 > 2 and z2 > 2:
                lines.append(f"  CROSSOVER between tier {t1} ({w1}) and tier {t2} ({w2})")
    (Path(folder) / "crossovers.txt").write_text("\n".join(lines) + "\n")
    return lines


def main(folder):
    df, gcols = load(folder)
    s = summarize(df, gcols)
    s.to_csv(Path(folder) / "summary.csv", index=False)
    fig_budget(s, folder)
    fig_adapt_retention(s, folder)
    fig_grid(s, gcols, folder)
    for line in crossovers(s, gcols, folder):
        print(line)
    cols = ["model", "tier"] + gcols + ["budget_param_bits", "p_overall", "p_overall_std", "n_seeds"]
    print(s[[c for c in cols if c in s]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
