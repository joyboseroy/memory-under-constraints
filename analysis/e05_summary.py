"""Tables and figures for E05 from results/e05/results_*.csv.

    python analysis/e05_summary.py results/e05            # writes e05_tables.md and two figures there

Aggregation: score = normalised return, (reward - random policy) / (optimal - random policy), exact for
each regime. A "run" is one (cell, seed, arm, n_plan). Paired differences are taken within the same MDP
(same seed and cell) and summarised as mean +/- 1.96 standard errors over those units. The units share
MDPs across n_plan and are not fully independent, so treat the intervals as rough.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["grid", "c", "drift_kind", "drift_frac", "tier", "seed"]


def load(d):
    frames = [pd.read_csv(p) for p in sorted(Path(d).glob("results_*.csv")) if "smoke" not in p.name]
    return pd.concat(frames, ignore_index=True)


def per_run(df):
    g = df.groupby(KEY + ["arm", "n_plan"], as_index=False)
    out = g.agg(norm=("norm", "mean"), flops=("flops_per_step", "first"), param_bits=("param_bits", "first"))
    p = df.pivot_table(index=KEY + ["arm", "n_plan"], columns="phase", values=["norm", "adapt_steps", "kl", "ece",
                                                                               "norm_last_quarter"]).reset_index()
    p.columns = ["_".join(str(x) for x in c if x != "").strip("_") for c in p.columns]
    return out.merge(p, on=KEY + ["arm", "n_plan"])


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), len(x)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def fmt(m, h, n=None):
    return "n/a" if not np.isfinite(m) else f"{m:+.3f} ± {h:.3f}"


def paired(runs, a, b, col, n_plan, mask=None):
    r = runs if mask is None else runs[mask]
    ra = r[(r.arm == a) & (r.n_plan == (0 if a == "q_learning" else n_plan))].set_index(KEY)[col]
    rb = r[(r.arm == b) & (r.n_plan == (0 if b == "q_learning" else n_plan))].set_index(KEY)[col]
    d = (ra - rb).dropna()
    return ci(d.values)


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    return s + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"


def main(d):
    d = Path(d)
    df = load(d)
    runs = per_run(df)
    main_runs = runs[(runs.grid.isin(["main", "severity"])) & (runs.tier == "small")]
    arms = [a for a in ["q_learning", "replay", "replay_sa", "dyna_count", "dyna_count_noaction", "dyna_sdm",
                        "dyna_oracle", "dyna_oracle_full"] if a in set(runs.arm)]
    out = []
    n_cells = main_runs.groupby(KEY).ngroups
    out.append(f"Runs: {len(df.groupby(KEY + ['arm', 'n_plan']))} arm-runs over {n_cells} (cell, seed) MDPs "
               f"in the main and severity grids.\n")

    # 1. score by arm x drift kind at n_plan 20
    out.append("### Normalised return by arm and drift kind (n_plan = 20, mean over c, seeds and phases)\n")
    rows = []
    for a in arms:
        n = 0 if a == "q_learning" else 20
        r = main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.grid == "main")]
        cells = [f"{r[r.drift_kind == k].norm.mean():.3f}" for k in ("dynamics", "reward", "both")]
        rows.append([a] + cells + [f"{r.norm.mean():.3f}"])
    out.append(md_table(["arm", "dynamics", "reward", "both", "all"], rows))

    # 2. by c
    out.append("### Normalised return by control strength c (n_plan = 20, all drift kinds)\n")
    rows = []
    for a in arms:
        n = 0 if a == "q_learning" else 20
        r = main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.grid == "main")]
        rows.append([a] + [f"{r[r.c == c].norm.mean():.3f}" for c in (0.3, 0.6, 0.9)])
    out.append(md_table(["arm", "c = 0.3", "c = 0.6", "c = 0.9"], rows))

    # 3. n_plan sweep
    out.append("### Normalised return versus planning steps (main grid, all c and kinds)\n")
    rows = []
    for a in arms:
        if a == "q_learning":
            r = main_runs[(main_runs.arm == a) & (main_runs.grid == "main")]
            rows.append([a] + [f"{r.norm.mean():.3f}"] * 3)
        else:
            rows.append([a] + [f"{main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.grid == 'main')].norm.mean():.3f}"
                              for n in (5, 20, 50)])
    out.append(md_table(["arm", "n_plan 5", "n_plan 20", "n_plan 50"], rows))

    # 4. adaptation after drift, phase index 1 and 2
    out.append("### Steps to reach a rolling normalised score of 0.5 after each phase starts "
               "(n_plan = 20, mean over main grid; capped at 2000, so a value near 2000 means it often did not get there)\n")
    rows = []
    for a in arms:
        n = 0 if a == "q_learning" else 20
        r = main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.grid == "main")]
        rows.append([a] + [f"{r[f'adapt_steps_{p}'].mean():.0f}" for p in (0, 1, 2)])
    out.append(md_table(["arm", "phase 0 (initial)", "phase 1 (after drift)", "phase 2 (back to regime 0)"], rows))

    # 5. hypotheses
    out.append("### Hypothesis checks (paired differences within the same MDP, mean ± 1.96 SE)\n")
    def row(label, a, b, col, kind=None, n_plan=20, c=None):
        m = (main_runs.grid == "main")
        if kind:
            m &= main_runs.drift_kind == kind
        if c is not None:
            m &= main_runs.c == c
        mean, h, n = paired(main_runs, a, b, col, n_plan, m)
        return [label, a + " − " + b, col.replace("norm_", "").replace("_", " "), fmt(mean, h), n]
    rows = [
        row("E5-H1 reward drift", "dyna_count", "q_learning", "norm_1", "reward"),
        row("E5-H1 reward drift", "dyna_count", "replay", "norm_1", "reward"),
        row("E5-H1 reward drift", "dyna_count", "replay_sa", "norm_1", "reward"),
        row("E5-H1 reward drift", "dyna_oracle_full", "replay_sa", "norm_1", "reward"),
        row("E5-H2 dynamics drift", "dyna_count", "dyna_sdm", "norm_1", "dynamics"),
        row("E5-H2 dynamics drift", "dyna_count", "q_learning", "norm_1", "dynamics"),
        row("E5-H2 dynamics drift", "dyna_count", "replay", "norm_1", "dynamics"),
        row("E5-H4 small n_plan (5)", "dyna_count", "replay", "norm", None, 5),
        row("E5-H4 n_plan 20", "dyna_count", "replay", "norm", None, 20),
        row("E5-H4 large n_plan (50)", "dyna_count", "replay", "norm", None, 50),
        row("E5-H4b c = 0.3", "dyna_count", "dyna_count_noaction", "norm", None, 20, 0.3),
        row("E5-H4b c = 0.6", "dyna_count", "dyna_count_noaction", "norm", None, 20, 0.6),
        row("E5-H4b c = 0.9", "dyna_count", "dyna_count_noaction", "norm", None, 20, 0.9),
        row("E5-H5 null, all drift", "dyna_count", "dyna_sdm", "norm", None),
        row("E5-H5 null, all drift", "dyna_count", "replay_sa", "norm", None),
        row("E5-H5 null, all drift", "dyna_count", "replay", "norm", None),
        row("E5-H5 null, all drift", "dyna_sdm", "replay", "norm", None),
        row("coverage headroom", "dyna_oracle_full", "dyna_oracle", "norm", None),
        row("model headroom", "dyna_oracle", "dyna_count", "norm", None),
    ]
    out.append(md_table(["hypothesis", "difference", "score", "mean ± 1.96 SE", "units"], [r for r in rows if r[-1]]))

    # 6. KL / ECE versus score
    out.append("### Model quality versus planning quality (one point per model-arm run; phase 0 to 2 pooled)\n")
    mdl = df[(df.arm.isin(["dyna_count", "dyna_count_noaction", "dyna_sdm", "replay_sa"])) & (df.n_plan == 20) &
             (df.tier == "small") & (df.grid == "main") & (df.drift_kind.isin(["dynamics", "both"]))]
    rows = []
    for a, g in mdl.groupby("arm"):
        gg = g.dropna(subset=["kl", "norm"])
        rho = gg[["kl", "norm"]].corr(method="spearman").iloc[0, 1] if len(gg) > 3 else float("nan")
        e = g.dropna(subset=["ece"])
        rows.append([a, f"{g.kl.mean():.3f}", f"{g.ece.mean():.3f}" if len(e) else "n/a", f"{g.norm.mean():.3f}", f"{rho:+.2f}"])
    out.append(md_table(["arm", "mean KL(true, model)", "mean ECE", "mean score", "Spearman(KL, score) within arm"], rows))
    allm = mdl.dropna(subset=["kl", "norm"])
    if len(allm) > 3:
        out.append(f"Across all four model arms pooled, Spearman(KL, score) = "
                   f"{allm[['kl', 'norm']].corr(method='spearman').iloc[0, 1]:+.2f}.\n")

    # 7. cost
    out.append("### Cost: approximate FLOPs per real step, memory, and gain per FLOP (main grid, n_plan = 20)\n")
    q = main_runs[(main_runs.arm == "q_learning") & (main_runs.grid == "main")]
    rows = []
    for a in arms:
        n = 0 if a == "q_learning" else 20
        r = main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.grid == "main")]
        gain = r.norm.mean() - q.norm.mean()
        extra = r.flops.mean() - q.flops.mean()
        rows.append([a, f"{r.flops.mean():.0f}", f"{r.param_bits.mean():.0f}", f"{r.norm.mean():.3f}",
                     "n/a" if a == "q_learning" else f"{1000 * gain / extra:.3f}"])
    out.append(md_table(["arm", "FLOPs / real step", "model param_bits", "score", "score gain per 1000 FLOPs"], rows))

    # 8. tier
    t = runs[runs.grid == "tier"]
    if len(t):
        out.append("### Capacity tier (c = 0.6, n_plan = 20): small versus medium\n")
        rows = []
        for a in ["replay", "dyna_count", "dyna_sdm"]:
            s_ = main_runs[(main_runs.arm == a) & (main_runs.n_plan == 20) & (main_runs.c == 0.6) &
                           (main_runs.drift_kind.isin(["dynamics", "reward"])) & (main_runs.drift_frac == 0.5) &
                           (main_runs.grid == "main")]
            m_ = t[(t.arm == a)]
            rows.append([a, f"{s_.norm.mean():.3f}", f"{m_.norm.mean():.3f}", f"{s_.param_bits.mean():.0f}",
                         f"{m_.param_bits.mean():.0f}"])
        out.append(md_table(["arm", "score small", "score medium", "param_bits small", "param_bits medium"], rows))

    # 9. severity
    sv = runs[runs.grid == "severity"]
    if len(sv):
        out.append("### Drift severity at c = 0.6 (n_plan = 20): drift_frac 0.5 versus 1.0\n")
        rows = []
        for a in arms:
            n = 0 if a == "q_learning" else 20
            h = main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.c == 0.6) & (main_runs.grid == "main")]
            f = sv[(sv.arm == a) & (sv.n_plan == n)]
            rows.append([a, f"{h.norm_1.mean():.3f}", f"{f.norm_1.mean():.3f}"])
        out.append(md_table(["arm", "phase-1 score, drift_frac 0.5", "phase-1 score, drift_frac 1.0"], rows))

    (d / "e05_tables.md").write_text("\n".join(out))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for i, k in enumerate(("dynamics", "reward", "both")):
        for a in arms:
            if a == "q_learning":
                v = main_runs[(main_runs.arm == a) & (main_runs.grid == "main") & (main_runs.drift_kind == k)].norm.mean()
                ax[i].axhline(v, ls=":", color="gray", label="q_learning")
            else:
                ys = [main_runs[(main_runs.arm == a) & (main_runs.n_plan == n) & (main_runs.grid == "main") &
                                (main_runs.drift_kind == k)].norm.mean() for n in (5, 20, 50)]
                ax[i].plot((5, 20, 50), ys, marker="o", label=a)
        ax[i].set_title(f"drift: {k}")
        ax[i].set_xlabel("planning steps per real step")
        ax[i].set_xscale("log")
    ax[0].set_ylabel("normalised return")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "fig_e05_return_vs_nplan.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for a, g in mdl.groupby("arm"):
        ax.scatter(g.kl, g.norm, s=10, alpha=0.5, label=a)
    ax.set_xlabel("KL(true dynamics, model) over visited pairs")
    ax.set_ylabel("normalised return in that phase")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "fig_e05_return_vs_kl.png", dpi=130)
    plt.close(fig)
    return "\n".join(out)


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "results/e05"))
