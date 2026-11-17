"""Metrics computed from model predictions Q and the true conditional distributions P.

Definitions (also in docs/hypotheses.md, fixed before any serious run):

performance fraction  p = 1 - mean(KL(P||Q)) / mean(KL(P||Uniform)) over a window.
    p = 1 for the oracle, p ~ 0 for a uniform predictor.
adaptation time       events from the start of a phase until the rolling-window p first
    reaches p_star (absolute, default 0.5; 0.9 was unreachable in 2000-event phases, see
    docs/hypotheses.md deviations). `adapt_rel_time_i` uses rel_star (default 0.9) times the
    model's own end-of-phase p instead. Both are censored at the phase length.
retention             p of the model, frozen, on held-out sequences of regime 0 after the
    first drifted phase, divided by the same quantity right after phase 0.
"""
from __future__ import annotations

import numpy as np

QMIN = 1e-9


def _norm(Q):
    Qc = np.clip(Q, QMIN, None)
    return Qc / Qc.sum(axis=1, keepdims=True)


def kl_rows(P, Q):
    Qc = _norm(Q)
    Pc = np.clip(P, 1e-300, None)
    return np.sum(np.where(P > 0, P * (np.log(Pc) - np.log(Qc)), 0.0), axis=1)


def uniform_kl_rows(P):
    V = P.shape[1]
    Pc = np.clip(P, 1e-300, None)
    return np.sum(np.where(P > 0, P * (np.log(Pc) + np.log(V)), 0.0), axis=1)


def logloss_rows(Q, x):
    Qc = _norm(Q)
    return -np.log(Qc[np.arange(len(x)), x])


def entropy_rows(P):
    Pc = np.clip(P, 1e-300, None)
    return -np.sum(np.where(P > 0, P * np.log(Pc), 0.0), axis=1)


def ece(conf, correct, bins=10):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def perf_fraction(P, Q):
    klu = uniform_kl_rows(P).mean()
    if klu <= 1e-12:
        return float("nan")
    return float(1.0 - kl_rows(P, Q).mean() / klu)


def rolling_perf(kl, klu, w):
    """p over the trailing window of w events ending at each index (nan before w-1)."""
    ck = np.concatenate([[0.0], np.cumsum(kl)])
    cu = np.concatenate([[0.0], np.cumsum(klu)])
    T = len(kl)
    p = np.full(T, np.nan)
    if T >= w:
        sk = ck[w:] - ck[:-w]
        su = cu[w:] - cu[:-w]
        p[w - 1:] = 1.0 - sk / np.maximum(su, 1e-12)
    return p


def time_to_threshold(p, start, end, w, thresh):
    """Events from `start` until rolling p (windows fully inside [start,end)) reaches thresh.
    Returns (time, censored)."""
    first = start + w - 1
    for i in range(first, end):
        if p[i] >= thresh:
            return i - start + 1, False
    return end - start, True


def summarize(stream, Q, probes=None, window=200, p_star=0.5, rel_star=0.9):
    """Summarise one run. `probes` maps (phase_idx, regime) -> p of the frozen model."""
    P, x = stream.probs, stream.tokens
    T = stream.T
    kl = kl_rows(P, Q)
    klu = uniform_kl_rows(P)
    ll = logloss_rows(Q, x)
    oll = logloss_rows(P, x)
    pred = Q.argmax(1)
    acc = float((pred == x).mean())
    oacc = float((P.argmax(1) == x).mean())
    conf = _norm(Q).max(1)
    w = min(window, max(5, T // (4 * max(1, len(stream.starts)))))
    roll = rolling_perf(kl, klu, w)

    out = {
        "kl_mean": float(kl.mean()),
        "logloss": float(ll.mean()),
        "oracle_logloss": float(oll.mean()),
        "excess_logloss": float(ll.mean() - oll.mean()),
        "acc": acc,
        "oracle_acc": oacc,
        "acc_norm": acc / max(oacc, 1e-12),
        "agreement": float((pred == P.argmax(1)).mean()),
        "ece": ece(conf, pred == x),
        "p_overall": float(1.0 - kl.mean() / max(klu.mean(), 1e-12)),
        "window": int(w),
    }
    n_ph = len(stream.starts)
    ends = list(stream.starts[1:]) + [T]
    for i, (s, e) in enumerate(zip(stream.starts, ends)):
        t_ad, cens = time_to_threshold(roll, s, e, w, p_star)
        out[f"adapt_time_{i}"] = int(t_ad)
        out[f"adapt_censored_{i}"] = bool(cens)
        out[f"p_final_{i}"] = float(roll[e - 1])
        out[f"kl_final_{i}"] = float(kl[e - w:e].mean())
        pf = out[f"p_final_{i}"]
        if pf > 0.1:
            t_rel, c_rel = time_to_threshold(roll, s, e, w, rel_star * pf)
        else:
            t_rel, c_rel = e - s, True  # never reached a meaningful level
        out[f"adapt_rel_time_{i}"] = int(t_rel)
        out[f"adapt_rel_censored_{i}"] = bool(c_rel)
    if n_ph >= 2:
        out["adapt_time_drift"] = float(np.mean([out[f"adapt_time_{i}"] for i in range(1, n_ph)]))
        out["adapt_rel_time_drift"] = float(np.mean([out[f"adapt_rel_time_{i}"] for i in range(1, n_ph)]))
    if n_ph >= 3 and stream.phases[-1] == stream.phases[0]:
        out["relearn_ratio"] = out[f"adapt_time_{n_ph-1}"] / max(out["adapt_time_0"], 1)
    if probes:
        r0 = stream.phases[0]
        before = probes.get((0, r0))
        out["probe_before"] = before
        if n_ph >= 2 and stream.phases[1] != r0:
            after = probes.get((1, r0))
            out["probe_after"] = after
            if before is not None and after is not None and before > 1e-6:
                out["retention"] = after / before
                out["forgetting"] = before - after
    return out
