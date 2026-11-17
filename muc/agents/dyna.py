"""Dyna-style agents for E05. One tabular Q-learner, many ways to spend planning steps.

Everything except the planning source is identical across arms: the Q table, exploration,
the reward estimate R_hat (updated from real steps, last observed value), the planning
schedule and the recency-weighted sampling of visited (s, a) pairs.

Arms (docs/e05_dyna_spec.md):
    q_learning          no planning
    replay              Q-learning backups on stored real transitions, sampled uniformly from a bounded buffer
    replay_sa           expected backups over the empirical next-token counts of (s, a)  (uncompressed table)
    dyna_count          expected backups through ActionCountModel
    dyna_count_noaction the same model with the action dropped from its keys (degenerate control)
    dyna_sdm            expected backups through ActionSDMModel
    dyna_oracle         expected backups through the true dynamics, visited (s, a) pairs only
    dyna_oracle_full    as dyna_oracle, but planning samples every (s, a) pair uniformly. The upper bound that
                        separates model-accuracy headroom from coverage headroom

Planning for every other arm samples only (s, a) pairs the agent has actually visited, recency
weighted. That restriction is the spec's, and it matters: with a steerable source many
contexts are rare, so a large share of the Q table is never touched. A perfect model cannot
repair that unless planning is allowed to leave the visited set.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..models.action import ActionCountModel, ActionSDMModel, OracleDynamics
from ..models.base import bits_per_token

ARMS = ("q_learning", "replay", "replay_sa", "dyna_count", "dyna_count_noaction", "dyna_sdm", "dyna_oracle",
        "dyna_oracle_full")

TIERS = {  # model capacity; SDM locations chosen so param_bits match the count model within about 1 percent.
    # (The first run used 64 and 512 locations, about half the count model's bits. It was discarded, see docs/negative_results.md.)
    # Replay's buffer is sized to the count model's bits
    "small": dict(count_entries=128, sdm_locations=137),
    "medium": dict(count_entries=1024, sdm_locations=1100),
}


def make_agent(arm, mdp, n_plan, tier="small", seed=0, **kw):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    t = TIERS[tier]
    n, m = mdp.n, mdp.m
    model, buffer_cap = None, 0
    if arm == "dyna_count":
        model = ActionCountModel(n, m, max_entries=t["count_entries"], seed=seed)
    elif arm == "dyna_count_noaction":
        model = ActionCountModel(n, m, max_entries=t["count_entries"], use_action=False, seed=seed)
    elif arm == "dyna_sdm":
        model = ActionSDMModel(n, m, n_locations=t["sdm_locations"], seed=seed)
    elif arm in ("dyna_oracle", "dyna_oracle_full"):
        model = OracleDynamics(mdp)
    elif arm == "replay":
        ref = ActionCountModel(n, m, max_entries=t["count_entries"]).budget()["param_bits"]
        buffer_cap = max(1, ref // ((m + 2) * bits_per_token(n)))  # bits per stored (s, a, x')
    return DynaAgent(arm, mdp, model, 0 if arm == "q_learning" else n_plan, buffer_cap, seed=seed, **kw)


class DynaAgent:
    def __init__(self, arm, mdp, model, n_plan, buffer_cap=0, alpha=0.3, epsilon=0.1, tau=500.0, sa_decay=0.999,
                 seed=0):
        self.arm, self.mdp, self.model, self.n_plan = arm, mdp, model, int(n_plan)
        self.S, self.n, self.gamma = mdp.S, mdp.n, mdp.cfg.gamma
        self.alpha, self.epsilon, self.tau, self.sa_decay = alpha, epsilon, tau, sa_decay
        self.rng = np.random.default_rng([seed, 99])
        self.Q = np.zeros((self.S, self.n))
        self.Rhat = np.zeros((self.S, self.n))
        self.last_visit = np.full((self.S, self.n), -np.inf)
        self.t = 0
        self.buffer = deque(maxlen=buffer_cap) if arm == "replay" else None
        if arm == "replay_sa":
            self.sa = np.zeros((self.S, self.n, self.n))  # decayed counts of x' after (s, a)
            self.sa_t = np.zeros((self.S, self.n))
        self.plan_queries = 0
        self.kl_acc, self.ece_conf, self.ece_hit = [], [], []

    # ---- acting ----
    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    # ---- learning from one real step ----
    def observe(self, s, a, x, r, s2, regime):
        if self.arm in ("dyna_oracle", "dyna_oracle_full"):
            self.model.sync(regime)
        ctx = list(self.mdp.ctx[s])
        if self.model is not None and not self.arm.startswith("dyna_oracle"):
            p = self.model.predict_at(ctx, a)  # before learning: for calibration
            self.ece_conf.append(float(p.max()))
            self.ece_hit.append(float(int(p.argmax()) == x))
            self.model.update(ctx, a, x)
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s, a])
        self.Rhat[s, x] = r
        self.last_visit[s, a] = self.t
        if self.buffer is not None:
            self.buffer.append((s, a, x))
        if self.arm == "replay_sa":
            self.sa[s, a] *= self.sa_decay ** (self.t - self.sa_t[s, a])
            self.sa[s, a, x] += 1.0
            self.sa_t[s, a] = self.t
        self.t += 1
        if self.n_plan > 0:
            self._plan()

    # ---- planning ----
    def _sample_pairs(self, k):
        if self.arm == "dyna_oracle_full":
            idx = self.rng.integers(self.S * self.n, size=k)
            return [(int(i) // self.n, int(i) % self.n) for i in idx]
        age = self.t - self.last_visit
        w = np.exp(-age / self.tau)
        w[~np.isfinite(self.last_visit)] = 0.0
        flat = w.ravel()
        tot = flat.sum()
        if tot <= 0:
            return []
        idx = self.rng.choice(flat.size, size=k, p=flat / tot)
        return [(int(i) // self.n, int(i) % self.n) for i in idx]

    def _plan(self):
        if self.arm == "replay":
            if not self.buffer:
                return
            for i in self.rng.integers(len(self.buffer), size=self.n_plan):
                s, a, x = self.buffer[int(i)]
                s2 = self.mdp.ns[s, x]
                self.Q[s, a] += self.alpha * (self.Rhat[s, x] + self.gamma * self.Q[s2].max() - self.Q[s, a])
            return
        for s, a in self._sample_pairs(self.n_plan):
            if self.arm == "replay_sa":
                c = self.sa[s, a] * self.sa_decay ** (self.t - self.sa_t[s, a])
                p = c / c.sum()
            else:
                p = self.model.predict_at(list(self.mdp.ctx[s]), a)
                self.plan_queries += 1
            target = float((p * (self.Rhat[s] + self.gamma * self.Q[self.mdp.ns[s]].max(axis=1))).sum())
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    # ---- diagnostics ----
    def dynamics_kl(self, regime, pairs):
        """Mean KL(true || model) over (s, a) pairs. NaN for arms without a model."""
        if self.arm == "replay_sa":
            def q(s, a):
                c = self.sa[s, a] * self.sa_decay ** (self.t - self.sa_t[s, a])
                return (c + 1e-3) / (c + 1e-3).sum()
        elif self.model is None:
            return float("nan")
        else:
            def q(s, a):
                return np.clip(self.model.predict_at(list(self.mdp.ctx[s]), a), 1e-9, None)
        P = self.mdp.transition(regime)
        if not pairs:
            return float("nan")
        kl = [float((P[s, a] * np.log(P[s, a] / q(s, a))).sum()) for s, a in pairs]
        return float(np.mean(kl))

    def flops_per_step(self):
        """Approximate, per real step: the Q backup, the model update, and planning queries."""
        base = 4 * self.n
        if self.arm == "q_learning":
            return base
        if self.arm in ("replay", "replay_sa"):
            return base + self.n_plan * 4 * self.n
        b = self.model.budget()
        return base + b["approx_flops_per_event"] + self.n_plan * (b["flops_per_query"] + 4 * self.n)

    def budget(self):
        qbits = self.S * self.n * 32
        if self.arm == "replay":
            return {"param_bits": int(self.buffer.maxlen * (self.mdp.m + 2) * bits_per_token(self.n)),
                    "q_table_bits": qbits}
        if self.arm == "replay_sa":
            return {"param_bits": int(self.S * self.n * self.n * 32), "q_table_bits": qbits}
        if self.model is None:
            return {"param_bits": 0, "q_table_bits": qbits}
        return {"param_bits": self.model.budget()["param_bits"], "q_table_bits": qbits}


def ece(conf, hit, bins=10):
    conf, hit = np.asarray(conf), np.asarray(hit)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            tot += sel.mean() * abs(conf[sel].mean() - hit[sel].mean())
    return float(tot)


def run_agent(agent, mdp, seed, window=200, theta=0.5):
    """Run all phases. Returns per-phase and overall metrics as a flat dict."""
    cfg = mdp.cfg
    rng = np.random.default_rng([seed, 7])
    s = mdp.start_state(rng)
    rewards, norm_ref = [], []
    phase_rows = []
    for pi in range(len(cfg.phases)):
        regime = mdp.regime_of_phase(pi)
        sol = mdp.solve(regime)
        lo, hi = sol["rate_rand"], sol["rate_opt"]
        agent.ece_conf, agent.ece_hit = [], []
        visited = set()
        r_phase = []
        for _ in range(cfg.phase_len):
            a = agent.act(s)
            x, r, s2 = mdp.step(s, a, regime, rng)
            agent.observe(s, a, x, r, s2, regime)
            visited.add((s, a))
            r_phase.append(r)
            s = s2
        r_phase = np.array(r_phase)
        kernel = np.ones(window) / window
        roll = np.convolve(r_phase, kernel, mode="valid")
        norm_roll = (roll - lo) / max(hi - lo, 1e-9)
        hit = np.flatnonzero(norm_roll >= theta)
        adapt = int(hit[0]) + window if len(hit) else cfg.phase_len  # censored at phase_len
        q4 = r_phase[-cfg.phase_len // 4:].mean()
        phase_rows.append(dict(
            phase=pi, regime=str(regime), rate_opt=hi, rate_rand=lo,
            reward=float(r_phase.mean()), frac_opt=float(r_phase.mean() / hi),
            norm=float((r_phase.mean() - lo) / max(hi - lo, 1e-9)),
            norm_last_quarter=float((q4 - lo) / max(hi - lo, 1e-9)),
            adapt_steps=adapt, censored=int(len(hit) == 0),
            kl=agent.dynamics_kl(regime, sorted(visited)),
            ece=ece(agent.ece_conf, agent.ece_hit)))
        rewards.append(r_phase)
    out = dict(phases=phase_rows, flops_per_step=agent.flops_per_step(), **agent.budget(),
               plan_queries=agent.plan_queries)
    out["norm_overall"] = float(np.mean([p["norm"] for p in phase_rows]))
    return out
