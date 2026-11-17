"""E05 tests: exact MDP, action-conditioned memory contract, agents. Pure NumPy, runs without torch."""
import numpy as np
import pytest

from muc.agents import ARMS, make_agent, run_agent
from muc.envs import SteerConfig, SteerableMDP
from muc.models.action import ActionCountModel, ActionSDMModel, OracleDynamics, check_action_adapter
from muc.tasks import TaskConfig


def small_cfg(**kw):
    return SteerConfig(phase_len=400, **kw)


# ---- the environment ----
def test_c_zero_reduces_to_the_pilot_source():
    mdp = SteerableMDP(small_cfg(c=0.0), seed=1)
    P = mdp.transition((0, 0))
    rule = mdp.rule((0, 0))
    for a in range(mdp.n):
        assert np.allclose(P[:, a, :], rule)


def test_c_one_makes_the_action_the_next_token():
    mdp = SteerableMDP(small_cfg(c=1.0), seed=1)
    P = mdp.transition((0, 0))
    for a in range(mdp.n):
        assert np.allclose(P[:, a, a], 1.0)


def test_transitions_are_distributions():
    mdp = SteerableMDP(small_cfg(c=0.6), seed=2)
    for regime in [(0, 0), (1, 0), (0, 1)]:
        P = mdp.transition(regime)
        assert np.allclose(P.sum(axis=2), 1.0) and P.min() >= 0


def test_rule_matches_the_pilot_automaton():
    cfg = small_cfg(c=0.0)
    mdp = SteerableMDP(cfg, seed=3)
    from muc.tasks import build_variants
    auto = build_variants(cfg.source.replace(phases=(0, 1), drift_frac=cfg.drift_frac, n_distract=0), 3)[0]
    for s in (0, 5, 40):
        assert np.allclose(mdp.rule((0, 0))[s], auto.full_dist(list(mdp.ctx[s]))[: mdp.n])


def test_next_state_is_the_shifted_window():
    mdp = SteerableMDP(small_cfg(), seed=0)
    s = mdp.encode([1, 2, 3])
    assert list(mdp.ctx[mdp.ns[s, 0]]) == [2, 3, 0]


def test_value_iteration_agrees_with_monte_carlo_rollouts():
    mdp = SteerableMDP(small_cfg(c=0.6), seed=1)
    regime = (0, 0)
    sol = mdp.solve(regime)
    rng = np.random.default_rng(0)
    s, tot, N = mdp.start_state(rng), 0.0, 40000
    for _ in range(N):
        _, r, s = mdp.step(s, int(sol["pi"][s]), regime, rng)
        tot += r
    assert abs(tot / N - sol["rate_opt"]) < 0.02


def test_optimal_beats_random_and_control_helps():
    lo = SteerableMDP(small_cfg(c=0.0), seed=1).solve((0, 0))
    hi = SteerableMDP(small_cfg(c=0.9), seed=1).solve((0, 0))
    assert lo["rate_opt"] >= lo["rate_rand"] - 1e-9
    assert hi["rate_opt"] > lo["rate_opt"]


def test_returning_to_regime_zero_restores_the_mdp_exactly():
    for kind in ("dynamics", "reward", "both"):
        mdp = SteerableMDP(small_cfg(c=0.6, drift_kind=kind), seed=4)
        assert mdp.regime_of_phase(0) == mdp.regime_of_phase(2) == (0, 0)
        assert mdp.regime_of_phase(1) != (0, 0)


def test_reward_drift_changes_rewards_not_dynamics_and_keeps_density():
    mdp = SteerableMDP(small_cfg(c=0.6, drift_kind="reward", drift_frac=1.0), seed=5)
    r0, r1 = mdp.regime_of_phase(0), mdp.regime_of_phase(1)
    assert np.allclose(mdp.transition(r0), mdp.transition(r1))
    assert mdp.reward(r0).sum() == mdp.reward(r1).sum()
    assert not np.array_equal(mdp.reward(r0), mdp.reward(r1))


def test_dynamics_drift_changes_dynamics_not_rewards():
    mdp = SteerableMDP(small_cfg(c=0.6, drift_kind="dynamics", drift_frac=1.0), seed=5)
    r0, r1 = mdp.regime_of_phase(0), mdp.regime_of_phase(1)
    assert np.allclose(mdp.reward(r0), mdp.reward(r1))
    assert not np.allclose(mdp.transition(r0), mdp.transition(r1))


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        SteerableMDP(small_cfg(drift_kind="nope"), 0)
    with pytest.raises(ValueError):
        SteerableMDP(small_cfg(c=1.5), 0)


# ---- action-conditioned memories ----
@pytest.mark.parametrize("factory", [
    lambda: ActionCountModel(4, 3, max_entries=128),
    lambda: ActionCountModel(4, 3, max_entries=128, use_action=False),
    lambda: ActionSDMModel(4, 3, n_locations=64),
])
def test_action_adapter_contract(factory):
    assert check_action_adapter(factory)


def test_count_model_learns_a_deterministic_steerable_rule():
    m = ActionCountModel(4, 3, max_entries=512, decay=1.0)
    ctx = [0, 1, 2]
    for _ in range(50):
        m.update(ctx, 2, 2)  # the action 2 always produces 2
        m.update(ctx, 1, 3)  # the action 1 always produces 3
    assert m.predict_at(ctx, 2).argmax() == 2 and m.predict_at(ctx, 1).argmax() == 3


def test_action_input_is_what_separates_the_two_actions():
    with_a = ActionCountModel(4, 3, max_entries=512, decay=1.0)
    without = ActionCountModel(4, 3, max_entries=512, decay=1.0, use_action=False)
    for _ in range(60):
        for a in (0, 3):
            with_a.update([0, 1, 2], a, a)
            without.update([0, 1, 2], a, a)
    d_with = abs(with_a.predict_at([0, 1, 2], 0) - with_a.predict_at([0, 1, 2], 3)).sum()
    d_without = abs(without.predict_at([0, 1, 2], 0) - without.predict_at([0, 1, 2], 3)).sum()
    assert d_with > 1.0 and d_without < 1e-9


def test_oracle_returns_true_dynamics_after_sync():
    mdp = SteerableMDP(small_cfg(c=0.6), seed=1)
    o = OracleDynamics(mdp)
    o.sync((0, 0))
    s = 17
    assert np.allclose(o.predict_at(list(mdp.ctx[s]), 2), mdp.transition((0, 0))[s, 2])


# ---- agents ----
def test_every_arm_runs_and_reports():
    mdp = SteerableMDP(small_cfg(c=0.6, drift_kind="both"), seed=1)
    for arm in ARMS:
        out = run_agent(make_agent(arm, mdp, 5, "small", seed=1), mdp, 1, window=100)
        assert len(out["phases"]) == 3 and np.isfinite(out["norm_overall"])
        assert out["flops_per_step"] > 0 and out["q_table_bits"] == 64 * 4 * 32


def test_runs_are_reproducible():
    mdp = SteerableMDP(small_cfg(c=0.6), seed=2)
    a = run_agent(make_agent("dyna_count", mdp, 5, "small", seed=2), mdp, 2, window=100)
    b = run_agent(make_agent("dyna_count", mdp, 5, "small", seed=2), mdp, 2, window=100)
    assert [p["reward"] for p in a["phases"]] == [p["reward"] for p in b["phases"]]


def test_oracle_model_has_zero_dynamics_error():
    mdp = SteerableMDP(small_cfg(c=0.6), seed=1)
    ag = make_agent("dyna_oracle", mdp, 5, "small", seed=1)
    out = run_agent(ag, mdp, 1, window=100)
    assert all(p["kl"] < 1e-9 for p in out["phases"])


def test_planning_with_the_true_model_over_all_pairs_beats_no_planning():
    cfg = SteerConfig(c=0.6, phase_len=1500, phases=(0,))
    scores = {0: [], 20: []}
    for seed in range(4):
        mdp = SteerableMDP(cfg, seed)
        for n_plan in scores:
            arm = "q_learning" if n_plan == 0 else "dyna_oracle_full"
            out = run_agent(make_agent(arm, mdp, n_plan, "small", seed=seed), mdp, seed, window=100)
            scores[n_plan].append(out["phases"][0]["norm_last_quarter"])
    assert np.mean(scores[20]) > np.mean(scores[0]) + 0.1


def test_visited_only_planning_leaves_part_of_the_q_table_untouched():
    cfg = SteerConfig(c=0.6, phase_len=1500, phases=(0,))
    mdp = SteerableMDP(cfg, 0)
    ag = make_agent("dyna_oracle", mdp, 20, "small", seed=0)
    run_agent(ag, mdp, 0, window=100)
    assert np.isinf(ag.last_visit).sum() > 0  # some (s, a) pairs were never visited, so never planned over


def test_replay_buffer_is_bounded_by_the_budget():
    mdp = SteerableMDP(small_cfg(), seed=1)
    ag = make_agent("replay", mdp, 5, "small", seed=1)
    run_agent(ag, mdp, 1, window=100)
    assert len(ag.buffer) <= ag.buffer.maxlen


def test_replay_uses_current_rewards_not_stored_ones():
    # After a reward change the replayed backup must read the updated R_hat entry.
    mdp = SteerableMDP(small_cfg(), seed=1)
    ag = make_agent("replay", mdp, 1, "small", seed=1)
    ag.buffer.append((3, 1, 2))
    ag.Rhat[3, 2] = 1.0
    ag.Q[:] = 0
    ag.rng = np.random.default_rng(0)
    ag._plan()
    assert ag.Q[3, 1] > 0


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        make_agent("nope", SteerableMDP(small_cfg(), 0), 5)


@pytest.mark.parametrize("tier", ["small", "medium"])
def test_tiers_match_param_bits_across_model_arms(tier):
    mdp = SteerableMDP(small_cfg(), seed=0)
    bits = {a: make_agent(a, mdp, 5, tier).budget()["param_bits"] for a in ("replay", "dyna_count", "dyna_sdm")}
    ref = bits["dyna_count"]
    assert all(abs(b - ref) / ref < 0.02 for b in bits.values()), bits
