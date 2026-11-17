import unittest

from muc.economics.cost import CostModel, Ledger
from muc.economics.environment import World, drifting, stationary
from muc.economics.memory import ExactCache, NoMemory, OracleMemory
from muc.economics.policy import Recompute, Refresh, Retrieve, Verify
from muc.economics.sim import simulate


class LedgerTests(unittest.TestCase):
    def test_total_and_reweighting(self):
        led = Ledger(computes=2, reads=10, writes=3, verifies=4, errors=1)
        m = CostModel(compute=50, read=1, write=1, verify=5, error=100)
        self.assertEqual(m.total(led), 100 + 10 + 3 + 20 + 100)
        self.assertEqual(m.replace(error=0).total(led), 133)

    def test_add(self):
        a, b = Ledger(computes=1, reads=2), Ledger(computes=3, errors=1)
        c = a + b
        self.assertEqual((c.computes, c.reads, c.errors), (4, 2, 1))


class WorldTests(unittest.TestCase):
    def test_stream_independent_of_truth_access(self):
        a, b = drifting(0.5, 0.05, seed=3), drifting(0.5, 0.05, seed=3)
        keys_a, keys_b = [], []
        for _ in range(300):
            keys_a.append(a.next_query())
            b.truth(b.next_query())
            keys_b.append(b._n_keys)  # touch internal progress only
        keys_b = []
        c = drifting(0.5, 0.05, seed=3)
        for _ in range(300):
            keys_b.append(c.next_query())
            c.truth(keys_b[-1])
        self.assertEqual(keys_a, keys_b)

    def test_truth_does_not_depend_on_who_looks_or_how_often(self):
        a, b = drifting(0.5, 0.1, seed=1), drifting(0.5, 0.1, seed=1)
        seen_a, seen_b = [], []
        for _ in range(200):
            ka, kb = a.next_query(), b.next_query()
            self.assertEqual(ka, kb)
            for _ in range(3):  # b looks several times, a looks once
                b.truth(kb)
            seen_a.append(a.truth(ka))
            seen_b.append(b.truth(kb))
        self.assertEqual(seen_a, seen_b)

    def test_stationary_truth_is_constant(self):
        w = stationary(0.9, seed=0)
        first = {}
        for _ in range(500):
            k = w.next_query()
            first.setdefault(k, w.truth(k))
            self.assertEqual(w.truth(k), first[k])

    def test_recurrence_rate(self):
        w = World(recurrence=0.7, seed=5)
        seen, repeats, n = set(), 0, 5000
        for _ in range(n):
            k = w.next_query()
            repeats += k in seen
            seen.add(k)
        self.assertAlmostEqual(repeats / n, 0.7, delta=0.03)

    def test_change_probability_matches_rate(self):
        # P(changed at least once within t steps) = 1 - (1 - d)^t for geometric gaps.
        d, t, n = 0.1, 10, 2000
        w = World(drift=d, seed=4)
        initial = [w.truth(k) for k in range(n)]  # all read at t = 0
        w.t = t
        changed = sum(w.truth(k) != initial[k] for k in range(n))
        self.assertAlmostEqual(changed / n, 1 - (1 - d) ** t, delta=0.03)

    def test_changed_value_differs(self):
        w = World(recurrence=0.0, drift=1.0, seed=0)
        k = w.next_query()
        v1 = w.truth(k)
        w.t += 1
        self.assertNotEqual(w.truth(k), v1)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            World(recurrence=1.5)
        with self.assertRaises(ValueError):
            World(n_values=1)


class MemoryTests(unittest.TestCase):
    def test_lru_eviction_order(self):
        c = ExactCache(capacity=2)
        c.write(1, 10, 0)
        c.write(2, 20, 0)
        c.read(1)  # 2 is now least recent
        c.write(3, 30, 0)
        self.assertIsNone(c.read(2))
        self.assertEqual(c.read(1).value, 10)
        self.assertEqual(c.read(3).value, 30)

    def test_forget(self):
        c = ExactCache()
        c.write(1, 5, 0)
        c.forget(1)
        self.assertIsNone(c.read(1))
        c.forget(1)  # no error

    def test_noise_rate_and_wrong_value_differs(self):
        c = ExactCache(noise=0.3, seed=1)
        c.write(1, 7, 0)
        wrong, n = 0, 4000
        for _ in range(n):
            v = c.read(1).value
            wrong += v != 7
        self.assertAlmostEqual(wrong / n, 0.3, delta=0.03)

    def test_entry_records_write_time(self):
        c = ExactCache()
        c.write(1, 5, 42)
        self.assertEqual(c.read(1).written_at, 42)

    def test_oracle_always_current(self):
        w = drifting(0.8, 0.2, seed=0)
        m = OracleMemory(w.truth)
        res = simulate(Retrieve(), m, w, 500)
        self.assertEqual(res.ledger.errors, 0)

    def test_no_memory_never_hits(self):
        res = simulate(Retrieve(), NoMemory(), stationary(0.9), 200)
        self.assertEqual(res.hits, 0)
        self.assertEqual(res.ledger.computes, 200)

    def test_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            ExactCache(capacity=0)


class PolicyTests(unittest.TestCase):
    def test_recompute_counts_and_never_errs(self):
        res = simulate(Recompute(), ExactCache(), drifting(0.5, 0.2, seed=0), 300)
        self.assertEqual((res.ledger.computes, res.ledger.reads, res.ledger.errors), (300, 0, 0))

    def test_retrieve_has_no_errors_in_stationary_clean_world(self):
        res = simulate(Retrieve(), ExactCache(), stationary(0.6, seed=0), 1000)
        self.assertEqual(res.ledger.errors, 0)
        self.assertEqual(res.ledger.reads, 1000)

    def test_retrieve_errs_when_world_drifts(self):
        res = simulate(Retrieve(), ExactCache(), drifting(0.8, 0.2, seed=0), 1000)
        self.assertGreater(res.ledger.errors, 0)

    def test_verify_never_wrong_with_perfect_checking(self):
        res = simulate(Verify(), ExactCache(), drifting(0.8, 0.2, seed=0), 1000)
        self.assertEqual(res.ledger.errors, 0)

    def test_verify_pays_per_hit(self):
        res = simulate(Verify(), ExactCache(), stationary(0.6, seed=0), 1000)
        self.assertEqual(res.ledger.verifies, res.hits)

    def test_refresh_ttl_zero_recomputes_every_hit_of_older_entries(self):
        res = simulate(Refresh(ttl=0), ExactCache(), stationary(0.6, seed=0), 500)
        self.assertEqual(res.ledger.computes, 500)

    def test_refresh_huge_ttl_behaves_like_retrieve(self):
        a = simulate(Refresh(ttl=10**9), ExactCache(), drifting(0.6, 0.05, seed=2), 600)
        b = simulate(Retrieve(), ExactCache(), drifting(0.6, 0.05, seed=2), 600)
        self.assertEqual(a.ledger, b.ledger)

    def test_same_world_across_policies(self):
        # Recompute is exact whatever the world does, so its cost is just the compute count.
        for seed in range(3):
            r = simulate(Recompute(), ExactCache(), drifting(0.5, 0.1, seed=seed), 400)
            self.assertEqual(r.ledger.computes, 400)

    def test_capacity_reduces_hit_rate(self):
        big = simulate(Retrieve(), ExactCache(), stationary(0.9, seed=0), 2000)
        small = simulate(Retrieve(), ExactCache(capacity=2), stationary(0.9, seed=0), 2000)
        self.assertLess(small.hit_rate, big.hit_rate)

    def test_negative_ttl_rejected(self):
        with self.assertRaises(ValueError):
            Refresh(ttl=-1)


if __name__ == "__main__":
    unittest.main()
