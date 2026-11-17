import unittest

from muc.economics import __main__ as cli
from muc.economics.analytic import recompute_cost, retrieval_wins, retrieve_cost
from muc.economics.cost import CostModel
from muc.economics.experiments import drift, phase_diagram, static_baseline


class AnalyticTests(unittest.TestCase):
    def test_formulas(self):
        m = CostModel(compute=50, read=1, write=1, verify=5, error=100)
        self.assertEqual(recompute_cost(m), 50)
        # rho 0.6, q 0.9: 1 + 0.6*0.1*100 + 0.4*51
        self.assertAlmostEqual(retrieve_cost(0.6, 0.9, m), 1 + 6 + 20.4)

    def test_boundary_moves_the_right_way(self):
        m = CostModel()
        self.assertTrue(retrieval_wins(0.9, 1.0, m))
        self.assertFalse(retrieval_wins(0.9, 0.2, m))  # noisy memory loses
        cheap = m.replace(compute=1.0)
        self.assertFalse(retrieval_wins(0.9, 1.0, cheap))  # cheap compute loses


class H1Tests(unittest.TestCase):
    def test_simulator_matches_arithmetic(self):
        rows = static_baseline.run()
        self.assertTrue(static_baseline.passes(rows), [(r["compute"], r["noise"], r["z"]) for r in rows])

    def test_check_has_teeth(self):
        # A simulator with the wrong world (different recurrence than the model assumes) must fail.
        rows = static_baseline.run(steps=4000, seeds=(0, 1, 2), recurrence=0.6)
        for r in rows:
            r["sim_retrieve"] *= 1.15
            r["z"] = (r["sim_retrieve"] - r["pred_retrieve"]) / 0.05
        self.assertFalse(static_baseline.passes(rows))

    def test_grid_has_decided_cells_on_both_sides(self):
        rows = static_baseline.run(steps=1500, seeds=(0,), computes=(2, 100), noises=(0.0, 0.6))
        wins = {r["sign_ok"] for r in rows}
        self.assertIn(True, wins)


class DriftGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = drift.grid_ledgers(steps=600, seeds=(0,), recurrences=(0.5,), drifts=(0.0, 0.2))

    def test_cells_present_and_counts_consistent(self):
        self.assertEqual(set(self.grid), {(0.5, 0.0), (0.5, 0.2)})
        for cell in self.grid.values():
            self.assertEqual(set(cell), {"recompute", "retrieve", "verify", "refresh"})
            self.assertEqual(cell["recompute"][0].computes, 600)

    def test_rescoring_changes_costs_not_counts(self):
        cell = self.grid[(0.5, 0.2)]
        a = drift.scores(cell, CostModel())
        b = drift.scores(cell, CostModel().replace(error=0.0))
        self.assertLess(b["retrieve"], a["retrieve"])
        self.assertEqual(b["recompute"], a["recompute"])

    def test_extremes(self):
        self.assertEqual(drift.best(self.grid[(0.5, 0.0)], CostModel())[0], "retrieve")
        self.assertEqual(drift.best(self.grid[(0.5, 0.2)], CostModel())[0], "recompute")

    def test_phase_map_renders(self):
        text = phase_diagram.render(
            drift.grid_ledgers(steps=300, seeds=(0,), recurrences=drift.RECURRENCES, drifts=drift.DRIFTS),
            CostModel(),
        )
        self.assertIn("recurrence", text)
        self.assertEqual(len(text.splitlines()), 1 + len(drift.RECURRENCES))


class CliTests(unittest.TestCase):
    def test_usage(self):
        self.assertEqual(cli.main([]), 0)
        self.assertEqual(cli.main(["nonsense"]), 2)


if __name__ == "__main__":
    unittest.main()
