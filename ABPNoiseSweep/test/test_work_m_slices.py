"""Synthetic checks for the notebook's finite crossing-m work slices.

Run from the project root: python -B test/test_work_m_slices.py
The tests load the actual notebook functions without running its data discovery.
"""

import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

PROJECT = Path(__file__).resolve().parents[1]
notebook = json.loads((PROJECT / "plot_production_errors_work_theta.ipynb").read_text(encoding="utf-8"))
namespace = {"np": np, "pd": pd}
exec("".join(notebook["cells"][6]["source"]), namespace)
tree = ast.parse("".join(notebook["cells"][22]["source"]))
helper_definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(ast.Module(body=helper_definitions, type_ignores=[]), "notebook_work", "exec"), namespace)
WORK_PLOTS = "".join(notebook["cells"][24]["source"])


class WorkMSlices(unittest.TestCase):
    def setUp(self):
        self.case = dict(case_id="C001", file="synthetic.jld2", D=0.025, v=0.5, T=4)
        # Nonuniform widths. The center 1.0 lies exactly at one requested cut.
        self.edges = np.array([-0.1, 0.1, 0.5, 1.5, 2.5])
        self.mass = np.array([[9.0, 1.0, 3.0, 0.0],
                              [1.0, 3.0, 1.0, 0.0],
                              [0.0, 2.0, 0.0, 0.0]])
        self.values = np.array([[2.0, 20.0, 5.0, 0.0],
                                [10.0, 4.0, 9.0, 0.0],
                                [0.0, 7.0, 0.0, 0.0]])
        self.arrays = dict(mass=self.mass, mass2=self.mass,
                           sum_W=self.mass*self.values,
                           sum_W2=self.mass*self.values**2)

    def table(self, slices, case=None, arrays=None):
        return namespace["work_m_slice_table"](
            case or self.case, self.edges, arrays or self.arrays, slices).set_index("m_slice")

    def test_slices_pool_weighted_moments_before_ratios(self):
        row = self.table({"broad": (-0.05, 0.4)}).loc["broad"]
        num = (self.mass*self.values)[:, :2].sum(axis=1)
        den = self.mass[:, :2].sum(axis=1)
        expected_mean, expected_error = namespace["jackknife_ratio"](num, den)
        self.assertAlmostEqual(row.mean_active_work, num.sum()/den.sum())
        self.assertAlmostEqual(row.chain_jackknife_se, expected_error)
        # Neither unweighted bin means nor individual-chain means is the result.
        bin_means = (self.mass*self.values)[:, :2].sum(axis=0) / self.mass[:, :2].sum(axis=0)
        self.assertFalse(np.isclose(row.mean_active_work, bin_means.mean()))
        self.assertFalse(np.isclose(row.mean_active_work, (num/den).mean()))
        self.assertAlmostEqual(row.mean_normalized_active_work, expected_mean)
        self.assertEqual(row.n_crossing_y_bins, 2)
        self.assertAlmostEqual(row.effective_m_lo, -0.1)
        self.assertAlmostEqual(row.effective_m_hi, 0.5)

    def test_nonuniform_bins_use_center_half_open_selection(self):
        table = self.table({"below": (0.0, 1.0), "at_one": (1.0, 1.1)})
        # The 1.0 center is excluded from the first slice and included in second.
        self.assertEqual(table.loc["below", "n_crossing_y_bins"], 2)
        row = table.loc["at_one"]
        self.assertEqual(row.n_crossing_y_bins, 1)
        self.assertEqual(row.first_selected_m_center, 1.0)
        self.assertEqual(row.last_selected_m_center, 1.0)
        self.assertEqual(row.effective_m_lo, 0.5)
        self.assertEqual(row.effective_m_hi, 1.5)
        self.assertAlmostEqual(row.mean_active_work, 6.0)

    def test_narrow_slice_without_centers_is_explicitly_unresolved(self):
        row = self.table({"none": (0.11, 0.12)}).loc["none"]
        self.assertEqual(row.status, "no_bin_centers")
        self.assertEqual(row.n_crossing_y_bins, 0)
        for name in ["effective_m_lo", "effective_m_hi", "mean_active_work", "chain_jackknife_se", "slice_fraction"]:
            self.assertTrue(np.isnan(row[name]), name)

    def test_selected_but_unvisited_bins_do_not_invent_work(self):
        row = self.table({"empty": (1.9, 2.1)}).loc["empty"]
        self.assertEqual(row.status, "no_observed_mass")
        self.assertEqual(row.n_crossing_y_bins, 1)
        self.assertEqual(row.unbiased_mass, 0)
        self.assertTrue(np.isnan(row.mean_active_work))
        self.assertTrue(np.isnan(row.chain_jackknife_se))
        self.assertTrue(np.isnan(row.slice_fraction_se))

    def test_all_zero_masses_leave_all_ratios_undefined(self):
        arrays = {name: np.zeros_like(values) for name, values in self.arrays.items()}
        row = self.table({"near_zero": (-0.05, 0.05)}, arrays=arrays).loc["near_zero"]
        self.assertTrue(np.isnan(row.mean_active_work))
        self.assertTrue(np.isnan(row.slice_fraction))

    def test_zero_speed_retains_raw_work_and_masks_normalization(self):
        case = dict(self.case, v=0)
        row = self.table({"near_zero": (-0.05, 0.05)}, case=case).loc["near_zero"]
        self.assertAlmostEqual(row.mean_active_work, 2.8)
        self.assertTrue(np.isnan(row.mean_normalized_active_work))
        self.assertTrue(np.isnan(row.normalized_chain_jackknife_se))

    def test_chain_with_no_slice_mass_stays_in_jackknife(self):
        row = self.table({"near_zero": (-0.05, 0.05)}).loc["near_zero"]
        expected_mean, expected_error = namespace["jackknife_ratio"](
            self.arrays["sum_W"][:, 0], self.mass[:, 0])
        self.assertEqual(row.n_chains_total, 3)
        self.assertEqual(row.n_chains_with_mass, 2)
        self.assertAlmostEqual(row.mean_active_work, expected_mean)
        self.assertAlmostEqual(row.chain_jackknife_se, expected_error)

    def test_invalid_slice_bounds_and_moments_are_rejected(self):
        for limits in [(1, 1), (1, -1), (np.nan, 1)]:
            with self.assertRaises(ValueError):
                self.table({"invalid": limits})
        invalid = dict(self.arrays, mass=np.full_like(self.mass, -1))
        with self.assertRaises(ValueError):
            self.table({"valid": (-0.05, 0.05)}, arrays=invalid)

    def test_plot_source_has_valid_mathtext_escapes_and_theory(self):
        ast.parse(WORK_PLOTS)
        self.assertNotIn("\r", WORK_PLOTS)
        self.assertIn("raw_theory = theory_displacement * theory_v", WORK_PLOTS)
        self.assertIn("where=theory_scale > 0", WORK_PLOTS)

    def test_plot_theory_matches_raw_and_normalized_work_for_both_sweeps(self):
        slices = {"near_m0": (-0.05, 0.05), "near_m1": (0.95, 1.05)}
        for sweep in ("v", "D"):
            cases = [dict(self.case, case_id=f"C{i+1:03d}",
                          v=value if sweep == "v" else 0.5,
                          D=value if sweep == "D" else 0.025)
                     for i, value in enumerate((0.3, 0.5) if sweep == "v" else (0.01, 0.025))]
            summaries = [namespace["work_m_slice_table"](case, self.edges, self.arrays, slices)
                         for case in cases]
            profiles = {}
            for case in cases:
                den = self.mass.sum(axis=0)
                mean, se = namespace["jackknife_ratio"](self.arrays["sum_W"], self.mass)
                scale = case["v"]**2 * case["T"]
                profiles[case["case_id"]] = pd.DataFrame(dict(
                    crossing_y_center=(self.edges[:-1] + self.edges[1:])/2,
                    mean_active_work=mean, chain_jackknife_se=se,
                    mean_normalized_active_work=mean/scale,
                    normalized_chain_jackknife_se=se/scale))
            figures = []

            def finish(fig, name):
                fig.canvas.draw()  # Exercise mathtext labels and real plotting calls.
                figures.append(fig)

            plotting = dict(namespace, plt=plt, CASES=cases, WORK_PROFILES=profiles,
                            WORK_M_SLICE_SUMMARY=pd.concat(summaries, ignore_index=True),
                            ACTIVE_WORK_M_SLICES=slices, BAND_SIGMA=1.0,
                            SWEEP_PARAMETER=sweep, Y_DISPLAY=(-2, 2), FLAT_THEORY_DISPLACEMENT=2.1,
                            CASE_COLORS={case["case_id"]: f"C{i}" for i, case in enumerate(cases)},
                            add_speed_colorbar=lambda fig, axes: None,
                            finish_figure=finish, export_table=lambda table, name: None,
                            MaxNLocator=MaxNLocator, StrMethodFormatter=StrMethodFormatter)
            plotting.update(DETAIL_CASES=[], WORK_COMPONENT_PROFILES={},
                            WORK_COMPONENT_M_SLICE_SUMMARY=pd.DataFrame())
            try:
                exec(WORK_PLOTS, plotting)
                self.assertEqual(len(figures), 2)
                comparison = figures[-1]
                raw = next(line for line in comparison.axes[0].lines
                           if line.get_label() == r"Supplied flat theory $2.1v$")
                normalized = next(line for line in comparison.axes[1].lines
                                  if line.get_label() == r"Supplied flat theory $2.1v/(v^2T)$")
                velocities = np.array([case["v"] for case in cases])
                np.testing.assert_allclose(raw.get_ydata(), 2.1*velocities)
                np.testing.assert_allclose(normalized.get_ydata(), 2.1/(velocities*self.case["T"]))
                np.testing.assert_allclose(raw.get_xdata(), [case[sweep] for case in cases])
            finally:
                for fig in figures:
                    plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
