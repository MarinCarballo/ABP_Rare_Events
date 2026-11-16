"""Test actual notebook circular modes and the speed progression figure.

Run after applying the notebook patch:
    python -B test/test_theta_mode_tracking.py
Synthetic PDFs/figures stay in memory; existing CSVs are read only when present.
"""

import ast
import json
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((PROJECT / "plot_production_errors_work_theta.ipynb").read_text(encoding="utf-8"))


def actual_helpers():
    source = "".join(NOTEBOOK["cells"][10]["source"])
    parsed = ast.parse(source)
    definitions = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    namespace = dict(np=np, pd=pd, THETA_PEAK_MIN_PROMINENCE=0.03, THETA_PEAK_MIN_RELATIVE_HEIGHT=0.10)
    exec(compile(definitions, "notebook:cell10", "exec"), namespace)
    if "theta_modal_peaks" not in namespace:
        raise RuntimeError("Apply test/_patch_theta_progression.py before testing actual notebook helpers")
    return namespace


def pdf_table(density):
    edges = np.linspace(-np.pi, np.pi, len(density) + 1)
    widths = np.diff(edges)
    density = np.asarray(density, dtype=float)
    if np.isfinite(density).all() and np.sum(density*widths) > 0:
        density = density / np.sum(density*widths)
    return pd.DataFrame(dict(left=edges[:-1], right=edges[1:], center=(edges[:-1] + edges[1:])/2,
                             width=widths, unbiased_pdf=density,
                             unbiased_mass=1000*density*widths, chain_jackknife_se=0.01*density,
                             effective_x_lo=-0.4, effective_x_hi=-0.2))


class ThetaModeTracking(unittest.TestCase):
    def setUp(self):
        self.namespace = actual_helpers()
        self.theta = -np.pi + (np.arange(180) + 0.5)*(2*np.pi/180)

    def bimodal(self, angle=np.deg2rad(33)):
        return pdf_table(np.exp(60*np.cos(self.theta-angle)) + np.exp(60*np.cos(self.theta+angle)))

    def test_two_modes_use_exact_signed_bins_without_symmetry_fitting(self):
        table = self.bimodal()
        rows = self.namespace["theta_modal_peaks"](table)
        self.assertEqual(list(rows.branch), ["negative", "positive"])
        np.testing.assert_allclose(rows.theta_deg, [-33, 33], atol=1e-10)
        for row in rows.itertuples():
            source = table.iloc[np.argmin(np.abs(table.center - row.theta_rad))]
            self.assertEqual(row.theta_rad, source.center)
            self.assertEqual(row.theta_bin_left, source.left)
            self.assertEqual(row.theta_bin_right, source.right)
            self.assertAlmostEqual(np.rad2deg(row.theta_bin_width), 2)
            self.assertEqual(row.peak_pdf_se, source.chain_jackknife_se)
        self.assertNotIn("theta_se", rows.columns)

    def test_unimodal_zero_plateau_is_one_mode(self):
        rows = self.namespace["theta_modal_peaks"](pdf_table(np.exp(6*np.cos(self.theta))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0].branch, "central")

    def test_unimodal_seam_plateau_is_one_mode(self):
        rows = self.namespace["theta_modal_peaks"](pdf_table(np.exp(6*np.cos(self.theta-np.pi))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0].branch, "seam")

    def test_uniform_and_unobserved_pdfs_have_no_invented_modes(self):
        uniform = self.namespace["theta_modal_peaks"](pdf_table(np.ones(180)))
        self.assertTrue(uniform.empty)
        self.assertEqual(uniform.attrs["status"], "flat_pdf")
        missing = self.namespace["theta_modal_peaks"](pdf_table(np.full(180, np.nan)))
        self.assertTrue(missing.empty)
        self.assertEqual(missing.attrs["status"], "no_observed_pdf")

    def test_configured_filters_are_exposed_and_validated(self):
        with self.assertRaises(ValueError):
            self.namespace["theta_modal_peaks"](self.bimodal(), min_prominence=-0.1)
        config = "".join(NOTEBOOK["cells"][4]["source"])
        self.assertIn("THETA_PEAK_MIN_PROMINENCE", config)
        self.assertIn("THETA_PEAK_MIN_RELATIVE_HEIGHT", config)

    def test_exact_pdf_copies_are_marked_but_distinct_replicates_remain(self):
        table = self.bimodal()
        fingerprint = self.namespace["theta_pdf_fingerprint"](table)
        changed = table.copy()
        changed["unbiased_mass"] *= 2  # Same normalized PDF; a different ensemble.
        distinct = self.namespace["theta_pdf_fingerprint"](changed)
        self.assertNotEqual(fingerprint, distinct)
        rows = pd.DataFrame([dict(case_id=case, x_slice="upstream", D=0.025, v=0.39, T=40,
                                  pdf_fingerprint=fp, branch="positive", theta_rad=0.575)
                             for case, fp in [("C001", fingerprint), ("C002", fingerprint), ("C003", distinct)]])
        marked = self.namespace["theta_mode_plot_rows"](rows)
        self.assertEqual(marked.plot_duplicate.tolist(), [False, True, False])
        self.assertEqual(len(marked), 3)

    def test_actual_progression_plot_renders_without_pooling_copies_or_replicates(self):
        cases = [dict(case_id=f"C{i+1:03d}", file=f"case{i+1}.jld2", D=0.025, v=v, T=40)
                 for i, v in enumerate([0.32, 0.32, 0.39, 0.39])]
        tables = [self.bimodal(np.deg2rad(angle)) for angle in [21, 21, 33, 35]]
        path_data = {"theta": {case["case_id"]: dict(pdfs={"upstream": table})
                               for case, table in zip(cases, tables)}}
        figures = []

        def finish(fig, name):
            fig.canvas.draw()
            figures.append((fig, name))
            plt.close(fig)

        namespace = dict(self.namespace, plt=plt, MaxNLocator=MaxNLocator,
                         StrMethodFormatter=StrMethodFormatter, X_SLICES={"upstream": (-0.4, -0.2)},
                         CASES=cases, PATH_DATA=path_data, display=lambda *args: None,
                         export_table=lambda *args: None, finish_figure=finish)
        source = "".join(NOTEBOOK["cells"][20]["source"])
        source = source[source.index("# Theta mode progression:"):]
        exec(source, namespace)
        self.assertEqual(len(figures), 1)
        self.assertEqual(len(namespace["THETA_MODAL_PEAKS"]), 8)
        self.assertEqual(int(namespace["THETA_MODAL_PEAKS"].plot_duplicate.sum()), 2)
        self.assertTrue(all("distinct cases" in line.get_label()
                            for line in figures[0][0].axes[0].lines if len(line.get_xdata()) == 3))
        low, high = figures[0][0].axes[0].get_ylim()
        self.assertAlmostEqual(low, -high)
        self.assertLess(high, 1.0)  # Visible progression instead of a full ±pi axis.

    def test_present_highest_speed_upstream_data_gives_plus_minus_33_degrees(self):
        path = PROJECT / "data/plots_production_errors_work_theta/C029_path_theta_upstream_pdf.csv"
        if not path.exists():
            self.skipTest("Optional existing upstream CSV not present")
        rows = self.namespace["theta_modal_peaks"](pd.read_csv(path))
        np.testing.assert_allclose(rows.theta_deg, [-33, 33], atol=1e-9)

    def test_present_central_wiggle_is_not_a_third_mode(self):
        path = PROJECT / "data/plots_production_errors_work_theta/C007_path_theta_upstream_pdf.csv"
        if not path.exists():
            self.skipTest("Optional existing upstream CSV not present")
        rows = self.namespace["theta_modal_peaks"](pd.read_csv(path))
        self.assertEqual(len(rows), 2)
        np.testing.assert_allclose(rows.theta_deg, [-27, 27], atol=1e-9)

    def test_present_asymmetry_and_nonmonotonic_step_are_preserved(self):
        root = PROJECT / "data/plots_production_errors_work_theta"
        paths = [root / f"C{number:03d}_path_theta_upstream_pdf.csv" for number in [7, 9]]
        if not all(path.exists() for path in paths):
            self.skipTest("Optional existing upstream CSVs not present")
        before, after = [self.namespace["theta_modal_peaks"](pd.read_csv(path)) for path in paths]
        np.testing.assert_allclose(before.theta_deg, [-27, 27], atol=1e-9)
        np.testing.assert_allclose(after.theta_deg, [-25, 27], atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
