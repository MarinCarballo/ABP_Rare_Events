"""Validate actual notebook component means, chain errors, IO and plots.

Run from the project root with:
    python -B test/test_work_components_notebook.py
HDF5 fixtures and rendered figures stay in memory; no simulation is performed.
"""

import ast
import io
import json
import unittest
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((PROJECT / "plot_production_errors_work_theta.ipynb").read_text(encoding="utf-8"))


def notebook_functions(namespace, index):
    """Execute function definitions from the notebook without analysis loops."""
    parsed = ast.parse("".join(NOTEBOOK["cells"][index]["source"]))
    functions = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, f"notebook:cell{index}", "exec"), namespace)


class WorkComponentNotebook(unittest.TestCase):
    def setUp(self):
        self.namespace = dict(np=np, pd=pd, h5py=h5py)
        notebook_functions(self.namespace, 6)
        notebook_functions(self.namespace, 22)
        self.messages = []
        self.namespace.update(ACTIVE_WORK_M_SLICES={"near_m0": (-0.05, 0.05), "near_m1": (0.95, 1.05)},
                              note=lambda case, message: self.messages.append(message))
        self.edges = np.array([-0.1, 0.1, 0.5, 1.5])
        self.mass = np.array([[9.0, 1.0, 3.0], [1.0, 3.0, 1.0], [0.0, 2.0, 0.0]])
        self.wx = np.array([[2.0, 3.0, 5.0], [10.0, 4.0, -1.0], [0.0, 7.0, 0.0]])
        self.wy = np.array([[-1.0, 2.0, -2.0], [-9.0, -1.0, 3.0], [0.0, -5.0, 0.0]])
        self.total = self.wx + self.wy
        self.arrays = dict(mass=self.mass, mass2=self.mass,
                           sum_W=self.mass*self.total, sum_W2=self.mass*self.total**2,
                           sum_Wx=self.mass*self.wx, sum_Wy=self.mass*self.wy,
                           sum_Wx2=self.mass*self.wx**2, sum_Wy2=self.mass*self.wy**2,
                           sum_WxWy=self.mass*self.wx*self.wy)
        self.case = dict(case_id="C001", file="synthetic.jld2", D=0.025, v=0.5, T=4.0,
                         label="Synthetic", has_work=True)

    def products(self, slices=None, case=None, arrays=None):
        return self.namespace["work_component_products"](
            case or self.case, self.edges, arrays or self.arrays,
            slices or self.namespace["ACTIVE_WORK_M_SLICES"])

    def test_signed_components_use_same_normalizer_and_close_to_total(self):
        profile, summary, closure = self.products()
        near0 = summary[summary.m_slice == "near_m0"].set_index("component")
        self.assertAlmostEqual(near0.loc["x", "mean_active_work"], 2.8)
        self.assertAlmostEqual(near0.loc["y", "mean_active_work"], -1.8)
        self.assertAlmostEqual(near0.loc["total", "mean_active_work"], 1.0)
        self.assertEqual(set(near0.unbiased_mass), {10.0})
        self.assertTrue(closure.loc[closure.unbiased_mass > 0, "mean_closure_ok"].all())
        self.assertTrue(np.allclose(closure.mean_sum_minus_total, 0, atol=1e-12))
        self.assertTrue(np.allclose(closure.second_moment_sum_minus_total, 0, atol=1e-12))
        for _, group in profile.groupby("crossing_y_center"):
            mean = group.set_index("component").mean_active_work
            self.assertAlmostEqual(mean["x"] + mean["y"], mean["total"])

    def test_total_error_is_direct_ratio_not_quadrature_of_component_errors(self):
        _, summary, closure = self.products()
        near0 = summary[summary.m_slice == "near_m0"].set_index("component")
        se_x, se_y = near0.loc["x", "chain_jackknife_se"], near0.loc["y", "chain_jackknife_se"]
        self.assertGreater(np.hypot(se_x, se_y), 0)
        self.assertEqual(near0.loc["total", "chain_jackknife_se"], 0)
        row = closure[(closure.scope == "m_slice") & (closure.selection == "near_m0")].iloc[0]
        self.assertLess(row.weighted_covariance_x_y, 0)
        self.assertAlmostEqual(row.weighted_variance_total, row.weighted_variance_x_plus_y)
        self.assertAlmostEqual(row.weighted_variance_total, 0)

    def test_slice_moments_are_pooled_before_component_jackknife(self):
        _, summary, _ = self.products(slices={"wide": (-0.05, 0.4)})
        den = self.mass[:, :2].sum(axis=1)
        for component, name in [("x", "sum_Wx"), ("y", "sum_Wy"), ("total", "sum_W")]:
            num = self.arrays[name][:, :2].sum(axis=1)
            expected_mean, expected_se = self.namespace["jackknife_ratio"](num, den)
            row = summary[summary.component == component].iloc[0]
            self.assertAlmostEqual(row.mean_active_work, expected_mean)
            self.assertAlmostEqual(row.chain_jackknife_se, expected_se)
            self.assertEqual(row.n_chains_total, 3)

    def test_empty_slice_has_unknown_component_means(self):
        _, summary, closure = self.products(slices={"unresolved": (0.11, 0.12)})
        self.assertTrue(summary.mean_active_work.isna().all())
        self.assertTrue(summary.chain_jackknife_se.isna().all())
        self.assertEqual(set(summary.status), {"no_bin_centers"})
        rows = closure[closure.scope == "m_slice"]
        self.assertTrue(rows.mean_Wx.isna().all())
        self.assertFalse(rows.mean_closure_ok.any())

    def test_zero_speed_has_raw_components_but_no_normalized_estimate(self):
        profile, summary, _ = self.products(case=dict(self.case, v=0))
        self.assertTrue(np.isfinite(summary.mean_active_work).all())
        self.assertTrue(summary.mean_normalized_active_work.isna().all())
        self.assertTrue(profile.normalized_chain_jackknife_se.isna().all())

    def fixture(self, components=True, omit=None):
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as h5:
            h5["active_work/crossing_y_bins"] = self.edges
            h5["chain_statistics/n_chains"] = 3
            for chain in range(3):
                base = f"chain_statistics/chains/{chain + 1}/active_work/unbiased"
                for name, values in self.arrays.items():
                    if (not components and name not in {"mass", "mass2", "sum_W", "sum_W2"}) or name == omit:
                        continue
                    h5[f"{base}/{name}"] = values[chain]
        buffer.seek(0)
        return buffer

    def test_new_component_datasets_load(self):
        buffer = self.fixture()
        profile, summary, components = self.namespace["load_work_products"](dict(self.case, file=buffer))
        self.assertIsNotNone(components)
        self.assertEqual(set(components[1].component), {"x", "y", "total"})
        expected = summary.set_index("m_slice").mean_active_work
        measured = components[1].query("component == 'total'").set_index("m_slice").mean_active_work
        np.testing.assert_allclose(expected, measured)

    def test_legacy_total_data_retains_total_analysis_and_reports_missing_components(self):
        buffer = self.fixture(components=False)
        profile, summary, components = self.namespace["load_work_products"](dict(self.case, file=buffer))
        self.assertIsNone(components)
        self.assertFalse(profile.empty)
        self.assertFalse(summary.empty)
        self.assertTrue(any("component moments Wx/Wy unavailable" in message for message in self.messages))

    def test_partial_component_fields_are_not_reconstructed(self):
        buffer = self.fixture(omit="sum_WxWy")
        _, _, components = self.namespace["load_work_products"](dict(self.case, file=buffer))
        self.assertIsNone(components)

    def test_malformed_component_second_moments_rejected(self):
        broken = dict(self.arrays, sum_Wx2=-self.arrays["sum_Wx2"])
        with self.assertRaises(ValueError):
            self.products(arrays=broken)

    def test_actual_notebook_component_plots_render_in_memory(self):
        figures = []

        def finish(fig, name):
            fig.canvas.draw()
            figures.append(name)
            plt.close(fig)

        namespace = dict(self.namespace, plt=plt, MaxNLocator=MaxNLocator,
                         StrMethodFormatter=StrMethodFormatter, CASES=[self.case],
                         DETAIL_CASES=[self.case], CASE_COLORS={"C001": "C0"},
                         BAND_SIGMA=1, Y_DISPLAY=(-1, 1.5), SWEEP_PARAMETER="v",
                         export_table=lambda *args: None, finish_figure=finish,
                         add_speed_colorbar=lambda *args: None)
        buffer = self.fixture()
        total_profile, total_summary, components = namespace["load_work_products"](dict(self.case, file=buffer))
        namespace.update(WORK_PROFILES={"C001": total_profile}, WORK_M_SLICE_SUMMARY=total_summary,
                         WORK_COMPONENT_PROFILES={"C001": components[0]},
                         WORK_COMPONENT_M_SLICE_SUMMARY=components[1])
        exec("".join(NOTEBOOK["cells"][24]["source"]), namespace)
        self.assertEqual(len(figures), 5)
        self.assertTrue(any("x_y_total_profiles" in name for name in figures))
        self.assertTrue(any("x_y_total_near_m0" in name for name in figures))
        self.assertTrue(any("x_y_total_near_m1" in name for name in figures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
