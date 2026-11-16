"""Check the actual notebook's modal-angle geometric work reference.

Run from the project root: python -B test/test_work_cosine_theory.py
Only the helper definition and literal configuration values are extracted;
production data discovery and notebook plotting are never executed.
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
NOTEBOOK_PATH = PROJECT / "plot_production_errors_work_theta.ipynb"
THEORY_COLUMNS = (
    "theory_displacement", "cos_theta_star", "flat_work_theory", "sb_work_theory",
    "flat_work_theory_normalized", "sb_work_theory_normalized",
)


def load_actual_theory_helper():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    definitions = {}
    wanted_functions = {"cosine_work_theory", "plot_cosine_work_reference"}
    settings = {}
    wanted_settings = {"SB_THEORY_THETA_SLICE", "FLAT_THEORY_DISPLACEMENT"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK_PATH.name}:cell{index}")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
                definitions[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted_settings:
                        settings[target.id] = ast.literal_eval(node.value)
    if definitions.keys() != wanted_functions:
        raise AssertionError(f"Missing notebook helpers: {sorted(wanted_functions - definitions.keys())}")
    missing = wanted_settings - settings.keys()
    if missing:
        raise AssertionError(f"Missing literal theory settings: {sorted(missing)}")
    namespace = {"np": np, "pd": pd, **settings}
    exec(compile(ast.Module(body=list(definitions.values()), type_ignores=[]),
                 str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


class CosineWorkTheory(unittest.TestCase):
    def setUp(self):
        self.namespace = load_actual_theory_helper()
        self.theory = self.namespace["cosine_work_theory"]
        self.rows = pd.DataFrame([
            dict(case_id="A", file="a.jld2", D=0.025, v=0.4, T=10.0,
                 branch="negative", theta_rad=-np.pi / 3, x_slice="crossing",
                 plot_duplicate=False, pdf_fingerprint="same-A", peak_pdf_se=0.03),
            dict(case_id="A", file="a.jld2", D=0.025, v=0.4, T=10.0,
                 branch="positive", theta_rad=np.pi / 4, x_slice="crossing",
                 plot_duplicate=False, pdf_fingerprint="same-A", peak_pdf_se=0.02),
            dict(case_id="A_copy", file="copied-a.jld2", D=0.025, v=0.4, T=10.0,
                 branch="negative", theta_rad=-np.pi / 3, x_slice="crossing",
                 plot_duplicate=True, pdf_fingerprint="same-A", peak_pdf_se=0.03),
            dict(case_id="B", file="b.jld2", D=0.05, v=0.6, T=20.0,
                 branch="positive", theta_rad=np.pi / 6, x_slice="crossing",
                 plot_duplicate=False, pdf_fingerprint="B", peak_pdf_se=0.04),
            dict(case_id="A", file="a.jld2", D=0.025, v=0.4, T=10.0,
                 branch="positive", theta_rad=0.2, x_slice="upstream",
                 plot_duplicate=False, pdf_fingerprint="A-upstream", peak_pdf_se=0.01),
        ], index=[8, 3, 14, 6, 2])

    def test_actual_default_is_user_selected_2p1_and_crossing_slice(self):
        self.assertEqual(self.namespace["FLAT_THEORY_DISPLACEMENT"], 2.1)
        self.assertEqual(self.namespace["SB_THEORY_THETA_SLICE"], "crossing")
        table = self.theory(self.rows)
        self.assertEqual(len(table), 4)
        self.assertTrue((table.x_slice == "crossing").all())
        np.testing.assert_allclose(table.theory_displacement, 2.1)
        np.testing.assert_allclose(table.flat_work_theory, 2.1 * table.v)
        np.testing.assert_allclose(table.sb_work_theory, 2.1 * table.v * np.cos(table.theta_rad))

    def test_normalization_uses_v_squared_T_independently_of_displacement(self):
        for displacement in (2.0, 2.1, 3.7):
            with self.subTest(displacement=displacement):
                table = self.theory(self.rows, displacement=displacement)
                expected_flat = table.v.to_numpy() * displacement
                expected_sb = expected_flat * np.cos(table.theta_rad.to_numpy())
                expected_scale = table["v"].to_numpy() ** 2 * table["T"].to_numpy()
                np.testing.assert_allclose(table.flat_work_theory, expected_flat)
                np.testing.assert_allclose(table.sb_work_theory, expected_sb)
                np.testing.assert_allclose(table.flat_work_theory_normalized, expected_flat / expected_scale)
                np.testing.assert_allclose(table.sb_work_theory_normalized, expected_sb / expected_scale)
                # v**2.1 was an accidental earlier normalization and must not
                # return when the displacement happens to be set to 2.1.
                wrong_scale = table["v"].to_numpy() ** 2.1 * table["T"].to_numpy()
                self.assertFalse(np.allclose(table.sb_work_theory_normalized, expected_sb / wrong_scale))

    def test_both_asymmetric_branches_and_duplicate_metadata_are_retained(self):
        table = self.theory(self.rows)
        source = self.rows[self.rows.x_slice == "crossing"]
        pd.testing.assert_frame_equal(table[list(source.columns)].reset_index(drop=True),
                                      source.reset_index(drop=True))
        a = table[table.case_id == "A"].set_index("branch")
        self.assertEqual(set(a.index), {"negative", "positive"})
        self.assertAlmostEqual(a.loc["negative", "sb_work_theory"], 0.4 * 2.1 * 0.5)
        self.assertAlmostEqual(a.loc["positive", "sb_work_theory"], 0.4 * 2.1 / np.sqrt(2))
        self.assertFalse(np.isclose(a.loc["negative", "sb_work_theory"],
                                    a.loc["positive", "sb_work_theory"]))
        self.assertEqual(int(table.plot_duplicate.sum()), 1)
        self.assertEqual(set(table.case_id), {"A", "A_copy", "B"})

    def test_source_is_unchanged_and_result_is_an_independent_copy(self):
        before = self.rows.copy(deep=True)
        table = self.theory(self.rows)
        pd.testing.assert_frame_equal(self.rows, before)
        table.loc[:, "theta_rad"] = 0.0
        table.loc[:, "file"] = "changed-output-only"
        pd.testing.assert_frame_equal(self.rows, before)
        # Existing PDF height SE is metadata, not an angular uncertainty that
        # can be propagated as a work-theory confidence interval.
        self.assertEqual(set(table.columns) - set(self.rows.columns), set(THEORY_COLUMNS))

    def test_explicit_slice_never_joins_or_substitutes_other_slices(self):
        upstream = self.theory(self.rows, x_slice="upstream", displacement=2.0)
        self.assertEqual(len(upstream), 1)
        self.assertAlmostEqual(upstream.sb_work_theory.iloc[0], 0.8 * np.cos(0.2))
        self.assertEqual(upstream.pdf_fingerprint.iloc[0], "A-upstream")
        missing = self.theory(self.rows, x_slice="not_saved")
        self.assertTrue(missing.empty)
        self.assertTrue(set(THEORY_COLUMNS) <= set(missing.columns))
        self.assertTrue(set(self.rows.columns) <= set(missing.columns))

    def test_empty_input_does_not_invent_a_zero_angle_mode(self):
        for empty in (pd.DataFrame(), self.rows.iloc[:0]):
            with self.subTest(columns=list(empty.columns)):
                result = self.theory(empty)
                self.assertTrue(result.empty)
                self.assertTrue(set(THEORY_COLUMNS) <= set(result.columns))

    def test_signed_symmetric_central_and_seam_modes_use_radians(self):
        rows = pd.DataFrame([
            dict(case_id=f"K{i}", D=0.025, v=0.4, T=10, branch=branch,
                 theta_rad=theta, x_slice="crossing")
            for i, (branch, theta) in enumerate([
                ("negative", -np.pi / 3), ("positive", np.pi / 3),
                ("central", 0.0), ("seam", -np.pi), ("seam", np.pi),
            ])
        ])
        result = self.theory(rows, displacement=2.0)
        np.testing.assert_allclose(result.cos_theta_star, [0.5, 0.5, 1.0, -1.0, -1.0], atol=1e-15)
        np.testing.assert_allclose(result.sb_work_theory, [0.4, 0.4, 0.8, -0.8, -0.8], atol=1e-15)
        self.assertTrue((np.abs(result.sb_work_theory) <= result.flat_work_theory + 1e-15).all())
        self.assertEqual(result.branch.tolist(), rows.branch.tolist())

    def test_zero_speed_and_zero_duration_leave_raw_theory_and_mask_normalization(self):
        rows = self.rows.iloc[:2].copy()
        rows.loc[rows.index[0], "v"] = 0.0
        rows.loc[rows.index[1], "T"] = 0.0
        with np.errstate(divide="raise", invalid="raise"):
            result = self.theory(rows)
        np.testing.assert_allclose(result.flat_work_theory, [0.0, 0.84])
        self.assertEqual(float(result.sb_work_theory.iloc[0]), 0.0)
        self.assertTrue(result.flat_work_theory_normalized.isna().all())
        self.assertTrue(result.sb_work_theory_normalized.isna().all())

    def test_invalid_displacement_is_rejected(self):
        for displacement in (0.0, -2.0, np.nan, np.inf, -np.inf):
            with self.subTest(displacement=displacement), self.assertRaises(ValueError):
                self.theory(self.rows, displacement=displacement)

    def test_nonfinite_or_nonphysical_source_values_are_rejected(self):
        for column, value in (("D", np.nan), ("D", np.inf), ("v", np.inf),
                              ("v", -0.1), ("T", np.nan), ("T", -1),
                              ("theta_rad", np.nan), ("theta_rad", np.inf),
                              ("theta_rad", np.pi + 0.01), ("theta_rad", -np.pi - 0.01)):
            with self.subTest(column=column, value=value):
                invalid = self.rows.copy()
                invalid.loc[invalid.index[0], column] = value
                with self.assertRaises(ValueError):
                    self.theory(invalid)

    def test_nonempty_input_requires_source_identity_and_angle_columns(self):
        for column in ("case_id", "D", "v", "T", "branch", "theta_rad", "x_slice"):
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.theory(self.rows.drop(columns=column))

    def test_plot_deduplicates_inside_selected_cases_without_losing_signed_branches(self):
        records = []
        for case_id, speed, duplicate, fingerprint in (
            ("outside_representative", 0.4, False, "same-copy"),
            ("selected_copy", 0.4, True, "same-copy"),
            ("second_selected_copy", 0.4, True, "same-copy"),
            ("selected_later_speed", 0.6, False, "later"),
            ("unselected_other_run", 0.4, False, "other"),
        ):
            for branch, angle in (("negative", -0.6), ("positive", 0.4)):
                records.append(dict(case_id=case_id, D=0.025, T=40.0, v=speed,
                                    x_slice="crossing", branch=branch, theta_rad=angle,
                                    plot_duplicate=duplicate, pdf_fingerprint=fingerprint))
        rows = self.theory(pd.DataFrame(records))
        before = rows.copy(deep=True)
        cases = [dict(case_id=name, v=speed, D=0.025, T=40.0) for name, speed in (
            ("selected_copy", 0.4), ("second_selected_copy", 0.4),
            ("selected_later_speed", 0.6),
        )]
        fig, axes = plt.subplots(1, 2)
        try:
            self.namespace["plot_cosine_work_reference"](axes, rows, cases, sweep_parameter="v")
            for ax, normalized in zip(axes, (False, True)):
                lines = {line.get_marker(): line for line in ax.lines}
                self.assertEqual(set(lines), {"v", "^"})
                for marker, angle in (("v", -0.6), ("^", 0.4)):
                    line = lines[marker]
                    np.testing.assert_allclose(line.get_xdata(), [0.4, 0.6])
                    expected = 2.1 * np.array([0.4, 0.6]) * np.cos(angle)
                    if normalized:
                        expected /= np.array([0.4, 0.6]) ** 2 * 40.0
                    np.testing.assert_allclose(line.get_ydata(), expected)
                    self.assertNotEqual(line.get_linestyle(), "None")
            pd.testing.assert_frame_equal(rows, before)
        finally:
            plt.close(fig)

    def test_actual_standalone_figure_keeps_unresolved_speed_as_a_nan_gap(self):
        records = []
        for case_id, speed in (("first", 0.30), ("last", 0.40)):
            for branch, angle in (("negative", -0.6), ("positive", 0.4)):
                records.append(dict(case_id=case_id, D=0.025, T=40.0, v=speed,
                                    x_slice="crossing", branch=branch, theta_rad=angle,
                                    plot_duplicate=False, pdf_fingerprint=case_id))
        theory = self.theory(pd.DataFrame(records))
        cases = [dict(case_id=name, v=speed, D=D, T=T) for name, speed, D, T in (
            ("first", 0.30, 0.025, 40.0),
            ("no_resolved_modes", 0.35, 0.025, 40.0),
            ("last", 0.40, 0.025, 40.0),
            ("other_D", 0.45, 0.05, 40.0),
            ("other_T", 0.50, 0.025, 20.0),
        )]
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        marker = "# A separate geometric-reference figure"
        snippets = ["".join(cell["source"]).split(marker, 1)[1]
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
        self.assertEqual(len(snippets), 1)
        figures = []
        namespace = dict(self.namespace, np=np, pd=pd, plt=plt,
                         MaxNLocator=MaxNLocator, StrMethodFormatter=StrMethodFormatter,
                         SB_WORK_THEORY=theory, CASES=cases, theory_displacement=2.1,
                         finish_figure=lambda fig, name: figures.append((fig, name)))
        try:
            exec(compile(ast.parse(marker + snippets[0]), "notebook_standalone_cosine_plot", "exec"), namespace)
            self.assertEqual(len(figures), 1)
            fig, name = figures[0]
            self.assertIn("D0.025_T40", name)
            for ax, reduction_only in zip(fig.axes, (False, True)):
                lines = {line.get_marker(): line for line in ax.lines if line.get_marker() in {"v", "^"}}
                self.assertEqual(set(lines), {"v", "^"})
                for marker, angle in (("v", -0.6), ("^", 0.4)):
                    np.testing.assert_allclose(lines[marker].get_xdata(), [0.30, 0.35, 0.40])
                    values = np.asarray(lines[marker].get_ydata(), dtype=float)
                    self.assertTrue(np.isnan(values[1]))
                    expected = np.full(2, np.cos(angle))
                    if not reduction_only:
                        expected *= 2.1 * np.array([0.30, 0.40])
                    np.testing.assert_allclose(values[[0, 2]], expected)
            fig.canvas.draw()
        finally:
            for fig, _ in figures:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
