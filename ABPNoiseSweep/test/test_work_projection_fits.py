"""Analytic checks for unconstrained active-work displacement projection fits.

Run from the project root: python -B test/test_work_projection_fits.py
The actual low-level fitting function is extracted with AST; the notebook,
production-file discovery, exports, and plotting are not executed.
"""

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts" / "fit_active_work_xy_projection.py"


def actual_projection_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if sum(node.name == "fit_linear_work_projection" for node in definitions) != 1:
        raise AssertionError("Expected one fit_linear_work_projection in the actual fitting script")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class WorkProjectionFits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = actual_projection_functions()
        cls.fit = staticmethod(namespace["fit_linear_work_projection"])
        cls.wrapper = staticmethod(namespace["fit_active_work_projection"])

    def test_two_parameter_projection_recovers_both_unconstrained_displacements(self):
        for sign in (-1, 1):
            with self.subTest(branch_sign=sign):
                speed = np.array([0.30, 0.32, 0.35, 0.38, 0.40])
                theta = sign * np.array([0.10, 0.25, 0.45, 0.60, 0.75])
                design = speed[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])
                truth = np.array([2.1, -0.7])
                result = self.fit(design @ truth, design)
                np.testing.assert_allclose(result["coefficients"], truth, atol=2e-14)
                self.assertEqual(result["rank"], 2)
                self.assertEqual(result["n_fit"], 5)
                self.assertEqual(result["dof"], 3)
                self.assertEqual(result["status"], "ok")
                self.assertFalse(result["weighted"])
                self.assertTrue(np.isnan(result["chi2"]))
                self.assertTrue(np.isnan(result["reduced_chi2"]))

    def test_ols_covariance_uses_residual_scale_and_n_minus_two(self):
        design = np.array([[1, 0], [0, 1], [1, 1], [2, -1], [2, 1]], dtype=float)
        observed = design @ np.array([2.1, 0.8]) + np.array([0.3, -0.1, 0.2, -0.4, 0.1])
        expected = np.linalg.solve(design.T @ design, design.T @ observed)
        residual = observed - design @ expected
        rss = residual @ residual
        covariance = np.linalg.inv(design.T @ design) * rss / (len(observed) - 2)
        result = self.fit(observed, design)
        np.testing.assert_allclose(result["coefficients"], expected, atol=1e-14)
        np.testing.assert_allclose(result["covariance"], covariance, rtol=2e-13)
        np.testing.assert_allclose(result["parameter_se"], np.sqrt(np.diag(covariance)), rtol=2e-13)
        self.assertAlmostEqual(result["residual_sum_squares"], rss)
        self.assertAlmostEqual(result["rmse"], np.sqrt(rss / len(observed)))
        self.assertEqual(result["dof"], 3)

    def test_wls_covariance_is_absolute_and_not_residual_rescaled(self):
        speed = np.array([0.3, 0.32, 0.35, 0.38, 0.4])
        theta = np.array([0.1, 0.25, 0.45, 0.6, 0.75])
        design = speed[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])
        errors = np.array([0.02, 0.03, 0.01, 0.04, 0.025])
        observed = design @ np.array([2.1, 0.8]) + np.array([0.01, -0.025, 0.03, -0.01, 0.02])
        weighted_design = design / errors[:, None]
        weighted_observed = observed / errors
        information = weighted_design.T @ weighted_design
        expected = np.linalg.solve(information, weighted_design.T @ weighted_observed)
        covariance = np.linalg.inv(information)
        result = self.fit(observed, design, errors)
        residual = observed - design @ expected
        chi2 = np.sum((residual / errors) ** 2)
        self.assertTrue(result["weighted"])
        np.testing.assert_allclose(result["coefficients"], expected, rtol=1e-13)
        np.testing.assert_allclose(result["covariance"], covariance, rtol=1e-13)
        np.testing.assert_allclose(result["parameter_se"], np.sqrt(np.diag(covariance)), rtol=1e-13)
        self.assertAlmostEqual(result["chi2"], chi2)
        self.assertAlmostEqual(result["reduced_chi2"], chi2 / 3)
        self.assertAlmostEqual(result["condition_number"], np.linalg.cond(weighted_design))

    def test_flat_projection_is_one_parameter_through_the_origin(self):
        speed = np.array([0.2, 0.4, 0.7, 1.0])
        observed = 1.3 * speed + 0.2  # An offset must not silently become an intercept.
        expected = np.dot(speed, observed) / np.dot(speed, speed)
        result = self.fit(observed, speed)
        same = self.fit(observed, speed[:, None])
        self.assertEqual(result["coefficients"].shape, (1,))
        self.assertAlmostEqual(result["coefficients"][0], expected)
        self.assertGreater(abs(expected - 1.3), 0.1)
        self.assertEqual(result["dof"], len(speed) - 1)
        rss = np.sum((observed - speed * expected) ** 2)
        self.assertAlmostEqual(result["covariance"][0, 0], rss / 3 / np.dot(speed, speed))
        np.testing.assert_allclose(same["coefficients"], result["coefficients"])
        np.testing.assert_allclose(same["covariance"], result["covariance"])

    def test_negative_flat_displacement_is_not_forced_positive(self):
        speed = np.array([0.2, 0.4, 0.7])
        errors = np.full(3, 0.1)
        result = self.fit(-2.0 * speed, speed, errors)
        self.assertAlmostEqual(result["coefficients"][0], -2.0)
        self.assertAlmostEqual(result["parameter_se"][0], 0.1 / np.linalg.norm(speed))
        self.assertGreater(result["parameter_se"][0], 0.0)  # Absolute SE survives an exact fit.

    def test_partial_error_availability_selects_valid_weighted_rows(self):
        theta = np.linspace(0.1, 0.8, 6)
        design = 0.4 * np.column_stack([np.cos(theta), np.sin(theta)])
        observed = design @ np.array([2.1, -0.7])
        observed[2] = np.nan
        design[4, 0] = np.inf
        errors = np.array([0.1, np.nan, 0.2, 0.0, 0.1, 0.3])
        result = self.fit(observed, design, errors)
        np.testing.assert_array_equal(result["fit_mask"], [True, False, False, False, False, True])
        self.assertTrue(result["weighted"])
        self.assertEqual(result["n_fit"], 2)
        self.assertEqual(result["dof"], 0)
        np.testing.assert_allclose(result["coefficients"], [2.1, -0.7], atol=1e-14)
        self.assertTrue(np.isfinite(result["parameter_se"]).all())
        self.assertTrue(np.isnan(result["reduced_chi2"]))

    def test_all_unusable_errors_fall_back_to_ols_and_weighting_can_be_disabled(self):
        speed = np.array([0.2, 0.4, 0.7, 1.0])
        observed = np.array([0.3, 0.8, 1.0, 2.4])
        expected = self.fit(observed, speed)
        for errors, use_errors in (([0, np.nan, -1, np.inf], True),
                                   ([0.01, 10.0, 0.02, 20.0], False)):
            with self.subTest(use_errors=use_errors):
                result = self.fit(observed, speed, errors, use_jackknife_se=use_errors)
                self.assertFalse(result["weighted"])
                np.testing.assert_array_equal(result["fit_mask"], np.ones(4, dtype=bool))
                np.testing.assert_allclose(result["coefficients"], expected["coefficients"])
                np.testing.assert_allclose(result["covariance"], expected["covariance"])

    def test_nonfinite_observation_or_design_rows_are_masked_without_mutation(self):
        speed = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        observed = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        before_speed, before_observed = speed.copy(), observed.copy()
        result = self.fit(observed, speed)
        np.testing.assert_array_equal(result["fit_mask"], [True, False, False, False, True])
        self.assertEqual(result["n_fit"], 2)
        self.assertAlmostEqual(result["coefficients"][0], 1.0)
        np.testing.assert_array_equal(speed, before_speed)
        np.testing.assert_array_equal(observed, before_observed)

    def test_constant_angle_is_rank_deficient_even_when_speed_varies(self):
        speed = np.linspace(0.3, 0.4, 8)
        design = speed[:, None] * np.array([[np.cos(0.4), np.sin(0.4)]])
        observed = design @ np.array([2.1, 0.8])
        result = self.fit(observed, design)
        self.assertEqual(result["status"], "rank_deficient")
        self.assertEqual(result["rank"], 1)
        self.assertTrue(np.isnan(result["coefficients"]).all())
        self.assertTrue(np.isnan(result["covariance"]).all())
        self.assertTrue(np.isnan(result["parameter_se"]).all())
        self.assertGreater(result["condition_number"], 1e12)

    def test_narrow_angle_range_reports_weak_identification_and_large_correlated_errors(self):
        theta = 0.4 + np.linspace(-1, 1, 9) * 1e-9
        speed = np.linspace(0.3, 0.4, 9)
        design = speed[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])
        result = self.fit(design @ np.array([2.1, 0.8]), design, np.full(9, 0.1))
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["status"], "weakly_identified")
        self.assertGreater(result["condition_number"], 1e8)
        self.assertTrue(np.isfinite(result["coefficients"]).all())
        self.assertTrue((result["parameter_se"] > 1e5).all())
        covariance = result["covariance"]
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        self.assertGreater(abs(correlation), 0.999999)

    def test_exactly_determined_ols_has_no_residual_scale_error_estimate(self):
        for design, truth in ((np.array([[0.4]]), np.array([2.1])),
                              (np.array([[0.4, 0.1], [0.2, 0.3]]), np.array([2.1, -0.7]))):
            with self.subTest(parameters=len(truth)):
                result = self.fit(design @ truth, design)
                np.testing.assert_allclose(result["coefficients"], truth, atol=1e-14)
                self.assertEqual(result["dof"], 0)
                self.assertTrue(np.isnan(result["covariance"]).all())
                self.assertTrue(np.isnan(result["parameter_se"]).all())

    def test_too_few_finite_rows_do_not_return_a_minimum_norm_parameter_pair(self):
        for observed, design in (([], np.empty((0, 2))),
                                 ([1.0], np.array([[0.4, 0.2]])),
                                 ([np.nan, np.inf], np.array([[0.4, 0.2], [0.3, 0.1]]))):
            with self.subTest(observed=observed):
                result = self.fit(observed, design)
                self.assertEqual(result["status"], "insufficient_data")
                self.assertLess(result["n_fit"], 2)
                self.assertTrue(np.isnan(result["coefficients"]).all())
                self.assertTrue(np.isnan(result["covariance"]).all())

    def wrapper_data(self):
        modes, measured = [], []
        for i, (speed, theta) in enumerate(zip([0.30, 0.35, 0.40], [0.1, 0.4, 0.7])):
            case_id = f"sb{i}"
            work = speed * (2.1 * np.cos(theta) + 0.8 * np.sin(theta))
            measured.append(dict(case_id=case_id, file=f"{case_id}.jld2", D=0.025, v=speed, T=40.0,
                                 m_slice="near_m1", status="ok", mean_active_work=work,
                                 chain_jackknife_se=0.01, mean_normalized_active_work=999.0,
                                 unbiased_mass=100.0, effective_m_lo=0.5, effective_m_hi=1.05))
            for branch, signed_angle in (("negative", -theta), ("positive", theta)):
                modes.append(dict(case_id=case_id, D=0.025, v=speed, T=40.0,
                                  branch=branch, theta_rad=signed_angle, x_slice="crossing",
                                  pdf_fingerprint=case_id, identical_pdf_representative_case_id=case_id))
        for i, speed in enumerate([0.2, 0.3, 0.5]):
            measured.append(dict(case_id=f"flat{i}", file=f"flat{i}.jld2", D=0.025, v=speed, T=40.0,
                                 m_slice="near_m0", status="ok", mean_active_work=2.7 * speed,
                                 chain_jackknife_se=0.01, mean_normalized_active_work=999.0,
                                 unbiased_mass=100.0, effective_m_lo=-0.05, effective_m_hi=0.05))
        return pd.DataFrame(modes), pd.DataFrame(measured), {"near_m1": (0.5, 1.05), "near_m0": (-0.05, 0.05)}

    def test_wrapper_uses_raw_near_m1_and_independent_angle_free_near_m0_cases(self):
        modes, measured, slices = self.wrapper_data()
        extra = measured.iloc[[0]].copy()
        extra.loc[:, "case_id"] = "no_modal_angle_for_this_case"
        extra.loc[:, "mean_active_work"] = 1000.0
        wrong_slice = measured.iloc[[0]].copy()
        wrong_slice.loc[:, "m_slice"] = "other_slice"
        wrong_slice.loc[:, "mean_active_work"] = 2000.0
        measured = pd.concat([measured, extra, wrong_slice], ignore_index=True)
        before_modes, before_work = modes.copy(deep=True), measured.copy(deep=True)
        summary, predictions, flat, flat_predictions = self.wrapper(modes, measured, slices)
        self.assertEqual(set(summary.branch), {"negative", "positive"})
        for row in summary.itertuples():
            self.assertEqual(row.n_fit, 3)
            self.assertEqual(row.n_parameters, 2)
            self.assertEqual(row.m_slice, "near_m1")
            self.assertAlmostEqual(row.delta_x_fit, 2.1)
            self.assertAlmostEqual(row.delta_y_fit, -0.8 if row.branch == "negative" else 0.8)
        self.assertEqual(set(predictions.case_id), {"sb0", "sb1", "sb2"})
        self.assertTrue((predictions.m_slice == "near_m1").all())
        np.testing.assert_allclose(predictions.projected_active_work, predictions.mean_active_work, atol=1e-14)
        np.testing.assert_allclose(predictions.projected_active_work_normalized,
                                   predictions.projected_active_work / (predictions.v**2 * predictions["T"]))
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat.n_parameters.iloc[0], 1)
        self.assertEqual(flat.n_fit.iloc[0], 3)
        self.assertEqual(flat.dof.iloc[0], 2)
        self.assertAlmostEqual(flat.delta_x_fit.iloc[0], 2.7)
        self.assertEqual(set(flat_predictions.case_id), {"flat0", "flat1", "flat2"})
        self.assertTrue((flat_predictions.delta_y == 0).all())
        pd.testing.assert_frame_equal(modes, before_modes)
        pd.testing.assert_frame_equal(measured, before_work)
        # Flat data remain usable when no angle histograms exist at all.
        no_sb, _, flat_without_modes, _ = self.wrapper(pd.DataFrame(), measured, slices)
        self.assertTrue(no_sb.empty)
        self.assertAlmostEqual(flat_without_modes.delta_x_fit.iloc[0], 2.7)

    def test_wrapper_checks_physical_settings_after_exact_case_id_matching(self):
        modes, measured, slices = self.wrapper_data()
        for parameter, change in (("D", 0.01), ("v", 0.01), ("T", 1.0)):
            with self.subTest(parameter=parameter):
                wrong = measured.copy()
                wrong.loc[wrong.case_id == "sb0", parameter] += change
                with self.assertRaisesRegex(ValueError, "settings disagree"):
                    self.wrapper(modes, wrong, slices)

    def test_wrapper_counts_known_copies_once_only_when_work_statistics_agree(self):
        modes, measured, slices = self.wrapper_data()
        copied_modes = modes.loc[modes.case_id == "sb0"].copy()
        copied_modes.loc[:, "case_id"] = "sb0_copy"
        copied_work = measured.loc[measured.case_id == "sb0"].copy()
        copied_work.loc[:, "case_id"] = "sb0_copy"
        copied_work.loc[:, "file"] = "copied-sb0.jld2"
        all_modes = pd.concat([modes, copied_modes], ignore_index=True)
        all_work = pd.concat([measured, copied_work], ignore_index=True)
        summary, predictions, _, _ = self.wrapper(all_modes, all_work, slices)
        self.assertTrue((summary.n_matched == 4).all())
        self.assertTrue((summary.n_fit == 3).all())
        self.assertTrue((summary.n_known_copy_rows_excluded == 1).all())
        copied_predictions = predictions.loc[predictions.case_id == "sb0_copy"]
        self.assertTrue(copied_predictions.excluded_identical_copy.all())
        self.assertFalse(copied_predictions.used_in_fit.any())
        # A shared angular PDF alone cannot establish identical work samples.
        changed = all_work.copy()
        changed.loc[changed.case_id == "sb0_copy", "mean_active_work"] += 0.1
        changed_summary, _, _, _ = self.wrapper(all_modes, changed, slices)
        self.assertTrue((changed_summary.n_fit == 4).all())
        self.assertTrue((changed_summary.n_known_copy_rows_excluded == 0).all())

    def test_flat_only_wrapper_does_not_validate_or_refit_the_preceding_sb_inputs(self):
        modes, measured, slices = self.wrapper_data()
        # Deliberately conflicting SB settings and duplicate angle rows would
        # fail a requested SB fit. They must not obstruct the independent flat
        # fit when the notebook retains its preceding SB calculation.
        modes.loc[:, "D"] = 0.5
        modes = pd.concat([modes, modes.iloc[[0]]], ignore_index=True)
        sb, sb_predictions, flat, flat_predictions = self.wrapper(modes, measured, slices, fit_sb=False)
        self.assertTrue(sb.empty)
        self.assertTrue(sb_predictions.empty)
        self.assertEqual(len(flat), 1)
        self.assertAlmostEqual(flat.delta_x_fit.iloc[0], 2.7)
        self.assertEqual(flat.n_fit.iloc[0], 3)
        self.assertEqual(set(flat_predictions.case_id), {"flat0", "flat1", "flat2"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
