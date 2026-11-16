"""Synthetic checks of signed-angle functions extracted from the actual notebook.

Run from the project root with ``python -B test/validate_signed_angles.py``.
No production file, notebook patch script, or notebook top-level code is run.
"""

import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "plot_production_errors_work_theta.ipynb"


def notebook_functions():
    """Compile definitions only, so remote discovery and plotting stay idle."""
    wanted = {
        "signed_theta", "signed_theta_histogram", "theta_axis",
        "vector", "scalar", "checked_edges", "chain_ids", "read_chain_matrix",
        "jackknife_ratio", "pdf_from_mass", "load_path_products", "plot_path_map",
    }
    found = {}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK.name}:cell{index}")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                found[node.name] = node
    if found.keys() != wanted:
        raise AssertionError(f"Missing notebook definitions: {sorted(wanted - found.keys())}")
    namespace = {"np": np, "pd": pd, "LogNorm": LogNorm, "BAND_SIGMA": 1.0}
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


FUNCTIONS = notebook_functions()


class MemoryFile(dict):
    """Only the h5py interface used by the extracted readers, in memory."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def synthetic_product(edges, matrices):
    # The extracted functions retain their compilation namespace as __globals__.
    # Recompile into a fresh dictionary to isolate each test's synthetic source.
    namespace = notebook_functions()
    source = MemoryFile({
        "histograms/bins_path_x": np.array([-1.0, 0.0, 2.0]),
        "histograms/bins_path_theta": np.asarray(edges),
        "chain_statistics/chains": {str(i): {} for i in range(1, len(matrices) + 1)},
        "chain_statistics/n_chains": len(matrices),
    })
    for chain, matrix in enumerate(matrices, 1):
        base = f"chain_statistics/chains/{chain}/path_endpoint_conditions/x_gt_0p5/unbiased"
        source[f"{base}/path_x_theta_shape"] = np.asarray(matrix.shape)
        source[f"{base}/path_x_theta_flat_julia_column_major"] = matrix.ravel(order="F")
    notes = []
    namespace.update(
        h5py=SimpleNamespace(File=lambda *_args, **_kwargs: source),
        X_SLICES={"left": (-1.0, 0.0), "right": (0.0, 2.0)},
        note=lambda case, message: notes.append((case["case_id"], message)),
    )
    case = {"case_id": "S001", "file": "synthetic.jld2", "path_x_filter_min": -1.0}
    return namespace["load_path_products"](case, "theta"), namespace, notes


def independent_jackknife_pdf(masses, widths):
    """Delete original chains, never treat the seam pieces as new samples."""
    masses = np.asarray(masses, dtype=float)
    pooled = masses.sum(axis=0)
    density = pooled / pooled.sum() / widths
    deleted = []
    for chain in range(len(masses)):
        retained = np.delete(masses, chain, axis=0).sum(axis=0)
        deleted.append(retained / retained.sum() / widths)
    deleted = np.asarray(deleted)
    se = np.sqrt((len(masses) - 1) / len(masses)
                 * ((deleted - deleted.mean(axis=0)) ** 2).sum(axis=0))
    se[pooled == 0] = np.nan
    return density, se


class SignedAngles(unittest.TestCase):
    def test_signed_values_preserve_orientation_and_both_pi_endpoints(self):
        original = np.array([-9 * np.pi, -4 * np.pi, -3 * np.pi / 2, -np.pi,
                             -np.pi / 2, 0.0, np.pi / 2, np.pi, 3 * np.pi / 2,
                             2 * np.pi, 7 * np.pi])
        expected = np.array([-np.pi, 0, np.pi / 2, -np.pi, -np.pi / 2,
                             0, np.pi / 2, -np.pi, -np.pi / 2, 0, -np.pi])
        signed = FUNCTIONS["signed_theta"](original)
        np.testing.assert_allclose(signed, expected, atol=5e-15)
        np.testing.assert_allclose(np.cos(signed), np.cos(original), atol=5e-15)
        np.testing.assert_allclose(np.sin(signed), np.sin(original), atol=5e-15)
        self.assertTrue(np.all((-np.pi <= signed) & (signed < np.pi)))
        self.assertEqual(float(FUNCTIONS["signed_theta"](np.pi)), -np.pi)
        self.assertEqual(float(FUNCTIONS["signed_theta"](-np.pi)), -np.pi)

    def test_even_180_bins_are_an_exact_mass_permutation(self):
        edges = np.linspace(0, 2 * np.pi, 181)
        new_edges, order, fractions = FUNCTIONS["signed_theta_histogram"](edges)
        expected_order = np.r_[90:180, 0:90]
        np.testing.assert_array_equal(order, expected_order)
        np.testing.assert_array_equal(fractions, np.ones(180))
        np.testing.assert_allclose(new_edges, np.linspace(-np.pi, np.pi, 181), atol=2e-15)
        masses = np.vstack((np.arange(1, 181), np.arange(180, 0, -1),
                            1 + np.arange(180) % 7)).astype(float)
        moved = masses[:, order] * fractions
        np.testing.assert_array_equal(moved, masses[:, expected_order])
        np.testing.assert_array_equal(moved.sum(axis=1), masses.sum(axis=1))
        original_pdf = FUNCTIONS["pdf_from_mass"](masses, edges)
        signed_pdf = FUNCTIONS["pdf_from_mass"](moved, new_edges)
        for column in ("unbiased_mass", "unbiased_pdf", "chain_jackknife_se", "relative_se",
                       "band_low", "band_high", "n_chains_with_bin_mass"):
            np.testing.assert_allclose(signed_pdf[column], original_pdf[column].to_numpy()[order],
                                       rtol=3e-13, atol=1e-14, equal_nan=True)

    def test_odd_five_bins_split_the_seam_without_losing_mass(self):
        edges = np.linspace(0, 2 * np.pi, 6)
        new_edges, order, fractions = FUNCTIONS["signed_theta_histogram"](edges)
        np.testing.assert_array_equal(order, [2, 3, 4, 0, 1, 2])
        np.testing.assert_allclose(fractions, [0.5, 1, 1, 1, 1, 0.5], atol=1e-15)
        masses = np.array([[9, 1, 6, 2, 4], [1, 7, 2, 8, 1], [3, 2, 9, 1, 5]], dtype=float)
        moved = masses[:, order] * fractions
        np.testing.assert_allclose(moved.sum(axis=1), masses.sum(axis=1), atol=1e-14)
        density, se = independent_jackknife_pdf(moved, np.diff(new_edges))
        table = FUNCTIONS["pdf_from_mass"](moved, new_edges)
        np.testing.assert_allclose(table.unbiased_pdf, density, rtol=1e-14)
        np.testing.assert_allclose(table.chain_jackknife_se, se, rtol=1e-13)
        # Same source-bin uncertainty at both seam ends: splitting cannot make
        # either piece appear more precisely sampled than the original bin.
        original = FUNCTIONS["pdf_from_mass"](masses, edges)
        for index in (0, -1):
            self.assertAlmostEqual(table.unbiased_pdf.iloc[index], original.unbiased_pdf.iloc[2])
            self.assertAlmostEqual(table.chain_jackknife_se.iloc[index], original.chain_jackknife_se.iloc[2])
        self.assertAlmostEqual(float(np.sum(table.unbiased_pdf * table.width)), 1.0)

    def test_nonuniform_full_circle_shifted_and_already_centered(self):
        for edges in (
            np.array([0.0, 0.2, 1.7, 3.6, 4.1, 2 * np.pi]),
            np.array([0.37, 0.57, 2.07, 3.97, 4.47, 0.37 + 2 * np.pi]),
            np.array([-np.pi, -2.1, -0.6, 0.2, 2.6, np.pi]),
            np.array([-2 * np.pi + 0.37, -2 * np.pi + 0.57, -2 * np.pi + 2.07,
                      -2 * np.pi + 3.97, -2 * np.pi + 4.47, 0.37]),
        ):
            with self.subTest(edges=edges):
                new_edges, order, fractions = FUNCTIONS["signed_theta_histogram"](edges)
                self.assertEqual(new_edges[0], -np.pi)
                self.assertEqual(new_edges[-1], np.pi)
                self.assertTrue(np.all(np.diff(new_edges) > 0))
                mass_by_source = np.bincount(order, weights=fractions, minlength=len(edges) - 1)
                np.testing.assert_allclose(mass_by_source, np.ones(len(edges) - 1), atol=4e-15)
                original_density = np.array([0.5, 2.0, 0.8, 1.1, 3.0])
                mass = original_density * np.diff(edges)
                moved = mass[order] * fractions
                np.testing.assert_allclose(moved / np.diff(new_edges), original_density[order], rtol=3e-14)
                self.assertAlmostEqual(moved.sum(), mass.sum())
        centered = np.array([-np.pi, -2.1, -0.6, 0.2, 2.6, np.pi])
        new_edges, order, fractions = FUNCTIONS["signed_theta_histogram"](centered)
        np.testing.assert_allclose(new_edges, centered, atol=1e-15)
        np.testing.assert_array_equal(order, np.arange(5))
        np.testing.assert_array_equal(fractions, np.ones(5))

    def test_load_path_products_transforms_each_chain_before_uncertainty(self):
        edges = np.linspace(0, 2 * np.pi, 6)
        matrices = np.array([[[9, 1, 6, 2, 4], [2, 3, 1, 8, 4]],
                             [[1, 7, 2, 8, 1], [5, 3, 8, 1, 2]],
                             [[3, 2, 9, 1, 5], [4, 2, 3, 6, 1]]], dtype=float)
        product, namespace, notes = synthetic_product(edges, matrices)
        self.assertEqual(notes, [])
        new_edges, order, fractions = namespace["signed_theta_histogram"](edges)
        transformed = matrices[:, :, order] * fractions[None, None, :]
        np.testing.assert_allclose(product["pooled_mass"], transformed.sum(axis=0), atol=1e-14)
        for name, selected in (("all_saved_x", [0, 1]), ("left", [0]), ("right", [1])):
            mass = transformed[:, selected, :].sum(axis=1)
            expected_pdf, expected_se = independent_jackknife_pdf(mass, np.diff(new_edges))
            table = product["pdfs"][name]
            np.testing.assert_allclose(table.unbiased_pdf, expected_pdf, rtol=2e-14)
            np.testing.assert_allclose(table.chain_jackknife_se, expected_se, rtol=2e-13)
            self.assertTrue((table.theta_display_convention == "[-pi, pi)").all())
        diagnostics = {row["slice"]: row for row in product["diagnostics"]}
        self.assertAlmostEqual(diagnostics["all_saved_x"]["mass"], matrices.sum())
        self.assertEqual(diagnostics["all_saved_x"]["n_chains_total"], 3)
        self.assertAlmostEqual(diagnostics["all_saved_x"]["pdf_integral"], 1.0)

    def test_signed_ticks_and_path_map_keep_full_display_domain(self):
        edges = np.linspace(0, 2 * np.pi, 6)
        matrices = np.arange(1, 31, dtype=float).reshape(3, 2, 5)
        product, namespace, _ = synthetic_product(edges, matrices)
        namespace.update(LOG_HEATMAP=False, X_DISPLAY=None, Y_DISPLAY=None, X_SLICES={})
        fig, (ax_x, ax_y) = plt.subplots(1, 2)
        try:
            namespace["theta_axis"](ax_x)
            np.testing.assert_allclose(ax_x.get_xlim(), [-np.pi, np.pi])
            np.testing.assert_allclose(ax_x.get_xticks(), [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            mesh = namespace["plot_path_map"](ax_y, product, "theta")
            np.testing.assert_allclose(ax_y.get_ylim(), [-np.pi, np.pi])
            shown = np.asarray(mesh.get_array()).reshape(6, 2)
            expected = (product["pooled_mass"] / matrices.sum()
                        / np.diff(product["x_edges"])[:, None]
                        / np.diff(product["z_edges"])[None, :])
            np.testing.assert_allclose(shown, expected.T)
            fig.canvas.draw()  # Exercise the actual signed-pi mathtext labels.
        finally:
            plt.close(fig)

    def test_invalid_angular_edges_are_rejected(self):
        invalid = [[], [0.0], [0.0, np.pi], [0.0, 3 * np.pi],
                   [0.0, 0.0, 2 * np.pi], [0.0, 1.0, 0.5, 2 * np.pi],
                   [0.0, np.nan, 2 * np.pi], [-np.inf, 0.0, np.pi],
                   [[0.0, 2 * np.pi]], np.array(0.0)]
        for edges in invalid:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                FUNCTIONS["signed_theta_histogram"](edges)

    def test_extremely_narrow_valid_bin_is_preserved_or_explicitly_rejected(self):
        # A display helper must not silently delete an actual source bin merely
        # because its width is close to the numerical seam-deduplication tolerance.
        for edges in (np.array([0.0, 1e-14, 1.0, 2 * np.pi]),
                      np.array([0.0, np.pi - 1e-14, np.pi + 1e-14, 2 * np.pi])):
            with self.subTest(edges=edges):
                try:
                    _, order, fractions = FUNCTIONS["signed_theta_histogram"](edges)
                except ValueError:
                    continue  # Explicit resolution limits cannot silently discard mass.
                mass_by_source = np.bincount(order, weights=fractions, minlength=3)
                np.testing.assert_allclose(mass_by_source, np.ones(3), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
