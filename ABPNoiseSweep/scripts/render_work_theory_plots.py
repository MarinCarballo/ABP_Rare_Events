"""Render work comparisons and cosine references from notebook-exported CSVs.

Run from the project root: python -B scripts/render_work_theory_plots.py
Uses the current notebook settings and functions, without scanning JLD2 files
or running simulations. Existing measured work means and errors are read only.
"""

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd


def main():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "plot_production_errors_work_theta.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "__work_theory_export__"}
    for index in (2, 4):
        exec("".join(notebook["cells"][index]["source"]), namespace)
    directory = namespace["OUTPUT_DIR"]
    inventory = pd.read_csv(directory / "case_inventory.csv")
    cases = inventory.to_dict("records")
    if not cases:
        raise ValueError("The exported inventory has no cases")
    namespace.update(CASES=cases, inventory=inventory, MESSAGES=[], TABLES=[],
                     FIGURES=[], SHOW_FIGS=False, DETAIL_CASES=[])
    helpers = ast.parse("".join(notebook["cells"][10]["source"]))
    # Do not rewrite the source inventory from the lightweight renderer.
    helpers.body = [node for node in helpers.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "export_table")]
    exec(compile(helpers, "notebook_plot_helpers", "exec"), namespace)
    parsed = ast.parse("".join(notebook["cells"][22]["source"]))
    functions = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, "notebook_work_helpers", "exec"), namespace)

    # Extract modes afresh with the notebook's current filters, retaining the
    # effective x bounds and source fingerprint as in a full notebook run.
    namespace["PATH_DATA"] = {"theta": {}}
    for case in cases:
        tables = {}
        for name in namespace["X_SLICES"]:
            path = directory / f"{case['case_id']}_path_theta_{name}_pdf.csv"
            if path.is_file():
                tables[name] = pd.read_csv(path)
        if tables:
            namespace["PATH_DATA"]["theta"][case["case_id"]] = {"pdfs": tables}
    if not namespace["PATH_DATA"]["theta"]:
        raise FileNotFoundError("No exported angular PDFs found beside case_inventory.csv")
    modes = "".join(notebook["cells"][20]["source"])
    modes = modes[modes.index("# Theta mode progression:"):modes.index("display(THETA_MODAL_PEAKS)")]
    exec(modes, namespace)

    profiles = {}
    for case in cases:
        path = directory / f"{case['case_id']}_active_work_vs_crossing_y.csv"
        if path.is_file():
            profiles[case["case_id"]] = pd.read_csv(path)
    summary_path = directory / "active_work_m_slice_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    namespace.update(WORK_PROFILES=profiles, WORK_M_SLICE_SUMMARY=summary,
                     WORK_COMPONENT_PROFILES={}, WORK_COMPONENT_M_SLICE_SUMMARY=pd.DataFrame())
    exec("".join(notebook["cells"][24]["source"]), namespace)
    print(f"Reference: ell={namespace['FLAT_THEORY_DISPLACEMENT']:g}, theta slice={namespace['SB_THEORY_THETA_SLICE']}, "
          f"relative prominence={namespace['THETA_PEAK_MIN_PROMINENCE']:g}")
    for path in namespace["FIGURES"] + namespace["TABLES"]:
        print(path)


if __name__ == "__main__":
    main()
