"""Regenerate angle PDFs and modal-angle/speed plots from exported CSVs.

Run from the ABPNoiseSweep root after exporting the notebook's case inventory
and path-slice PDF tables. No simulation or JLD2 scan is performed. The drawing
and mode extraction code comes from the notebook, so its labels/settings match.
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
    namespace = {"__name__": "__theta_plot_export__"}
    exec("".join(notebook["cells"][2]["source"]), namespace)
    exec("".join(notebook["cells"][4]["source"]), namespace)
    directory = namespace["OUTPUT_DIR"]
    inventory = pd.read_csv(directory / "case_inventory.csv")
    namespace.update(CASES=inventory.to_dict("records"), inventory=inventory,
                     MESSAGES=[], TABLES=[], FIGURES=[], SHOW_FIGS=False)
    namespace["CASE_COLORS"] = {}
    helpers = ast.parse("".join(notebook["cells"][10]["source"]))
    # Reuse the notebook's setup without rewriting the input inventory CSV.
    helpers.body = [node for node in helpers.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "export_table")]
    exec(compile(helpers, "notebook_plot_helpers", "exec"), namespace)
    path_data = {"y": {}, "theta": {}}
    for case in namespace["CASES"]:
        for variable, products in path_data.items():
            tables = {}
            for name in ["all_saved_x", *namespace["X_SLICES"]]:
                path = directory / f"{case['case_id']}_path_{variable}_{name}_pdf.csv"
                if path.is_file():
                    tables[name] = pd.read_csv(path)
            if tables:
                products[case["case_id"]] = {"pdfs": tables}
    if not path_data["theta"]:
        raise FileNotFoundError("No exported angular PDF tables found beside case_inventory.csv")
    namespace["PATH_DATA"] = path_data
    exec("".join(notebook["cells"][20]["source"]), namespace)
    for path in namespace["FIGURES"]:
        print(path)
    for path in namespace["TABLES"]:
        print(path)


if __name__ == "__main__":
    main()
