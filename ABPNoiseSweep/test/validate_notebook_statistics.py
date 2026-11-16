"""Check notebook syntax and the actual chain-jackknife helper on known ratios."""

import ast
import json
from pathlib import Path

import numpy as np


project = Path(__file__).resolve().parents[1]
functions = {}
for filename in ("movies.ipynb", "plot_abp_TD_sweep.ipynb",
                 "plot_speed_sweep.ipynb", "plot_production_errors_work_theta.ipynb"):
    notebook = json.loads((project / filename).read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"{filename}:cell{index}")
        if filename == "plot_production_errors_work_theta.ipynb":
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    functions[node.name] = node

namespace = {"np": np}
exec(compile(ast.Module(body=[functions["jackknife_ratio"]], type_ignores=[]),
             "notebook_jackknife", "exec"), namespace)
ratio = namespace["jackknife_ratio"]

value, se = ratio([2, 4, 8], [1, 2, 4])
assert value == 2 and se == 0
value, se = ratio([1, 2, 5], [1, 1, 2])
deleted = np.array([7/3, 6/3, 3/2])
assert value == 2
assert np.isclose(se, np.sqrt(2/3 * np.sum((deleted - deleted.mean())**2)))
_, se = ratio([1], [2])
assert np.isnan(se)  # One chain cannot supply a chain-delete uncertainty.
value, se = ratio([0, 0], [0, 0])
assert np.isnan(value) and np.isnan(se)
try:
    ratio([1, 2], [1, -1])
except ValueError:
    pass
else:
    raise AssertionError("Negative reweighting denominator was accepted")

print("All four notebook code cells parse; chain-jackknife analytic checks passed.")
