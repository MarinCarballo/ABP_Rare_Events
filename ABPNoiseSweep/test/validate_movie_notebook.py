"""Validate movie loading and render tiny movies without simulation output.

Run from the project root with: python test/validate_movie_notebook.py
Temporary CSV, HDF5, GIF, and figure outputs are removed automatically.
"""

import ast
import json
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]


def source(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def validate_notebook_syntax():
    for name in ["movies.ipynb", "plot_production_errors_work_theta.ipynb"]:
        notebook = json.loads((PROJECT / name).read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{name}:cell{index}")


def validate_movies(directory):
    notebook = json.loads((PROJECT / "movies.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    exec(source(notebook, 1), namespace)
    exec(source(notebook, 2), namespace)
    namespace.update(FPS=4, DPI=30, SAVE_FORMAT="gif", X_LIM=None, Y_LIM=None)
    rows = [
        dict(movie_id=j, chain_id=1, mcmc_step=step, path_index=k,
             physical_time=t, x=10+t*j, y=4+t, theta=t, endpoint_x=10+0.2*j,
             endpoint_y=0.2, iteration=j, sampling_step=step,
             unbias_weight_shifted=float("nan"))
        for j, step in enumerate([1, 51, 101], 1)
        for k, t in enumerate([0.0, 0.1, 0.2], 1)
    ]
    path = directory / "muca_movie_trajectories_long.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    namespace.update(MOVIE_CSV=path.resolve(), SAVE_MOVIES=False)
    exec(source(notebook, 3), namespace)
    assert len(namespace["summary"]) == 3
    assert namespace["keep_endpoints"](8, 3).tolist() == [0, 3, 6, 7]
    assert namespace["keep_endpoints"](8, 1, 1).tolist() == [7]
    fig, ax = plt.subplots()
    namespace["landscape"](ax)
    assert ax.get_xlim()[1] > 10.6 and ax.get_xlim()[0] < 10
    assert ax.get_ylim()[1] > 4.2 and ax.get_ylim()[0] < 4
    plt.close(fig)
    namespace["SAVE_MOVIES"] = True
    namespace["movie_out_dir"].mkdir(parents=True)
    exec(source(notebook, 4), namespace)
    exec(source(notebook, 5), namespace)
    # The trajectory overlay is a separate figure with a theta-axis label.
    exec(source(notebook, 7), namespace)
    plt.gcf().canvas.draw()
    plt.close("all")
    outputs = list(namespace["movie_out_dir"].glob("*.gif"))
    assert len(outputs) == 2 and all(p.stat().st_size > 0 for p in outputs)

    # Legacy production CSVs lack the adaptive-MUCA metadata columns.
    old = path.with_name("movie_trajectories_long.csv")
    pd.DataFrame(rows).drop(columns=["iteration", "sampling_step"]).to_csv(old, index=False)
    legacy = namespace["read_movie_csv"](old)
    assert legacy.iteration.eq(0).all()
    assert legacy.movie_id.nunique() == 3
    namespace["MOVIE_CSV"] = old
    try:
        namespace["find_movie_csv"]()
    except ValueError:
        pass
    else:
        raise AssertionError("A production CSV was silently treated as MUCA")


def validate_production_panels(directory):
    notebook = json.loads((PROJECT / "plot_production_errors_work_theta.ipynb").read_text(encoding="utf-8"))
    path = directory / "examples.jld2"
    with h5py.File(path, "w") as h5:
        h5["metadata/dt"] = 0.1
        h5["metadata/saved_path_time_thin"] = 3
        # Last interval is shorter than the thinning cadence: explicit times
        # must keep the actual endpoint at 0.7, rather than inventing 0.9.
        base = "saved_trajectories/all/1"
        for name, data in dict(x=[-1, -0.5, 0.5, 1], y=[0, 0.1, 0.1, 0],
                               thetas=[0, 0.2, 0.3, 0.4], thetas_unwrapped=[0, 0.2, 0.3, 0.4],
                               physical_times=[0, 0.3, 0.6, 0.7],
                               chain_id=1, mcmc_step=100).items():
            h5[f"{base}/{name}"] = data
    figures = []

    def finish_figure(fig, name):
        figures.append(fig)
        fig.canvas.draw()

    namespace = dict(DETAIL_CASES=[dict(file=path, case_id="C001", label="Synthetic")],
                     EXAMPLE_PATH_WINDOW="all", MAX_EXAMPLE_PATHS=12,
                     SHOW_INDIVIDUAL_PATHS=True, h5py=h5py, np=np, plt=plt,
                     vector=lambda h5, key: np.asarray(h5[key], dtype=float).reshape(-1),
                     scalar=lambda h5, key: float(np.asarray(h5[key])),
                     note=lambda case, message: None,
                     theta_axis=lambda ax, axis: None, finish_figure=finish_figure)
    helpers = [node for node in ast.parse(source(notebook, 10)).body
               if isinstance(node, ast.FunctionDef) and node.name in {"signed_theta", "theta_axis"}]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "notebook_angles", "exec"), namespace)
    exec(source(notebook, 26), namespace)
    assert len(figures) == 2
    assert namespace["saved"][0]["t"][-1] == 0.7
    assert namespace["saved"][0]["x"][-1] == 1
    assert np.all(namespace["saved"][0]["theta"] >= -np.pi)
    assert np.all(namespace["saved"][0]["theta"] < np.pi)
    assert np.allclose(figures[0].axes[1].get_ylim(), [-np.pi, np.pi])
    assert np.allclose(figures[1].axes[3].get_ylim(), [-np.pi, np.pi])
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    validate_notebook_syntax()
    with tempfile.TemporaryDirectory(prefix="abp_movie_validation_") as temp:
        directory = Path(temp)
        validate_movies(directory)
        validate_production_panels(directory)
    print("Movie/legacy CSV loading, both GIF renders, and full production path panels passed.")
