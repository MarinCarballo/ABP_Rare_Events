# ABP rare-event noise and speed sweeps

This Julia package samples complete active Brownian particle trajectories in a
two-dimensional potential. Multicanonical (MUCA) sampling learns a bias in the
endpoint `x(T)`; production freezes that bias and saves reweighted histograms,
chain statistics, active work, and optional trajectory snapshots.

- [Physics, equations, algorithms, errors, and usage](docs/physics_and_usage.md)
- [Brief change and audit report](REPORT.md)

The default potential (`lambda=2.5`), force, noise convention, and Euler–Maruyama
update are preserved. The new `lambda=8` submission makes the requested
potential-parameter change. Crank–Nicolson is an explicit option. Its
finite-step trajectories differ from Euler, so comparisons require a time-step
convergence check.

## Install and check

Run commands from this directory, with Julia and the project dependencies
installed before submitting cluster jobs:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. scripts/run_noise_sweep.jl --dry-run
julia --project=. test/runtests.jl
```

The project pins a compatible `MonteCarloX` 0.2 source revision in
`Project.toml` (Julia 1.10 or newer). The newer 0.3 API removed interfaces
used here. See the detailed installation notes if an existing manifest
resolves an incompatible version; record the resolved environment with results.

## Run

```bash
# Review the small-workload configuration without sampling.
julia --project=. --threads=4 scripts/run_noise_sweep.jl --smoke --dry-run

# Explicit optional numerical update; Euler remains the default.
ABP_INTEGRATOR=crank_nicolson julia --project=. scripts/run_noise_sweep.jl --dry-run

# Optional Crank–Nicolson proposal in NOISE space, retaining Euler dynamics.
ABP_NOISE_PROPOSAL=pcn julia --project=. scripts/run_noise_sweep.jl --dry-run

# Fixed D=0.025: one speed per array task, followed immediately by production.
mkdir -p errors data
sbatch run/submit_fixed_D_speed_landscape.sh

# Dedicated MUCA-learning / production trajectory movie data.
sbatch run/run_abp_movie_paths_array_nariman.sh
```

The speed array covers `0.280:0.005:0.390`, including speeds below and above the
user-specified critical speed `0.32`. The supplied cluster wrappers contain
large research workloads; inspect their environment defaults before submission.
Cluster paths and the Julia executable must match your installation.

For the new two-potential comparison, deploy the updated `src/` and fixed-D
runner together with these submission files, then run from the project root:

```bash
mkdir -p errors
sbatch run/submit_production_work_theta_lambda2p5.sh
sbatch run/submit_production_work_theta_lambda8.sh
```

Each array has 29 speeds: `0.200, 0.240, 0.260`, `0.280:0.005:0.390`, and
`0.400, 0.450, 0.500`, at `D=0.025`, `T=40`, and `dt=0.01`. Both retain
Euler, random-walk proposals, the existing workload, 128 CPUs, 96 GB, and a
seven-day limit. They write separate lambda-tagged run directories. Load one
lambda run at a time in notebooks, whose grouping currently uses `(D,v,T)`.
The `0.32` critical-speed reference is supplied for lambda 2.5; lambda 8 may
have a different threshold. No cluster execution or completion time has been
verified, and no job was submitted.

Configuration preflight accepts `bash <submission-file> --dry-run` with Slurm
array variables set and exits before result collection; see the
[full example](docs/physics_and_usage.md#9-entry-points-and-cluster-use).
The scripts default to the cluster Julia path. Use
`sbatch <submission-file> --integrator=crank_nicolson` for the optional update.
Older backend code ignores the new lambda setting, so copying only the shell
scripts would not produce the requested lambda 8 potential.

## Analyze

Set the run directories in the configuration cells, then run:

| Notebook | Purpose |
| --- | --- |
| `movies.ipynb` | Evolution of sampled paths during MUCA or production; physical-time path movies; saved production path inspection |
| `plot_production_errors_work_theta.ipynb` | Chain jackknife errors, active work by crossing route, orientation occupation, trajectory examples |
| `plot_speed_sweep.ipynb` | Fixed-noise speed comparisons |
| `plot_abp_TD_sweep.ipynb` | Duration/noise comparisons |

JLD2 is the complete result format. CSVs are plotting exports; pooled CSVs
alone do not contain the chain statistics needed for the notebook error bands.
The [active-work component notes](docs/active_work_components.md) explain the
new full-path x/y measurements, the constant-angle argument, and component
comparisons in the configured crossing-y slices. Existing total-only files
remain usable; exact ensemble component means require new production moments.
The movie and production-error notebooks use NumPy, pandas, Matplotlib, h5py,
and Pillow in their Python kernel. MP4 export additionally requires FFmpeg.

The lightweight notebook checks can be rerun without research data:

```bash
python test/validate_notebook_statistics.py
python test/validate_movie_notebook.py
```

## Audit essentials

Active work uses a full-resolution midpoint line integral with unwrapped
angles. The theta histogram is an endpoint-conditioned **path occupation**
histogram, with angles wrapped to `[0,2π)` when binned. It is not an endpoint
theta distribution.

MUCA flatness in the inspected `MonteCarloX` implementation ignores empty
bins, and the recorded `flatness_threshold` does not stop the run. Stopping
uses boundary-traversal counts. Existing move bookkeeping also omits
Gaussian-prior rejections from the MUCA learning histogram and distorts the
reported acceptance rate. These existing behaviors are documented, with the
sampling and learning policy preserved; flatness alone is not evidence of
full sampling or validated convergence. See the detailed documentation and
report before interpreting results.
