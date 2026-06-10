# ABPNoiseSweep

Data-only ABP path-sampling project for cluster runs. This code was made with the help of GPT 5.5 translating the original ipynb file to a cluster workflow. 
## Layout

```text
ABPNoiseSweep/
├── Project.toml
├── src/                    # Model, moves, MUCA, production, I/O, CLI
├── scripts/run_noise_sweep.jl
├── run/run_abp_noise_sweep_data_only.sh
├── test/runtests.jl
└── docs/cluster_usage.md
```

## Move weights

We have 4 types of moves.

```text
[reflection_update, theta0_update, block_noise_update, local_noise_update]
```

Default:

```text
[0.0, 0.05, 0.55, 0.40]
```

Example with 5% reflection moves:

```bash
ABP_MOVE_WEIGHTS=0.05,0.05,0.50,0.40 julia --project=. scripts/run_noise_sweep.jl --smoke
```

## Iteration scaling with D

The reference number of MUCA iterations is `ABP_N_ITER`. If `ABP_SCALE_N_ITER_WITH_D=true`, the actual number used for each D is

```text
n_iter_effective(D) = round(ABP_N_ITER * sqrt(ABP_D_SCALING_REFERENCE / D))
```

Check this before running a long job with:

```bash
julia --project=. scripts/run_noise_sweep.jl --dry-run
```

## Cluster example

```bash
sbatch --export=ALL,ABP_D_VALUES=0.1,0.01,0.005,ABP_OUTPUT_DIR=run_name,ABP_N_ITER=90 run/run_abp_noise_sweep_data_only.sh
```
### Or directly change run_abp_noise_sweep_data_only.sh
For a smoke test:

```bash
./run/run_abp_noise_sweep_data_only.sh --smoke
```


## First-time setup

From the project directory, install dependencies once:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

Then check the configuration without running the simulation:

```bash
julia --project=. scripts/run_noise_sweep.jl --dry-run
```

The cluster script can also instantiate/precompile when requested with `ABP_INSTANTIATE=true` and `ABP_PRECOMPILE=true`, but keep those false for normal production jobs after setup.
