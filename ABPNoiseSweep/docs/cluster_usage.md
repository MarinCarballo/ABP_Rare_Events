# Cluster usage

Run all commands from the project directory `ABPNoiseSweep`.


## First run on a new cluster or computer

Install the Julia package environment once from the project directory:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

Optional, but useful before a long job:

```bash
julia --project=. -e 'import Pkg; Pkg.precompile()'
```

You can also ask the cluster script to do this before the job starts:

```bash
sbatch --export=ALL,ABP_INSTANTIATE=true,ABP_PRECOMPILE=true run/run_abp_noise_sweep_data_only.sh --dry-run
```

For normal production submissions after the environment is installed, leave `ABP_INSTANTIATE=false` and `ABP_PRECOMPILE=false`.

## Check configuration without starting the simulation

```bash
julia --project=. scripts/run_noise_sweep.jl --dry-run
```

This prints the effective MUCA iteration count for each value of `D`.

## Submit a configured job

```bash
sbatch --export=ALL,ABP_D_VALUES=0.1,0.01,0.005,ABP_OUTPUT_DIR=meaningful_run_name,ABP_MOVE_WEIGHTS=0.0,0.05,0.55,0.40,ABP_N_ITER=90,ABP_N_ITER_STEPS_PER_ITER=120000000 run/run_abp_noise_sweep_data_only.sh
```

The command above is one terminal command. You can also split it across lines with backslashes.

## Reflection moves

Reflection is not a separate run dimension. Change only `ABP_MOVE_WEIGHTS`.

Order:

```text
[reflection, theta0, block_noise, local_noise]
```

Default, no reflection:

```text
0.0,0.05,0.55,0.40
```

With 5% reflection:

```text
0.05,0.05,0.50,0.40
```
