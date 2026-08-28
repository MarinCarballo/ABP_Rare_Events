# Fixed-noise speed landscape

This is the simple workflow requested:

```text
15 speeds at D=0.025
    -> one Slurm array task per speed
    -> MUCA
    -> production immediately in the same Julia process
```

There are no checkpoint handoffs, separate production jobs, or noise-dependent
workload rules.

## Install

From the `ABPNoiseSweep` repository root:

```bash
cp run_fixed_D_speed_landscape.jl scripts/run_fixed_D_speed_landscape.jl
cp production.jl src/production.jl
mkdir -p errors data
```

Then submit:

```bash
sbatch submit_fixed_D_speed_landscape.sh
```

The velocity grid is `0.320:0.005:0.390`, with `D_t=D_r=0.025` for every task.

The combined plotting directory is:

```text
data/abp_speed_D0p025_array_JOBID/combined/
```

## Default first-look workload per speed

```text
MUCA iterations                 30
final-iteration MUCA moves      25,000,000
production moves                10,000,000
Julia threads                   32
production chains               16
whole-path observation stride   1,000
```

The endpoint histograms are updated every production move.  Whole-path
occupancy is scanned only every 1,000 production moves.

