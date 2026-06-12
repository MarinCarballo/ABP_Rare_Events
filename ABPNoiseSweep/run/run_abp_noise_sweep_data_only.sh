#!/bin/bash
    #SBATCH --job-name=abp_noise_sweep_1
    #SBATCH --partition=titan
    #SBATCH --time=7-00:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=32
    #SBATCH --cpus-per-task=2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs

: "${JULIA_CMD:=julia}"
: "${JULIA_NUM_THREADS:=${SLURM_CPUS_PER_TASK:-1}}"

# Dependency setup. Keep these false if
# already installed the Julia environment once with Pkg.instantiate().
: "${ABP_INSTANTIATE:=false}"
: "${ABP_PRECOMPILE:=false}"

: "${ABP_D_VALUES:=0.1,0.01}"
: "${ABP_MOVE_WEIGHTS:=run/move_weights.json}" # JSON file with the default move weights.
: "${ABP_OUTPUT_DIR:=abp_noise_sweep_endpoint_conditioned}" #: "${ABP_OUTPUT_DIR:=/scratch/$USER/ABP_runs/run_001}"
: "${ABP_SAVE_CSV:=true}"

: "${ABP_N_ITER:=20}"
: "${ABP_N_ITER_STEPS_PER_ITER:=10000000}"
: "${ABP_N_THERM_MUCA:=100000}"
: "${ABP_D_SCALING_REFERENCE:=0.01}"
: "${ABP_SCALE_N_ITER_WITH_D:=true}"
: "${ABP_BLOCK_DXI:=0.05}"
: "${ABP_LOCAL_DXI:=0.8}"

: "${ABP_N_PROD_OBS_TOTAL:=6000000}"
: "${ABP_N_PROD_CHAINS:=${JULIA_NUM_THREADS}}"
: "${ABP_N_THERM_PROD:=100000}"
: "${ABP_PROD_STRIDE:=10000}"
: "${ABP_ROUNDTRIP_STRIDE:=10000}"

export JULIA_NUM_THREADS ABP_INSTANTIATE ABP_PRECOMPILE
export ABP_D_VALUES ABP_MOVE_WEIGHTS ABP_OUTPUT_DIR ABP_SAVE_CSV
export ABP_N_ITER ABP_N_ITER_STEPS_PER_ITER ABP_N_THERM_MUCA
export ABP_D_SCALING_REFERENCE ABP_SCALE_N_ITER_WITH_D ABP_BLOCK_DXI ABP_LOCAL_DXI
export ABP_N_PROD_OBS_TOTAL ABP_N_PROD_CHAINS ABP_N_THERM_PROD ABP_PROD_STRIDE ABP_ROUNDTRIP_STRIDE

printf 'Running ABPNoiseSweep from %s\n' "$PROJECT_DIR"
printf '  JULIA_NUM_THREADS: %s\n' "$JULIA_NUM_THREADS"
printf '  ABP_INSTANTIATE: %s\n' "$ABP_INSTANTIATE"
printf '  ABP_PRECOMPILE: %s\n' "$ABP_PRECOMPILE"
printf '  ABP_D_VALUES: %s\n' "$ABP_D_VALUES"
printf '  ABP_MOVE_WEIGHTS: %s\n' "$ABP_MOVE_WEIGHTS"
printf '  ABP_OUTPUT_DIR: %s\n' "$ABP_OUTPUT_DIR"
printf '  ABP_N_ITER: %s\n' "$ABP_N_ITER"
printf '  ABP_D_SCALING_REFERENCE: %s\n' "$ABP_D_SCALING_REFERENCE"
printf '  ABP_SCALE_N_ITER_WITH_D: %s\n' "$ABP_SCALE_N_ITER_WITH_D"

exec "$JULIA_CMD" --project=. scripts/run_noise_sweep.jl "$@"
