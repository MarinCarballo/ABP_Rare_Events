#!/bin/bash

#SBATCH --job-name=abp_TD_sweep
#SBATCH --partition=nariman
#SBATCH --exclude=nariman[09-12]
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --exclusive

# 3 integration times x 3 noise values = 9 independent jobs.
# Index mapping:
#   0: T=10, D=0.5
#   1: T=10, D=0.1
#   2: T=10, D=0.05
#   3: T=20, D=0.5
#   4: T=20, D=0.1
#   5: T=20, D=0.05
#   6: T=40, D=0.5
#   7: T=40, D=0.1
#   8: T=40, D=0.05
#SBATCH --array=0-8

#SBATCH --error=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_TD_%A_%a.err
#SBATCH --output=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_TD_%A_%a.out

set -euo pipefail

# ============================================================
# Project
# ============================================================

PROJECT_DIR="/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep"
cd "$PROJECT_DIR"

# Slurm needs the log directory to exist before submission, but keeping this
# here is harmless for ordinary runtime-created files.
mkdir -p "$PROJECT_DIR/errors"

# ============================================================
# Required source support for T sweep
# ============================================================

if ! grep -q "ABP_TRAJECTORY_T" "$PROJECT_DIR/src/cli.jl"; then
    echo "ERROR: src/cli.jl does not parse ABP_TRAJECTORY_T yet."
    echo "Add this line inside abp_apply_env_overrides!(cfg), near ABP_D_VALUES:"
    echo '    haskey(ENV, "ABP_TRAJECTORY_T") && (cfg.trajectory_T = parse(Float64, ENV["ABP_TRAJECTORY_T"]))'
    exit 2
fi

# ============================================================
# Select one (T, D) pair for this Slurm array task
# ============================================================

T_VALUES=("10" "20" "40")
D_VALUES=("0.5" "0.1" "0.05")

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"
N_D="${#D_VALUES[@]}"
N_T="${#T_VALUES[@]}"
N_TASKS=$(( N_D * N_T ))

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= N_TASKS )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${ARRAY_INDEX}; expected 0..$((N_TASKS - 1))"
    exit 1
fi

T_INDEX=$(( ARRAY_INDEX / N_D ))
D_INDEX=$(( ARRAY_INDEX % N_D ))

TRAJECTORY_T="${T_VALUES[$T_INDEX]}"
D="${D_VALUES[$D_INDEX]}"

T_TAG="${TRAJECTORY_T//./p}"
T_TAG="${T_TAG//-/m}"
D_TAG="${D//./p}"
D_TAG="${D_TAG//-/m}"

# Each array task runs exactly one noise value and one integration time.
ABP_TRAJECTORY_T="$TRAJECTORY_T"
ABP_D_VALUES="$D"

# ============================================================
# Output directories
# ============================================================

RUN_ROOT="${PROJECT_DIR}/data/abp_TD_sweep_${SLURM_ARRAY_JOB_ID}"
TASK_OUTPUT_DIR="${RUN_ROOT}/task_${ARRAY_INDEX}_T${T_TAG}_D${D_TAG}"
COMBINED_RUN_DIR="${RUN_ROOT}/combined"

mkdir -p "$TASK_OUTPUT_DIR"
mkdir -p "$COMBINED_RUN_DIR"
mkdir -p "$COMBINED_RUN_DIR/data"

ABP_OUTPUT_DIR="$TASK_OUTPUT_DIR"

# ============================================================
# Julia settings
# ============================================================

: "${JULIA_CMD:=/home/nst/amarin/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
: "${JULIA_NUM_THREADS:=${SLURM_CPUS_PER_TASK:-1}}"

# Compute nodes usually have no internet.
: "${ABP_INSTANTIATE:=false}"
: "${ABP_PRECOMPILE:=false}"

# ============================================================
# General ABP settings
# ============================================================

: "${ABP_MOVE_WEIGHTS:=run/move_weights.json}"
: "${ABP_SAVE_CSV:=true}"

# ============================================================
# MUCA settings for a relatively cheap T/D diagnostic scan
# ============================================================

: "${ABP_N_ITER:=80}"
: "${ABP_N_ITER_STEPS_PER_ITER:=100000000}"
: "${ABP_N_THERM_MUCA:=300000}"

# Reference near the middle of the cheap D range.
: "${ABP_D_SCALING_REFERENCE:=0.1}"
: "${ABP_SCALE_N_ITER_WITH_D:=true}"

: "${ABP_BLOCK_DXI:=0.05}"
: "${ABP_LOCAL_DXI:=0.8}"

# ============================================================
# MUCA roundtrip stopping condition
# ============================================================

: "${ABP_ROUNDTRIP_AVG_TARGET_FRACTION:=0.5}"
: "${ABP_ROUNDTRIP_CONVERGENCE_HITS:=3}"
: "${ABP_ROUNDTRIP_TARGET:=50}"

# ============================================================
# Production settings for comparison across T
# ============================================================

: "${ABP_N_PROD_OBS_TOTAL:=100000000}"
: "${ABP_N_PROD_CHAINS:=${JULIA_NUM_THREADS}}"
: "${ABP_N_THERM_PROD:=300000}"
: "${ABP_PROD_STRIDE:=10000}"
: "${ABP_ROUNDTRIP_STRIDE:=10000}"

# ============================================================
# Whole-path occupation histogram settings
# ============================================================

: "${ABP_PATH_OBSERVATION_STRIDE:=100}"
: "${ABP_PATH_TIME_STRIDE:=5}"
: "${ABP_PATH_FILTER_X_MIN:=-0.6}"

# ============================================================
# Export environment variables
# ============================================================

export JULIA_NUM_THREADS
export ABP_INSTANTIATE ABP_PRECOMPILE

export ABP_TRAJECTORY_T
export ABP_D_VALUES
export ABP_MOVE_WEIGHTS
export ABP_OUTPUT_DIR
export ABP_SAVE_CSV

export ABP_N_ITER
export ABP_N_ITER_STEPS_PER_ITER
export ABP_N_THERM_MUCA
export ABP_D_SCALING_REFERENCE
export ABP_SCALE_N_ITER_WITH_D
export ABP_BLOCK_DXI
export ABP_LOCAL_DXI

export ABP_ROUNDTRIP_AVG_TARGET_FRACTION
export ABP_ROUNDTRIP_CONVERGENCE_HITS
export ABP_ROUNDTRIP_TARGET

export ABP_N_PROD_OBS_TOTAL
export ABP_N_PROD_CHAINS
export ABP_N_THERM_PROD
export ABP_PROD_STRIDE
export ABP_ROUNDTRIP_STRIDE

export ABP_PATH_OBSERVATION_STRIDE
export ABP_PATH_TIME_STRIDE
export ABP_PATH_FILTER_X_MIN

# ============================================================
# Run information
# ============================================================

echo "============================================================"
echo "ABP T-D sweep array task"
echo "============================================================"
printf 'Project:                         %s\n' "$PROJECT_DIR"
printf 'Array job ID:                    %s\n' "$SLURM_ARRAY_JOB_ID"
printf 'Array task ID:                   %s\n' "$SLURM_ARRAY_TASK_ID"
printf 'T index, D index:                %s, %s\n' "$T_INDEX" "$D_INDEX"
printf 'Trajectory T:                    %s\n' "$ABP_TRAJECTORY_T"
printf 'Noise D:                         %s\n' "$ABP_D_VALUES"
printf 'Julia executable:                %s\n' "$JULIA_CMD"
printf 'Julia threads:                   %s\n' "$JULIA_NUM_THREADS"
printf 'Task output directory:           %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output directory:       %s\n' "$COMBINED_RUN_DIR"
printf 'MUCA iterations:                 %s\n' "$ABP_N_ITER"
printf 'MUCA steps per iteration:        %s\n' "$ABP_N_ITER_STEPS_PER_ITER"
printf 'MUCA thermalization:             %s\n' "$ABP_N_THERM_MUCA"
printf 'D scaling reference:             %s\n' "$ABP_D_SCALING_REFERENCE"
printf 'Scale n_iter with D:             %s\n' "$ABP_SCALE_N_ITER_WITH_D"
printf 'Roundtrip target fraction:       %s\n' "$ABP_ROUNDTRIP_AVG_TARGET_FRACTION"
printf 'Roundtrip consecutive hits:      %s\n' "$ABP_ROUNDTRIP_CONVERGENCE_HITS"
printf 'Production observations:         %s\n' "$ABP_N_PROD_OBS_TOTAL"
printf 'Production chains:               %s\n' "$ABP_N_PROD_CHAINS"
printf 'Production stride:               %s\n' "$ABP_PROD_STRIDE"
printf 'Production roundtrip stride:     %s\n' "$ABP_ROUNDTRIP_STRIDE"
printf 'Path-observation stride:         %s\n' "$ABP_PATH_OBSERVATION_STRIDE"
printf 'Path-time stride:                %s\n' "$ABP_PATH_TIME_STRIDE"
printf 'Path x(t) minimum:               %s\n' "$ABP_PATH_FILTER_X_MIN"
echo "============================================================"

"$JULIA_CMD" --version

# ============================================================
# Run Julia
# ============================================================

"$JULIA_CMD" \
    --project=. \
    scripts/run_noise_sweep.jl \
    "$@"

# ============================================================
# Copy this task's result into a combined plotting-ready directory.
# The source case directory name only contains D, so the destination name
# must include both T and D to avoid overwriting cases with the same D.
# ============================================================

shopt -s nullglob

case_dirs_found=0

for case_dir in "$TASK_OUTPUT_DIR"/data/abp_endpoint_conditioned_*; do
    if [[ -d "$case_dir" ]]; then
        dest_dir="$COMBINED_RUN_DIR/data/abp_endpoint_conditioned_T${T_TAG}_D${D_TAG}"
        rm -rf "$dest_dir"
        mkdir -p "$dest_dir"
        cp -a "$case_dir"/. "$dest_dir"/
        case_dirs_found=$((case_dirs_found + 1))
    fi
done

jld2_files_found=0

for result_file in "$TASK_OUTPUT_DIR"/*.jld2; do
    if [[ -f "$result_file" ]]; then
        cp -a "$result_file" "$COMBINED_RUN_DIR/abp_endpoint_conditioned_T${T_TAG}_D${D_TAG}.jld2"
        jld2_files_found=$((jld2_files_found + 1))
    fi
done

# ============================================================
# Completion information
# ============================================================

echo "============================================================"
echo "Completed ABP T-D sweep task"
echo "============================================================"
printf 'Trajectory T:                    %s\n' "$TRAJECTORY_T"
printf 'Noise D:                         %s\n' "$D"
printf 'Array task:                      %s\n' "$ARRAY_INDEX"
printf 'Case directories copied:         %s\n' "$case_dirs_found"
printf 'JLD2 files copied:               %s\n' "$jld2_files_found"
printf 'Task output:                     %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output:                 %s\n' "$COMBINED_RUN_DIR"
echo "============================================================"
