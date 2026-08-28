#!/bin/bash
#SBATCH --job-name=abp_noise_array_T40_scaled_nariman
#SBATCH --partition=nariman
#SBATCH --exclude=nariman[09-12]
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --exclusive

# 13 independent noise jobs
#SBATCH --array=0-12
#SBATCH --error=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_noise_T40_scaled_%A_%a.err
#SBATCH --output=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_noise_T40_scaled_%A_%a.out

set -euo pipefail


# ============================================================
# Project
# ============================================================

PROJECT_DIR="/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/errors"
mkdir -p "$PROJECT_DIR/data"


# ============================================================
# Select one noise value for this Slurm array task
# ============================================================

D_VALUES=(
    "0.5"
    "0.1"
    "0.08"
    "0.06"
    "0.05"
    "0.04"
    "0.03"
    "0.025"
    "0.02"
    "0.015"
    "0.01"
    "0.0075"
    "0.005"
)

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#D_VALUES[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${ARRAY_INDEX}"
    exit 1
fi

D="${D_VALUES[$ARRAY_INDEX]}"

# Convert, for example:
#   0.05  -> 0p05
#   0.005 -> 0p005
D_TAG="${D//./p}"
D_TAG="${D_TAG//-/m}"

# Each array task runs exactly one noise value.
ABP_D_VALUES="$D"


# ============================================================
# Output directories
#
# These must be defined before ABP_OUTPUT_DIR is assigned.
# ============================================================

RUN_ROOT="${PROJECT_DIR}/data/abp_noise_T40_scaled_array_${SLURM_ARRAY_JOB_ID}"
TASK_OUTPUT_DIR="${RUN_ROOT}/task_${ARRAY_INDEX}_D${D_TAG}"
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

# Keep these false because the compute nodes do not have internet.
# The environment must already have been instantiated and precompiled.
: "${ABP_INSTANTIATE:=false}"
: "${ABP_PRECOMPILE:=false}"


# ============================================================
# General ABP settings
# ============================================================

: "${ABP_MOVE_WEIGHTS:=run/move_weights.json}"
: "${ABP_SAVE_CSV:=true}"

# Integration time.
# Requires cli.jl to contain:
# haskey(ENV, "ABP_TRAJECTORY_T") && (cfg.trajectory_T = parse(Float64, ENV["ABP_TRAJECTORY_T"]))
: "${ABP_TRAJECTORY_T:=40}"


# ============================================================
# MUCA settings
# ============================================================

: "${ABP_N_ITER:=150}"
: "${ABP_N_ITER_STEPS_PER_ITER:=800000000}"
: "${ABP_N_THERM_MUCA:=1000000}"

: "${ABP_D_SCALING_REFERENCE:=0.05}"
: "${ABP_SCALE_N_ITER_WITH_D:=true}"

: "${ABP_BLOCK_DXI:=0.05}"
: "${ABP_LOCAL_DXI:=0.8}"


# ============================================================
# MUCA roundtrip stopping condition
#
# This controls MUCA learning, not production length.
# ============================================================

: "${ABP_ROUNDTRIP_AVG_TARGET_FRACTION:=0.8}"
: "${ABP_ROUNDTRIP_CONVERGENCE_HITS:=5}"
: "${ABP_ROUNDTRIP_TARGET:=50}"


# ============================================================
# Production-length scaling with noise D
#
# Original fit:
#   steps(D) = 1.167e10 * exp(-97.32 * D)
#
# The raw exponential fit undershoots high noise badly.
# For D=0.1 it gives only ~6.9e5, so we impose a much
# larger production floor.
#
# Recommended for serious T=40 statistics:
#   min production = 5e9
#   max production = 2e10
# ============================================================

: "${ABP_SCALE_PRODUCTION_WITH_D:=true}"

: "${ABP_PROD_STEPS_MODEL_A:=1.167e10}"
: "${ABP_PROD_STEPS_MODEL_B:=97.32}"

: "${ABP_MIN_PROD_OBS_TOTAL:=5000000000}"
: "${ABP_MAX_PROD_OBS_TOTAL:=20000000000}"

if [[ "$ABP_SCALE_PRODUCTION_WITH_D" == "true" || "$ABP_SCALE_PRODUCTION_WITH_D" == "1" ]]; then
    ABP_N_PROD_OBS_TOTAL="$(
        awk \
            -v D="$D" \
            -v A="$ABP_PROD_STEPS_MODEL_A" \
            -v B="$ABP_PROD_STEPS_MODEL_B" \
            -v MIN_STEPS="$ABP_MIN_PROD_OBS_TOTAL" \
            -v MAX_STEPS="$ABP_MAX_PROD_OBS_TOTAL" \
            'BEGIN {
                raw = A * exp(-B * D);

                if (raw < MIN_STEPS) {
                    steps = MIN_STEPS;
                } else if (raw > MAX_STEPS) {
                    steps = MAX_STEPS;
                } else {
                    steps = raw;
                }

                printf "%.0f\n", steps;
            }'
    )"
else
    : "${ABP_N_PROD_OBS_TOTAL:=5000000000}"
fi


# ============================================================
# Production settings
# ============================================================

: "${ABP_N_PROD_CHAINS:=${JULIA_NUM_THREADS}}"
: "${ABP_N_THERM_PROD:=1000000}"

# Store/use one production observation every this many MCMC steps,
# according to the production implementation.
: "${ABP_PROD_STRIDE:=10000}"

# Frequency at which the production roundtrip state is recorded.
: "${ABP_ROUNDTRIP_STRIDE:=10000}"


# ============================================================
# Whole-path occupation histogram settings
#
# For T=40, dt=0.01 gives about 4000 physical-time points
# per trajectory. With scaled production, PATH_TIME_STRIDE=1
# and PATH_OBSERVATION_STRIDE=100 would be extremely expensive.
#
# These defaults still give useful path-density diagnostics
# but avoid counting too many trajectory points.
# ============================================================

# Only accumulate a whole trajectory every N production MCMC steps.
: "${ABP_PATH_OBSERVATION_STRIDE:=1000}"

# Within an accumulated trajectory, count every Nth integration point.
: "${ABP_PATH_TIME_STRIDE:=1}"

# Keep only path points satisfying x(t) > this threshold.
: "${ABP_PATH_FILTER_X_MIN:=-0.8}"


# ============================================================
# Export environment variables
# ============================================================

export JULIA_NUM_THREADS
export ABP_INSTANTIATE
export ABP_PRECOMPILE

export ABP_D_VALUES
export ABP_MOVE_WEIGHTS
export ABP_OUTPUT_DIR
export ABP_SAVE_CSV
export ABP_TRAJECTORY_T

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

export ABP_SCALE_PRODUCTION_WITH_D
export ABP_PROD_STEPS_MODEL_A
export ABP_PROD_STEPS_MODEL_B
export ABP_MIN_PROD_OBS_TOTAL
export ABP_MAX_PROD_OBS_TOTAL

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
echo "ABP noise-array task, T=40, D-scaled production"
echo "============================================================"

printf 'Project:                         %s\n' "$PROJECT_DIR"
printf 'Array job ID:                    %s\n' "$SLURM_ARRAY_JOB_ID"
printf 'Array task ID:                   %s\n' "$SLURM_ARRAY_TASK_ID"
printf 'Noise D:                         %s\n' "$ABP_D_VALUES"
printf 'Trajectory T:                    %s\n' "$ABP_TRAJECTORY_T"

printf 'Julia executable:                %s\n' "$JULIA_CMD"
printf 'Julia threads:                   %s\n' "$JULIA_NUM_THREADS"

printf 'Task output directory:           %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output directory:       %s\n' "$COMBINED_RUN_DIR"

printf 'MUCA iterations:                 %s\n' "$ABP_N_ITER"
printf 'MUCA steps per iteration:        %s\n' "$ABP_N_ITER_STEPS_PER_ITER"
printf 'MUCA thermalization:             %s\n' "$ABP_N_THERM_MUCA"

printf 'MUCA D scaling reference:        %s\n' "$ABP_D_SCALING_REFERENCE"
printf 'Scale MUCA iterations with D:    %s\n' "$ABP_SCALE_N_ITER_WITH_D"

printf 'Roundtrip target fraction:       %s\n' "$ABP_ROUNDTRIP_AVG_TARGET_FRACTION"
printf 'Roundtrip consecutive hits:      %s\n' "$ABP_ROUNDTRIP_CONVERGENCE_HITS"
printf 'Diagnostic roundtrip target:     %s\n' "$ABP_ROUNDTRIP_TARGET"

printf 'Scale production with D:         %s\n' "$ABP_SCALE_PRODUCTION_WITH_D"
printf 'Production scaling model A:      %s\n' "$ABP_PROD_STEPS_MODEL_A"
printf 'Production scaling model B:      %s\n' "$ABP_PROD_STEPS_MODEL_B"
printf 'Min production observations:     %s\n' "$ABP_MIN_PROD_OBS_TOTAL"
printf 'Max production observations:     %s\n' "$ABP_MAX_PROD_OBS_TOTAL"

printf 'Chosen production observations:  %s\n' "$ABP_N_PROD_OBS_TOTAL"
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
#
# Do not use "exec" here. The shell must remain alive after Julia
# finishes so that it can copy the result into the combined folder.
# ============================================================

"$JULIA_CMD" \
    --project=. \
    scripts/run_noise_sweep.jl \
    "$@"


# ============================================================
# Copy this task's unique per-noise results into one common,
# plotting-ready directory.
#
# Each array task has a different D-dependent case directory and
# JLD2 filename, so the copies should not overwrite each other.
# ============================================================

shopt -s nullglob

case_dirs_found=0

for case_dir in "$TASK_OUTPUT_DIR"/data/abp_endpoint_conditioned_*; do
    if [[ -d "$case_dir" ]]; then
        cp -a "$case_dir" "$COMBINED_RUN_DIR/data/"
        case_dirs_found=$((case_dirs_found + 1))
    fi
done

jld2_files_found=0

for result_file in "$TASK_OUTPUT_DIR"/*.jld2; do
    if [[ -f "$result_file" ]]; then
        cp -a "$result_file" "$COMBINED_RUN_DIR/"
        jld2_files_found=$((jld2_files_found + 1))
    fi
done


# ============================================================
# Completion information
# ============================================================

echo "============================================================"
echo "Completed ABP noise-array task, T=40, D-scaled production"
echo "============================================================"

printf 'Noise D:                         %s\n' "$D"
printf 'Array task:                      %s\n' "$ARRAY_INDEX"
printf 'Chosen production observations:  %s\n' "$ABP_N_PROD_OBS_TOTAL"
printf 'Case directories copied:         %s\n' "$case_dirs_found"
printf 'JLD2 files copied:               %s\n' "$jld2_files_found"
printf 'Task output:                     %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output:                 %s\n' "$COMBINED_RUN_DIR"

echo "Output size:"
du -sh "$TASK_OUTPUT_DIR" "$COMBINED_RUN_DIR" 2>/dev/null || true

echo "============================================================"