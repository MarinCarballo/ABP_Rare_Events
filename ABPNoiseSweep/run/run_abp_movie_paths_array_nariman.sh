#!/bin/bash
#SBATCH --job-name=abp_movie_paths_nariman
#SBATCH --partition=nariman
#SBATCH --exclude=nariman[09-12]
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --exclusive

# Low-noise movie jobs:
#   task 0: D = 0.05
#   task 1: D = 0.01
#   task 2: D = 0.005
#SBATCH --array=0-3
#SBATCH --error=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_movie_%A_%a.err
#SBATCH --output=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_movie_%A_%a.out

set -euo pipefail


# ============================================================
# Project
# ============================================================

PROJECT_DIR="/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/errors"
mkdir -p "$PROJECT_DIR/data"


# ============================================================
# Select one low-noise value for this Slurm array task
# ============================================================

D_VALUES=(
    "0.1"
    "0.05"
    "0.01"
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
# Movie diagnostics are stored in a dedicated folder:
#
#   ABPNoiseSweep/data/movie_runs/abp_movie_paths_<JOBID>/
#
# This keeps movie data separate from normal production/sweep data.
# ============================================================

MOVIE_DATA_ROOT="${PROJECT_DIR}/data/movie_runs"

RUN_ROOT="${MOVIE_DATA_ROOT}/abp_movie_paths_${SLURM_ARRAY_JOB_ID}"
TASK_OUTPUT_DIR="${RUN_ROOT}/task_${ARRAY_INDEX}_D${D_TAG}"
COMBINED_RUN_DIR="${RUN_ROOT}/combined"

mkdir -p "$MOVIE_DATA_ROOT"
mkdir -p "$TASK_OUTPUT_DIR"
mkdir -p "$COMBINED_RUN_DIR"
mkdir -p "$COMBINED_RUN_DIR/data"

ABP_OUTPUT_DIR="$TASK_OUTPUT_DIR"

# ============================================================
# Julia settings
# ============================================================

: "${JULIA_CMD:=/home/nst/amarin/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
: "${JULIA_NUM_THREADS:=${SLURM_CPUS_PER_TASK:-1}}"

# Keep these false because compute nodes should not need internet.
: "${ABP_INSTANTIATE:=false}"
: "${ABP_PRECOMPILE:=false}"


# ============================================================
# General ABP settings
# ============================================================

: "${ABP_MOVE_WEIGHTS:=run/move_weights.json}"
: "${ABP_SAVE_CSV:=true}"

# Integration time for the physical ABP trajectory.
# This requires the ABP_TRAJECTORY_T line in cli.jl.
: "${ABP_TRAJECTORY_T:=40}"


# ============================================================
# MUCA settings
# ============================================================

: "${ABP_N_ITER:=130}"
: "${ABP_N_ITER_STEPS_PER_ITER:=500000000}"
: "${ABP_N_THERM_MUCA:=1000000}"

: "${ABP_D_SCALING_REFERENCE:=0.05}"
: "${ABP_SCALE_N_ITER_WITH_D:=true}"

: "${ABP_BLOCK_DXI:=0.05}"
: "${ABP_LOCAL_DXI:=0.8}"


# ============================================================
# MUCA roundtrip stopping condition
# ============================================================

: "${ABP_ROUNDTRIP_AVG_TARGET_FRACTION:=0.8}"
: "${ABP_ROUNDTRIP_CONVERGENCE_HITS:=5}"
: "${ABP_ROUNDTRIP_TARGET:=50}"


# ============================================================
# Production settings for long roundtrip movies
#
# Important:
#   The movie saves only ABP_MOVIE_CHAIN_ID.
#   Therefore, to see long Markov-chain evolution, we want
#   many MCMC steps in each individual production chain.
#
#   per-chain steps ≈ ABP_N_PROD_OBS_TOTAL / ABP_N_PROD_CHAINS
# ============================================================

: "${ABP_N_PROD_OBS_TOTAL:=2000000000}"

# Use fewer production chains than 128 so the saved chain runs longer.
# With 2e9 total and 16 chains:
#   chain 1 gets about 1.25e8 MCMC steps.
: "${ABP_N_PROD_CHAINS:=16}"

: "${ABP_N_THERM_PROD:=1000000}"

: "${ABP_PROD_STRIDE:=10000}"

# Record roundtrip diagnostics more finely than before.
: "${ABP_ROUNDTRIP_STRIDE:=5000}"


# ============================================================
# Whole-path occupation histogram settings
#
# These are not the movie settings.
# They only affect the path-density histogram CSVs.
# For a movie-focused job, keep these cheap.
# ============================================================

: "${ABP_PATH_OBSERVATION_STRIDE:=10000}"
: "${ABP_PATH_TIME_STRIDE:=10}"
: "${ABP_PATH_FILTER_X_MIN:=-0.6}"


# ============================================================
# Raw movie trajectory settings for long roundtrip movies
#
# These save a long sequence of full trajectories from one
# production Markov chain.
#
# The production-chain movie will show:
#   frame 1: trajectory at one MCMC step
#   frame 2: trajectory at a later MCMC step
#   ...
# together with the endpoint x(T) trace.
# ============================================================

: "${ABP_SAVE_MOVIE_PATHS:=true}"

# Save one chain only.
: "${ABP_MOVIE_CHAIN_ID:=1}"

# Save one trajectory every this many MCMC steps of the saved chain.
# With chain length ≈ 1.25e8 and stride=50000,
# you get around 2500 saved trajectory snapshots.
: "${ABP_MOVIE_STRIDE:=50000}"

# Inside each saved physical trajectory, save every Nth integration point.
# For T=20, dt=0.01, thin=4 gives about 500 physical frames.
# For T=40, dt=0.01, thin=4 gives about 1000 physical frames.
: "${ABP_MOVIE_PATH_TIME_THIN:=4}"

# Maximum number of saved trajectories.
# This should be large enough not to stop too early.
: "${ABP_MOVIE_MAX_TRAJECTORIES:=3000}"

# Save all endpoints. We filter later in Python.
: "${ABP_MOVIE_ENDPOINT_X_MIN:=-Inf}"
: "${ABP_MOVIE_ENDPOINT_X_MAX:=Inf}"

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

export ABP_N_PROD_OBS_TOTAL
export ABP_N_PROD_CHAINS
export ABP_N_THERM_PROD
export ABP_PROD_STRIDE
export ABP_ROUNDTRIP_STRIDE

export ABP_PATH_OBSERVATION_STRIDE
export ABP_PATH_TIME_STRIDE
export ABP_PATH_FILTER_X_MIN

export ABP_SAVE_MOVIE_PATHS
export ABP_MOVIE_CHAIN_ID
export ABP_MOVIE_STRIDE
export ABP_MOVIE_PATH_TIME_THIN
export ABP_MOVIE_MAX_TRAJECTORIES
export ABP_MOVIE_ENDPOINT_X_MIN
export ABP_MOVIE_ENDPOINT_X_MAX


# ============================================================
# Run information
# ============================================================

echo "============================================================"
echo "ABP movie-path task"
echo "============================================================"

printf 'Project:                         %s\n' "$PROJECT_DIR"
printf 'Array job ID:                    %s\n' "$SLURM_ARRAY_JOB_ID"
printf 'Array task ID:                   %s\n' "$SLURM_ARRAY_TASK_ID"
printf 'Noise D:                         %s\n' "$ABP_D_VALUES"
printf 'Trajectory T:                    %s\n' "$ABP_TRAJECTORY_T"

printf 'Julia executable:                %s\n' "$JULIA_CMD"
printf 'Julia threads:                   %s\n' "$JULIA_NUM_THREADS"

printf 'Run root:                        %s\n' "$RUN_ROOT"
printf 'Task output directory:           %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output directory:       %s\n' "$COMBINED_RUN_DIR"

printf 'MUCA iterations:                 %s\n' "$ABP_N_ITER"
printf 'MUCA steps per iteration:        %s\n' "$ABP_N_ITER_STEPS_PER_ITER"
printf 'MUCA thermalization:             %s\n' "$ABP_N_THERM_MUCA"

printf 'Roundtrip target fraction:       %s\n' "$ABP_ROUNDTRIP_AVG_TARGET_FRACTION"
printf 'Roundtrip consecutive hits:      %s\n' "$ABP_ROUNDTRIP_CONVERGENCE_HITS"
printf 'Diagnostic roundtrip target:     %s\n' "$ABP_ROUNDTRIP_TARGET"

printf 'Production observations:         %s\n' "$ABP_N_PROD_OBS_TOTAL"
printf 'Production chains:               %s\n' "$ABP_N_PROD_CHAINS"
printf 'Production stride:               %s\n' "$ABP_PROD_STRIDE"
printf 'Production roundtrip stride:     %s\n' "$ABP_ROUNDTRIP_STRIDE"

printf 'Path-observation stride:         %s\n' "$ABP_PATH_OBSERVATION_STRIDE"
printf 'Path-time stride:                %s\n' "$ABP_PATH_TIME_STRIDE"
printf 'Path x(t) minimum:               %s\n' "$ABP_PATH_FILTER_X_MIN"

printf 'Save movie paths:                %s\n' "$ABP_SAVE_MOVIE_PATHS"
printf 'Movie chain ID:                  %s\n' "$ABP_MOVIE_CHAIN_ID"
printf 'Movie stride:                    %s\n' "$ABP_MOVIE_STRIDE"
printf 'Movie path time thin:            %s\n' "$ABP_MOVIE_PATH_TIME_THIN"
printf 'Movie max trajectories:          %s\n' "$ABP_MOVIE_MAX_TRAJECTORIES"
printf 'Movie endpoint x min:            %s\n' "$ABP_MOVIE_ENDPOINT_X_MIN"
printf 'Movie endpoint x max:            %s\n' "$ABP_MOVIE_ENDPOINT_X_MAX"

printf 'Movie data root:                 %s\n' "$MOVIE_DATA_ROOT"
printf 'Run root:                        %s\n' "$RUN_ROOT"
printf 'Task output directory:           %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output directory:       %s\n' "$COMBINED_RUN_DIR"

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
# Copy this task's unique per-noise results into one common,
# plotting-ready directory.
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
echo "Completed ABP movie-path task"
echo "============================================================"

printf 'Noise D:                         %s\n' "$D"
printf 'Array task:                      %s\n' "$ARRAY_INDEX"
printf 'Case directories copied:         %s\n' "$case_dirs_found"
printf 'JLD2 files copied:               %s\n' "$jld2_files_found"
printf 'Task output:                     %s\n' "$TASK_OUTPUT_DIR"
printf 'Combined output:                 %s\n' "$COMBINED_RUN_DIR"

echo "Movie CSV check:"
find "$COMBINED_RUN_DIR" -name "movie_trajectories_long.csv" -print || true

echo "Output size:"
du -sh "$RUN_ROOT" || true

echo "============================================================"