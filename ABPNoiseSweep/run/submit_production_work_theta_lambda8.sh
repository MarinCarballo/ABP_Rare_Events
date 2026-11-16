#!/usr/bin/env bash
# Submit from ABPNoiseSweep after creating errors/ (Slurm opens logs first):
#   mkdir -p errors
#   sbatch run/submit_production_work_theta_lambda8.sh
#SBATCH --job-name=abp_work_theta_lambda8_D0p025
#SBATCH --partition=nariman
#SBATCH --exclude=nariman[09-12]
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G
#SBATCH --array=0-28
#SBATCH --error=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_work_theta_lambda8_%A_%a.err
#SBATCH --output=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_work_theta_lambda8_%A_%a.out

set -euo pipefail

# Same production setup in both lambda submissions; the selected potential
# parameter and output namespace are their only differences.
PROJECT_DIR="${ABP_PROJECT_DIR:-/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/errors"

# Fix the physical setup even if a previous shell exported different values.
ABP_POTENTIAL_LAMBDA=8.0
ABP_TRAJECTORY_T=40.0
ABP_DT=0.01
ABP_INTEGRATOR=euler
ABP_NOISE_PROPOSAL=random_walk
ABP_SAVE_MOVIE_PATHS=false
ABP_SAVE_MUCA_MOVIE_PATHS=false
export ABP_POTENTIAL_LAMBDA ABP_TRAJECTORY_T ABP_DT
export ABP_SAVE_MOVIE_PATHS ABP_SAVE_MUCA_MOVIE_PATHS

# Safe configuration inspection without simulation or result collection:
#   SLURM_ARRAY_JOB_ID=preflight SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK=1 \
#     bash run/submit_production_work_theta_lambda8.sh --dry-run
# Optional Crank-Nicolson: sbatch this_script.sh --integrator=crank_nicolson
DRY_RUN=false
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=true ;;
        --integrator=*) ABP_INTEGRATOR="${argument#*=}" ;;
        --potential-lambda=*)
            echo "ERROR: lambda is fixed by this submission file; use the other lambda script." >&2
            exit 1 ;;
        *) echo "ERROR: unsupported runner argument: $argument" >&2; exit 1 ;;
    esac
done

# Retain 0.280:0.005:0.390 and add outer speeds for wider coverage.
# Relative to vc=0.32: 0.200 is 37.5% below and 0.500 is 56.25% above.
# vc=0.32 is the supplied lambda=2.5 reference, not an established lambda=8 threshold.
# If changing the grid, update #SBATCH --array as well (29 speeds: 0 through 28).
V_VALUES=(
    0.200 0.240 0.260 0.280 0.285 0.290
    0.295 0.300 0.305 0.310 0.315 0.320
    0.325 0.330 0.335 0.340 0.345 0.350
    0.355 0.360 0.365 0.370 0.375 0.380
    0.385 0.390 0.400 0.450 0.500
)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this script using sbatch --array.}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID.}"
if [[ ! "$ARRAY_INDEX" =~ ^[0-9]+$ ]] || (( ARRAY_INDEX >= ${#V_VALUES[@]} )); then
    echo "ERROR: invalid array index: $ARRAY_INDEX" >&2
    exit 1
fi

ABP_V="${V_VALUES[$ARRAY_INDEX]}"
V_TAG="${ABP_V//./p}"
RUN_ROOT="$PROJECT_DIR/data/abp_work_theta_lambda8_D0p025_array_${ARRAY_JOB_ID}"
TASK_OUTPUT_DIR="$RUN_ROOT/task_${ARRAY_INDEX}_v${V_TAG}"
COMBINED_RUN_DIR="$RUN_ROOT/combined"
mkdir -p "$TASK_OUTPUT_DIR" "$COMBINED_RUN_DIR/data"

# Explicit assignment is essential: exporting an unset ABP_OUTPUT_DIR would
# let the runner write into its legacy fallback directory.
ABP_OUTPUT_DIR="$TASK_OUTPUT_DIR"

: "${JULIA_CMD:=/home/nst/amarin/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-128}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

if ! command -v "$JULIA_CMD" >/dev/null 2>&1; then
    echo "ERROR: Julia executable not found: $JULIA_CMD" >&2
    exit 1
fi
if [[ ! -f scripts/run_fixed_D_speed_landscape.jl ]]; then
    echo "ERROR: fixed-D Julia runner not found in $PROJECT_DIR" >&2
    exit 1
fi

# Same long production workload as the current speed submission.
# Seven days is the requested time limit, not a measured completion guarantee.
: "${ABP_N_ITER:=200}"
: "${ABP_N_ITER_STEPS_PER_ITER:=800000000}"
: "${ABP_N_THERM_MUCA:=1000000}"
: "${ABP_BLOCK_DXI:=0.05}"
: "${ABP_LOCAL_DXI:=0.8}"

: "${ABP_N_PROD_OBS_TOTAL:=55000000000}"
: "${ABP_N_PROD_CHAINS:=$JULIA_NUM_THREADS}"
: "${ABP_N_THERM_PROD:=1000000}"
: "${ABP_PROD_STRIDE:=10000}"
: "${ABP_ROUNDTRIP_STRIDE:=10000}"
: "${ABP_PATH_OBSERVATION_STRIDE:=1000}"
: "${ABP_PATH_TIME_STRIDE:=1}"
: "${ABP_ACTIVE_WORK_OBSERVATION_STRIDE:=10000}"
: "${ABP_PATH_FILTER_X_MIN:=-1}"
: "${ABP_N_PATH_THETA_BINS:=180}"

: "${ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION:=0.5}"
: "${ABP_ROUNDTRIP_CONVERGENCE_HITS:=5}"
: "${ABP_ROUNDTRIP_TARGET:=50}"
: "${ABP_SAVE_CSV:=true}"

# Euler is fixed above unless --integrator=... was explicitly supplied.
: "${ABP_INTEGRATOR:=euler}"
export ABP_INTEGRATOR
# Keep the existing random-walk path-sampling kernel in both submissions.
: "${ABP_NOISE_PROPOSAL:=random_walk}"
: "${ABP_BLOCK_PCN_BETA:=0.05}"
: "${ABP_LOCAL_PCN_BETA:=0.8}"
export ABP_NOISE_PROPOSAL ABP_BLOCK_PCN_BETA ABP_LOCAL_PCN_BETA

# Storage-only settings: keep complete production examples for the notebook.
: "${ABP_SAVED_PATH_TIME_THIN:=1}"
: "${ABP_MAX_SAVED_PATHS_PER_WINDOW:=120}"
export ABP_SAVED_PATH_TIME_THIN ABP_MAX_SAVED_PATHS_PER_WINDOW

if (( ABP_N_PROD_CHAINS < 1 || ABP_N_PROD_CHAINS > JULIA_NUM_THREADS )); then
    echo "ERROR: production chains must be between 1 and allocated Julia threads." >&2
    exit 1
fi
if (( ABP_PATH_OBSERVATION_STRIDE < 1 || ABP_ACTIVE_WORK_OBSERVATION_STRIDE < 1 )); then
    echo "ERROR: observation strides must be positive." >&2
    exit 1
fi
if (( ABP_ACTIVE_WORK_OBSERVATION_STRIDE % ABP_PATH_OBSERVATION_STRIDE != 0 )); then
    echo "ERROR: active-work stride must be a multiple of the path-observation stride." >&2
    exit 1
fi

export JULIA_NUM_THREADS ABP_V ABP_OUTPUT_DIR
export ABP_N_ITER ABP_N_ITER_STEPS_PER_ITER ABP_N_THERM_MUCA
export ABP_BLOCK_DXI ABP_LOCAL_DXI
export ABP_N_PROD_OBS_TOTAL ABP_N_PROD_CHAINS ABP_N_THERM_PROD
export ABP_PROD_STRIDE ABP_ROUNDTRIP_STRIDE
export ABP_PATH_OBSERVATION_STRIDE ABP_PATH_TIME_STRIDE
export ABP_ACTIVE_WORK_OBSERVATION_STRIDE ABP_PATH_FILTER_X_MIN ABP_N_PATH_THETA_BINS
export ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION ABP_ROUNDTRIP_CONVERGENCE_HITS ABP_ROUNDTRIP_TARGET
export ABP_SAVE_CSV

echo "MUCA -> production with the installed move kernel"
echo "lambda=$ABP_POTENTIAL_LAMBDA | D=0.025 | v=$ABP_V | array task=$ARRAY_INDEX"
echo "Physical duration=$ABP_TRAJECTORY_T | dt=$ABP_DT | noise proposal=$ABP_NOISE_PROPOSAL"
echo "Integrator=$ABP_INTEGRATOR"
echo "Threads=$JULIA_NUM_THREADS | production chains=$ABP_N_PROD_CHAINS"
echo "MUCA iterations=$ABP_N_ITER | final-iteration moves=$ABP_N_ITER_STEPS_PER_ITER"
echo "Production moves=$ABP_N_PROD_OBS_TOTAL"
echo "Path stride=$ABP_PATH_OBSERVATION_STRIDE | physical-time stride=$ABP_PATH_TIME_STRIDE"
echo "Active-work stride=$ABP_ACTIVE_WORK_OBSERVATION_STRIDE | theta bins=$ABP_N_PATH_THETA_BINS"
echo "Path-point filter: x(t)>$ABP_PATH_FILTER_X_MIN"
echo "Task output: $ABP_OUTPUT_DIR"

"$JULIA_CMD" --startup-file=no --project=. --threads="$JULIA_NUM_THREADS" scripts/run_fixed_D_speed_landscape.jl "$@"

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run completed: configuration checked; no simulation or result collection."
    exit 0
fi

# All task files use a D-only basename. Add v when collecting them for plotting.
shopt -s nullglob
result_files=("$TASK_OUTPUT_DIR"/abp_endpoint_conditioned_*.jld2)
if (( ${#result_files[@]} != 1 )); then
    echo "ERROR: expected one JLD2 in $TASK_OUTPUT_DIR; found ${#result_files[@]}." >&2
    exit 1
fi
cp -a "${result_files[0]}" "$COMBINED_RUN_DIR/abp_endpoint_conditioned_lambda8_D0p025_v${V_TAG}.jld2"

case_dirs=("$TASK_OUTPUT_DIR"/data/abp_endpoint_conditioned_*)
if (( ${#case_dirs[@]} == 1 )); then
    CSV_DEST="$COMBINED_RUN_DIR/data/abp_endpoint_conditioned_lambda8_D0p025_v${V_TAG}"
    mkdir -p "$CSV_DEST"
    cp -a "${case_dirs[0]}/." "$CSV_DEST/"
fi

echo "Completed lambda=$ABP_POTENTIAL_LAMBDA, D=0.025, v=$ABP_V"
echo "Notebook RUN_ROOTS: [Path(\"$COMBINED_RUN_DIR\")]"
