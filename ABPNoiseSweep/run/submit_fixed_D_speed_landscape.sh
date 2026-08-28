#!/usr/bin/env bash
#SBATCH --job-name=abp_speed_D0p025
#SBATCH --partition=nariman
#SBATCH --exclude=nariman[09-12]
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G
#SBATCH --array=0-14
#SBATCH --error=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_speed_D0p025_%A_%a.err
#SBATCH --output=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_speed_D0p025_%A_%a.out

set -euo pipefail

PROJECT_DIR="/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/errors" "$PROJECT_DIR/data"

# One velocity per Slurm array task.  Noise is fixed at D=0.025 in the Julia
# runner.  This gives 15 independent MUCA -> production simulations.
V_VALUES=(
    "0.320"
    "0.325"
    "0.330"
    "0.335"
    "0.340"
    "0.345"
    "0.350"
    "0.355"
    "0.360"
    "0.365"
    "0.370"
    "0.375"
    "0.380"
    "0.385"
    "0.390"
)

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#V_VALUES[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${ARRAY_INDEX}"
    exit 1
fi

ABP_V="${V_VALUES[$ARRAY_INDEX]}"
V_TAG="${ABP_V//./p}"

RUN_ROOT="${PROJECT_DIR}/data/abp_speed_D0p025_array_${SLURM_ARRAY_JOB_ID}"
TASK_OUTPUT_DIR="${RUN_ROOT}/task_${ARRAY_INDEX}_v${V_TAG}"
COMBINED_RUN_DIR="${RUN_ROOT}/combined"

mkdir -p "$TASK_OUTPUT_DIR" "$COMBINED_RUN_DIR/data"

: "${JULIA_CMD:=/home/nst/amarin/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia}"
JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Full long-run settings
: "${ABP_N_ITER:=200}"
: "${ABP_N_ITER_STEPS_PER_ITER:=800000000}"
: "${ABP_N_THERM_MUCA:=1000000}"

: "${ABP_N_PROD_OBS_TOTAL:=55000000000}"
: "${ABP_N_PROD_CHAINS:=128}"
: "${ABP_N_THERM_PROD:=1000000}"
: "${ABP_PROD_STRIDE:=10000}"
: "${ABP_ROUNDTRIP_STRIDE:=10000}"

: "${ABP_PATH_OBSERVATION_STRIDE:=1000}"
: "${ABP_PATH_TIME_STRIDE:=1}"
: "${ABP_PATH_FILTER_X_MIN:=-1}"

: "${ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION:=0.5}"
: "${ABP_ROUNDTRIP_CONVERGENCE_HITS:=5}"
: "${ABP_ROUNDTRIP_TARGET:=50}"

: "${ABP_SAVE_CSV:=true}"

export JULIA_NUM_THREADS ABP_V ABP_OUTPUT_DIR
export ABP_N_ITER ABP_N_ITER_STEPS_PER_ITER ABP_N_THERM_MUCA
export ABP_N_PROD_OBS_TOTAL ABP_N_PROD_CHAINS ABP_N_THERM_PROD
export ABP_PATH_OBSERVATION_STRIDE ABP_PATH_TIME_STRIDE
export ABP_PATH_FILTER_X_MIN ABP_SAVE_CSV
export JULIA_NUM_THREADS ABP_V ABP_OUTPUT_DIR
export ABP_N_ITER ABP_N_ITER_STEPS_PER_ITER ABP_N_THERM_MUCA
export ABP_N_PROD_OBS_TOTAL ABP_N_PROD_CHAINS ABP_N_THERM_PROD
export ABP_PROD_STRIDE ABP_ROUNDTRIP_STRIDE
export ABP_PATH_OBSERVATION_STRIDE ABP_PATH_TIME_STRIDE
export ABP_PATH_FILTER_X_MIN
export ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION
export ABP_ROUNDTRIP_CONVERGENCE_HITS ABP_ROUNDTRIP_TARGET
export ABP_SAVE_CSV

echo "============================================================"
echo "Fixed-noise speed task: MUCA -> production"
echo "D = 0.025 | v = ${ABP_V} | array task = ${ARRAY_INDEX}"
echo "Threads = ${JULIA_NUM_THREADS} | production chains = ${ABP_N_PROD_CHAINS}"
echo "MUCA iterations = ${ABP_N_ITER}"
echo "MUCA final-iteration moves = ${ABP_N_ITER_STEPS_PER_ITER}"
echo "Production moves = ${ABP_N_PROD_OBS_TOTAL}"
echo "Task output = ${TASK_OUTPUT_DIR}"
echo "============================================================"

"$JULIA_CMD" --project=. --threads="$JULIA_NUM_THREADS" \
    scripts/run_fixed_D_speed_landscape.jl

# Collect plotting-ready results without filename collisions between speeds.
shopt -s nullglob

result_files=("$TASK_OUTPUT_DIR"/*.jld2)
if (( ${#result_files[@]} != 1 )); then
    echo "ERROR: expected one JLD2 result, found ${#result_files[@]}"
    exit 1
fi

cp -a "${result_files[0]}" \
    "$COMBINED_RUN_DIR/abp_endpoint_conditioned_D0p025_v${V_TAG}.jld2"

case_dirs=("$TASK_OUTPUT_DIR"/data/abp_endpoint_conditioned_*)
if (( ${#case_dirs[@]} == 1 )); then
    cp -a "${case_dirs[0]}" \
        "$COMBINED_RUN_DIR/data/abp_endpoint_conditioned_D0p025_v${V_TAG}"
fi

echo "============================================================"
echo "COMPLETED: D = 0.025 | v = ${ABP_V}"
echo "Task output: $TASK_OUTPUT_DIR"
echo "Combined output: $COMBINED_RUN_DIR"
echo "============================================================"

