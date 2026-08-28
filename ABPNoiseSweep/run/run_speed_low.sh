#!/usr/bin/env bash
#SBATCH --job-name=abp_speed_D0p025_lowv
#SBATCH --partition=nariman
#SBATCH --exclude=nariman[09-12]
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --exclusive
#SBATCH --array=0-5
#SBATCH --error=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_speed_D0025_lowv_%A_%a.err
#SBATCH --output=/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep/errors/abp_speed_D0025_lowv_%A_%a.out

set -euo pipefail

PROJECT_DIR="/home/nst/amarin/Desktop/ABP_Rare_Events/ABPNoiseSweep"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/errors" "$PROJECT_DIR/data"

V_VALUES=("0.290" "0.295" "0.300" "0.305" "0.310" "0.315")

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#V_VALUES[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${ARRAY_INDEX}"
    exit 1
fi

V="${V_VALUES[$ARRAY_INDEX]}"
V_TAG="${V//./p}"
V_TAG="${V_TAG//-/m}"

export ABP_V="$V"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ABP_INSTANTIATE=false
export ABP_PRECOMPILE=false

export ABP_N_ITER=200
export ABP_N_ITER_STEPS_PER_ITER=800000000
export ABP_N_THERM_MUCA=1000000

export ABP_N_PROD_OBS_TOTAL=55000000000
export ABP_N_PROD_CHAINS="${SLURM_CPUS_PER_TASK}"
export ABP_N_THERM_PROD=1000000
export ABP_PROD_STRIDE=1000
export ABP_ROUNDTRIP_STRIDE=1000

export ABP_PATH_OBSERVATION_STRIDE=1000
export ABP_PATH_TIME_STRIDE=1
export ABP_PATH_FILTER_X_MIN=-1.0

export ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION=0.5
export ABP_ROUNDTRIP_CONVERGENCE_HITS=5
export ABP_ROUNDTRIP_TARGET=50
export ABP_SAVE_CSV=true

RUN_ROOT="${PROJECT_DIR}/data/abp_speed_D0p025_array_${SLURM_ARRAY_JOB_ID}"
TASK_OUTPUT_DIR="${RUN_ROOT}/task_${ARRAY_INDEX}_v${V_TAG}"
mkdir -p "$TASK_OUTPUT_DIR"

export ABP_OUTPUT_DIR="$TASK_OUTPUT_DIR"

RUNNER_FILE="${PROJECT_DIR}/scripts/run_fixed_D_speed_landscape.jl"

if [[ ! -f "$RUNNER_FILE" ]]; then
    echo "ERROR: runner not found: $RUNNER_FILE"
    echo "Set RUNNER_FILE to the combined MUCA -> production Julia runner used for job 1669022."
    exit 1
fi

JULIA_CMD="/home/nst/amarin/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"

echo "============================================================"
echo "Fixed-noise low-speed task: MUCA -> production"
echo "D = 0.025 | v = ${V} | array task = ${ARRAY_INDEX}"
echo "Job ID = ${SLURM_ARRAY_JOB_ID}"
echo "Threads = ${JULIA_NUM_THREADS}"
echo "MUCA iterations = ${ABP_N_ITER}"
echo "MUCA final-iteration moves = ${ABP_N_ITER_STEPS_PER_ITER}"
echo "Production moves = ${ABP_N_PROD_OBS_TOTAL}"
echo "Production chains = ${ABP_N_PROD_CHAINS}"
echo "Path-observation stride = ${ABP_PATH_OBSERVATION_STRIDE}"
echo "Path filter x(t) > ${ABP_PATH_FILTER_X_MIN}"
echo "Task output = ${TASK_OUTPUT_DIR}"
echo "Runner = ${RUNNER_FILE}"
echo "============================================================"

"$JULIA_CMD" --startup-file=no --project="$PROJECT_DIR" "$RUNNER_FILE"

echo
echo "Produced files:"
find "$TASK_OUTPUT_DIR" -maxdepth 4 -type f \( -name '*.jld2' -o -name '*.csv' \) -print | sort