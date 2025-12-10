#!/bin/bash
# ===================================================================
# ✧ SLURM Job Array for PROD Pre-training v1.0
# ✧ 1 TASK / 1 GPU / 1 EXPERIMENT
# ===================================================================

# --- ALLOCATION DIRECTIVES (ARRAY MODE: 1 JOB / 1 GPU) ---
#SBATCH --job-name=qapt_preentrainement_array # Job name
#SBATCH -A XXX                          # H100 project account
#SBATCH -C h100                         # H100 constraint
#SBATCH --partition=gpu_p6              # Partition H100
#SBATCH --qos=qos_gpu_h100-t3           # H100 partition
#SBATCH --time=10:00:00                 # Max time (adjust after 1st run)
#SBATCH --hint=nomultithread

# --- LOG MANAGEMENT (ARRAY MODE) ---
# %A = Job ID (ex: 305813), %a = Task ID (ex: 0, 1, 2...)
# Ensure the logs/slurm/ folder exists
# mkdir -p logs/slurm
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

# --- ARRAY PARAMETERS (1 GPU per task) ---
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # 1 single main process
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # 1 single GPU
#SBATCH --cpus-per-task=24              # CPU per GPU ratio (24 cores / 1 GPU)

# --- EXPERIMENT GRID DEFINITION ---
# (Based on config.yml: 1 model * 8 schemes = 8 jobs)
# We run max 24 concurrent jobs (JZ courtesy limit)
#SBATCH --array=0-19%20

set -x -e

# ========= ✧ EXPERIMENT GRID DEFINITION ✧ =========
# (Must match config.yml)
MODELS=(
   "camembert-base"
   "Dr-BERT/DrBERT-7GB"
)

# Schemes with 'enabled: true' in config.yml
QUANT_SCHEMES=(
    "FP32_Baseline"
    "LSQ_E8W8A8"
    "LSQ_E4W4A4"
    "LSQ_E6W2A6"
    "LSQ_E2W2A2"
    "BitNet_E6W1.58A6"
)

# --- Build combinations ---
COMBOS=()
for model in "${MODELS[@]}"; do
    for quant in "${QUANT_SCHEMES[@]}"; do
        # Format: "model_id|quant_name"
        COMBOS+=("$model|$quant")
    done
done
TOTAL=${#COMBOS[@]}
echo "TOTAL COMBINATIONS: $TOTAL"

# --- Handle array (logic from tuto_jeanzay_1 example) ---
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "Index ${SLURM_ARRAY_TASK_ID} >= TOTAL ${TOTAL} → nothing to do."
    exit 0
fi

IFS='|' read -r MODEL_ID QUANT_NAME <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================================"
echo "✧ Launching Task ${SLURM_ARRAY_TASK_ID}/${TOTAL}"
echo "✧ Model        : ${MODEL_ID}"
echo "✧ Quant.       : ${QUANT_NAME}"
echo "======================================================"
# ============================================================


# --- ENVIRONMENT MANAGEMENT (Identical to 05_...) ---
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

eecho "Compute nodes allocated. Aligning base modules..."
module purge
module load arch/h100
module load miniforge/24.9.0

cd ${WORK}/Quantization-Aware-Pre-Training

echo "🔥 Activating pre-existing sovereign environment..."
conda activate ./conda_envs/qapt_env

echo "🗺️ Updating PATH to include our sovereign tool base..."
export PYTHONUSERBASE=${WORK}/Quantization-Aware-Pre-Training/python_user_base
export PATH=${PYTHONUSERBASE}/bin:$PATH

echo "✅ Environment ready."

# --- ✧ SINGLE TASK LAUNCH ✧ ---
# Export variables that the Python script will read
export MODEL_ID="$MODEL_ID"
export QUANT_NAME="$QUANT_NAME"

echo "🚀 Launching orchestrator in SINGLE TASK mode..."
srun poetry run python -m src.orchestrators.run_pretraining_array

echo "✅ Task ${SLURM_ARRAY_TASK_ID} (${MODEL_ID} / ${QUANT_NAME}) completed."