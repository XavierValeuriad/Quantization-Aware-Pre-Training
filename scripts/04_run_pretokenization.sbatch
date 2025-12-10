#!/bin/bash
# ===================================================================
# SLURM Job for PROD Tokenization v1.0
# ===================================================================

# --- OFFICIAL ALLOCATION DIRECTIVES
#SBATCH --job-name=qapt_pretokenization
#SBATCH -A XXX                          # Account and CPU partition specified
#SBATCH -C h100                         # Explicit hardware constraint H100
#SBATCH --partition=gpu_p6              # Partition specific to H100.
#SBATCH --qos=qos_gpu_h100-dev          # [OPTIMIZATION] QoS for short jobs (< 2h).
#SBATCH --gres=gpu:1                    # Request 1 GPU (which unlocks access to the node)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

# --- LOG MANAGEMENT ---
#SBATCH --output=logs/slurm/%x-%j.out    # Log files renamed for clarity
#SBATCH --error=logs/slurm/%x-%j.err     # Job name (%x) and ID (%j) will be used

set -x -e

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "Compute nodes allocated. Aligning base modules..."
module purge
module load arch/h100
module load miniforge/24.9.0

cd ${WORK}/Quantization-Aware-Pre-Training

echo "🔥 Activating pre-existing sovereign environment..."
conda activate ./conda_envs/qapt_env

echo "🗺️ Updating PATH to include our sovereign tool base..."
# We add the directory of executables
# installed via PYTHONUSERBASE to the system PATH.
export PYTHONUSERBASE=${WORK}/Quantization-Aware-Pre-Training/python_user_base
export PATH=${PYTHONUSERBASE}/bin:$PATH

echo "✅ Environment ready. Launching via direct designation..."

echo "🚀 Launching pre-tokenization orchestrator..."

# [FINAL & ROBUST COMMAND]
# We use the absolute path to the 'poetry' executable
# which was installed EXACTLY ONCE in our environment.
srun poetry run python -m src.orchestrators.run_pretokenization

echo "✅ Pre-tokenization pipeline completed."