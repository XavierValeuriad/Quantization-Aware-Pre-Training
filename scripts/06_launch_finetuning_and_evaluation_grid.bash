#!/bin/bash
set -e
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
echo "--- Triggering the fine-tuning and evaluation orchestrator ---"
scripts/utils/_setup_env.sh
.venv/bin/python -m src.orchestrators.run_task_pipeline_grid
echo "--- Fine-Tuning and Evaluation orchestrator completed ---"
