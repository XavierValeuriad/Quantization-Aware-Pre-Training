#!/bin/bash
set -e
echo "--- Triggering model acquisition orchestrator ---"
scripts/utils/_setup_env.sh
.venv/bin/python -m src.orchestrators.download_models
echo "--- Model acquisition orchestrator completed ---"