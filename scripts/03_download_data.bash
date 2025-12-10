#!/bin/bash
set -e
echo "--- Triggering data acquisition orchestrator ---"
scripts/utils/_setup_env.sh
.venv/bin/python -m src.orchestrators.download_data
echo "--- Data acquisition orchestrator completed ---"