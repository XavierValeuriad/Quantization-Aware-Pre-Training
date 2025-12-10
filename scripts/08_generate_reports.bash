#!/bin/bash
set -e
echo "--- Triggering report generation orchestrator ---"
scripts/utils/_setup_env.sh
.venv/bin/python -m src.orchestrators.generate_reports
.venv/bin/python -m src.orchestrators.run_reporting_grid
echo "--- Report generation orchestrator completed ---"