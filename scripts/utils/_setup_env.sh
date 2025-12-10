#!/bin/bash
# This script is intended to be sourced, not executed.
set -e

# Activates the Python virtual environment
source .venv/bin/activate

# Loads local environment variables (e.g., HUGGING_FACE_HUB_TOKEN)
if [ -f ".env" ]; then
    echo "INFO: .env file found. Loading environment variables."
    source .env
fi