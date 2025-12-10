#!/usr/bin/env bash

# ===================================================================
# Bootstrap Script v4.0 (Hybrid: uv + Poetry)
#
# This version leverages 'uv' for rapid environment setup while
# relying on the standard and robust tools of the Python ecosystem
# (`venv`, `pip`) orchestrated by `poetry`. This is the most stable
# configuration for an HPC environment like Jean-Zay.
# ===================================================================

set -euo pipefail

echo "==> Configuring bootstrap environment..."

# --- SYMMETRIC CACHE LOGIC (BASH) ---
PROJECT_NAME=$(basename "$PWD")
ENV_FOLDER="prod"
DEV_CONFIG_FILE="configs/dev_config.yaml"
if [ -f "$DEV_CONFIG_FILE" ]; then
    if grep -qE "^\s*dev\s*:\s*true\s*$" "$DEV_CONFIG_FILE"; then
        ENV_FOLDER="dev"
    fi
fi
echo "Environment mode for cache: $ENV_FOLDER"

if [ -n "${SCRATCH-}" ]; then
    CACHE_ROOT="$SCRATCH/$ENV_FOLDER/$PROJECT_NAME/cache"
    echo "HPC environment detected. Cache root: $CACHE_ROOT"
else
    PROJECT_PARENT_DIR=$(dirname "$PWD")
    CACHE_ROOT="$PROJECT_PARENT_DIR/../SCRATCH/$ENV_FOLDER/$PROJECT_NAME/cache"
    echo "Local environment detected. Cache root: $CACHE_ROOT"
fi

# Cache configuration for pip and poetry only.
mkdir -p "$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export POETRY_CACHE_DIR="$CACHE_ROOT/poetry"
export POETRY_CONFIG_DIR="$CACHE_ROOT/poetry_config"
echo "Cache variables (pip, poetry) configured."
# ---------------------------------------------

# --- CHECKING FOR 'uv' ---
if ! command -v uv >/dev/null; then
    echo "==> 'uv' not found. Attempting installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" 
else
    echo "==> 'uv' is already installed."
fi

echo "==> Creating virtual environment '.venv' with Python 3.10.4"

# --- VIRTUAL ENVIRONMENT CREATION ---
uv venv .venv --python 3.10.4

# --- ENVIRONMENT ACTIVATION ---
echo "==> Activating .venv environment"
source .venv/bin/activate

# --- ORCHESTRATOR INSTALLATION ---
echo "==> Installing 'poetry' into the virtual environment using 'pip'"
# Using pip (via uv), the standard, to install the management tool.
uv pip install --upgrade pip
uv pip install --upgrade "poetry>=1.5"

# --- CONFIGURATION AND INSTALLATION VIA POETRY ---
# The following logic is correct as it uses 'poetry' as an orchestrator
# which, internally, will rely on 'pip' for installation.

echo "==> Configuring Poetry to use the ACTIVE environment"
poetry config virtualenvs.create false --local

echo "==> Synchronizing poetry.lock with pyproject.toml..."

if [ ! -f "poetry.lock" ]; then
    echo "==> 'poetry.lock' not found. Generating lock file..."
    poetry lock
else
    echo "==> Existing 'poetry.lock' found. Generation skipped to ensure reproducibility."
fi

echo "==> Installing project dependencies via Poetry (using pip as backend)"
poetry install --no-interaction --no-root

echo ""
echo "✅ ==> Bootstrap completed successfully."
echo "The '.venv' environment is ready and contains all dependencies."
echo ""
echo "To activate your environment for a work session, run:"
echo "source .venv/bin/activate"
echo ""
python -V