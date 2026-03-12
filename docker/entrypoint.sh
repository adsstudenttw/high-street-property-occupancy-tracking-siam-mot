#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export CONDA_DIR="${CONDA_DIR:-/opt/conda}"
export CONDA_ENV="${CONDA_ENV:-siammot}"

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    # Activate the project env so every docker run command uses the conda stack.
    . "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    export PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:${PATH}"
fi

exec "$@"
