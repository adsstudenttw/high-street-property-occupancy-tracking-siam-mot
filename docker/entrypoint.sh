#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

exec "$@"
