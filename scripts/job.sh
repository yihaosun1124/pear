#!/usr/bin/env bash
set -euo pipefail
# Cluv supplies --chdir and the per-cluster resources from pyproject.toml.
exec /usr/bin/python3 -B scripts/launch_pear.py "$@"
