#!/bin/bash
# Run a python script with a fresh, node-local Triton/Inductor cache:
#   scripts/run/eval_shim.sh scripts/dualsps/eval_runs.py <run> ...
# A Triton cache shared across jobs on a network filesystem can return wrong kernels
# without any error (measured on one checkpoint: val NLL 9.71 through the shared cache,
# 3.11 through a clean one). Every GPU evaluation and analysis goes through this shim.
set -euo pipefail
BASE="/tmp/${USER:-$(id -un)}/dualsps_eval_$$"
export TRITON_CACHE_DIR="$BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$BASE/inductor"
export DUALSPS_NODE_LOCAL_CACHE=1   # checked by scripts/analysis/common.assert_node_local_triton
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
trap 'rm -rf "$BASE"' EXIT
"${PYTHON:-.venv/bin/python}" "$@"
