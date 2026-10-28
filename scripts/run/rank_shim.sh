#!/bin/bash
# torchrun --no-python target: one node-local Triton/Inductor cache per rank.
#   torchrun --standalone --nproc_per_node=8 --no-python scripts/run/rank_shim.sh scripts/train.py ...
# Ranks sharing one cache on a network filesystem race inside torch.compile
# (FileNotFoundError on .json/.llir/.cubin).
set -euo pipefail
BASE="/tmp/${USER:-$(id -un)}/dualsps_${TORCHELASTIC_RUN_ID:-$PPID}/rank${LOCAL_RANK:-0}"
export TRITON_CACHE_DIR="$BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$BASE/inductor"
export DUALSPS_NODE_LOCAL_CACHE=1   # checked by scripts/analysis/common.assert_node_local_triton
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exec "${PYTHON:-.venv/bin/python}" "$@"
