# Shared environment for the commands in the README ("Reproduce from scratch").
# Source it from the repo root:   source scripts/run/env.sh
#
# Every value comes from the environment, falling back to <repo>/.env (KEY=value lines,
# see .env.example). A variable already set in the environment wins over .env.
#   DUALSPS_DATA_ROOT  required: tokenized corpus at <root>/data/<dataset>/{train,val}.bin
#   DUALSPS_OUT_ROOT   run outputs at <root>/out/<run>/ (default: DUALSPS_DATA_ROOT)
#   DUALSPS_LOG_DIR    training logs, read by eval_runs.py (default: <DATA_ROOT>/logs)
#   VENV               python environment (default: <repo>/.venv); its bin/ goes first on PATH
# The python scripts read the same variables, and .env, through src/repo_paths.py.

if [ ! -f scripts/train.py ]; then
    echo "scripts/run/env.sh: $(pwd) is not the repo root; source it from there" >&2
    return 1 2>/dev/null || exit 1
fi

if [ -f .env ]; then
    while IFS='=' read -r key value; do
        [[ $key =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        [ -n "${!key:-}" ] || export "$key=$value"
    done < .env
fi

if [ -z "${DUALSPS_DATA_ROOT:-}" ]; then
    echo "scripts/run/env.sh: set DUALSPS_DATA_ROOT (environment or .env, see .env.example)" >&2
    return 1 2>/dev/null || exit 1
fi
export DUALSPS_DATA_ROOT
export DUALSPS_OUT_ROOT="${DUALSPS_OUT_ROOT:-$DUALSPS_DATA_ROOT}"
export DUALSPS_LOG_DIR="${DUALSPS_LOG_DIR:-$DUALSPS_DATA_ROOT/logs}"
export VENV="${VENV:-$(pwd)/.venv}"
export PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export WANDB_MODE="${WANDB_MODE:-offline}"
# bound torchelastic's exit barrier so a finished run does not sit idle holding the GPUs
export TORCHELASTIC_EXIT_BARRIER_TIMEOUT="${TORCHELASTIC_EXIT_BARRIER_TIMEOUT:-60}"
