"""The one place that says where things live.

Repo-relative paths come from this file's location. Machine-specific locations come from
environment variables, and there are no machine-specific defaults: a script that needs one
of them fails with a message naming the variable. The variables can also be set in
<repo>/.env (KEY=value lines, see .env.example), which this module loads at import with the
same rule as scripts/run/env.sh: a variable already set in the environment wins over .env.

    DUALSPS_DATA_ROOT   required: base of the tokenized corpus,
                        <DATA_ROOT>/data/<dataset>/{train,val}.bin
    DUALSPS_OUT_ROOT    base of run outputs: <OUT_ROOT>/out/<run>/*.pt (same meaning as Hydra's
                        system.out_root) and the live result files <OUT_ROOT>/results/*.jsonl;
                        default DATA_ROOT
    DUALSPS_LOG_DIR     logs of the training runs, read by scripts/dualsps/eval_runs.py
                        for throughput and wall-clock; default <DATA_ROOT>/logs
    DUALSPS_HF_REPO     Hugging Face repo holding the checkpoints (scripts/dualsps/hf_export.py);
                        default ryankim17920/mechanistic-sps, the paper's public checkpoints
    DUALSPS_RESULTS     paper inputs (ledger.jsonl, wallclock.jsonl, analysis JSONs);
                        default <repo>/scripts/analysis/results, the snapshot in git

The paper is built from the snapshot (LEDGER, WALLCLOCK). Evaluation and benchmark jobs
append to the live files (live_ledger(), live_wallclock()); `make snapshot` copies those
into the snapshot, an explicit step.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "conf"
DOTENV = REPO / ".env"


def _load_dotenv(path: Path = DOTENV) -> None:
    """KEY=value lines of `path` into os.environ, never overriding a variable already set.
    The same parsing as scripts/run/env.sh: no quotes, no spaces around '=', other lines ignored."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) and not os.environ.get(key):
            os.environ[key] = value


_load_dotenv()


def _required(var: str, what: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set. It is {what}. Export it or set it in {DOTENV} "
                           f"(copy .env.example).")
    return value


def data_root() -> Path:
    return Path(_required("DUALSPS_DATA_ROOT", "the base directory of the tokenized corpus "
                                               "(<root>/data/<dataset>/{train,val}.bin)"))


def out_root() -> Path:
    return Path(os.environ.get("DUALSPS_OUT_ROOT") or data_root())


def run_dir(run: str) -> Path:
    """<OUT_ROOT>/out/<run>, where scripts/train.py writes that run's checkpoints."""
    return out_root() / "out" / run


def log_dir() -> Path:
    return Path(os.environ.get("DUALSPS_LOG_DIR") or data_root() / "logs")


HF_REPO_DEFAULT = "ryankim17920/mechanistic-sps"   # public: the paper's final checkpoints
# public: the checkpoint ladders of a17_trajectory.py part B and the embedding ladders of
# a21_emb_divergence.py (scripts/dualsps/hf_export.py download --extra)
HF_EXTRA_REPO = "ryankim17920/mechanistic-sps-extra"


def hf_repo() -> str:
    """The Hugging Face repo holding the checkpoints: $DUALSPS_HF_REPO, else the public one."""
    return os.environ.get("DUALSPS_HF_REPO") or HF_REPO_DEFAULT


def live_results() -> Path:
    """Where evaluation and benchmark jobs append their rows (not the snapshot in git)."""
    return out_root() / "results"


def live_ledger() -> Path:
    return live_results() / "ledger.jsonl"


def live_wallclock() -> Path:
    return live_results() / "wallclock.jsonl"


RESULTS = Path(os.environ.get("DUALSPS_RESULTS") or REPO / "scripts" / "analysis" / "results")
LEDGER = RESULTS / "ledger.jsonl"        # snapshot of live_ledger(), read by the paper build
WALLCLOCK = RESULTS / "wallclock.jsonl"  # snapshot of live_wallclock(), read by the paper build
