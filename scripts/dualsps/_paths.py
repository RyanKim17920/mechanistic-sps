"""Run-output lookup for the scripts that run outside Hydra, on top of src/repo_paths.py.

    <DUALSPS_OUT_ROOT>/out/<run>/*.pt     checkpoints (same layout as train.py's system.out_root)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import repo_paths  # noqa: E402


def run_dir(run: str) -> str:
    return str(repo_paths.run_dir(run))


def run_glob(run: str, pattern: str) -> list[str]:
    """Sorted glob of `pattern` inside the run's output directory."""
    return sorted(str(p) for p in repo_paths.run_dir(run).glob(pattern))
