"""Where W&B keeps its local run files."""

from __future__ import annotations

import os
from pathlib import Path


def prepare_wandb_dir_from_config(cfg) -> str | None:
    """The local W&B directory: ``logging.wandb_dir`` (default ``<out_root>/wandb``), else
    ``$WANDB_DIR``. The directory is created, and ``WANDB_DIR`` is set to it unless it is
    already set. Returns None when neither is given."""
    raw = cfg.logging.get("wandb_dir") or os.environ.get("WANDB_DIR")
    if not raw or not str(raw).strip():
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(raw).strip())))
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(path))
    return str(path)
