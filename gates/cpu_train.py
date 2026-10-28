"""Run scripts/train.py unchanged on a CPU-only host, for gate G4.

One shim, outside the code under test: clip_grad_norm_ prints each step's exact pre-clip
gradient norm as `G4 grad_norm <hex>`. Arguments are passed through to train.py as Hydra
overrides.
"""
import runpy
import sys
from pathlib import Path

import torch

_clip = torch.nn.utils.clip_grad_norm_


def _clip_and_print(*args, **kwargs):
    norm = _clip(*args, **kwargs)
    print(f"G4 grad_norm {float(norm).hex()}", flush=True)
    return norm


torch.nn.utils.clip_grad_norm_ = _clip_and_print

train = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
sys.argv = [str(train), *sys.argv[1:]]
runpy.run_path(str(train), run_name="__main__")
