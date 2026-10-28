"""Smoke test: run the real scripts/train.py for a few optimizer steps per model family.

    python scripts/smoke_train.py          # CPU, reduced width and context (a few minutes)
    python scripts/smoke_train.py --gpu    # one GPU, the paper's model size, bf16, compiled

It only shows that training does not break: each family trains a few steps through Hydra on
a synthetic token file written to a temporary directory, runs its final evaluation and
writes its final checkpoint. Nothing is compared against the paper's numbers (that is what
`make check` does). The freeze-and-retrain arm loads the frozen state tower from the final
checkpoint of the Sequential 6+6 smoke run, exactly as it loads the trained source run.

On CPU the SPS arm replaces its Triton attention kernel (CUDA-only) with an equivalent
masked softmax in PyTorch; everything else is the code that trains the paper's models.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

# family -> experiment config. Sequential 6+6 runs before `frz`, whose frozen state tower
# is that run's final checkpoint.
FAMILIES = {
    "Transformer": "s_full_attention_20b_fw100",
    "SPS": "s_sps_w64_20b_fw100",
    "Two-tower": "s_two_tower_w0_equal_20b",
    "Sequential": "s_two_tower_seq12_20b",
    "Sequential 6+6": "s_two_tower_seq6_20b",
    "Shared": "s_two_tower_w0_shared_20b",
    "AF-SPS": "s_two_tower_afsps_faithful_20b",
    "12+6": "s_two_tower_asym_20b",
    "frz": "s_two_tower_frz_seq6src_early_20b",
}

# CPU: every model at reduced width (the depths, read maps and sharing stay the paper's).
CPU_WIDTH = {"hidden_size": 128, "n_head": 2, "intermediate_size": 384}
CPU_TWO_TOWER_WIDTH = {"state_intermediate": 384, "pred_intermediate": 384}
FINAL_LINE = re.compile(r"^final \| iter\s+\d+ \| val nll: (\S+)", re.M)


def write_corpus(root: Path, n_train: int, n_val: int) -> None:
    """Random GPT-2 token ids at <root>/data/<dataset>/{train,val}.bin, as prepare.py writes."""
    dataset = yaml.safe_load((REPO / "conf" / "data" / "fineweb-edu-100bt.yaml").read_text())["dataset"]
    d = root / "data" / dataset
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        rng.integers(0, 50257, n, dtype=np.uint16).tofile(d / f"{split}.bin")


def overrides(name: str, root: Path, gpu: bool, steps: int) -> list[str]:
    block = 4096 if gpu else 128
    micro, batch = 2, 4
    tokens = steps * batch * block
    ov = [f"+experiment={name}", f"system.data_root={root}", "logging.wandb_log=false",
          "training.init_from=scratch", "training.world_size=null",
          f"model.config.block_size={block}", f"training.micro_batch_size={micro}",
          f"training.global_batch_size={batch}", f"training.max_tokens={tokens}",
          f"scheduler.warmup_tokens={tokens // 2}", f"scheduler.lr_decay_tokens={tokens // 2}",
          f"training.eval_total_tokens={batch * block}", f"training.eval_interval_tokens={tokens}",
          f"training.save_every={10 * tokens}", f"training.rolling_save_every={10 * tokens}"]
    if not gpu:
        ov += ["system.device=cpu", "system.dtype=float32", "system.compile=false",
               "system.backend=gloo"]
        ov += [f"model.config.{k}={v}" for k, v in CPU_WIDTH.items()]
        if "two_tower" in name:
            ov += [f"model.config.{k}={v}" for k, v in CPU_TWO_TOWER_WIDTH.items()]
            ov.append("++model.config.flex_compile=false")   # compiled flex has no CPU lowering
    return ov


def run_family(label: str, name: str, root: Path, gpu: bool, steps: int, threads: int) -> str:
    """Train one config; return a one-line report. Raises with the log tail on failure."""
    t0 = time.time()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", *overrides(name, root, gpu, steps)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]),
               WANDB_MODE="disabled")
    if not gpu:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads))
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(k, None)
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    log = p.stdout + p.stderr
    (root / f"{name}.log").write_text(log)
    m = FINAL_LINE.search(p.stdout)
    finals = list((root / "out" / name).glob("ckpt_tokens_*_final.pt"))
    if p.returncode != 0 or m is None or len(finals) != 1 or not math.isfinite(float(m.group(1))):
        raise RuntimeError(f"{label} ({name}) failed, exit code {p.returncode}:\n"
                           + "\n".join(log.splitlines()[-40:]))
    return (f"{label:15s} {name:36s} final val nll {float(m.group(1)):.3f}  "
            f"({time.time() - t0:.0f}s)")


def child(hydra_args: list[str]) -> None:
    """Run scripts/train.py in this process (on CPU, with the SPS kernel replaced)."""
    import runpy

    import torch

    if not torch.cuda.is_available():
        import modeling.models.sps.core as sps_core
        sps_core.triton_sps_sliding_attention = sps_attention_reference
    train = REPO / "scripts" / "train.py"
    sys.argv = [str(train), *hydra_args]
    runpy.run_path(str(train), run_name="__main__")


def sps_attention_reference(q, k, v, scale, window, warp_specialize=False,
                            documents_idx_BxT=None):
    """The SPS kernel's attention as a masked softmax over the interleaved 2T slots: slot
    s sees state (even) slot 2k when 2k <= s, and <predict> (odd) slot 2k+1 when
    2k+1 <= s and the tokens are at most `window` apart; both within one document."""
    import torch
    two_t = q.shape[2]
    qs = torch.arange(two_t, device=q.device).view(-1, 1)
    ks = torch.arange(two_t, device=q.device).view(1, -1)
    vis = (ks <= qs) & ((ks % 2 == 0) | (qs // 2 - ks // 2 <= window))
    vis = vis.view(1, 1, two_t, two_t)
    if documents_idx_BxT is not None:
        d = documents_idx_BxT
        vis = vis & (d.unsqueeze(-1) == d.unsqueeze(-2)).unsqueeze(1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    probs = torch.softmax(scores.masked_fill(~vis, float("-inf")), dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v.float()).to(q.dtype)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="one GPU at the paper's model size")
    ap.add_argument("--steps", type=int, default=3, help="optimizer steps per family (default 3)")
    ap.add_argument("--only", nargs="+", choices=list(FAMILIES), metavar="FAMILY",
                    help=f"a subset of: {', '.join(FAMILIES)}")
    ap.add_argument("--jobs", type=int, default=4, help="CPU: families trained in parallel")
    ap.add_argument("--keep", metavar="DIR",
                    help="write data, checkpoints and logs to DIR (must not exist) and keep them")
    args = ap.parse_args()
    families = {k: v for k, v in FAMILIES.items() if not args.only or k in args.only}
    if "frz" in families:
        families["Sequential 6+6"] = FAMILIES["Sequential 6+6"]
    jobs = 1 if args.gpu else max(1, args.jobs)
    threads = max(1, min(8, (os.cpu_count() or 1) // jobs))

    with tempfile.TemporaryDirectory(prefix="smoke_") as td:
        root = Path(args.keep) if args.keep else Path(td)
        root.mkdir(parents=True, exist_ok=not args.keep)
        write_corpus(root, n_train=1 << 20, n_val=1 << 18)
        print(f"smoke: {len(families)} families, {args.steps} steps each, "
              f"{'1 GPU' if args.gpu else 'CPU'}, under {root}", flush=True)
        failed = []

        def run(label):
            try:
                print(f"  ok    {run_family(label, families[label], root, args.gpu, args.steps, threads)}",
                      flush=True)
            except RuntimeError as e:
                print(f"  FAIL  {e}", flush=True)
                failed.append(label)

        # frz reads the final checkpoint of the Sequential 6+6 run, so it goes last.
        with ThreadPoolExecutor(jobs) as ex:
            list(ex.map(run, [k for k in families if k != "frz"]))
        if "frz" in families:
            run("frz")
        if failed:
            sys.exit(f"smoke: FAILED: {', '.join(failed)}")
        print("smoke: all families trained", flush=True)


if __name__ == "__main__":
    main()
