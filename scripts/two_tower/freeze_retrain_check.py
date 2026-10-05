"""Unit check for the freeze-and-retrain arms (paper §6.3). Needs 1 GPU.

For each frozen state source (Sequential 6+6 seed-1, Two-tower 6+6 seed-1) and each read
map (post / final / early-final explicit), builds the model from the REAL experiment yaml,
applies ``freeze_state_tower`` from the source checkpoint, and asserts:

  1. the frozen state tower's per-level outputs (every residual level the pred tower can
     read) are bit-identical to the source model's on a real val batch (bf16 autocast, as
     in training);
  2. the read levels are the intended ones -- final == the Sequential original's [6]*6,
     post == the Two-tower original's [1..6], early == [6,6,6,4,5,6];
  3. end-to-end read semantics: copying the source's PRED tower into the matching arm
     (seq source + final, tt source + post) reproduces the source model's logits exactly,
     so state_kv/pred_proj read wiring is identical to the originals;
  4. frozen params: requires_grad False, no .grad after backward, excluded from the
     optimizer; every pred-tower param is trainable.

The source checkpoints are the final checkpoints of the configs' training.freeze_state_from
runs under <DUALSPS_OUT_ROOT>/out/; the val batch comes from <DUALSPS_DATA_ROOT>/data/
<cfg.data.dataset>/val.bin.

Usage (one GPU): scripts/run/eval_shim.sh scripts/two_tower/freeze_retrain_check.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import repo_paths
from modeling.models.model import config_args
from modeling.models.two_tower.core import TwoTowerConfig, TwoTowerModel
from training import final_checkpoint_path

# frozen source (the prefix of the frz config names) -> the read map of the source run
SOURCES = {"seq6src": "final", "tt6src": "post"}
EXPECT = {"post": [1, 2, 3, 4, 5, 6], "final": [6] * 6, "early": [6, 6, 6, 4, 5, 6]}


def compose_run(run):
    with initialize_config_dir(config_dir=str(repo_paths.CONF), version_base=None):
        return compose("config", overrides=[f"+experiment={run}",
                                            f"system.data_root={repo_paths.data_root()}",
                                            f"system.out_root={repo_paths.out_root()}"])


def strip(sd):
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    sd.pop("freqs_cis", None)
    return sd


def batch(val_path, n=2, T=4096, off=12345):
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    X = torch.stack([torch.from_numpy(data[off + i * T: off + (i + 1) * T].astype(np.int64))
                     for i in range(n)]).cuda()
    Y = torch.stack([torch.from_numpy(data[off + i * T + 1: off + (i + 1) * T + 1].astype(np.int64))
                     for i in range(n)]).cuda()
    return X, Y


def main():
    torch.manual_seed(0)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    for sk, src_map in SOURCES.items():
        first = compose_run(f"s_two_tower_frz_{sk}_{next(iter(EXPECT))}_20b")
        src_run = first.training.freeze_state_from
        X, Y = batch(Path(first.system.data_root) / "data" / first.data.dataset / "val.bin")
        ck = torch.load(final_checkpoint_path(repo_paths.run_dir(src_run)), map_location="cpu",
                        weights_only=False, mmap=True)
        src = TwoTowerModel(TwoTowerConfig(**config_args(TwoTowerConfig, ck["model_args"])))
        src = src.cuda().eval()
        src_sd = strip(ck["model"])
        r = src.load_state_dict(src_sd, strict=False)
        assert not r.missing_keys, r.missing_keys
        assert src.config.read_map == src_map
        with ac:
            _, src_res = src.state_levels(X)
            src_logits = src(X)
        for mk, lv in EXPECT.items():
            run = f"s_two_tower_frz_{sk}_{mk}_20b"
            cfg = compose_run(run)
            assert cfg.training.freeze_state_from == src_run, (run, cfg.training.freeze_state_from)
            m = instantiate(cfg.model).cuda()
            print(m.freeze_state_tower(src_sd))
            # (2) read levels
            assert m.read_levels == lv, (run, m.read_levels)
            if mk == "final":
                assert m.read_levels == [6] * 6
                if sk == "seq6src":
                    assert m.read_levels == src.read_levels
            if mk == "post" and sk == "tt6src":
                assert m.read_levels == src.read_levels
            # (1) frozen state outputs bit-identical, every level
            m.eval()
            with ac:
                _, res = m.state_levels(X)
            assert len(res) == len(src_res) == 7
            for j, (a, b) in enumerate(zip(res, src_res)):
                assert torch.equal(a, b), f"{run}: state level {j} differs"
            # the levels this arm actually reads, vs what the Sequential original reads
            if mk == "final" and sk == "seq6src":
                for i, l in enumerate(m.read_levels):
                    assert torch.equal(res[l], src_res[src.read_levels[i]])
            # (4) freezing: grads + optimizer membership
            m.train()
            frozen = set(m.state_tower_param_names())
            assert frozen and all(n.startswith(("transformer.wte.", "transformer.state_h."))
                                  for n in frozen)
            for n, p in m.named_parameters():
                assert p.requires_grad == (n not in frozen), n
            opt = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda")
            opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
            named = dict(m.named_parameters())
            assert not any(id(named[n]) in opt_ids for n in frozen)
            assert all(id(p) in opt_ids for n, p in named.items() if n not in frozen)
            with ac:
                _, loss, _ = m(X[:1], Y[:1])
            loss.backward()
            assert all(named[n].grad is None for n in frozen)
            assert all(p.grad is not None for n, p in named.items() if n not in frozen)
            before = {n: named[n].detach().clone() for n in frozen}
            opt.step()
            assert all(torch.equal(before[n], named[n]) for n in frozen)
            # (3) end-to-end equivalence for the arm whose read map matches its source
            if mk == src_map:
                m.eval()
                with torch.no_grad():
                    for n, p in m.named_parameters():
                        if n not in frozen:
                            p.copy_(src_sd[n])
                with ac:
                    lg = m(X)
                assert torch.equal(lg, src_logits), f"{run}: logits differ from source"
                print(f"  {run}: source pred tower in -> logits bit-identical to {src_run}")
            n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in m.parameters())
            print(f"PASS {run}: read_levels={m.read_levels} frozen_params={n_all - n_tr:,} "
                  f"trainable={n_tr:,} total={n_all:,} (source total "
                  f"{sum(p.numel() for p in src.parameters()):,})", flush=True)
            del m, opt
            torch.cuda.empty_cache()
        del src, ck, src_sd
    print("ALL FREEZE-RETRAIN CHECKS PASSED")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
