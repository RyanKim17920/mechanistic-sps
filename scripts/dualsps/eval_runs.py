"""The paper's metric of record: val_nll_full_sweep, appended to the ledger.

The in-training `val nll` line samples the val set randomly with replacement and is a
progress indicator only. This script scores a run's final checkpoint (and every
ckpt_tokens_* checkpoint, for the loss-vs-compute curve) with a full, deterministic,
non-overlapping sweep over the entire val set of the run's own dataset, identical for
every run. It also records params, FLOPs/token, throughput and wall-clock (from the
training job's log, see repo_paths.log_dir()) and the architecture.

Each evaluated run appends one provenanced record (git commit and dirty flag, resolved
model config, checkpoint sha256, host, Triton-cache state) to the live ledger,
<DUALSPS_OUT_ROOT>/results/ledger.jsonl (repo_paths.live_ledger()), or to --ledger. The
paper reads the snapshot in scripts/analysis/results/, which only `make snapshot` updates.
The exit status is nonzero when any requested run could not be evaluated.

Usage (GPU, through scripts/run/eval_shim.sh):
        python scripts/dualsps/eval_runs.py <run> [<run> ...] [--ledger PATH]
        python scripts/dualsps/ledger.py --list | --show N     # read the ledger
"""
import argparse
import math
import os
import re
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from _paths import run_dir, run_glob   # first: puts src/ on sys.path
import ledger
import repo_paths
from modeling.models.model import config_args

CONF = str(repo_paths.CONF)
BLOCK, MB = 4096, 6     # val sequence length and eval micro-batch of every ledger number


def val_path_of(cfg):
    """The val.bin for THIS run's own dataset, where src/data/prepare.py writes it:
    <data_root>/data/<dataset>/val.bin.

    Every run is scored against its own dataset's val set, never a hardcoded one --
    runs at the 20B/fineweb-edu-100bt budget must not silently get scored against the
    old 2B/fineweb-edu val.bin (a different, smaller val set). Raises loudly if the
    resolved path does not exist rather than falling back to some other val.bin.
    """
    data_root = cfg.system.data_root
    dataset = cfg.data.dataset
    path = os.path.join(data_root, "data", dataset, "val.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"val.bin not found for dataset '{dataset}' at {path} -- refusing to fall "
            f"back to a different dataset's val set")
    return path


def assert_config_matches_checkpoint(run, model, ck):
    """Fail loudly when the on-disk experiment YAML no longer describes this checkpoint.

    `load_model` rebuilds the model from conf/experiment/<run>.yaml and then loads the
    checkpoint weights with `strict=False`. Nothing in that path checked that the YAML still
    says what it said at training time, and the dangerous fields are exactly the ones that
    change NO tensor shape -- `cross_attn_mode`, `window_size`, the split/per-stream MLP
    widths when they happen to keep the same shapes -- so an edited config loads cleanly and
    silently scores a DIFFERENT architecture than the one that was trained, then writes that
    wrong geometry into the ledger. Training's resume path already treats
    `checkpoint['model_args']` as authoritative (scripts/train.py:164-170); eval had no
    equivalent guard.

    Compares the recomposed live config -- `asdict(model.config)`, byte-for-byte the thing
    train.py stores as `model_args` -- against the checkpoint's own copy, preferring
    `model_args` and falling back to the full training config at `config.model.config`.
    Any disagreement RAISES. Fields a legacy checkpoint predates are warned about instead,
    and nothing is ever coerced: the config is an INPUT to the run, so a disagreement is a
    question for the operator, not something to paper over by silently adopting one side.
    """
    live = asdict(model.config)
    ck_args, src, partial = ck.get("model_args"), "checkpoint['model_args']", False
    if not ck_args:
        # Older checkpoints predate `model_args`; the full training config carries the same
        # values, but only the keys the YAML spells out, so absences there are expected
        # rather than reportable.
        ck_args = ((ck.get("config") or {}).get("model") or {}).get("config")
        src, partial = "checkpoint['config']['model']['config']", True
    if not ck_args:
        print(f"    WARNING: {run}: checkpoint stores neither model_args nor config.model."
              f"config -- config/checkpoint agreement is UNVERIFIED for this run", flush=True)
        return
    ck_args = config_args(type(model.config), dict(ck_args))   # drops removed fields

    # The two-tower configs pin attn_backend to `flex`, which is what `auto` resolved to
    # when they trained (gates/ALLOWED_CONFIG_DIFFS.md); their checkpoints record `auto`.
    if ck_args.get("attn_backend") == "auto" and live.get("attn_backend") == "flex":
        ck_args["attn_backend"] = "flex"
    mismatched = [(k, ck_args[k], live[k]) for k in sorted(live) if k in ck_args
                  and live[k] != ck_args[k]]
    if mismatched:
        lines = "\n".join(f"      {k}: checkpoint={c!r}  conf/experiment/{run}.yaml={y!r}"
                          for k, c, y in mismatched)
        raise ValueError(
            f"{run}: the experiment config no longer matches the trained checkpoint "
            f"({src}):\n{lines}\n"
            f"    These fields need not change any tensor shape, so the weights would still "
            f"load and the run would be scored -- and its geometry recorded -- as an "
            f"architecture that was never trained. Refusing to evaluate. Restore the config "
            f"to what trained this checkpoint, or evaluate a checkpoint of the current "
            f"config; the mismatch is NOT resolved by adopting either side automatically.")

    unknown = sorted(set(ck_args) - set(live))
    if unknown:
        print(f"    WARNING: {run}: checkpoint records model fields the current config class "
              f"does not define: {', '.join(unknown)} -- these are unverified", flush=True)
    if not partial:
        absent = sorted(set(live) - set(ck_args))
        if absent:
            print(f"    WARNING: {run}: checkpoint predates model fields {', '.join(absent)} "
                  f"-- their current values are unverified against training", flush=True)


def load_model(run):
    with initialize_config_dir(config_dir=CONF, version_base=None):
        cfg = compose("config", overrides=[f"+experiment={run}",
                                           f"system.data_root={repo_paths.data_root()}"])
    m = instantiate(cfg.model).cuda().eval()
    cands = run_glob(run, "*final*.pt") or [os.path.join(run_dir(run), "ckpt.pt")]
    all_ckpts = run_glob(run, "ckpt_tokens_*.pt")
    ck = torch.load(cands[0], map_location="cpu", weights_only=False)
    assert_config_matches_checkpoint(run, m, ck)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    sd.pop("freqs_cis", None)
    m.load_state_dict(sd, strict=False)
    return m, cfg, ck, cands[0], all_ckpts


def full_sweep_nll(model, val_path):
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    starts = list(range(0, len(data) - BLOCK - 1, BLOCK))
    s = c = 0.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(starts), MB):
            b = starts[i:i + MB]
            X = torch.stack([torch.from_numpy(data[j:j + BLOCK].astype(np.int64)) for j in b]).cuda()
            Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + BLOCK].astype(np.int64)) for j in b]).cuda()
            _, _, st = model(X, Y)
            s += float(st["token_nll_sum"]); c += float(st["token_nll_count"])
    return s / c, int(c), len(starts)


def arch_of(cfg):
    mc = cfg.model.config
    tgt = str(cfg.model._target_)
    if "two_tower" in tgt:
        # The two towers have independent depth/width/FFN width and no longer share a key
        # set, so there is no single (depth, d_model) that describes the run; the geometry
        # travels as a dict straight into `plotting.flops.two_tower_flops_per_token`.
        d = int(mc.hidden_size)
        geom = {
            "state_n_layer": int(mc.state_n_layer),
            "pred_n_layer": int(mc.pred_n_layer),
            "state_hidden": int(mc.get("state_hidden", None) or d),
            "pred_hidden": int(mc.get("pred_hidden", None) or d),
            "state_intermediate": int(mc.intermediate_size if mc.get("state_intermediate", None) is None
                                      else mc.state_intermediate),
            "pred_intermediate": int(mc.intermediate_size if mc.get("pred_intermediate", None) is None
                                     else mc.pred_intermediate),
            "read_map": str(mc.get("read_map", "post")),
            "head_dim": d // int(mc.n_head),
            "block_size": int(mc.block_size),
        }
        rs = geom["read_source"] = str(mc.read_source)
        if geom["read_map"] == "explicit":
            geom["read_levels"] = [int(v) for v in mc.read_levels]
        # `depth_per_slot` stays meaningful as the depth a token traverses in each tower.
        return "two_tower", max(geom["state_n_layer"], geom["pred_n_layer"]), {
            "state_n_layer": geom["state_n_layer"], "pred_n_layer": geom["pred_n_layer"],
            "read_map": geom["read_map"], "read_source": rs,
            "state_intermediate": geom["state_intermediate"],
            "pred_intermediate": geom["pred_intermediate"],
            "two_tower": geom}
    fam = "sps" if "sps" in tgt else "standard"
    return fam, int(mc.n_layer), {"n_layer": int(mc.n_layer)}


def log_stats(run, cfg):
    """throughput / wall-clock from the training log."""
    out = {}
    # Several logs can carry the same exp= tag (e.g. launches that died during
    # torch.compile). Pick the one with the most training iterations, not the first.
    cands = []
    tokens_per_iter = int(cfg.training.global_batch_size) * int(cfg.model.config.block_size)
    for f in repo_paths.log_dir().glob("*.out"):
        txt = open(f, errors="ignore").read()
        if f"exp={run} " in txt:
            cands.append((txt.count("\ntrain | iter"), txt))
    if not cands:
        return out
    for _n, txt in [max(cands, key=lambda z: z[0])]:
        ms = [float(x) for x in re.findall(r"\| (\d+)ms \| tokens:", txt)]
        its = re.findall(r"^train \| iter\s+(\d+) \|.*?tokens: ([\d,]+)", txt, re.M)
        if ms:
            med = sorted(ms)[len(ms) // 2]
            out["ms_per_iter_median"] = med
            out["tokens_per_iter"] = tokens_per_iter
            out["throughput_tok_s"] = tokens_per_iter / (med / 1000.0)
        if its:
            out["final_iter"] = int(its[-1][0])
            out["tokens_processed"] = int(its[-1][1].replace(",", ""))
            if ms:
                out["train_wall_clock_hours"] = out["final_iter"] * med / 1000.0 / 3600.0
                out["total_ms_sum_hours"] = sum(ms) / 1000.0 / 3600.0
        tr = [float(x) for x in re.findall(r"^train \| iter .*? nll: ([\d.]+)", txt, re.M)]
        if tr:
            out["train_nll_last100_mean"] = sum(tr[-100:]) / len(tr[-100:])
    return out


def print_summary(rows):
    print(f"\n{'run':42s} {'params':>12s} {'FLOPs/tok':>11s} {'val NLL':>9s} {'tok/s':>10s}")
    for r in sorted(rows, key=lambda x: x["val_nll_full_sweep"]):
        print(f"{r['run']:42s} {r['params']:>12,} {r['fwd_flops_per_token']/1e6:>10.1f}M "
              f"{r['val_nll_full_sweep']:>9.4f} {r.get('throughput_tok_s', 0):>10,.0f}")


def main(runs, ledger_path):
    from plotting.flops import forward_flops_per_token
    rows, skipped = [], []
    for run in runs:
        print(f"--- {run}", flush=True)
        try:
            m, cfg, ck, ckpath, all_ckpts = load_model(run)
        except Exception as e:
            print(f"    SKIP ({type(e).__name__}: {e})")
            skipped.append(run)
            continue
        fam, depth, extra = arch_of(cfg)
        d = int(cfg.model.config.hidden_size)
        intermediate_size = cfg.model.config.get("intermediate_size", None)
        val_path = val_path_of(cfg)
        t0 = time.time()
        nll, ntok, nseq = full_sweep_nll(m, val_path)
        fwd = float(forward_flops_per_token(fam, depth, d, intermediate_size=intermediate_size,
                                            two_tower=extra.get("two_tower")))
        toks = ck.get("tokens_seen") or ck.get("tokens") or 0
        row = dict(run=run, checkpoint=os.path.basename(ckpath), family=fam, depth_per_slot=depth,
                   hidden_size=d, **extra,
                   params=int(sum(p.numel() for p in m.parameters())),
                   val_nll_full_sweep=round(nll, 6), val_ppl=round(math.exp(nll), 4),
                   val_tokens=ntok, val_seqs=nseq,
                   fwd_flops_per_token=fwd,
                   train_flops_per_token=3.0 * fwd,
                   tokens_seen_ckpt=toks, eval_seconds=round(time.time() - t0, 1),
                   **log_stats(run, cfg))
        n_tok = row.get("tokens_processed") or toks
        row["total_train_flops"] = 3.0 * fwd * n_tok if n_tok else None

        # loss-vs-compute curve over every saved checkpoint
        curve = []
        for cp in all_ckpts:
            mt = re.search(r"ckpt_tokens_(\d+)", cp)
            if not mt:
                continue
            tk = int(mt.group(1))
            c2 = torch.load(cp, map_location="cpu", weights_only=False)
            sd2 = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in c2["model"].items()}
            sd2.pop("freqs_cis", None)
            m.load_state_dict(sd2, strict=False)
            n2, _, _ = full_sweep_nll(m, val_path)
            curve.append({"tokens": tk, "val_nll": round(n2, 6),
                          "train_flops": 3.0 * fwd * tk})
            print(f"      curve: {tk/1e9:.2f}B tokens -> val_nll {n2:.4f}", flush=True)
            del c2, sd2
        row["curve"] = curve
        row["curve_checkpoints"] = [os.path.basename(c) for c in all_ckpts]

        # Provenance. Recorded per-run (not once per invocation) because a single
        # invocation can span a rebase-free but dirty tree edit, and because each run
        # has its own checkpoint bytes to pin.
        print("      hashing checkpoint...", flush=True)
        record = {"schema": ledger.SCHEMA, "timestamp": ledger.utc_now(),
                  "backfilled": False, "run": run}
        record.update(ledger.git_provenance())
        record["checkpoint_path"] = ckpath
        record["checkpoint_sha256"] = ledger.sha256_file(ckpath)
        record["model_config"] = OmegaConf.to_container(cfg.model, resolve=True)
        record["dataset"] = cfg.data.dataset
        record["token_budget"] = int(cfg.training.max_tokens)
        record["seed"] = int(cfg.training.seed)
        record["chunk_shuffle_seed"] = int(cfg.training.chunk_shuffle_seed)
        record.update(ledger.env_provenance())
        record.update(row)
        ledger.append(record, ledger_path)
        rows.append(record)
        print(f"    val_nll={nll:.6f} ppl={math.exp(nll):.3f} params={row['params']:,}", flush=True)
        if not record["triton_cache_node_local"]:
            print("    WARNING: TRITON_CACHE_DIR is not a node-local /tmp path -- a shared "
                  "cache returns wrong kernels SILENTLY. Re-run via scripts/run/eval_shim.sh.",
                  flush=True)
        del m; torch.cuda.empty_cache()

    if rows:
        print_summary(rows)
    print(f"\nappended {len(rows)} record(s) to {ledger_path}")
    if skipped:
        print(f"FAILED to evaluate {len(skipped)} run(s): {' '.join(skipped)}", file=sys.stderr)
    return 1 if skipped else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="experiment names (conf/experiment/<run>.yaml)")
    ap.add_argument("--ledger", type=str, default=None,
                    help="ledger to append to (default: repo_paths.live_ledger())")
    a = ap.parse_args()
    sys.exit(main(a.runs, a.ledger or repo_paths.live_ledger()))
