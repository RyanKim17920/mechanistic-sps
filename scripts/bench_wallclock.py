#!/usr/bin/env python3
"""Wall-clock benchmark for every architecture arm of the paper, on identical settings.

Question this answers: do the FLOPs/token differences (flops.py) turn into real speed?
Every arm is built from the EXACT Hydra experiment config its 20B run used (random init --
timing does not need trained weights) and measured on ONE H100 with the same settings:

  1. TRAINING: fwd + bwd + grad-clip + fused AdamW step, bf16 autocast, the model wrapped in
     `torch.compile(model)` (default mode) -- exactly the `system.compile=true` path every
     20B run took in scripts/train.py. Fixed micro-batch at T=4096 on real val tokens of the
     config's own dataset (so the document masks the kernels see are the ones training
     saw). tokens/s = mb*T / median step time.
  2. PREFILL: the same compiled model, eval + no_grad, the full forward train.py's
     estimate_loss runs (LM head over all T positions + CE; the standard model's forward
     REQUIRES targets, so every arm is given them for parity), same batch.
  3. KV/state cache bytes per token: analytic from the config, cross-checked for two-tower
     against an instrumented forward that counts the DISTINCT full-length k/v tensors the
     attention calls consume.
  4. FLOPs/token from src/plotting/flops.py (not the ledger, which is wrong for asym/afsps).

Decode latency is not measured by this release. The paper's snapshot rows
(scripts/analysis/results/wallclock.jsonl) also carry decode timings for the Transformer and
SPS, measured by an earlier version of this code with KV-cached decode paths, which this
release does not ship; the paper's tables read only the train/prefill numbers and the decode `status`.
A row records `decode.status` = `no_decode_path` for the two-tower family (it never had an
incremental path) and `not_measured` otherwise, so tab:efficiency's footnote is unchanged.

Each arm runs in its OWN subprocess (fresh dynamo state, clean peak-memory accounting).
The sweep driver runs every arm `--reps` times in interleaved order (rep 0 forward, rep 1
reversed, ...) so that contention from other jobs on the machine shows up as rep-to-rep spread.

Usage (one GPU; scripts/run/eval_shim.sh scripts/bench_wallclock.py ...). The paper's
table reads the rows tagged final_mb12 of the snapshot repo_paths.WALLCLOCK; runs append to
the live file repo_paths.live_wallclock() (the default --out), and `make snapshot` copies it:
    python scripts/bench_wallclock.py --sweep --reps 2 --mb 12 --tag final_mb12
    python scripts/bench_wallclock.py --arm tt_seq6_slim --rep 0 --out x.jsonl   # one arm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import repo_paths  # noqa: E402

# arm label -> experiment config used for its 20B run, and extra Hydra overrides per arm.
# The kernel-control arms (*_flex, not trained arms) run the standard model on its own
# flex_attention path instead of the repo's Triton full-attention kernel, i.e. the attention
# implementation family the two-tower arms run on.  Both lists are paper_manifest.yaml's.
_WC = yaml.safe_load((_REPO / "scripts" / "analysis" / "paper_manifest.yaml").read_text())["wallclock"]
ARMS = dict(_WC["arms"])
ARM_OVERRIDES = dict(_WC["overrides"])

BF16_BYTES = 2


# --------------------------------------------------------------------------------------
# model construction + static accounting
# --------------------------------------------------------------------------------------
def build(exp: str, overrides=()):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(config_dir=str(repo_paths.CONF), version_base=None):
        cfg = compose("config", overrides=[f"+experiment={exp}", *overrides])
    model = instantiate(cfg.model)
    return model, cfg


def family(cfg) -> str:
    tgt = str(cfg.model._target_)
    if "two_tower" in tgt:
        return "two_tower"
    if ".sps" in tgt:
        return "sps"
    if "full_attention" in tgt:
        return "standard"
    raise ValueError(tgt)


def flops_per_token(model, cfg) -> float:
    import plotting.flops as F

    mc = cfg.model.config
    fam = family(cfg)
    d = int(mc.hidden_size)
    if fam == "standard":
        return F.forward_flops_per_token("standard", int(mc.n_layer), d, int(mc.intermediate_size))
    if fam == "sps":
        return F.forward_flops_per_token("sps", int(mc.n_layer), d, int(mc.intermediate_size))
    # two_tower: take the geometry the model actually resolved (read_source "auto" etc.)
    spb = mc.get("state_intermediate_per_block", None)
    return F.two_tower_flops_per_token(
        state_n_layer=model.state_n_layer, pred_n_layer=model.pred_n_layer,
        state_hidden=model.state_hidden, pred_hidden=model.pred_hidden,
        state_intermediate=int(mc.state_intermediate if mc.get("state_intermediate") is not None
                               else mc.intermediate_size),
        pred_intermediate=int(mc.pred_intermediate if mc.get("pred_intermediate") is not None
                              else mc.intermediate_size),
        read_map=str(mc.read_map), read_source=model.read_source,
        share_ffn_across_towers=bool(mc.get("share_ffn_across_towers", False)),
        state_intermediate_per_block=(list(spb) if spb is not None else None),
    )


def param_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())  # de-duplicated by identity
    emb = 0
    seen = set()
    for name, mod in model.named_modules():
        import torch.nn as nn
        if isinstance(mod, (nn.Embedding,)) or name == "lm_head":
            w = mod.weight
            if id(w) not in seen:
                seen.add(id(w))
                emb += w.numel()
    return {"params_total": int(total), "params_nonemb": int(total - emb)}


def kv_bytes_analytic(model, cfg) -> dict:
    """Per-token bytes of the cache an incremental decoder would keep (bf16 k and v).

    Returns the per-token part and the per-SEQUENCE constant part (SPS's predict-slot window).
    """
    mc = cfg.model.config
    fam = family(cfg)
    d = int(mc.hidden_size)
    if fam == "standard":
        n_kv = int(mc.n_layer)
        return {"kv_levels": n_kv, "kv_bytes_per_token": n_kv * 2 * d * BF16_BYTES,
                "kv_bytes_const_per_seq": 0,
                "kv_note": f"{n_kv} layers x (k,v) x d={d}"}
    if fam == "sps":
        L = int(mc.n_layer)
        w = int(mc.window_size)
        return {"kv_levels": L, "kv_bytes_per_token": L * 2 * d * BF16_BYTES,
                "kv_bytes_const_per_seq": L * 2 * w * d * BF16_BYTES,
                "kv_note": (f"{L} layers x (k,v) of the state/token slot, every token; plus a "
                            f"ring of the last W={w} <predict>-slot k/v per layer (constant)")}
    # two-tower: state tower keeps every level's (k, v); pred reads either reuse them
    # (state_kv) or project their own (pred_proj -> one (k, v) per pred block).
    L_s, L_p = model.state_n_layer, model.pred_n_layer
    n_state = L_s
    n_head_lvl = 1 if "state_read_head" in model.transformer else 0
    n_read = L_p if model.read_source == "pred_proj" else 0
    n_kv = n_state + n_head_lvl + n_read
    note = f"state {L_s} levels + read_head {n_head_lvl} + pred read-proj {n_read}"
    return {"kv_levels": n_kv, "kv_bytes_per_token": n_kv * 2 * d * BF16_BYTES,
            "kv_bytes_const_per_seq": 0, "kv_note": note}


def kv_levels_from_code_two_tower(model, X):
    """Instrument `_attend`: count DISTINCT full-length (k, v) tensors the forward consumes.

    Each one is something an incremental decoder would have to cache per token (or recompute
    from a cached residual). Dedup by storage pointer, so `state_kv` reuse is counted once.
    """
    import torch

    seen = set()
    alive = []  # hold every k so a freed tensor's address cannot be reused and undercount
    orig = model._attend

    def spy(q, k, v, masks, stream):
        alive.append(k)
        seen.add((k.data_ptr(), tuple(k.shape)))
        return orig(q, k, v, masks, stream)

    model._attend = spy
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(X[:1, :256])
    finally:
        del model._attend
    return len(seen)


# --------------------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------------------
def _stats(ms: list[float]) -> dict:
    s = sorted(ms)
    n = len(s)
    return {"median_ms": statistics.median(s), "mean_ms": statistics.fmean(s),
            "p10_ms": s[max(0, int(0.1 * n) - 1)] if n >= 10 else s[0],
            "p90_ms": s[min(n - 1, int(math.ceil(0.9 * n)) - 1)], "n": n}


def val_bin(cfg) -> Path:
    """<DUALSPS_DATA_ROOT>/data/<cfg.data.dataset>/val.bin, the config's own val split."""
    path = repo_paths.data_root() / "data" / cfg.data.dataset / "val.bin"
    if not path.is_file():
        raise FileNotFoundError(f"no val.bin for dataset {cfg.data.dataset!r} at {path}")
    return path


def load_batches(val_path: Path, mb: int, T: int, n_batches: int, seed: int = 1234):
    import numpy as np
    import torch

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_batches):
        ix = rng.integers(0, len(data) - T - 1, size=mb)
        x = np.stack([data[i:i + T].astype(np.int64) for i in ix])
        y = np.stack([data[i + 1:i + 1 + T].astype(np.int64) for i in ix])
        xs.append(torch.from_numpy(x).cuda())
        ys.append(torch.from_numpy(y).cuda())
    return xs, ys


def time_train(model, cmodel, xs, ys, warmup: int, steps: int):
    import torch

    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda")  # conf/optimizer/adamw.yaml
    model.train()

    def step(i):
        # clone: the standard model masks `targets` IN PLACE (EOS -> ignore), and train.py
        # hands it a fresh Y every step; reusing a mutated Y breaks its logits.view().
        X, Y = xs[i % len(xs)], ys[i % len(ys)].clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = cmodel(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    t_c = time.perf_counter()
    step(0)
    torch.cuda.synchronize()
    first_step_s = time.perf_counter() - t_c
    for i in range(1, warmup):
        step(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = []
    last = None
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = step(warmup + i)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    loss_val = float(last.detach().float().item())
    return ms, peak, peak_res, first_step_s, loss_val


def time_prefill(model, cmodel, xs, ys, warmup: int, steps: int):
    import torch

    model.eval()
    with torch.no_grad():
        for i in range(warmup):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = cmodel(xs[i % len(xs)], ys[i % len(ys)].clone())
            del out
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ms = []
        for i in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = cmodel(xs[(warmup + i) % len(xs)], ys[(warmup + i) % len(ys)].clone())
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1e3)
            del out
    peak = torch.cuda.max_memory_allocated()
    model.train()
    return ms, peak


# --------------------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------------------
def run_arm(args) -> dict:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True  # as scripts/train.py sets them
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1337)
    exp = ARMS[args.arm]
    rec = {"arm": args.arm, "exp": exp, "rep": args.rep, "order_idx": args.order_idx,
           "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "micro_batch": args.mb, "seq_len": args.T, "compile": "torch.compile(model) default mode",
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        rec["git_commit"] = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    rec["overrides"] = ARM_OVERRIDES.get(args.arm, [])
    model, cfg = build(exp, rec["overrides"])
    rec["family"] = family(cfg)
    rec.update(param_counts(model))
    rec["flops_fwd_per_token"] = flops_per_token(model, cfg)
    rec.update(kv_bytes_analytic(model, cfg))
    model = model.cuda()
    xs, ys = load_batches(val_bin(cfg), args.mb, args.T, n_batches=4)
    if rec["family"] == "two_tower":
        from modeling.models.two_tower import attention as A
        rec["tt_attn_backend"] = A.resolve_backend(model.config.attn_backend, needs_document_mask=True)
        rec["kv_levels_code"] = kv_levels_from_code_two_tower(model, xs[0])
    rec["frac_batches_with_doc_boundary"] = float(
        sum(bool((x == model.config.eos_token_id).any()) for x in xs) / len(xs))

    cmodel = torch.compile(model) if not args.no_compile else model
    # 1. training
    ms, peak, peak_res, first_s, loss = time_train(model, cmodel, xs, ys, args.warmup, args.steps)
    tok = args.mb * args.T
    rec["train"] = {**_stats(ms), "tokens_per_s": tok * 1e3 / statistics.median(ms),
                    "peak_mem_gib": peak / 2**30, "peak_reserved_gib": peak_res / 2**30,
                    "first_step_incl_compile_s": first_s, "loss_last": loss, "step_ms_all": ms}
    print(f"[{args.arm}] train {rec['train']['tokens_per_s']:,.0f} tok/s "
          f"median {rec['train']['median_ms']:.1f} ms peak {peak/2**30:.1f} GiB", flush=True)
    # 2. prefill
    try:
        ms, peak = time_prefill(model, cmodel, xs, ys, args.warmup, args.steps)
        rec["prefill"] = {**_stats(ms), "tokens_per_s": tok * 1e3 / statistics.median(ms),
                          "peak_mem_gib": peak / 2**30, "step_ms_all": ms}
        print(f"[{args.arm}] prefill {rec['prefill']['tokens_per_s']:,.0f} tok/s", flush=True)
    except Exception as e:  # keep the training numbers
        traceback.print_exc()
        rec["prefill"] = {"error": repr(e)[:500]}
    # 3. decode: not measured (see the module docstring)
    if rec["family"] == "two_tower":
        rec["decode"] = {"status": "no_decode_path",
                         "reason": "the two-tower models have no KV-cached incremental decode path"}
    else:
        rec["decode"] = {"status": "not_measured",
                         "reason": "this release ships no incremental decode code"}
    return rec


# --------------------------------------------------------------------------------------
# report: jsonl -> markdown table
# --------------------------------------------------------------------------------------
def report(path: str, tag: str) -> str:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get("tag") == tag and "error" not in r]
    by = {}
    for r in recs:
        by.setdefault(r["arm"], []).append(r)
    ref = statistics.median(r["train"]["tokens_per_s"] for r in by["standard_tied"])
    ref_pf = statistics.median(r["prefill"]["tokens_per_s"] for r in by["standard_tied"])

    def f_k(x):
        return f"{x/1e3:,.0f}k"

    rows = ["| arm | FLOPs/tok (fwd) | params total / non-emb | train tok/s rep0 / rep1 | train rel. std-tied | "
            "prefill tok/s rep0 / rep1 | prefill rel. | KV bytes/tok | peak mem train (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm not in by:
            continue
        rs = sorted(by[arm], key=lambda r: r["rep"])
        r0 = rs[0]
        tr = [r["train"]["tokens_per_s"] for r in rs]
        pf = [r["prefill"]["tokens_per_s"] for r in rs if "tokens_per_s" in r["prefill"]]
        kv = f"{r0['kv_bytes_per_token']:,}"
        if r0.get("kv_bytes_const_per_seq"):
            kv += f" (+{r0['kv_bytes_const_per_seq']/2**20:.2f} MiB/seq)"
        if "kv_levels_code" in r0 and r0["kv_levels_code"] != r0["kv_levels"]:
            kv += f" [code: {r0['kv_levels_code']} lvls vs {r0['kv_levels']}]"
        rows.append(
            f"| {arm} | {r0['flops_fwd_per_token']/1e6:,.1f}M | {r0['params_total']/1e6:.1f}M / "
            f"{r0['params_nonemb']/1e6:.1f}M | {' / '.join(f_k(t) for t in tr)} | "
            f"{statistics.median(tr)/ref:.2f}x | {' / '.join(f_k(t) for t in pf)} | "
            f"{(statistics.median(pf)/ref_pf) if pf else float('nan'):.2f}x | {kv} | "
            f"{max(r['train']['peak_mem_gib'] for r in rs):.1f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--order-idx", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None, help="subset for --sweep")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=None, help="jsonl to append to (default: repo_paths.live_wallclock())")
    ap.add_argument("--mb", type=int, default=6)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--report", action="store_true", help="print the markdown table for --tag from --out")
    args = ap.parse_args()
    args.out = args.out or str(repo_paths.live_wallclock())

    if args.report:
        print(report(args.out, args.tag))
        return
    if args.sweep:
        arms = args.arms or list(ARMS)
        passthru = ["--out", args.out, "--mb", str(args.mb), "--T", str(args.T),
                    "--warmup", str(args.warmup), "--steps", str(args.steps), "--tag", args.tag]
        if args.no_compile:
            passthru.append("--no-compile")
        k = 0
        for rep in range(args.reps):
            order = arms if rep % 2 == 0 else arms[::-1]
            for arm in order:
                t0 = time.time()
                cmd = [sys.executable, __file__, "--arm", arm, "--rep", str(rep),
                       "--order-idx", str(k), *passthru]
                print(f"=== rep {rep} [{k}] {arm}", flush=True)
                rc = subprocess.run(cmd).returncode
                print(f"=== rep {rep} {arm} rc={rc} {time.time()-t0:.0f}s", flush=True)
                if rc != 0:
                    with open(args.out, "a") as f:
                        f.write(json.dumps({"arm": arm, "rep": rep, "tag": args.tag,
                                            "error": f"subprocess rc={rc}"}) + "\n")
                k += 1
        return

    assert args.arm, "--arm or --sweep required"
    try:
        rec = run_arm(args)
    except Exception as e:
        traceback.print_exc()
        rec = {"arm": args.arm, "rep": args.rep, "error": repr(e)[:1000]}
    rec["tag"] = args.tag
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if "error" in rec:
        sys.exit(1)


if __name__ == "__main__":
    main()
