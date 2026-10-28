"""A12 -- per-block depth lesions of each tower: which computations each stream requires.

A2 and A8 cut ACROSS the towers (which memory keys / which heads the readout may use).
Neither says anything about DEPTH: how much of each tower's 12 blocks is load-bearing for
language modelling at all.  A12 answers that with the cheapest possible causal probe --
delete one block's contribution and re-score.

The intervention, for every block i in 0..L-1 of each tower:

    attn    the block's attention branch writes nothing into its residual
            (`finish_attn(x, y) -> x`).  `bias=False` everywhere in this family, so this
            is EXACTLY "zero the attention output": c_proj(0) == 0.
    mlp     the block's MLP branch writes nothing (`mlp_step(x) -> x`).  Same argument:
            TowerMLP returns zeros, so the residual would be unchanged anyway.
    block   both -- the block becomes the identity map on its residual stream.

Why the lesion is applied to `finish_attn` / `mlp_step` and NOT by deleting the block from
the ModuleList: the STATE tower's blocks are two things at once.  Block j updates the
state residual AND (via `block.qkv`, or via `res_levels[j]` under
`read_source="pred_proj"`) supplies the k/v the pred tower READS at level j.  Removing the
module would delete the read interface as well as the computation, confounding "this
block's compute is dead" with "this read level does not exist".  Lesioning the residual
writes leaves every read level in place and changes only what flows INTO the levels above
it -- which is the question.  (A13 asks the complementary read-interface question.)

Nothing in `src/` is touched: the lesions are instance-attribute overrides of the two
public per-block steps `StateBlock`/`PredBlock` already expose for exactly this reason
(they were split out so the lockstep forward could interleave them), restored after every
arm.

Scoring is the 512-sequence sub-sweep (2.1 M tokens), NOT the 13.1 M-token full sweep:
72 lesions x 2 models x the full sweep is ~6x the budget, and every number here is a
PAIRED delta (one checkpoint, one val slice, one deterministic order, only the lesion
differing), whose resolvable floor is ~0.0005 -- far below any effect this analysis is
looking for.  The control's own full-sweep NLL is reported alongside so the sub-sweep's
offset is visible, exactly as A2 does it.

GATE: a no-op lesion (the patch installed, but calling straight through to the original
method) must reproduce the unpatched sub-sweep NLL bit-for-bit, and the patch counter must
be non-zero.  A patch that never fires reports every delta as a fake 0.000.

MEAN-ABLATION (`--ablation mean`, default `zero`).  Zeroing a write puts the residual
off-distribution; mean-ablation instead replaces the lesioned write, at EVERY position of
the lesioned stream (two-tower) or slot type (joint), with that write's per-channel mean
over a HELD-OUT set of training-distribution sequences: val sequences strided over
[sub_seqs, n_val), i.e. disjoint from the scored sub-sweep by construction (asserted).
The means are taken on the UN-lesioned model (clean-run means, the standard definition),
one vector per (stream|slot, block, attn|mlp).  `block` uses both means.  Output goes to
`a12_depth_lesion_<run>_mean.json`; the zero-mode file is never touched, and the zero path
is byte-for-byte the original code (the mean branch is a separate closure).

STANDARD branch (single-stream Transformer, family "standard").  The baseline for the
"state work is front-loaded / layer-1 MLP is critical" reading: the same attn/mlp/block
lesions, zero or mean, of the one stream (`StandardLesion`, stream name "single"), with
the same no-op gate plus an identity-rebuild gate, and the same held-out mean set.

Usage:
  a12_depth_lesion.py --run <run|role> [--sub-seqs 512] [--full-sweep-control]
                      [--kinds attn,mlp,block] [--streams state,pred]
                      [--ablation zero|mean] [--mean-seqs 256]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402  (also used by the joint slot lesion)

import common as C  # noqa: E402

KINDS = ("attn", "mlp", "block")
STREAMS = ("state", "pred")
ABLATIONS = ("zero", "mean")
MEAN_SEQS = 256          # held-out sequences for the mean-ablation vectors (~1.05 M tok)


# --------------------------------------------------------------------------------------
# mean-ablation: held-out sequences + clean-run write means
# --------------------------------------------------------------------------------------
def mean_seq_starts(n_tokens: int, n_eval: int, n_mean: int, block: int = None):
    """Starts of the held-out sequences the ablation means are taken over.

    Same grid as `common.sweep_nll` (non-overlapping BLOCK-strided starts from 0); the
    scored sub-sweep is `grid[:n_eval]`, so the mean set is drawn evenly-strided from
    `grid[n_eval:]` and is disjoint from it by construction."""
    block = block or C.BLOCK
    grid = list(range(0, n_tokens - block - 1, block))
    pool = grid[n_eval:]
    assert pool, f"no held-out sequences beyond the first {n_eval}"
    stride = max(1, len(pool) // max(1, n_mean))
    out = pool[::stride][:n_mean]
    assert not (set(out) & set(grid[:n_eval])), "mean set overlaps the scored sequences"
    return out


def _batches(val_path, starts):
    import numpy as np
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    for i in range(0, len(starts), C.MB):
        b = starts[i:i + C.MB]
        X = torch.stack([torch.from_numpy(data[j:j + C.BLOCK].astype(np.int64)) for j in b])
        Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + C.BLOCK].astype(np.int64))
                         for j in b])
        yield X.cuda(), Y.cuda()


class _MeanAcc:
    """float64 per-channel running sum/count, keyed (stream, block, kind)."""

    def __init__(self):
        self.s, self.n, self.sq = {}, {}, {}

    def add(self, key, w, mask=None):
        w = w.detach().double()
        if mask is not None:                            # (1, T, 1) bool row selector
            w = w[mask.expand_as(w)].view(-1, w.shape[-1])
        else:
            w = w.reshape(-1, w.shape[-1])
        self.s[key] = self.s.get(key, 0) + w.sum(0)
        self.sq[key] = self.sq.get(key, 0) + (w * w).sum()
        self.n[key] = self.n.get(key, 0) + w.shape[0]

    def means(self):
        return {k: (self.s[k] / self.n[k]).float() for k in self.s}

    def stats(self):
        """per key: ||mean||, write RMS norm (sqrt E||w||^2), rows."""
        out = {}
        for k in self.s:
            mu = self.s[k] / self.n[k]
            out[f"{k[0]}_{k[1]:02d}_{k[2]}"] = dict(
                mean_norm=round(float(mu.norm()), 6),
                write_rms_norm=round(float((self.sq[k] / self.n[k]).sqrt()), 6),
                rows=int(self.n[k]))
        return out


def two_tower_write_means(model, batches):
    """Clean-run per-channel means of every state/pred block's attention and MLP write.
    Installs capture patches that return exactly what the class methods return (the
    attention write is `finish_attn(0, y)`, i.e. the class's own projection path)."""
    acc = _MeanAcc()
    patched = []
    for stream, blocks in (("state", list(model.transformer.state_h)),
                           ("pred", list(model.transformer.pred_h))):
        for i, blk in enumerate(blocks):
            cls = type(blk)

            def finish(x, y, _b=blk, _k=(stream, i, "attn"), _f=cls.finish_attn):
                w = _f(_b, torch.zeros_like(x), y)
                acc.add(_k, w)
                return x + w

            def mlp(x, _b=blk, _k=(stream, i, "mlp")):
                w = _b.mlp(_b.mlp_norm(x))
                acc.add(_k, w)
                return x + w
            blk.finish_attn, blk.mlp_step = finish, mlp
            patched.append(blk)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for X, Y in batches:
                model(X, Y)
    finally:
        for blk in patched:
            for n in ("finish_attn", "mlp_step"):
                blk.__dict__.pop(n, None)
    return acc


def joint_write_means(model, batches):
    """Clean-run per-channel means of every shared-stack block's attention and MLP write,
    separately over state slots (2p) and pred slots (2p+1)."""
    acc = _MeanAcc()
    blocks = C.joint_blocks(model)
    for i, blk in enumerate(blocks):
        def fwd(x, freqs_cis, documents_idx_Bx2T=None, _b=blk, _i=i):
            s_m, p_m = C.joint_slot_masks(x.shape[1], x.device)
            a = _b.attn(_b.attention_norm(x), freqs_cis, documents_idx_Bx2T=documents_idx_Bx2T)
            acc.add(("state", _i, "attn"), a, s_m)
            acc.add(("pred", _i, "attn"), a, p_m)
            x = x + a
            m = _b.mlp(_b.mlp_norm(x))
            acc.add(("state", _i, "mlp"), m, s_m)
            acc.add(("pred", _i, "mlp"), m, p_m)
            return x + m
        blk.forward = fwd
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for X, Y in batches:
                model(X, Y)
    finally:
        for blk in blocks:
            blk.__dict__.pop("forward", None)
    return acc


def compute_means(model, family, val_path, n_eval, n_mean):
    import numpy as np
    n_tok = len(np.memmap(val_path, dtype=np.uint16, mode="r"))
    starts = mean_seq_starts(n_tok, n_eval, n_mean)
    t = time.time()
    fn = {"sps": joint_write_means,
          "standard": standard_write_means}.get(family, two_tower_write_means)
    acc = fn(model, _batches(val_path, starts))
    meta = dict(source="val.bin (same file as the scored sub-sweep)",
                selection=f"BLOCK-strided grid[{n_eval}:], evenly strided, first {n_mean}",
                n_seq=len(starts), first_start=int(starts[0]), last_start=int(starts[-1]),
                disjoint_from_scored=True, seconds=round(time.time() - t, 1),
                per_write=acc.stats())
    print(f"  mean-ablation vectors: {len(starts)} held-out seqs "
          f"({time.time()-t:.0f}s)", flush=True)
    return acc.means(), meta


# --------------------------------------------------------------------------------------
class DepthLesion:
    """Installs/removes one per-block lesion on a TwoTowerModel.  One model, one instance.

    The originals are captured ONCE at construction (they are the bound class methods) and
    every `clear()` restores by deleting the instance attribute, so no arm can inherit the
    previous arm's patch -- the failure mode that would make an entire table monotone.
    """

    def __init__(self, model):
        self.model = model
        self.state_blocks = list(model.transformer.state_h)
        self.pred_blocks = list(model.transformer.pred_h)
        self.calls = 0
        self._patched: list = []

    def blocks(self, stream):
        return self.state_blocks if stream == "state" else self.pred_blocks

    # -- installation -------------------------------------------------------------
    def _set(self, block, name, fn):
        self._patched.append((block, name))
        setattr(block, name, fn)

    def clear(self):
        for block, name in self._patched:
            try:
                delattr(block, name)          # falls back to the class method
            except AttributeError:
                pass
        self._patched = []

    def apply(self, stream, idx, kind, noop=False, means=None):
        """Install the lesion.  `noop=True` installs a patch that calls the ORIGINAL
        method -- the control that proves the patch path is live without changing maths.
        `means`: mean-ablation -- a dict (stream, idx, "attn"|"mlp") -> tensor broadcastable
        to the write (per-channel (C,) in practice); the lesioned write is REPLACED by it
        at every position instead of being zeroed.  None = zero-ablation (original path)."""
        self.clear()
        block = self.blocks(stream)[idx]
        cls = type(block)
        orig_finish, orig_mlp = cls.finish_attn, cls.mlp_step

        if noop:
            def finish(x, y, _b=block):
                self.calls += 1
                return orig_finish(_b, x, y)

            def mlp(x, _b=block):
                self.calls += 1
                return orig_mlp(_b, x)
            self._set(block, "finish_attn", finish)
            self._set(block, "mlp_step", mlp)
            return

        if means is not None:
            if kind in ("attn", "block"):
                mu_a = means[(stream, idx, "attn")]

                def finish(x, y, _mu=mu_a):
                    self.calls += 1
                    return x + _mu.to(device=x.device, dtype=x.dtype)
                self._set(block, "finish_attn", finish)
            if kind in ("mlp", "block"):
                mu_m = means[(stream, idx, "mlp")]

                def mlp(x, _mu=mu_m):
                    self.calls += 1
                    return x + _mu.to(device=x.device, dtype=x.dtype)
                self._set(block, "mlp_step", mlp)
            return

        if kind in ("attn", "block"):
            def finish(x, y):
                self.calls += 1
                return x                      # bias=False -> c_proj(0) == 0 exactly
            self._set(block, "finish_attn", finish)
        if kind in ("mlp", "block"):
            def mlp(x):
                self.calls += 1
                return x
            self._set(block, "mlp_step", mlp)


# --------------------------------------------------------------------------------------
# JOINT (tied-SPS) branch: slot lesions on the ONE weight-shared stack
# --------------------------------------------------------------------------------------
class SlotLesion:
    """Per-block SLOT lesion on the tied-SPS model (one stack, interleaved 2T slots).

    Same semantics as the two-tower `DepthLesion` (the one used for seq12_tied): a
    lesioned branch writes NOTHING into the residual -- but here only at one slot type.
    For block i and slot type S in {state, pred}:

        attn    x <- x + where(S-slot, 0, attn(norm(x)))      (bias=False: c_proj(0)=0)
        mlp     x <- x + where(S-slot, 0, mlp(norm(x)))
        block   both

    Exactly as in the two-tower lesion the READ interface is left intact: the lesioned
    slots still emit their keys/values (computed from their un-lesioned INPUT residual)
    to every query that may read them; only what flows into their residual above block
    i changes.  Zero-ablation, not mean-ablation, because that is the one-tower lesion's
    semantics and the only one under which "the block writes nothing" is exact.

    Implemented as an instance-level `forward` override on one `SPSBlock` (restored by
    `clear()`), built from the block's own submodules in the block's own op order, so
    the un-lesioned rows are bit-identical to the unpatched model's.
    """

    def __init__(self, model):
        self.blocks = C.joint_blocks(model)
        self.calls = 0
        self._patched = []

    def clear(self):
        for blk in self._patched:
            try:
                del blk.forward
            except AttributeError:
                pass
        self._patched = []

    def apply(self, stream, idx, kind, noop=False, identity_mask=False, means=None):
        """`noop`: call straight through to the class forward (patch liveness).
        `identity_mask`: go through the LESION code path with a keep-everything mask --
        the control that proves the masked path itself is exact (delta must be 0).
        `means`: mean-ablation; (slot, idx, "attn"|"mlp") -> tensor broadcastable to the
        write, substituted at the lesioned slot rows instead of zero.  None = zero."""
        self.clear()
        blk = self.blocks[idx]
        orig = type(blk).forward
        if means is not None and not noop:
            mu_a, mu_m = means[(stream, idx, "attn")], means[(stream, idx, "mlp")]

            def fwd(x, freqs_cis, documents_idx_Bx2T=None, _b=blk):
                self.calls += 1
                s_m, p_m = C.joint_slot_masks(x.shape[1], x.device)
                cut = torch.zeros_like(s_m) if identity_mask else (s_m if stream == "state" else p_m)
                a = _b.attn(_b.attention_norm(x), freqs_cis, documents_idx_Bx2T=documents_idx_Bx2T)
                if kind in ("attn", "block"):
                    a = torch.where(cut, mu_a.to(device=a.device, dtype=a.dtype), a)
                x = x + a
                m = _b.mlp(_b.mlp_norm(x))
                if kind in ("mlp", "block"):
                    m = torch.where(cut, mu_m.to(device=m.device, dtype=m.dtype), m)
                return x + m
            blk.forward = fwd
            self._patched.append(blk)
            return
        if noop:
            def fwd(x, freqs_cis, documents_idx_Bx2T=None, _b=blk):
                self.calls += 1
                return orig(_b, x, freqs_cis, documents_idx_Bx2T=documents_idx_Bx2T)
        else:
            def fwd(x, freqs_cis, documents_idx_Bx2T=None, _b=blk):
                self.calls += 1
                s_m, p_m = C.joint_slot_masks(x.shape[1], x.device)
                cut = torch.zeros_like(s_m) if identity_mask else (s_m if stream == "state" else p_m)
                a = _b.attn(_b.attention_norm(x), freqs_cis, documents_idx_Bx2T=documents_idx_Bx2T)
                if kind in ("attn", "block"):
                    a = torch.where(cut, torch.zeros_like(a), a)
                x = x + a
                m = _b.mlp(_b.mlp_norm(x))
                if kind in ("mlp", "block"):
                    m = torch.where(cut, torch.zeros_like(m), m)
                return x + m
        blk.forward = fwd
        self._patched.append(blk)


def run_joint(args, run, out_path, kinds, streams, model, cfg, family, ckpath):
    """A12 for the tied-SPS model.  "stream" = slot type; see `SlotLesion`.

    GATES (all must pass before any delta is recorded):
      1. no-op patch reproduces the unpatched sub-sweep NLL bit-for-bit, calls > 0
         (identical to the two-tower gate);
      2. identity lesion -- the lesion code path with an empty cut mask -- reproduces it
         bit-for-bit (proves the masked re-implementation of the block is exact);
      3. dead-end control: the LAST block's state-slot writes are read by nothing (the
         LM head reads pred slots only and no block follows), so lesioning them must
         give delta == 0 exactly.  A non-zero value means the slot parity is wrong.
    """
    val_path = C.val_path_of(cfg)
    L = len(C.joint_blocks(model))
    print(f"run={run} family={family} L={L} window={cfg.model.config.window_size} "
          f"ckpt={ckpath}", flush=True)
    t0 = time.time()
    native, ntok, nseq = C.sweep_nll(model, val_path, args.sub_seqs)
    print(f"  native sub-sweep nll={native:.6f} ({ntok} tok, {nseq} seq, "
          f"{time.time()-t0:.0f}s)", flush=True)
    res = dict(
        analysis="a12_depth_lesion", run=run, model_id=C.mid_of(run), label=C.label_of(run),
        family=family, branch="joint", checkpoint=ckpath,
        lesion_semantics=("slot lesion on the shared stack: zero the block's attention "
                          "and/or MLP residual write at state-slot (2p) or pred-slot "
                          "(2p+1) rows only; keys/values of lesioned slots untouched"),
        geometry=dict(n_layer=L, hidden=int(cfg.model.config.hidden_size),
                      intermediate=int(cfg.model.config.intermediate_size),
                      window_size=int(cfg.model.config.window_size),
                      tie_lm_head=bool(model.config.tie_lm_head)),
        scoring=dict(kind="sub-sweep", n_seq=nseq, n_tokens=ntok,
                     val_bin=val_path, paired_floor=C.PAIRED_FLOOR,
                     note="paired deltas: one ckpt, one val slice, one order; only the "
                          "lesion differs"),
        native_sub_sweep_nll=round(native, 6),
        lesions={},
    )
    means = _attach_ablation(res, args, model, family, val_path, nseq)
    if args.full_sweep_control:
        t = time.time()
        fn, ft, fs = C.sweep_nll(model, val_path, None)
        res["native_full_sweep_nll"] = round(fn, 6)
        res["scoring"]["full_sweep_tokens"] = ft
        res["scoring"]["full_sweep_seqs"] = fs
        res["scoring"]["sub_sweep_offset"] = round(native - fn, 6)
        print(f"  native FULL sweep nll={fn:.6f} ({ft} tok, {time.time()-t:.0f}s); "
              f"sub-sweep offset {native - fn:+.6f}", flush=True)
    C.save_json(out_path, res)

    les = SlotLesion(model)
    gates = {}
    for tag, kw in (("noop", dict(stream="pred", idx=0, kind="block", noop=True)),
                    ("identity_lesion", dict(stream="pred", idx=0, kind="block",
                                             identity_mask=True, means=means)),
                    ("dead_end_last_state_block", dict(stream="state", idx=L - 1,
                                                       kind="block", means=means))):
        c0 = les.calls
        les.apply(**kw)
        nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
        les.clear()
        d = nll - native
        gates[tag] = dict(nll=round(nll, 6), delta=d, calls=les.calls - c0,
                          passed=bool(abs(d) <= args.gate_tol and les.calls > c0))
        if tag == "dead_end_last_state_block" and means is not None:
            # Mean mode: the structural zero is REPORTED, not enforced (the task allows
            # it to be non-zero); the patch must still have fired.
            gates[tag]["enforced"] = False
            gates[tag]["passed"] = bool(les.calls > c0)
        print(f"  GATE {tag}: delta={d:+.8f} calls={les.calls - c0} "
              f"{'PASS' if gates[tag]['passed'] else 'FAIL'}", flush=True)
    res["control_noop_nll"] = gates["noop"]["nll"]
    res["control_delta_vs_native"] = round(gates["noop"]["delta"], 6)
    res["patched_calls"] = int(les.calls)
    res["gates"] = gates
    passed = all(g["passed"] for g in gates.values())
    res["control_gate_passed"] = passed
    if not passed:
        C.save_json(out_path, res)
        print("\nA12 (joint) FAILED a control gate; no lesion recorded.", flush=True)
        sys.exit(1)

    n_done = 0
    for stream in streams:
        for idx in range(L):
            for kind in kinds:
                tag = f"{stream}_{idx:02d}_{kind}"
                les.apply(stream, idx, kind, means=means)
                t = time.time()
                nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
                les.clear()
                d = nll - native
                res["lesions"][tag] = dict(stream=stream, block=idx, kind=kind,
                                           val_nll=round(nll, 6), delta=round(d, 6),
                                           seconds=round(time.time() - t, 1))
                n_done += 1
                print(f"  {tag:22s} nll={nll:.6f}  delta={d:+.6f}  "
                      f"({time.time()-t:.0f}s)  [{n_done}]", flush=True)
                C.save_json(out_path, res)

    summary = {}
    for stream in streams:
        for kind in kinds:
            ds = [res["lesions"][f"{stream}_{i:02d}_{kind}"]["delta"] for i in range(L)
                  if f"{stream}_{i:02d}_{kind}" in res["lesions"]]
            if not ds:
                continue
            summary[f"{stream}_{kind}"] = dict(
                total_delta=round(sum(ds), 6),
                max_delta=round(max(ds), 6),
                argmax_block=int(max(range(len(ds)), key=lambda i: ds[i])),
                min_delta=round(min(ds), 6),
                argmin_block=int(min(range(len(ds)), key=lambda i: ds[i])),
                n_below_1e_2=int(sum(1 for d in ds if d < 0.01)),
                blocks_below_1e_2=[i for i, d in enumerate(ds) if d < 0.01],
            )
    res["summary"] = summary
    C.save_json(out_path, res)
    print("\nA12 (joint) done.", flush=True)


# --------------------------------------------------------------------------------------
# STANDARD (single-stream Transformer) branch: the baseline for "state is front-loaded"
# --------------------------------------------------------------------------------------
def standard_blocks(model):
    """The one block stack of the single-stream Transformer (`full_attention_model`)."""
    return list(model.transformer.h)


def standard_write_means(model, batches):
    """Clean-run per-channel means of every block's attention and MLP write (all
    positions; the single stream has no slot/stream split).  Keyed ("single", i, kind)."""
    acc = _MeanAcc()
    blocks = standard_blocks(model)
    for i, blk in enumerate(blocks):
        def fwd(x, freqs_cis, _b=blk, _i=i, **kw):
            a = _b.attn(_b.attention_norm(x), freqs_cis, **kw)
            acc.add(("single", _i, "attn"), a)
            x = x + a
            m = _b.mlp(_b.mlp_norm(x))
            acc.add(("single", _i, "mlp"), m)
            return x + m
        blk.forward = fwd
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for X, Y in batches:
                model(X, Y.clone())
    finally:
        for blk in blocks:
            blk.__dict__.pop("forward", None)
    return acc


class StandardLesion:
    """Per-block lesion of the single-stream Transformer: the SAME semantics as the
    two-tower `DepthLesion` and the joint `SlotLesion`, on the only stream there is.

        attn    x <- x + 0                (bias=False: c_proj(0) == 0, exact)
        mlp     x <- x + 0
        block   both -- the block is the identity map on the residual

    Mean mode replaces the lesioned write, at every position, by its clean-run per-channel
    mean ("single", i, kind).  Implemented as an instance-level `forward` override on one
    `Block`/`TritonFullAttentionBlock`, rebuilt from the block's own submodules in the
    block's own op order (`attn(attention_norm(x))`, residual add, `mlp(mlp_norm(x))`,
    residual add), so an un-lesioned rebuild is bit-identical to the class forward (the
    `identity` gate proves it).  The lesion never changes what the block's attention
    READS.
    """

    STREAM = "single"

    def __init__(self, model):
        self.blocks = standard_blocks(model)
        self.calls = 0
        self._patched = []

    def clear(self):
        for blk in self._patched:
            try:
                del blk.forward
            except AttributeError:
                pass
        self._patched = []

    def apply(self, stream, idx, kind, noop=False, identity=False, means=None):
        """`noop`: straight through to the class forward (patch liveness).
        `identity`: the REBUILT forward with nothing removed -- proves the rebuild exact.
        `means`: mean-ablation dict ("single", idx, "attn"|"mlp") -> (C,) tensor."""
        assert stream == self.STREAM, f"standard family has one stream, got {stream!r}"
        self.clear()
        blk = self.blocks[idx]
        orig = type(blk).forward
        if noop:
            def fwd(x, freqs_cis, _b=blk, **kw):
                self.calls += 1
                return orig(_b, x, freqs_cis, **kw)
        else:
            cut_a = (not identity) and kind in ("attn", "block")
            cut_m = (not identity) and kind in ("mlp", "block")
            mu_a = means[(stream, idx, "attn")] if (means is not None and cut_a) else None
            mu_m = means[(stream, idx, "mlp")] if (means is not None and cut_m) else None

            def fwd(x, freqs_cis, _b=blk, **kw):
                self.calls += 1
                a = _b.attn(_b.attention_norm(x), freqs_cis, **kw)
                if cut_a:
                    a = (torch.zeros_like(a) if mu_a is None
                         else mu_a.to(device=a.device, dtype=a.dtype).expand_as(a))
                x = x + a
                m = _b.mlp(_b.mlp_norm(x))
                if cut_m:
                    m = (torch.zeros_like(m) if mu_m is None
                         else mu_m.to(device=m.device, dtype=m.dtype).expand_as(m))
                return x + m
        blk.forward = fwd
        self._patched.append(blk)


def run_standard(args, run, out_path, kinds, model, cfg, family, ckpath):
    """A12 for the single-stream Transformer -- the reference the separated models' and
    SPS's depth profiles are read against ("is the Transformer's layer 1 just as
    dominant?").

    GATES (both must pass before any delta is recorded):
      1. no-op patch reproduces the unpatched sub-sweep NLL bit-for-bit, calls > 0;
      2. identity lesion (the rebuilt block forward, nothing removed) reproduces it
         bit-for-bit, calls > 0.
    """
    val_path = C.val_path_of(cfg)
    blocks = standard_blocks(model)
    L = len(blocks)
    stream = StandardLesion.STREAM
    print(f"run={run} family={family} L={L} block={type(blocks[0]).__name__} "
          f"ckpt={ckpath}", flush=True)
    t0 = time.time()
    native, ntok, nseq = C.sweep_nll(model, val_path, args.sub_seqs)
    print(f"  native sub-sweep nll={native:.6f} ({ntok} tok, {nseq} seq, "
          f"{time.time()-t0:.0f}s)", flush=True)
    mc = cfg.model.config
    res = dict(
        analysis="a12_depth_lesion", run=run, model_id=C.mid_of(run), label=C.label_of(run),
        family=family, branch="standard", checkpoint=ckpath,
        lesion_semantics=("single-stream lesion: zero the block's attention and/or MLP "
                          "residual write at every position; keys/values untouched"),
        geometry=dict(n_layer=L, hidden=int(mc.hidden_size),
                      intermediate=int(mc.intermediate_size),
                      tie_lm_head=bool(getattr(model.config, "tie_lm_head", True)),
                      block_class=type(blocks[0]).__name__),
        scoring=dict(kind="sub-sweep", n_seq=nseq, n_tokens=ntok,
                     val_bin=val_path, paired_floor=C.PAIRED_FLOOR,
                     note="paired deltas: one ckpt, one val slice, one order; only the "
                          "lesion differs"),
        native_sub_sweep_nll=round(native, 6),
        lesions={},
    )
    means = _attach_ablation(res, args, model, family, val_path, nseq)
    if args.full_sweep_control:
        t = time.time()
        fn, ft, fs = C.sweep_nll(model, val_path, None)
        res["native_full_sweep_nll"] = round(fn, 6)
        res["scoring"]["full_sweep_tokens"] = ft
        res["scoring"]["full_sweep_seqs"] = fs
        res["scoring"]["sub_sweep_offset"] = round(native - fn, 6)
        print(f"  native FULL sweep nll={fn:.6f} ({ft} tok, {time.time()-t:.0f}s); "
              f"sub-sweep offset {native - fn:+.6f}", flush=True)
    C.save_json(out_path, res)

    les = StandardLesion(model)
    gates = {}
    for tag, kw in (("noop", dict(idx=0, kind="block", noop=True)),
                    ("identity_lesion", dict(idx=0, kind="block", identity=True,
                                             means=means))):
        c0 = les.calls
        les.apply(stream, **kw)
        nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
        les.clear()
        d = nll - native
        gates[tag] = dict(nll=round(nll, 6), delta=d, calls=les.calls - c0,
                          passed=bool(abs(d) <= args.gate_tol and les.calls > c0))
        print(f"  GATE {tag}: delta={d:+.8f} calls={les.calls - c0} "
              f"{'PASS' if gates[tag]['passed'] else 'FAIL'}", flush=True)
    res["control_noop_nll"] = gates["noop"]["nll"]
    res["control_delta_vs_native"] = round(gates["noop"]["delta"], 6)
    res["patched_calls"] = int(les.calls)
    res["gates"] = gates
    passed = all(g["passed"] for g in gates.values())
    res["control_gate_passed"] = passed
    if not passed:
        C.save_json(out_path, res)
        print("\nA12 (standard) FAILED a control gate; no lesion recorded.", flush=True)
        sys.exit(1)

    n_done = 0
    for idx in range(L):
        for kind in kinds:
            tag = f"{stream}_{idx:02d}_{kind}"
            les.apply(stream, idx, kind, means=means)
            t = time.time()
            nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
            les.clear()
            d = nll - native
            res["lesions"][tag] = dict(stream=stream, block=idx, kind=kind,
                                       val_nll=round(nll, 6), delta=round(d, 6),
                                       seconds=round(time.time() - t, 1))
            n_done += 1
            print(f"  {tag:22s} nll={nll:.6f}  delta={d:+.6f}  "
                  f"({time.time()-t:.0f}s)  [{n_done}]", flush=True)
            C.save_json(out_path, res)

    summary = {}
    for kind in kinds:
        ds = [res["lesions"][f"{stream}_{i:02d}_{kind}"]["delta"] for i in range(L)
              if f"{stream}_{i:02d}_{kind}" in res["lesions"]]
        if not ds:
            continue
        tot = sum(ds)
        summary[f"{stream}_{kind}"] = dict(
            total_delta=round(tot, 6),
            max_delta=round(max(ds), 6),
            argmax_block=int(max(range(len(ds)), key=lambda i: ds[i])),
            min_delta=round(min(ds), 6),
            argmin_block=int(min(range(len(ds)), key=lambda i: ds[i])),
            n_below_1e_2=int(sum(1 for d in ds if d < 0.01)),
            blocks_below_1e_2=[i for i, d in enumerate(ds) if d < 0.01],
            block0_share_of_total=round(ds[0] / tot, 6) if tot else None,
            first_third_share_of_total=(round(sum(ds[:max(1, L // 3)]) / tot, 6)
                                        if tot else None),
        )
    res["summary"] = summary
    C.save_json(out_path, res)
    print("\nA12 (standard) done.", flush=True)


# --------------------------------------------------------------------------------------
def default_out_path(run, ablation="zero"):
    """zero -> the committed a12_depth_lesion_<run>.json (unchanged);
    mean -> a12_depth_lesion_<run>_mean.json (never the zero file)."""
    zero = C.result_path("a12_depth_lesion", run)
    if ablation == "zero":
        return zero
    out = zero[:-len(".json")] + f"_{ablation}.json"
    assert out != zero
    return out


def _attach_ablation(res, args, model, family, val_path, n_eval):
    """Record the ablation mode on the result; in mean mode compute the held-out means.
    Returns the means dict (mean mode) or None (zero mode -> original code path).  Zero
    mode adds NO key to the result, so the zero JSON is reproduced byte-for-byte."""
    if args.ablation == "zero":
        return None
    means, meta = compute_means(model, family, val_path, n_eval, args.mean_seqs)
    res["ablation"] = "mean"
    res["lesion_semantics_ablation"] = (
        "MEAN-ablation: the lesioned write is replaced, at every position of the lesioned "
        "stream/slot type, by its per-channel clean-run mean over held-out val sequences "
        "disjoint from the scored sub-sweep")
    res["mean_ablation"] = meta
    return means


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sub-seqs", type=int, default=C.SUB_SWEEP_SEQS)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--streams", default=",".join(STREAMS))
    ap.add_argument("--full-sweep-control", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--gate-tol", type=float, default=1e-6)
    ap.add_argument("--ablation", choices=ABLATIONS, default="zero")
    ap.add_argument("--mean-seqs", type=int, default=MEAN_SEQS)
    args = ap.parse_args()

    run = C.resolve(args.run)
    out_path = args.out or default_out_path(run, args.ablation)
    kinds = [k for k in args.kinds.split(",") if k.strip()]
    streams = [s for s in args.streams.split(",") if s.strip()]
    assert all(k in KINDS for k in kinds), f"kinds must be a subset of {KINDS}"
    assert all(s in STREAMS for s in streams), f"streams must be a subset of {STREAMS}"
    C.assert_node_local_triton()

    model, cfg, family, ckpath = C.load(run)
    if family == "sps":
        return run_joint(args, run, out_path, kinds, streams, model, cfg, family, ckpath)
    if family == "standard":
        return run_standard(args, run, out_path, kinds, model, cfg, family, ckpath)
    if family != "two_tower":
        C.save_json(out_path, dict(analysis="a12_depth_lesion", run=run,
                                   model_id=C.mid_of(run), family=family,
                                   checkpoint=ckpath, skipped=True,
                                   reason="A12 lesions the two-tower per-tower block "
                                          "stacks; the joint families have one stack"))
        return

    val_path = C.val_path_of(cfg)
    L_s, L_p = int(model.state_n_layer), int(model.pred_n_layer)
    print(f"run={run} family={family} L_state={L_s} L_pred={L_p} "
          f"read_map={cfg.model.config.read_map} read_source={model.read_source} "
          f"ckpt={ckpath}", flush=True)

    t0 = time.time()
    native, ntok, nseq = C.sweep_nll(model, val_path, args.sub_seqs)
    print(f"  native sub-sweep nll={native:.6f} ({ntok} tok, {nseq} seq, "
          f"{time.time()-t0:.0f}s)", flush=True)

    res = dict(
        analysis="a12_depth_lesion", run=run, model_id=C.mid_of(run), label=C.label_of(run),
        family=family, checkpoint=ckpath,
        geometry=dict(state_n_layer=L_s, pred_n_layer=L_p,
                      state_hidden=int(model.state_hidden), pred_hidden=int(model.pred_hidden),
                      state_intermediate=int(model.state_intermediate),
                      pred_intermediate=int(model.pred_intermediate),
                      read_map=str(cfg.model.config.read_map),
                      read_source=str(model.read_source),
                      read_levels=list(model.read_levels)),
        scoring=dict(kind="sub-sweep", n_seq=nseq, n_tokens=ntok,
                     val_bin=val_path, paired_floor=C.PAIRED_FLOOR,
                     note="paired deltas: one ckpt, one val slice, one order; only the "
                          "lesion differs"),
        native_sub_sweep_nll=round(native, 6),
        lesions={},
    )
    means = _attach_ablation(res, args, model, family, val_path, nseq)
    if args.full_sweep_control:
        t = time.time()
        fn, ft, fs = C.sweep_nll(model, val_path, None)
        res["native_full_sweep_nll"] = round(fn, 6)
        res["scoring"]["full_sweep_tokens"] = ft
        res["scoring"]["full_sweep_seqs"] = fs
        res["scoring"]["sub_sweep_offset"] = round(native - fn, 6)
        print(f"  native FULL sweep nll={fn:.6f} ({ft} tok, {time.time()-t:.0f}s); "
              f"sub-sweep offset {native - fn:+.6f}", flush=True)
    C.save_json(out_path, res)

    les = DepthLesion(model)

    # --- GATE: the patch path is live AND mathematically inert when it should be.
    les.apply("pred", 0, "attn", noop=True)
    ctrl, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
    les.clear()
    res["control_noop_nll"] = round(ctrl, 6)
    res["control_delta_vs_native"] = round(ctrl - native, 6)
    res["patched_calls"] = int(les.calls)
    passed = abs(ctrl - native) <= args.gate_tol and les.calls > 0
    res["control_gate_passed"] = bool(passed)
    print(f"  GATE noop-vs-native delta = {ctrl - native:+.8f} "
          f"(tol {args.gate_tol}), patch calls = {les.calls}", flush=True)
    if not passed:
        C.save_json(out_path, res)
        print("\nA12 FAILED: the no-op lesion either never fired (every delta would be a "
              "fake zero) or changed the loss (the patch is not inert).", flush=True)
        sys.exit(1)

    # --- the sweep ---------------------------------------------------------------
    n_done = 0
    for stream in streams:
        n_blocks = L_s if stream == "state" else L_p
        for idx in range(n_blocks):
            for kind in kinds:
                tag = f"{stream}_{idx:02d}_{kind}"
                les.apply(stream, idx, kind, means=means)
                t = time.time()
                nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
                les.clear()
                d = nll - native
                res["lesions"][tag] = dict(stream=stream, block=idx, kind=kind,
                                           val_nll=round(nll, 6), delta=round(d, 6),
                                           seconds=round(time.time() - t, 1))
                n_done += 1
                print(f"  {tag:22s} nll={nll:.6f}  delta={d:+.6f}  "
                      f"({time.time()-t:.0f}s)  [{n_done}]", flush=True)
                # Written after every arm: a preemption then costs one arm.
                C.save_json(out_path, res)

    # --- per-stream summary: the "which computations each stream requires" number ----
    summary = {}
    for stream in streams:
        n_blocks = L_s if stream == "state" else L_p
        for kind in kinds:
            tags = [f"{stream}_{i:02d}_{kind}" for i in range(n_blocks)]
            ds = [res["lesions"][t]["delta"] for t in tags if t in res["lesions"]]
            if not ds:
                continue
            summary[f"{stream}_{kind}"] = dict(
                total_delta=round(sum(ds), 6),
                max_delta=round(max(ds), 6),
                argmax_block=int(max(range(len(ds)), key=lambda i: ds[i])),
                min_delta=round(min(ds), 6),
                argmin_block=int(min(range(len(ds)), key=lambda i: ds[i])),
                n_below_1e_2=int(sum(1 for d in ds if d < 0.01)),
                blocks_below_1e_2=[i for i, d in enumerate(ds) if d < 0.01],
            )
    res["summary"] = summary
    C.save_json(out_path, res)
    print("\nA12 done.", flush=True)


if __name__ == "__main__":
    main()
