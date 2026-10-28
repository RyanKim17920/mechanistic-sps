#!/usr/bin/env python
"""A18 -- decompose the STATE tower's gradient by READ EDGE.

THE IDENTITY
------------
In the clean two-tower family (``_forward_towers``, core.py:1068) the state tower is run
to completion and the ONLY place the loss ever touches a state parameter is the read:

    core.py:1087   lvl = self.read_levels[i]
    core.py:1089   k_s, v_s = block.read_state(res_levels[lvl], freqs_cis)

(``read_levels`` built at core.py:835 from ``read_level``, core.py:260-291: ``post`` ->
f(i)=i+1, ``final`` -> f(i)=L_s; ``PredBlock.read_state`` at core.py:540-548.)

So with ``u_k`` the tensor handed to ``read_state`` at pred block k,

    dL/dtheta_state = sum_k J_k^T (dL/du_k)  =:  sum_k g_k                        (*)

exactly -- the state tower has no second consumer.  A18 measures every ``g_k``.

HOW ONE EDGE IS ISOLATED (the forward is BIT-IDENTICAL)
------------------------------------------------------
``PredBlock.read_state`` is tapped so that its input is first wrapped in
``x.view_as(x)`` -- a pure view, zero arithmetic, so the forward VALUES are unchanged --
and the wrapper is recorded.  The wrapper matters: under ``read_map='final'`` all twelve
edges read the SAME tensor object (core.py:1082), and without a distinct autograd node
per edge there is nothing to differentiate with respect to.  Then

  1. one forward;
  2. ``autograd.grad(loss, reads)``            -> the twelve ``dL/du_k``;
  3. per k, ``autograd.grad(u_k, state_params, grad_outputs=dL/du_k)`` -> ``g_k``.
     The pred tower is never traversed, so each of these costs ~half a model backward.
  4. a SEPARATE, UNPATCHED forward + ``loss.backward()`` -> ``g_total`` (and the
     bit-identity check on the loss).

GATES (both mandatory, both aborting)
-------------------------------------
  GATE 1a (EXACTNESS, the one that can catch a lost path).  A few micro-batches in TRUE
    fp32 -- TF32 off, ``float32_matmul_precision('highest')`` -- where
    ``max_W ||sum_k g_k - g_total|| / ||g_total||`` must be <= 1e-4.  If any path from a
    state parameter to the loss did not go through a read, this is wrong by O(1) and no
    precision fixes it.  Abort on failure.
  GATE 1b (additivity under the PRODUCTION arithmetic).  The measurement itself runs in
    the bf16 autocast training used, where the two paths round differently: the plain
    backward adds the twelve edge gradients into the state RESIDUAL and propagates once,
    the decomposition propagates each separately and adds at the PARAMETER.  On the
    RMSNorm gains -- 768-element sums over 13 M tokens with near-total cancellation --
    that shows up as a per-tensor residual of order 1e-2 with a run-to-run repro floor of
    exactly 0, i.e. it is arithmetic and not noise.  The bound that matters for the
    reported cosines is therefore the NORM-WEIGHTED GLOBAL residual, which must be
    <= ``gate_tol`` (A16's bar); per-tensor offenders are listed, not hidden.
  GATE 2 (bit-identity): the mean loss under the ``view_as`` tap equals the unpatched
    mean loss bit-for-bit, per micro-batch.  ``view_as`` cannot change a number; if it
    did, the tap is not what this docstring says it is.

WHAT IS REPORTED  (per state block i, per parameter tensor)
-----------------------------------------------------------
    g_read(i)   = g_k for the edge whose read level is i+1, i.e. the edge that reads
                  the output of state block i.  Under ``post`` that is k = i.
    g_stack(i)  = sum over edges with read level > i+1 -- everything that reaches block i
                  by passing THROUGH the rest of the state stack.
    cos(g_stack(i), g_read(i))  <- the headline: does serving the direct reader of level
                  i+1 agree with serving the deeper readers?
plus norms, ||g_read||/||g_stack||, the fraction of tensors per block with cos < 0, the
full edge-pair cosine matrix per block, cos(g_k, g_total), and the batch-split (disjoint
half) control that is the gradient-noise floor every cosine must be read against.

Under ``read_map='final'`` every edge reads level L_s, so there is no direct reader of
level i+1 for any i < L_s and the stack-vs-read split is UNDEFINED BY CONSTRUCTION -- the
JSON says so in ``stack_read_defined``/``note`` rather than reporting a number.  For that
arm the per-edge cosines (edge vs edge, edge vs total) are still measured and are what the
figure shows; they are NOT comparable to the interleaved arm's pair cosines, because there
they all pass through one shared Jacobian.

Usage
-----
    scripts/run/eval_shim.sh scripts/analysis/a18_read_edge_grad.py --run s_two_tower_w0_equal_20b
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

ANALYSIS = "a18_read_edge_grad"
GATE_TOL = 1e-2          # A16's bar for the same additivity gate


# ======================================================================================
# measurement
# ======================================================================================
def run_measurement(args):
    import torch
    import common as C

    # common.py disables autograd process-wide (every other analysis here is eval-only).
    # This one IS a gradient measurement; re-enable explicitly rather than silently.
    torch.set_grad_enabled(True)

    from modeling.models.two_tower.core import PredBlock

    C.assert_node_local_triton()
    run = C.resolve(args.run)
    torch.manual_seed(0)

    if getattr(args, "untrained", False):
        # INIT CONTROL: same config, fresh random init (C.load_untrained, as A16 --untrained).
        model, cfg, family, ckpt = C.load_untrained(run, seed=args.seed)
    else:
        model, cfg, family, ckpt = C.load(run)
    assert family == "two_tower", f"{run}: A18 is a two-tower analysis, got {family}"
    model.eval()
    if args.attn_backend:
        model.config.attn_backend = args.attn_backend
    assert str(model.read_source) == "pred_proj", (
        f"{run}: A18 taps PredBlock.read_state, which requires read_source=pred_proj")
    assert not getattr(model, "share_ffn_across_towers", False), (
        f"{run}: shared-FFN arms do not go through _forward_towers")
    assert not bool(model.config.tie_lm_head), (
        f"{run}: tie_lm_head makes wte a pred-tower consumer too; the identity breaks")

    for p in model.parameters():
        p.requires_grad_(True)

    read_levels = [int(v) for v in model.read_levels]
    n_edge = len(read_levels)
    L_s = int(model.config.state_n_layer)

    # ---------------------------------------------------------------- state parameters
    # The state tower's parameters are the embedding (its input; untied, so state-only)
    # and the state blocks.  Nothing else in the graph is upstream of a read.
    sparams = [("transformer.wte", model.transformer.wte.weight)]
    for i, sb in enumerate(model.transformer.state_h):
        for sub, p in sb.named_parameters():
            sparams.append((f"state_h.{i}.{sub}", p))
    pname_list = [n for n, _ in sparams]
    plist = [p for _, p in sparams]
    pshape = {n: tuple(p.shape) for n, p in sparams}

    # reporting specs: (name, kind, block, param_name, row slice or None)
    inner = model.n_head_state * model.head_dim
    specs = [("transformer.wte", "embed", -1, "transformer.wte", None)]
    for i in range(L_s):
        c = f"state_h.{i}"
        specs += [
            (f"{c}.c_attn[q]", "attn_q", i, f"{c}.c_attn.weight", slice(0, inner)),
            (f"{c}.c_attn[k]", "attn_k", i, f"{c}.c_attn.weight", slice(inner, 2 * inner)),
            (f"{c}.c_attn[v]", "attn_v", i, f"{c}.c_attn.weight", slice(2 * inner, 3 * inner)),
            (f"{c}.c_proj", "attn_out", i, f"{c}.c_proj.weight", None),
            (f"{c}.mlp.gate_proj", "ffn_gate", i, f"{c}.mlp.gate_proj.weight", None),
            (f"{c}.mlp.up_proj", "ffn_up", i, f"{c}.mlp.up_proj.weight", None),
            (f"{c}.mlp.down_proj", "ffn_down", i, f"{c}.mlp.down_proj.weight", None),
            (f"{c}.attention_norm", "norm_attn", i, f"{c}.attention_norm.weight", None),
            (f"{c}.mlp_norm", "norm_mlp", i, f"{c}.mlp_norm.weight", None),
        ]
    specs = [s for s in specs if s[3] in pshape]

    # ---------------------------------------------------------------------- the tap
    READS: list = []
    _ORIG_READ = PredBlock.read_state

    def _read_state_tapped(self, state_residual_BxTxC, freqs_cis):
        # view_as is a pure view: zero arithmetic, forward bit-identical.  It exists only
        # to give each read edge its OWN autograd node -- essential under read_map=final,
        # where all edges are handed the same tensor object.
        u = state_residual_BxTxC.view_as(state_residual_BxTxC)
        READS.append(u)
        return _ORIG_READ(self, u, freqs_cis)

    # ------------------------------------------------------------------------- batches
    data = C.val_memmap(cfg)
    starts = C.seq_starts(data, args.n_seq)
    n_seq = len(starts)
    assert n_seq >= 2, f"need >= 2 sequences, got {n_seq}"
    mb = args.micro_batch
    n_mb = (n_seq + mb - 1) // mb
    half_of = lambda bi: 0 if bi < n_mb // 2 else 1  # noqa: E731

    # ------------------------------------------------------------------- accumulators
    # fp32 on the GPU: 13 (12 edges + total) x 2 halves x ~130 M params x 4 B = 13.5 GB,
    # which the H100 holds alongside one mb=4 forward.  Allocated lazily, so the edges
    # that are structurally disconnected under read_map=post cost nothing.
    acc: dict = {}

    def add(tag, half, name, g):
        key = (tag, half)
        d = acc.setdefault(key, {})
        flat = g.reshape(-1).float()
        if name in d:
            d[name] += flat
        else:
            d[name] = flat.clone()

    def batch_at(bi):
        b = starts[bi * mb:(bi + 1) * mb]
        X = torch.stack([torch.from_numpy(data[j:j + C.BLOCK].astype(np.int64))
                         for j in b]).cuda()
        Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + C.BLOCK].astype(np.int64))
                         for j in b]).cuda()
        return X, Y

    def one_micro_batch(X, Y, sink, use_ac, repro):
        """-> (loss_tapped, loss_plain).  ``sink(tag, name, grad)`` receives everything."""
        # ---- tapped pass: the per-edge gradients
        READS.clear()
        PredBlock.read_state = _read_state_tapped
        try:
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                _, loss, _ = model(X, Y)
            assert len(READS) == n_edge, f"tapped {len(READS)} reads, expected {n_edge}"
            l_tap = float(loss.detach())
            gu = torch.autograd.grad(loss, READS, retain_graph=True)
            for k in range(n_edge):
                gk = torch.autograd.grad(READS[k], plist, grad_outputs=gu[k],
                                         retain_graph=True, allow_unused=True)
                for name, g in zip(pname_list, gk):
                    if g is not None:
                        sink(k, name, g)
                del gk
            del gu, loss
            READS.clear()
        finally:
            PredBlock.read_state = _ORIG_READ

        # ---- plain pass: g_total, and GATE 2 on the loss
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            _, loss2, _ = model(X, Y)
        l_plain = float(loss2.detach())
        loss2.backward()
        for name, p in sparams:
            if p.grad is not None:
                sink("total", name, p.grad)
        model.zero_grad(set_to_none=True)
        del loss2

        # ---- REPRO pass: a SECOND identical plain backward.  Not a mode -- it measures
        # the run-to-run numerical floor of this pipeline.
        if repro:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                _, loss3, _ = model(X, Y)
            loss3.backward()
            for name, p in sparams:
                if p.grad is not None:
                    sink("total2", name, p.grad)
            model.zero_grad(set_to_none=True)
            del loss3
        return l_tap, l_plain

    # ------------------------------------------------ GATE 1a: the EXACTNESS probe
    # The production measurement runs in the bf16 autocast training used, where the
    # decomposed path (propagate each edge separately, add at the parameter) and the
    # plain path (add the edges into the state residual, propagate once) round
    # differently -- visibly so on the RMSNorm gains, whose gradient is a 13 M-token sum
    # with near-total cancellation.  That is arithmetic, not a lost path, and the way to
    # SHOW it is to take the arithmetic away: a few micro-batches in true fp32 (TF32 off,
    # matmul precision 'highest').  If a path were missing, this probe would be wrong by
    # O(1) no matter the precision.
    probe = dict(n_micro_batches=0)
    if args.gate_mb > 0:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_prec = torch.get_float32_matmul_precision()
        # core.py:248/993-997 casts q/k/v to `attn_dtype` (bfloat16) INSIDE the model, not
        # via autocast, so disabling autocast alone leaves attention in bf16 and the probe
        # is not an fp32 probe at all (measured: residual stuck at 3e-3 on c_attn).
        prev_attn_dtype = model.attn_dtype
        model.attn_dtype = None
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        pacc: dict = {}

        def psink(tag, name, g):
            d = pacc.setdefault(tag, {})
            f = g.reshape(-1).double()
            d[name] = d[name] + f if name in d else f.clone()
        try:
            for bi in range(min(args.gate_mb, n_mb)):
                X, Y = batch_at(bi)
                one_micro_batch(X, Y, psink, use_ac=False, repro=False)
                probe["n_micro_batches"] += 1
                del X, Y
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.backends.cudnn.allow_tf32 = prev_tf32
            torch.set_float32_matmul_precision(prev_prec)
            model.attn_dtype = prev_attn_dtype
        pworst, pworst_name, prows = 0.0, None, []
        for name, _p in sparams:
            gt = pacc.get("total", {}).get(name)
            if gt is None:
                continue
            s = torch.zeros_like(gt)
            scale = 0.0
            for k in range(n_edge):
                v = pacc.get(k, {}).get(name)
                if v is not None:
                    s = s + v
                    scale += float(v.norm())
            den = float(gt.norm())
            if den == 0:
                continue
            r = float((s - gt).norm()) / den
            prows.append(dict(name=name, rel_err=r, norm_total=den, sum_norm_k=scale,
                              cancellation=scale / den))
            if r > pworst:
                pworst, pworst_name = r, name
        prows.sort(key=lambda d: -d["rel_err"])
        for d in [r for r in prows if "norm" not in r["name"]][:8] + prows[:4]:
            print(f"  probe {d['name']:38s} rel={d['rel_err']:.3e} "
                  f"|gtot|={d['norm_total']:.4g} sum|gk|={d['sum_norm_k']:.4g} "
                  f"cancel={d['cancellation']:.3g}", flush=True)
        probe["per_tensor_top"] = prows[:20]
        probe.update(max_rel_err=pworst, worst_tensor=pworst_name, tol=args.probe_tol,
                     passed=bool(pworst <= args.probe_tol),
                     criterion=("in true fp32 (TF32 off, matmul precision 'highest'), "
                                "max_W ||sum_k g_k - g_total|| / ||g_total|| <= probe_tol"))
        del pacc
        torch.cuda.empty_cache()
        print(f"GATE1a exactness (fp32, {probe['n_micro_batches']} mb) "
              f"max_rel_err={pworst:.3e} worst={pworst_name} tol={args.probe_tol:.1e} "
              f"{'PASS' if probe['passed'] else 'FAIL'}", flush=True)
        if not probe["passed"]:
            print("ABORT: the decomposition is not exact even in fp32 -- a path is lost.",
                  flush=True)
            C.save_json(args.out or C.result_path(ANALYSIS, run),
                        dict(analysis=ANALYSIS, run=run, checkpoint=str(ckpt),
                             ABORTED="GATE1a failed", gate=dict(gate1a_exactness=probe)))
            sys.exit(2)

    # ------------------------------------------------------------------ the measurement
    losses_tapped, losses_plain, loss_bit_mismatch = [], [], 0
    import time
    t0 = time.time()

    for bi in range(n_mb):
        X, Y = batch_at(bi)
        h = half_of(bi)
        l_tap, l_plain = one_micro_batch(X, Y, lambda t, n, g: add(t, h, n, g),
                                         use_ac=not args.fp32, repro=not args.skip_repro)
        del X, Y
        losses_tapped.append(l_tap)
        losses_plain.append(l_plain)
        if l_tap != l_plain:
            loss_bit_mismatch += 1
        if bi % 50 == 0:
            print(f"  mb {bi}/{n_mb} loss={l_plain:.6f} t={time.time() - t0:.0f}s", flush=True)

    seconds = time.time() - t0
    gate2_ok = loss_bit_mismatch == 0
    print(f"GATE2 bit-identical loss: {n_mb - loss_bit_mismatch}/{n_mb} micro-batches "
          f"{'PASS' if gate2_ok else 'FAIL'}", flush=True)

    # ------------------------------------------------------------------- vector access
    def vec(tag, name, sl, half=None):
        """fp64 flat view of one (edge, tensor) gradient; None if structurally absent."""
        if half is None:
            a = acc.get((tag, 0), {}).get(name)
            bb = acc.get((tag, 1), {}).get(name)
            if a is None and bb is None:
                return None
            v = (a if bb is None else (bb if a is None else a + bb))
        else:
            v = acc.get((tag, half), {}).get(name)
            if v is None:
                return None
        v = v.double()
        if sl is not None:
            v = v.view(pshape[name])[sl].reshape(-1)
        return v

    def cos(a, b):
        if a is None or b is None:
            return None
        na, nb = float(a.norm()), float(b.norm())
        if na == 0.0 or nb == 0.0:
            return None
        return float(torch.dot(a, b) / (na * nb))

    def zeros_like_of(name, sl):
        n = int(np.prod(pshape[name])) if sl is None else \
            int(np.prod(pshape[name][1:])) * (sl.stop - sl.start)
        return torch.zeros(n, dtype=torch.float64, device="cuda")

    # ------------------------------------------------------- GATE 1: sum_k g_k == g_total
    gate_rows, worst, worst_repro, worst_ratio = [], 0.0, 0.0, 0.0
    num2 = den2 = 0.0
    bad = []
    for name, kind, blk, pn, sl in specs:
        gt = vec("total", pn, sl)
        if gt is None:
            continue
        s = zeros_like_of(pn, sl)
        scale = 0.0
        for k in range(n_edge):
            gk = vec(k, pn, sl)
            if gk is not None:
                s = s + gk
                scale += float(gk.norm())
        den = float(gt.norm())
        err = float((s - gt).norm())
        rel = err / den if den > 0 else 0.0
        # The two paths differ ONLY in where the twelve edge contributions are summed:
        # the plain backward adds them into the state RESIDUAL and propagates once; the
        # decomposition propagates each separately and adds at the PARAMETER.  Both are
        # exact in exact arithmetic, so the residual is round-off on terms of size
        # ``sum_k ||g_k||`` -- and on the RMSNorm gains that sum is 10-100x ||g_total||
        # (the edges very nearly cancel), which inflates ``rel_err`` by exactly that
        # factor without any path being lost.  ``rel_err_scaled`` divides the inflation
        # out and is the quantity that is genuinely at the arithmetic floor.
        rel_scaled = err / scale if scale > 0 else 0.0
        g2 = vec("total2", pn, sl)
        repro = (float((g2 - gt).norm()) / den) if (g2 is not None and den > 0) else None
        num2 += err ** 2
        den2 += den ** 2
        worst = max(worst, rel)
        worst_ratio = max(worst_ratio, rel_scaled)
        if repro is not None:
            worst_repro = max(worst_repro, repro)
        if rel_scaled > args.gate_tol:
            bad.append(name)
        gate_rows.append(dict(name=name, kind=kind, block=blk, rel_err=rel,
                              rel_err_scaled=rel_scaled, repro_rel_err=repro,
                              norm_total=den, sum_norm_k=scale,
                              cancellation=(scale / den) if den > 0 else None))
    global_rel = (num2 ** 0.5) / (den2 ** 0.5) if den2 > 0 else 0.0
    # GATE 1a has already shown the decomposition is EXACT (fp32 residual ~1e-6), so what
    # GATE 1 must establish is only that bf16 arithmetic has not corrupted the reported
    # numbers.  The reported cosines are dominated by the weight matrices -- the
    # norm-weighted GLOBAL residual is therefore the bound that matters; a per-tensor
    # `rel_err` above tol is flagged (`tensors_over_tol`) and excluded from nothing, but
    # it is a statement about that tensor's cancellation, not about the method.
    gate1_ok = bool(global_rel <= args.gate_tol)
    print(f"GATE1 additivity max_rel_err={worst:.3e} max_rel_err_scaled={worst_ratio:.3e} "
          f"repro_floor={worst_repro:.3e} global_rel_err={global_rel:.3e} "
          f"n_over_tol={len(bad)} tol={args.gate_tol:.1e} "
          f"{'PASS' if gate1_ok else 'FAIL'}", flush=True)

    read_map = str(model.config.read_map)
    out = dict(
        analysis=ANALYSIS, run=run, checkpoint=str(ckpt), read_map=read_map,
        read_levels=read_levels, state_n_layer=L_s, n_edge=n_edge,
        n_seq=n_seq, micro_batch=mb, n_micro_batches=n_mb, tokens=n_seq * C.BLOCK,
        seconds=seconds, fp32=bool(args.fp32),
        loss_by_mode=dict(tapped=float(np.mean(losses_tapped)),
                          plain=float(np.mean(losses_plain))),
        gate=dict(
            gate1a_exactness=probe,
            gate1_additivity=dict(max_rel_err=worst, max_rel_err_scaled=worst_ratio,
                                  max_repro_rel_err=worst_repro,
                                  global_rel_err=global_rel, tol=args.gate_tol,
                                  n_tensors_over_tol=len(bad), tensors_over_tol=bad,
                                  passed=gate1_ok,
                                  criterion=("the norm-weighted GLOBAL residual "
                                             "sqrt(sum_W ||sum_k g_k - g_total||^2) / "
                                             "sqrt(sum_W ||g_total||^2) <= tol, given "
                                             "GATE 1a's fp32 proof that the "
                                             "decomposition itself is exact"),
                                  per_tensor=gate_rows),
            gate2_bit_identity=dict(n_micro_batches=n_mb, n_mismatch=loss_bit_mismatch,
                                    passed=gate2_ok,
                                    criterion="float(loss_tapped) == float(loss_plain) "
                                              "for every micro-batch"),
            passed=bool(gate1_ok and gate2_ok and probe.get("passed", True))),
    )
    if not out["gate"]["passed"]:
        out["ABORTED"] = ("gate failed; no cosine from this run may be interpreted "
                          "(see gate.gate1_additivity / gate.gate2_bit_identity)")
        C.save_json(args.out or C.result_path(ANALYSIS, run), out)
        print(json.dumps(out["gate"]["gate1_additivity"]["passed"]), flush=True)
        sys.exit(2)

    # ------------------------------------------------------------------ the measurement
    # Under 'post', f(k) = k+1, so the direct reader of state block i's output is edge i
    # and the stack is edges k > i.  Under 'final', f(k) = L_s for every k: there is no
    # direct reader of level i+1 for i < L_s-1, and at i = L_s-1 there are TWELVE of them,
    # so "the" direct read does not exist.  The split is undefined by construction.
    stack_read_defined = (read_map == "post")

    def edges_at_level(lvl):
        return [k for k in range(n_edge) if read_levels[k] == lvl]

    tensors, pairs, per_block = [], [], []
    for i in range(-1, L_s):
        sel = [s for s in specs if s[2] == i]
        if not sel:
            continue
        direct = edges_at_level(i + 1) if i >= 0 else []
        stack_ks = [k for k in range(n_edge) if read_levels[k] > i + 1] if i >= 0 else \
            list(range(n_edge))
        blk_cos, blk_ratio = [], []
        for name, kind, blk, pn, sl in sel:
            gt = vec("total", pn, sl)
            if gt is None:
                continue
            per_edge = {k: vec(k, pn, sl) for k in range(n_edge)}
            live = [k for k in range(n_edge) if per_edge[k] is not None]

            g_read = None
            if stack_read_defined and len(direct) == 1 and per_edge[direct[0]] is not None:
                g_read = per_edge[direct[0]]
            g_stack = None
            for k in stack_ks:
                if per_edge[k] is None:
                    continue
                g_stack = per_edge[k] if g_stack is None else g_stack + per_edge[k]

            c_sr = cos(g_stack, g_read) if (stack_read_defined and g_read is not None) else None
            row = dict(
                name=name, kind=kind, block=blk,
                edges_live=live,
                direct_edge=(direct[0] if len(direct) == 1 else None),
                stack_edges=[k for k in stack_ks if per_edge[k] is not None],
                norm_total=float(gt.norm()),
                norm_read=(float(g_read.norm()) if g_read is not None else None),
                norm_stack=(float(g_stack.norm()) if g_stack is not None else None),
                ratio_read_over_stack=(
                    float(g_read.norm() / g_stack.norm())
                    if (g_read is not None and g_stack is not None
                        and float(g_stack.norm()) > 0) else None),
                cos_stack_vs_read=c_sr,
                cos_k_vs_total={str(k): cos(per_edge[k], gt) for k in live},
                norm_k={str(k): float(per_edge[k].norm()) for k in live},
                # CONTROL: same edge, disjoint halves of the val set.  This is the
                # gradient-noise floor -- the number every cosine above is read against.
                control_batch_split_total=cos(vec("total", pn, sl, 0),
                                              vec("total", pn, sl, 1)),
                control_batch_split_k={
                    str(k): cos(vec(k, pn, sl, 0), vec(k, pn, sl, 1)) for k in live},
            )
            if c_sr is not None:
                # cross-half version of the same cosine: kills "the sign is noise
                # cancellation within one half".
                sA = None
                for k in stack_ks:
                    v = vec(k, pn, sl, 0)
                    if v is not None:
                        sA = v if sA is None else sA + v
                row["control_stack_read_cross_half"] = cos(sA, vec(direct[0], pn, sl, 1))
                blk_cos.append(c_sr)
                if row["ratio_read_over_stack"] is not None:
                    blk_ratio.append(row["ratio_read_over_stack"])
            tensors.append(row)

            for a in range(len(live)):
                for bnd in range(a + 1, len(live)):
                    k, l = live[a], live[bnd]
                    pairs.append(dict(name=name, kind=kind, block=blk, k=k, l=l,
                                      cos=cos(per_edge[k], per_edge[l])))
            del per_edge

        pb = dict(block=i, n_tensors=len(sel),
                  stack_read_defined=bool(stack_read_defined and len(direct) == 1 and i >= 0),
                  direct_edge=(direct[0] if len(direct) == 1 else None),
                  n_stack_edges=len(stack_ks))
        if blk_cos:
            pb.update(median_cos_stack_read=float(np.median(blk_cos)),
                      mean_cos_stack_read=float(np.mean(blk_cos)),
                      min_cos_stack_read=float(np.min(blk_cos)),
                      max_cos_stack_read=float(np.max(blk_cos)),
                      p10_cos_stack_read=float(np.percentile(blk_cos, 10)),
                      frac_neg=float(np.mean(np.asarray(blk_cos) < 0)),
                      median_ratio_read_over_stack=(float(np.median(blk_ratio))
                                                    if blk_ratio else None))
        else:
            if i >= 0 and stack_read_defined and len(direct) == 1 and not stack_ks:
                pb["note"] = (f"no stack here by construction: block {i} is the last state "
                              "block, so no edge reads a level deeper than its output")
            elif i < 0:
                pb["note"] = ("the embedding is upstream of every read; it has no 'own "
                              "level' and so no direct reader")
            else:
                pb["note"] = ("stack-vs-read is UNDEFINED BY CONSTRUCTION here: no single "
                              f"edge reads level {i + 1} under read_map={read_map} "
                              f"(read_levels={sorted(set(read_levels))})")
        # the per-edge picture, which IS defined in both arms
        ec = [r["cos_k_vs_total"] for r in tensors if r["block"] == i]
        flat = [v for d in ec for v in d.values() if v is not None]
        if flat:
            pb.update(median_cos_k_vs_total=float(np.median(flat)),
                      min_cos_k_vs_total=float(np.min(flat)),
                      frac_neg_cos_k_vs_total=float(np.mean(np.asarray(flat) < 0)),
                      n_cos_k_vs_total=len(flat))
        pp = [p["cos"] for p in pairs if p["block"] == i and p["cos"] is not None]
        if pp:
            pb.update(n_pairs=len(pp), median_cos_pair=float(np.median(pp)),
                      min_cos_pair=float(np.min(pp)),
                      p10_cos_pair=float(np.percentile(pp, 10)),
                      frac_neg_pair=float(np.mean(np.asarray(pp) < 0)))
        per_block.append(pb)

    out["stack_read_defined"] = bool(stack_read_defined)
    out["note"] = (
        "read_map=post: f(k)=k+1, so state block i's output has exactly one direct reader "
        "(edge i) and the stack is edges k>i; cos(g_stack, g_read) is defined at every i."
        if stack_read_defined else
        "read_map=final: f(k)=L_s for every k, so NO state level below L_s has a direct "
        "reader and at L_s there are n_edge of them. cos(g_stack, g_read) is UNDEFINED BY "
        "CONSTRUCTION for every i and is reported as null, not as a number. The per-edge "
        "quantities (cos_k_vs_total, the edge-pair matrix) are defined but pass through "
        "ONE shared Jacobian, so they are not comparable to the interleaved arm's.")
    out["tensors"] = tensors
    out["pairs"] = pairs
    out["per_block"] = per_block

    allc = [r["cos_stack_vs_read"] for r in tensors if r["cos_stack_vs_read"] is not None]
    allp = [p["cos"] for p in pairs if p["cos"] is not None]
    allt = [v for r in tensors for v in r["cos_k_vs_total"].values() if v is not None]
    allctl = [r["control_batch_split_total"] for r in tensors
              if r["control_batch_split_total"] is not None]

    def agg(v):
        if not v:
            return None
        v = np.asarray(v, dtype=float)
        return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                    min=float(v.min()), max=float(v.max()),
                    p10=float(np.percentile(v, 10)), frac_neg=float((v < 0).mean()))

    out["aggregate"] = dict(cos_stack_vs_read=agg(allc), cos_pair=agg(allp),
                            cos_k_vs_total=agg(allt),
                            control_batch_split_total=agg(allctl))
    path = args.out or C.result_path(ANALYSIS, run)
    C.save_json(path, out)
    print(json.dumps(out["aggregate"], indent=2), flush=True)
    print(json.dumps(out["per_block"], indent=2)[:6000], flush=True)


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-seq", type=int, default=100000,
                    help="capped at the val set's 3198 non-overlapping sequences")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--gate-tol", type=float, default=GATE_TOL)
    ap.add_argument("--attn-backend", default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--gate-mb", type=int, default=2,
                    help="GATE 1a: micro-batches for the true-fp32 exactness probe")
    ap.add_argument("--probe-tol", type=float, default=1e-4)
    ap.add_argument("--skip-repro", action="store_true",
                    help="skip the second identical plain backward (the repro floor)")
    ap.add_argument("--untrained", action="store_true",
                    help="INIT CONTROL: same measurement on a fresh random init of the config")
    ap.add_argument("--seed", type=int, default=1234, help="init seed for --untrained")
    ap.add_argument("--out", default=None)
    run_measurement(ap.parse_args())


if __name__ == "__main__":
    main()
