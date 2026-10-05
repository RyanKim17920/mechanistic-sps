#!/usr/bin/env python
"""A16 -- do the two streams pull a SHARED weight in uncorrelated directions?

The sharing axis of this study (untied two-tower -> tied attention -> tied attention +
shared gated FFN) costs progressively more NLL.  The thesis-critical question behind that
axis is mechanistic, not empirical: when ONE tensor serves both the memory (state) stream
and the readout (pred) stream, are the two streams' gradient demands on that tensor
*aligned* (so sharing is nearly free and the cost must come from capacity) or
*uncorrelated* (so sharing forces one tensor to serve two objectives that carry no
information about each other, and separate parameters are justified from first
principles)?

WHAT IS MEASURED
----------------
For every shared parameter tensor W we decompose the total training gradient

    g_total(W) = dL/dW

into the part that arrives through the STATE tower's OCCURRENCE of W and the part that
arrives through the PREDICTION tower's OCCURRENCE of W, and report cos(g_state, g_pred).

The decomposition is the ordinary multivariate chain rule over occurrences: if W appears
at sites s_1..s_n, then dL/dW = sum_i dL/dW_i where W_i is an independent variable bound
to site i alone.  We realise dL/dW_i by running the model with the LIVE parameter at the
sites of interest and a DETACHED COPY (``W.detach()``, same numbers, no graph edge) at
every other site.  This is exact and additive because a detached copy still propagates
gradient to its INPUT ACTIVATIONS -- no activation edge is ever cut, so cross-stream
paths like "state occurrence -> state residual -> the pred tower's read of it -> loss"
are retained in g_state exactly as they are in g_total.  ``g_state + g_pred == g_total``
is therefore an identity, and we check it numerically as a correctness GATE before any
cosine is reported (see ``--gate-tol``).

The fused shared FFN needs one extra step.  ``SharedGatedMLP`` evaluates the FFN ONCE and
hands the single output ``f`` to both streams, so ``f``'s weights have a single
occurrence and a site split does not exist as written.  We restore one by evaluating the
branch twice -- numerically identical, but with the state-consumed branch and the
pred-consumed branch reading separate (live / detached) copies of the same weights.  That
turns "which stream CONSUMES this FFN evaluation" into a genuine occurrence split, which
is additive for the same reason as above (again, no activation edge is cut: both branches
still backpropagate into ``h_state`` and ``h_pred``).  ``gate_state`` and ``gate_pred``
are consumed by exactly one stream each, so their opposite-stream component is
identically zero and their cosine is undefined; they are reported as stream-exclusive.

WHICH TENSORS ARE SHARED
------------------------
  * ``tie_attn_across_towers``  -- state block i's fused ``c_attn`` (rows q | k | v) and
    its ``c_proj``.  The state occurrence is ``StateBlock.qkv`` / ``StateBlock.finish_attn``;
    the pred occurrence is ``PredBlock.query`` (q rows), ``PredBlock.read_state`` (k, v
    rows, applied to the STATE residual at level f(i)) and ``PredBlock.finish_attn``.
    ``read_state`` is attributed to the PRED tower because it is the readout block's own
    operation, even though its input is the state residual.
  * ``share_ffn_across_towers`` -- ``shared_mlp[i].mlp.{gate,up,down}_proj``, the pooling
    gate ``gate_in``, and the two output gates.
  * ``tie_lm_head`` -- ``transformer.wte.weight IS lm_head.weight``, so the state tower's
    input embedding and the readout's output head are ONE tensor.  (Only the AFSPS arm
    sets this; the tied-attention and untied arms leave it false.)

CONTROLS
--------
  1. INITIALISATION (``--untrained``).  The same measurement on a fresh random init of
     the same config.  If the trained cosine is ~0 but the init cosine is ~0 too, the
     measurement says nothing about training.  Run this and report it.
  2. BATCH SPLIT (always reported).  The 32 val sequences are split in half and
     ``cos(g_state[A], g_state[B])`` -- SAME stream, disjoint data -- is reported per
     tensor.  This is the gradient-noise floor of the estimate: it is the number the
     cross-stream cosine has to be compared against.
  3. UNTIED REFERENCE (``--untied``, on ``s_two_tower_w0_equal_20b``).  There the towers
     own separate tensors, so there is no split to make; instead we report the cosine
     between state block i's attention gradient and pred block i's corresponding
     (same-shaped, different) tensor's gradient.  That is what "two unrelated tensors"
     looks like in these units.

JOINT FAMILY (tied SPS, ``--run s_sps_w64_*``)
----------------------------------------------
One weight-shared stack over interleaved state (2p) / pred (2p+1) slots.  The split is
by ROW-occurrence: each slot's use of W is an exact leaf copy; k/v are attributed to the
stream of the query that READS them (the two-tower tied-attn convention), with the
producer split and a 2x2 k/v table also reported.  Full statement: ``run_joint``.

Usage
-----
    scripts/run/eval_shim.sh scripts/analysis/a16_grad_orthogonality.py --run <run|role>
    scripts/run/eval_shim.sh scripts/analysis/a16_grad_orthogonality.py --run <run> --untrained
    scripts/run/eval_shim.sh scripts/analysis/a16_grad_orthogonality.py --run s_two_tower_w0_equal_20b --untied
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import common as C  # noqa: E402  (also fixes sys.path for src/ and scripts/dualsps)

# common.py disables autograd process-wide -- every OTHER analysis in this directory is
# eval-only and must never build a graph.  This one is the exception: the whole
# measurement IS a gradient.  Re-enable explicitly rather than silently.
torch.set_grad_enabled(True)

from modeling.models.two_tower.core import SharedGatedMLP  # noqa: E402

ANALYSIS = "a16_grad_orthogonality"
GATE_TOL = 1e-3
COS_TOL = 0.02


# ======================================================================================
# detached stand-ins
# ======================================================================================
class _DetachedLinear:
    """Callable stand-in for an ``nn.Linear`` whose weight/bias are detached.

    Same numbers, no graph edge into the parameter -- but the gradient still flows into
    the INPUT, which is what keeps the occurrence decomposition additive.
    """

    def __init__(self, lin):
        self.weight = lin.weight.detach()
        self.bias = None if lin.bias is None else lin.bias.detach()

    def __call__(self, x):
        return F.linear(x, self.weight, self.bias)


class _DetachedEmbedding:
    def __init__(self, emb):
        self.weight = emb.weight.detach()

    def __call__(self, idx):
        return F.embedding(idx, self.weight)


class _DetachedMLP:
    """``TowerMLP`` with detached weights (plain tied FFN, ``tie_ffn_across_towers``)."""

    def __init__(self, mlp):
        self.mlp = mlp

    def __call__(self, x):
        return _mlp_apply(self.mlp, x, True)


class _DetachedNorm:
    """``TowerRMSNorm`` with a detached weight (``tie_norms_across_towers``)."""

    def __init__(self, norm):
        self.weight = norm.weight.detach()
        self.eps = norm.eps

    def __call__(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def _lin(layer, x, det: bool):
    w = layer.weight.detach() if det else layer.weight
    b = None if layer.bias is None else (layer.bias.detach() if det else layer.bias)
    return F.linear(x, w, b)


def _mlp_apply(mlp, x, det: bool):
    """``TowerMLP.forward`` with optionally-detached weights (dropout is 0 / eval)."""
    if mlp.intermediate == 0:
        return torch.zeros_like(x)
    return _lin(mlp.down_proj, F.silu(_lin(mlp.gate_proj, x, det)) * _lin(mlp.up_proj, x, det), det)


# ======================================================================================
# the shared gated FFN, split by CONSUMING stream
# ======================================================================================
_FFN_MODE = {"m": "total"}
_ORIG_SHARED_FORWARD = SharedGatedMLP.forward


def _shared_gated_forward_split(self, h_state_BxTxC, h_pred_BxTxC):
    mode = _FFN_MODE["m"]
    if mode == "total":
        return _ORIG_SHARED_FORWARD(self, h_state_BxTxC, h_pred_BxTxC)
    # The branch the OTHER stream consumes reads detached copies of every shared weight.
    det_state = mode == "pred"
    det_pred = mode == "state"
    cat = torch.cat([h_state_BxTxC, h_pred_BxTxC], dim=-1)

    def branch(det):
        w_in = self.gate_in.weight.detach() if det else self.gate_in.weight
        g = torch.sigmoid(F.linear(cat, w_in))
        return _mlp_apply(self.mlp, g * h_state_BxTxC + (1.0 - g) * h_pred_BxTxC, det)

    f_s = branch(det_state)
    f_p = branch(det_pred)
    w_gs = self.gate_state.weight.detach() if det_state else self.gate_state.weight
    w_gp = self.gate_pred.weight.detach() if det_pred else self.gate_pred.weight
    return (torch.sigmoid(F.linear(h_state_BxTxC, w_gs)) * f_s,
            torch.sigmoid(F.linear(h_pred_BxTxC, w_gp)) * f_p)


@contextlib.contextmanager
def split_mode(model, mode: str):
    """Bind the live parameter to ONE tower's occurrences; detach it at the other's.

    ``mode='total'`` is the unmodified model.  Forward VALUES are identical in all three
    modes by construction (detach changes the graph, never the numbers), which the caller
    asserts on the loss.
    """
    assert mode in ("total", "state", "pred")
    if mode == "total":
        yield
        return

    undo = []

    def shadow(obj, name, val):
        # Instance-__dict__ shadowing: nn.Module.__getattr__ only fires when normal
        # attribute lookup fails, so this hides the registered child WITHOUT touching
        # _parameters/_modules (and so without disturbing state_dict on restore).
        # `PredBlock.bind_tied` ALREADY lives in the instance __dict__ (it is installed
        # with object.__setattr__ precisely to stay out of _modules), so teardown must
        # RESTORE the previous value, not delete the name -- deleting it unbinds the
        # model's own tie and every later forward raises.
        had = name in obj.__dict__
        prev = obj.__dict__.get(name)
        object.__setattr__(obj, name, val)
        undo.append((obj, name, had, prev))

    try:
        if getattr(model, "tie_attn_across_towers", False):
            for i, pb in enumerate(model.transformer.pred_h):
                sb = model.transformer.state_h[i]
                if mode == "state":
                    shadow(pb, "_tied", SimpleNamespace(c_attn=_DetachedLinear(sb.c_attn),
                                                        c_proj=_DetachedLinear(sb.c_proj)))
                else:
                    # Pin the pred occurrences to the REAL modules FIRST, so the state-side
                    # shadowing below cannot leak through `_tied`.
                    shadow(pb, "_tied", SimpleNamespace(c_attn=sb.c_attn, c_proj=sb.c_proj))
            if mode == "pred":
                for sb in model.transformer.state_h:
                    shadow(sb, "c_attn", _DetachedLinear(sb.c_attn))
                    shadow(sb, "c_proj", _DetachedLinear(sb.c_proj))

        if bool(model.config.tie_lm_head):
            # wte.weight IS lm_head.weight: state = embedding lookup, pred = output head.
            if mode == "state":
                shadow(model, "lm_head", _DetachedLinear(model.lm_head))
            else:
                shadow(model.transformer, "wte", _DetachedEmbedding(model.transformer.wte))

        # Plain tied FFN / tied norms: pred block i's `mlp` / norms ARE state block i's
        # modules (registered under both parents), each tower still evaluating them on
        # its own residual -- two genuine occurrences, split by instance shadowing on the
        # side whose occurrence is detached.  Shadowing the STATE block's attribute does
        # not touch the pred block's lookup (which resolves through its own _modules).
        tied_leaf = []
        if getattr(model, "tie_ffn_across_towers", False):
            tied_leaf.append(("mlp", _DetachedMLP))
        if getattr(model, "tie_norms_across_towers", False):
            tied_leaf += [("attention_norm", _DetachedNorm), ("mlp_norm", _DetachedNorm)]
        for attr, stand_in in tied_leaf:
            for i, pb in enumerate(model.transformer.pred_h):
                sb = model.transformer.state_h[i]
                assert getattr(pb, attr) is getattr(sb, attr), f"{attr} not tied at block {i}"
                if mode == "state":
                    shadow(pb, attr, stand_in(getattr(sb, attr)))
                else:
                    shadow(sb, attr, stand_in(getattr(sb, attr)))

        if getattr(model, "share_ffn_across_towers", False):
            SharedGatedMLP.forward = _shared_gated_forward_split
            _FFN_MODE["m"] = mode
        yield
    finally:
        _FFN_MODE["m"] = "total"
        SharedGatedMLP.forward = _ORIG_SHARED_FORWARD
        for obj, name, had, prev in reversed(undo):
            if had:
                object.__setattr__(obj, name, prev)
            else:
                object.__delattr__(obj, name)


# ======================================================================================
# which tensors to probe
# ======================================================================================
def shared_specs(model):
    """-> [(name, kind, block, param, row_slice_or_None)] for every SHARED tensor."""
    specs = []
    inner = model.n_head_state * model.head_dim
    if getattr(model, "tie_attn_across_towers", False):
        for i, sb in enumerate(model.transformer.state_h):
            w = sb.c_attn.weight
            specs.append((f"state_h.{i}.c_attn[q]", "attn_q", i, w, slice(0, inner)))
            specs.append((f"state_h.{i}.c_attn[k]", "attn_k", i, w, slice(inner, 2 * inner)))
            specs.append((f"state_h.{i}.c_attn[v]", "attn_v", i, w, slice(2 * inner, 3 * inner)))
            specs.append((f"state_h.{i}.c_proj", "attn_out", i, sb.c_proj.weight, None))
    if getattr(model, "share_ffn_across_towers", False):
        for i, m in enumerate(model.transformer["shared_mlp"]):
            specs.append((f"shared_mlp.{i}.gate_proj", "ffn_gate_proj", i, m.mlp.gate_proj.weight, None))
            specs.append((f"shared_mlp.{i}.up_proj", "ffn_up_proj", i, m.mlp.up_proj.weight, None))
            specs.append((f"shared_mlp.{i}.down_proj", "ffn_down_proj", i, m.mlp.down_proj.weight, None))
            specs.append((f"shared_mlp.{i}.gate_in", "ffn_pool_gate", i, m.gate_in.weight, None))
            specs.append((f"shared_mlp.{i}.gate_state", "ffn_out_gate_state", i, m.gate_state.weight, None))
            specs.append((f"shared_mlp.{i}.gate_pred", "ffn_out_gate_pred", i, m.gate_pred.weight, None))
    if getattr(model, "tie_ffn_across_towers", False):
        for i, sb in enumerate(model.transformer.state_h):
            m = sb.mlp
            specs.append((f"state_h.{i}.mlp.gate_proj", "tffn_gate_proj", i, m.gate_proj.weight, None))
            specs.append((f"state_h.{i}.mlp.up_proj", "tffn_up_proj", i, m.up_proj.weight, None))
            specs.append((f"state_h.{i}.mlp.down_proj", "tffn_down_proj", i, m.down_proj.weight, None))
    if getattr(model, "tie_norms_across_towers", False):
        for i, sb in enumerate(model.transformer.state_h):
            specs.append((f"state_h.{i}.attention_norm", "norm_attn", i, sb.attention_norm.weight, None))
            specs.append((f"state_h.{i}.mlp_norm", "norm_mlp", i, sb.mlp_norm.weight, None))
    if bool(model.config.tie_lm_head):
        specs.append(("transformer.wte==lm_head", "embed_head", -1, model.transformer.wte.weight, None))
    return specs


def untied_pair_specs(model):
    """-> [(name, kind, block, state_param, state_slice, pred_param, pred_slice)].

    The untied arm's pred block owns ``q_proj`` (d x d), ``read_kv`` (2d x d, k then v)
    and ``c_proj``; the state block's fused ``c_attn`` is (3d x d) in q|k|v order.  The
    pairs below are therefore same-shaped, same-role, DIFFERENT tensors.
    """
    pairs = []
    inner = model.n_head_state * model.head_dim
    for i, (sb, pb) in enumerate(zip(model.transformer.state_h, model.transformer.pred_h)):
        sw = sb.c_attn.weight
        pairs.append((f"block{i}.q", "attn_q", i, sw, slice(0, inner), pb.q_proj.weight, None))
        pairs.append((f"block{i}.k", "attn_k", i, sw, slice(inner, 2 * inner),
                      pb.read_kv.weight, slice(0, inner)))
        pairs.append((f"block{i}.v", "attn_v", i, sw, slice(2 * inner, 3 * inner),
                      pb.read_kv.weight, slice(inner, 2 * inner)))
        pairs.append((f"block{i}.out", "attn_out", i, sb.c_proj.weight, None,
                      pb.c_proj.weight, None))
    return pairs


# ======================================================================================
# gradient accumulation
# ======================================================================================
def _take(param, sl):
    g = param.grad
    if g is None:
        return None
    return (g if sl is None else g[sl]).reshape(-1).double()


_AUTOCAST = {"on": True}


def accumulate(model, batches, params_and_slices, mode, half_of):
    """Run fwd+bwd over every micro-batch; return {half: {key: fp64 grad vector}}, losses."""
    acc = {0: {}, 1: {}}
    losses = []
    with split_mode(model, mode):
        for bi, (X, Y) in enumerate(batches):
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=bool(_AUTOCAST["on"])):
                _, loss, _ = model(X, Y)
            loss.backward()
            losses.append(float(loss.detach()))
            h = half_of(bi)
            for key, (p, sl) in params_and_slices.items():
                v = _take(p, sl)
                if v is None:
                    continue
                if key in acc[h]:
                    acc[h][key] += v
                else:
                    acc[h][key] = v.clone()
            del loss
    model.zero_grad(set_to_none=True)
    return acc, losses


def dot(a, b):
    return float(torch.dot(a, b))


def disattenuated_cos(sA, sB, pA, pB):
    """cos of the EXPECTED gradients, corrected for minibatch-noise attenuation.

    A minibatch gradient is ``g = gbar + e`` with independent noise across disjoint data
    shards, so for two disjoint halves A and B::

        E<g_s^A, g_s^B> = ||gbar_s||^2 ,  E<g_s^A, g_p^B> = <gbar_s, gbar_p>

    i.e. every CROSS-HALF inner product is an unbiased estimate of the corresponding
    inner product of the noise-free gradients, while the same-half quantities that the
    naive cosine uses are inflated by ``E||e||^2``.  The naive cosine is therefore
    attenuated by roughly ``sqrt(r_s * r_p)`` where ``r`` is the within-stream half-split
    cosine; this estimator divides that factor back out (the split-half / Spearman
    correction).  Returns None when either within-stream estimate is non-positive -- that
    means the expected gradient is not resolved above the noise at this batch size and NO
    cross-stream cosine from this run can be interpreted.
    """
    ss = 0.5 * (dot(sA, sB) + dot(sB, sA))
    pp = 0.5 * (dot(pA, pB) + dot(pB, pA))
    if ss <= 0 or pp <= 0:
        return None
    sp = 0.5 * (dot(sA, pB) + dot(sB, pA))
    return sp / (ss ** 0.5 * pp ** 0.5)


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return None
    return float(torch.dot(a, b) / (na * nb))


# ======================================================================================
# JOINT (tied-SPS) branch
# ======================================================================================
# In tied SPS there is ONE block stack; every weight of block i is applied, in one
# matmul, to all 2T interleaved slots.  "Which stream uses W" is therefore a question
# about ROWS of that matmul, not about which module is called.  The decomposition below
# makes each row-occurrence an independent variable (an exact leaf copy of W), runs a
# numerically identical forward, and reads the gradient of every leaf in ONE backward.
# See `run_joint` for the full statement of what is and is not attributed to each slot.
_KV_CELLS = ("ss", "sp", "ps", "pp")      # (producer slot, consumer slot)


def _leaf(p):
    return p.detach().clone().requires_grad_(True)


def joint_make_leaves(model, shared=False):
    """-> {(block, key): {cell: leaf}}.  Every leaf is a bit-exact copy of the live weight.

    c_attn gets FOUR leaves, indexed by (producer, consumer) slot type: the q rows are
    only ever read through `ss`/`pp` (a query is consumed by its own slot), while the k/v
    rows are read through all four (a key produced at a state slot is read by later state
    AND pred queries, and likewise for pred keys).  Every other per-slot weight gets two
    leaves, `s` and `p`.  The embedding table gets `s` (state-slot lookups of the real
    tokens) and `p` (pred-slot lookups of <predict> -- and, when tie_lm_head, the LM head,
    which reads pred slots only).

    `shared=True` binds every cell of a tensor to ONE leaf: the same (duplicated-attention)
    graph, but with no split -- the reference the exact-additivity gate compares against.
    """
    if shared:
        one = joint_make_leaves(model)
        return {k: dict.fromkeys(cells, next(iter(cells.values()))) for k, cells in one.items()}
    lv = {}
    for i, blk in enumerate(C.joint_blocks(model)):
        lv[(i, "c_attn")] = {c: _leaf(blk.attn.c_attn.weight) for c in _KV_CELLS}
        lv[(i, "c_proj")] = {c: _leaf(blk.attn.c_proj.weight) for c in "sp"}
        lv[(i, "gate")] = {c: _leaf(blk.mlp.gate_proj.weight) for c in "sp"}
        lv[(i, "up")] = {c: _leaf(blk.mlp.up_proj.weight) for c in "sp"}
        lv[(i, "down")] = {c: _leaf(blk.mlp.down_proj.weight) for c in "sp"}
        lv[(i, "norm_attn")] = {c: _leaf(blk.attention_norm.weight) for c in "sp"}
        lv[(i, "norm_mlp")] = {c: _leaf(blk.mlp_norm.weight) for c in "sp"}
    lv[(-1, "embed")] = {c: _leaf(model.transformer.wte.weight) for c in "sp"}
    return lv


def _rows(s_m, a, b):
    """state rows from `a`, pred rows from `b` (both full-shape, identical numbers)."""
    return torch.where(s_m, a, b)


def _split_norm(norm, x, w, s_m):
    n = norm._norm(x.float()).type_as(x)             # RMSNorm.forward, op for op
    return _rows(s_m, n * w["s"], n * w["p"])


def _split_lin(x, w, s_m):
    return _rows(s_m, F.linear(x, w["s"]), F.linear(x, w["p"]))


def _split_block(blk, i, x, freqs_cis, docs, lv, s_m):
    """`SPSBlock.forward` + `SPSFlashAttention.forward`, op for op, with per-row leaves."""
    import modeling.models.sps.core as sps_core
    a = blk.attn
    b, two_t, c = x.shape
    nh, hd = a.n_head, c // a.n_head
    h = _split_norm(blk.attention_norm, x, lv[(i, "norm_attn")], s_m)
    W = lv[(i, "c_attn")]
    out = {cell: F.linear(h, W[cell]) for cell in _KV_CELLS}

    def part(t, j):
        return t[..., j * c:(j + 1) * c]
    q = _rows(s_m, part(out["ss"], 0), part(out["pp"], 0))
    # keys/values as read by STATE queries (A) and by PRED queries (B)
    kA = _rows(s_m, part(out["ss"], 1), part(out["ps"], 1))
    vA = _rows(s_m, part(out["ss"], 2), part(out["ps"], 2))
    kB = _rows(s_m, part(out["sp"], 1), part(out["pp"], 1))
    vB = _rows(s_m, part(out["sp"], 2), part(out["pp"], 2))

    def attend(k, v):
        qq = q.view(b, two_t, nh, hd)
        kk = k.view(b, two_t, nh, hd)
        vv = v.view(b, two_t, nh, hd).transpose(1, 2)
        qq, kk = C.apply_rotary_emb(qq, kk, freqs_cis=freqs_cis)
        y = sps_core.triton_sps_sliding_attention(
            qq.transpose(1, 2).to(torch.bfloat16), kk.transpose(1, 2).to(torch.bfloat16),
            vv.to(torch.bfloat16), 1.0 / math.sqrt(hd), a.window_size,
            warp_specialize=a.warp_specialize, documents_idx_BxT=docs)
        return y.transpose(1, 2).contiguous().view(b, two_t, c)
    y = _rows(s_m, attend(kA, vA), attend(kB, vB)).to(a.c_proj.weight.dtype)
    x = x + a.resid_dropout(_split_lin(y, lv[(i, "c_proj")], s_m))
    h2 = _split_norm(blk.mlp_norm, x, lv[(i, "norm_mlp")], s_m)
    g = _split_lin(h2, lv[(i, "gate")], s_m)
    u = _split_lin(h2, lv[(i, "up")], s_m)
    m = blk.mlp.dropout(_split_lin(F.silu(g) * u, lv[(i, "down")], s_m))
    return x + m


def joint_split_loss(model, X, Y, lv):
    """`SPSModelBase.forward(X, Y)` re-expressed over the per-row leaves.

    Forward VALUES are identical to the model's (every leaf holds the same numbers and
    every matmul is evaluated at full shape); the caller asserts the loss matches the
    unmodified model bit-for-bit before any gradient is used.
    """
    from modeling.models.model import masked_lm_loss
    mode = getattr(model.config, "predict_embedding", "constant")
    assert mode in ("constant", "shared"), f"predict_embedding={mode} not supported"
    is_real, docs_T, docs_2T = model._expand_real_and_document_idx(X)
    idx2 = model.add_predict_tokens(X)
    b, two_t = idx2.shape
    t = two_t // 2
    dev = idx2.device
    pos = torch.arange(t, device=dev).repeat_interleave(2).unsqueeze(0).expand(b, -1)
    freqs_cis = model.freqs_cis.to(dev)[pos]
    s_m, _p_m = C.joint_slot_masks(two_t, dev)
    E = lv[(-1, "embed")]
    x = _rows(s_m, F.embedding(idx2, E["s"]), F.embedding(idx2, E["p"]))
    x = model.transformer.drop(x)
    for i, blk in enumerate(C.joint_blocks(model)):
        x = _split_block(blk, i, x, freqs_cis, docs_2T, lv, s_m)
    x = model.transformer.output_norm(x)
    head = E["p"] if bool(model.config.tie_lm_head) else model.lm_head.weight
    logits = F.linear(x[:, 1::2], head)
    loss, _ = masked_lm_loss(logits, X, Y, is_real, docs_T, model.config.eos_token_id)
    return loss


def joint_specs(model):
    """-> [(name, kind, block, live_param, row_slice, leaf_key)] -- every shared tensor."""
    specs = []
    c = int(model.config.hidden_size)
    for i, blk in enumerate(C.joint_blocks(model)):
        w = blk.attn.c_attn.weight
        for j, r in enumerate("qkv"):
            specs.append((f"h.{i}.c_attn[{r}]", f"attn_{r}", i, w, slice(j * c, (j + 1) * c),
                          (i, "c_attn")))
        specs.append((f"h.{i}.c_proj", "attn_out", i, blk.attn.c_proj.weight, None, (i, "c_proj")))
        specs.append((f"h.{i}.mlp.gate_proj", "ffn_gate_proj", i, blk.mlp.gate_proj.weight, None, (i, "gate")))
        specs.append((f"h.{i}.mlp.up_proj", "ffn_up_proj", i, blk.mlp.up_proj.weight, None, (i, "up")))
        specs.append((f"h.{i}.mlp.down_proj", "ffn_down_proj", i, blk.mlp.down_proj.weight, None, (i, "down")))
        specs.append((f"h.{i}.attention_norm", "norm_attn", i, blk.attention_norm.weight, None, (i, "norm_attn")))
        specs.append((f"h.{i}.mlp_norm", "norm_mlp", i, blk.mlp_norm.weight, None, (i, "norm_mlp")))
    if bool(model.config.tie_lm_head):
        specs.append(("transformer.wte==lm_head", "embed_head", -1, model.transformer.wte.weight,
                      None, (-1, "embed")))
    else:
        specs.append(("transformer.wte", "embed", -1, model.transformer.wte.weight, None, (-1, "embed")))
    return specs


def _leaf_grad(leaf, sl):
    g = leaf.grad
    if g is None:
        return None
    return (g if sl is None else g[sl]).reshape(-1).double()


def joint_decompose(lv, key, sl):
    """-> {"S","P","Sprod","Pprod", and for c_attn the four cells} as fp64 vectors."""
    cells = lv[key]
    g = {c: _leaf_grad(l, sl) for c, l in cells.items()}
    ref = next(v for v in g.values() if v is not None)
    z = torch.zeros_like(ref)
    g = {c: (z if v is None else v) for c, v in g.items()}
    if "ss" in g:
        return dict(S=g["ss"] + g["ps"], P=g["sp"] + g["pp"],
                    Sprod=g["ss"] + g["sp"], Pprod=g["ps"] + g["pp"],
                    cell_ss=g["ss"], cell_sp=g["sp"], cell_ps=g["ps"], cell_pp=g["pp"])
    return dict(S=g["s"], P=g["p"], Sprod=g["s"], Pprod=g["p"])


def joint_accumulate(model, batches, specs, mode, half_of, lv=None, log_every=0):
    """mode 'total': the unmodified model, live params.  mode 'split': the leaf forward.
    mode 'replica': the leaf forward with shared leaves (`joint_make_leaves(shared=True)`).
    -> ({half: {key: fp64}}, losses).  Keys: `name` (total) or `name::<part>` (split)."""
    import time as _time
    acc = {0: {}, 1: {}}
    losses = []
    t0 = _time.time()
    for bi, (X, Y) in enumerate(batches):
        model.zero_grad(set_to_none=True)
        if lv is not None:
            for cells in lv.values():
                for leaf in cells.values():
                    leaf.grad = None
        # cache_enabled=False for the leaf passes: autocast otherwise caches ONE bf16 cast
        # of a leaf that is used several times and sums its weight-gradients IN BF16
        # before the cast's backward, which would make the shared-leaf reference round
        # differently from the split (measured 3e-3 rel).  Forward values are unaffected.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(_AUTOCAST["on"]),
                                cache_enabled=(mode == "total")):
            if mode == "total":
                _, loss, _ = model(X, Y)
            else:
                loss = joint_split_loss(model, X, Y, lv)
        loss.backward()
        losses.append(float(loss.detach()))
        h = half_of(bi)
        for name, _k, _b, p, sl, key in specs:
            if mode == "total":
                parts = {name: _take(p, sl)}
            elif mode == "replica":
                parts = {f"{name}::F": _leaf_grad(next(iter(lv[key].values())), sl)}
            else:
                parts = {f"{name}::{k}": v for k, v in joint_decompose(lv, key, sl).items()}
            for k, v in parts.items():
                if v is None:
                    continue
                if k in acc[h]:
                    acc[h][k] += v
                else:
                    acc[h][k] = v.clone()
        del loss
        if log_every and (bi + 1) % log_every == 0:
            el = _time.time() - t0
            print(f"  [{mode}] {bi + 1}/{len(batches)} micro-batches  {el:.0f}s  "
                  f"eta {el / (bi + 1) * (len(batches) - bi - 1):.0f}s", flush=True)
    model.zero_grad(set_to_none=True)
    return acc, losses


def run_joint(args, model, cfg, family, ckpt, run):
    """A16 for the JOINT family (tied SPS): state slots vs pred slots on shared weights.

    THE DECOMPOSITION.  Tied SPS has one stack; block i's weights act on every one of the
    2T interleaved slots.  An OCCURRENCE of W is therefore one ROW of one matmul -- W
    applied to one slot's activation -- and the multivariate chain rule over occurrences
    (the same identity the two-tower branch uses) gives

        dL/dW = sum over rows r of dL/dW_r.

    Each row-occurrence is made an independent leaf (`joint_make_leaves`: bit-exact
    copies of W), the forward is re-expressed over those leaves with identical values
    (`joint_split_loss`, asserted equal to the model's own loss), and ONE backward gives
    every part.  No activation edge is cut, so every cross-slot path -- a state slot's
    key read by a later pred query, a state residual feeding a later state slot that a
    pred query reads -- stays in the part it originates from.  Additivity is exact by
    construction and is MEASURED twice: (a) against the gradient of a single shared leaf
    in the SAME graph (exact up to fp32 summation order), and (b) against an independent
    pass through the unmodified model.  (b) differs by a rounding floor only: the model
    casts q/k/v to bf16, so the kernel backward rounds dk/dv once for the summed key
    gradient in the model but once per consumer here (smoke, 8 seqs: <=3.6% rel L2 on the
    smallest-norm tensors, <=1% on the rest).

    WHICH ROWS BELONG TO WHICH STREAM.  For q, the attention output projection, the three
    FFN matrices and both RMSNorm weights, the occurrence at slot r is consumed by slot r
    alone, so "state part" = rows at state slots (2p), "pred part" = rows at pred slots
    (2p+1).  Keys/values are the one place where production and consumption differ: the
    k/v row of a STATE slot is read by later state queries AND by pred queries (and a
    pred slot's k/v by pred queries and by later state queries in the window).  The
    HEADLINE (`cos_state_pred`) attributes k/v by CONSUMER -- the stream of the query
    that reads it -- because that is exactly the two-tower tied-attention split (the
    state tower's own k/v for state queries = state; `PredBlock.read_state`, the pred
    tower's k/v re-projection of the state residual, = pred), so the number is
    comparable to tied-attn / afsps / seq12_tied.  Consumer attribution for k/v needs the
    attention evaluated twice (state queries against the "consumed-by-state" k/v leaves,
    pred queries against the "consumed-by-pred" leaves); this is numerically the same
    attention.  The PRODUCER split (k/v by the slot that computes it) is also reported
    (`producer.*`), as is the full 2x2 (producer x consumer) norm table for k and v.
    Only k/v differ between the two splits; for every other tensor they coincide.

    RMSNorm weights are attributed by the row they act on (producer).  For `mlp_norm`
    that is also the consumer; `attention_norm`'s state-row output additionally feeds the
    state-produced k/v that pred queries read, which a row split cannot separate
    without splitting the norm three ways -- stated, not hidden.

    EMBEDDING.  wte's state part = lookups of the real tokens at state slots; its pred
    part = the <predict> lookup at pred slots plus, when tie_lm_head, the whole LM head
    (which reads pred slots only).  Untied head: the pred part touches ONE row
    (<predict>) and the state part never touches it, so that cosine is 0 by support, not
    by training; it is reported but excluded from `ALL_BLOCKS`.  `output_norm` and an
    untied `lm_head` act on pred slots only (stream-exclusive) and are not probed.

    WHAT IS NOT A SPLIT.  SPS state slots carry no loss.  The state part is therefore
    entirely "gradient that reaches W through the state rows' activations and then
    through what later reads them", never a state-slot loss term.  Also note the last
    block's state rows are read by nothing downstream except (for k/v) the same block's
    queries; under the consumer split the whole state part of block L-1 is exactly zero
    (checked below as a structural control: `dead_end_gate`).

    CONTROLS -- as in the two-tower branch: batch-split halves (noise floor), the
    split-half disattenuated cosine, the cross-stream-cross-half cosine, `--untrained`
    (init), the second total pass (reproducibility floor), and the three-estimator gate.
    """
    import numpy as np
    import time as _time
    model.eval()
    _AUTOCAST["on"] = not args.fp32
    for p in model.parameters():
        p.requires_grad_(True)
    data = C.val_memmap(cfg)
    starts = C.seq_starts(data, args.n_seq)
    n_seq = len(starts)
    assert n_seq >= 2
    mb = args.micro_batch
    batches = []
    for i in range(0, n_seq, mb):
        b = starts[i:i + mb]
        X = torch.stack([torch.from_numpy(data[j:j + C.BLOCK].astype(np.int64)) for j in b]).cuda()
        Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + C.BLOCK].astype(np.int64)) for j in b]).cuda()
        batches.append((X, Y))
    n_mb = len(batches)
    half_of = lambda bi: 0 if bi < n_mb // 2 else 1  # noqa: E731
    specs = joint_specs(model)
    lv = joint_make_leaves(model)
    L = len(C.joint_blocks(model))
    out = dict(analysis=ANALYSIS, run=run, family=family, branch="joint",
               checkpoint=str(ckpt), untrained=bool(args.untrained), untied_reference=False,
               n_seq=n_seq, micro_batch=mb, n_micro_batches=n_mb, tokens=n_seq * C.BLOCK,
               decomposition=dict(
                   headline="consumer", unit="row-occurrence (one slot's use of W)",
                   kv="consumer = stream of the reading query; producer = slot computing it",
                   note="see run_joint docstring"),
               config=dict(tie_lm_head=bool(model.config.tie_lm_head),
                           window_size=int(cfg.model.config.window_size), n_layer=L))
    res, losses = {}, {}
    lv_shared = joint_make_leaves(model, shared=True)
    passes = [("total", "total"), ("split", "split"), ("replica", "replica")]
    if not args.skip_repro:
        passes.append(("total", "total2"))
    log_every = max(1, n_mb // 8)
    for mode, tag in passes:
        t0 = _time.time()
        res[tag], losses[tag] = joint_accumulate(model, batches, specs, mode, half_of,
                                                 lv={"split": lv, "replica": lv_shared}.get(mode),
                                                 log_every=log_every)
        print(f"mode={tag} loss={np.mean(losses[tag]):.8f} ({_time.time() - t0:.0f}s)",
              flush=True)
    lm = {m: float(np.mean(v)) for m, v in losses.items()}
    out["loss_by_mode"] = lm
    out["loss_max_spread"] = max(lm.values()) - min(lm.values())
    out["loss_mean"] = lm["total"]
    out["loss_split_bitexact"] = bool(losses["split"] == losses["total"]
                                      and losses["replica"] == losses["total"])
    tot = {m: {k: res[m][0][k] + res[m][1][k] for k in res[m][0]} for m in res}

    def part(name, which, half=None):
        src = tot["split"] if half is None else res["split"][half]
        return src.get(f"{name}::{which}")

    # -------- CORRECTNESS GATE.
    # (a) EXACT ADDITIVITY: the parts sum to the gradient of ONE leaf in the SAME graph
    #     (`replica`, shared leaves).  Only fp32/fp64 summation order separates them.
    # (b) FIDELITY: that same-graph gradient vs the UNMODIFIED model's.  The model casts
    #     q/k/v to bf16, so dk/dv are rounded to bf16 by the kernel backward: once for the
    #     summed key gradient in the model, once per consumer in the duplicated-attention
    #     graph.  This is a rounding floor, not a lost path, and it is reported per tensor.
    # (c) the original three-estimator spread against the model's own total (reported;
    #     it includes the rounding floor of (b)).
    gate_rows, est_rows = [], []
    worst = worst_p = worst_repro = spread_max = worst_fid = 0.0
    for name, kind, _blk, _p, _sl, _key in specs:
        gt = tot["total"][name]
        gf = tot["replica"][f"{name}::F"]
        gs, gp = part(name, "S"), part(name, "P")
        gsp, gpp = part(name, "Sprod"), part(name, "Pprod")
        den = float(gf.norm())
        rel = float((gs + gp - gf).norm()) / den if den > 0 else 0.0
        relp = float((gsp + gpp - gf).norm()) / den if den > 0 else 0.0
        dt = float(gt.norm())
        fid = float((gf - gt).norm()) / dt if dt > 0 else 0.0
        g2 = tot.get("total2", {}).get(name)
        repro = float((g2 - gt).norm()) / dt if (g2 is not None and dt > 0) else 0.0
        worst, worst_p, worst_repro = max(worst, rel), max(worst_p, relp), max(worst_repro, repro)
        worst_fid = max(worst_fid, fid)
        gate_rows.append(dict(name=name, kind=kind, rel_err=rel, rel_err_producer=relp,
                              fidelity_rel_err=fid, cos_replica_vs_model=cos(gf, gt),
                              repro_rel_err=repro, norm_total=dt))
        vals = [cos(gs, gp), cos(gt - gp, gp), cos(gs, gt - gs)]
        ok_vals = [v for v in vals if v is not None]
        spread = (max(ok_vals) - min(ok_vals)) if len(ok_vals) == 3 else None
        if spread is not None:
            spread_max = max(spread_max, spread)
        est_rows.append(dict(name=name, kind=kind, cos_split=vals[0], cos_total_minus_pred=vals[1],
                             cos_total_minus_state=vals[2], spread=spread))
    # structural control: consumer-state part of the LAST block is read by nothing
    # (norm_attn is excluded: its state-row output also feeds the state-produced keys
    # that PRED queries read, and a norm weight is attributed by the row it acts on.)
    dead = [float(part(n, "S").abs().max()) for n, k, b, _p, _s, _y in specs
            if b == L - 1 and k != "norm_attn"]
    dead_ok = bool(max(dead) == 0.0)
    add_ok = bool(worst <= args.add_tol and worst_p <= args.add_tol)
    fid_ok = bool(worst_fid <= args.fid_tol)
    ok = bool(out["loss_split_bitexact"] and add_ok and fid_ok and dead_ok
              and (args.skip_repro or worst_repro == 0.0))
    out["gate"] = dict(applicable=True, max_rel_err=worst, max_rel_err_producer=worst_p,
                       exact_additivity=dict(tol=args.add_tol, passed=add_ok),
                       max_fidelity_rel_err=worst_fid,
                       fidelity=dict(tol=args.fid_tol, passed=fid_ok),
                       estimator_spread_within_cos_tol=bool(spread_max <= args.cos_tol),
                       max_repro_rel_err=worst_repro, max_estimator_spread=spread_max,
                       tol=args.gate_tol, cos_tol=args.cos_tol,
                       loss_split_bitexact=out["loss_split_bitexact"],
                       dead_end_gate=dict(max_abs_state_grad_last_block=max(dead), passed=dead_ok),
                       criterion=("split/replica loss == model loss bit-for-bit AND repro "
                                  "floor == 0 AND parts sum to the same-graph gradient within "
                                  "add_tol AND same-graph vs model gradient within fid_tol "
                                  "(bf16 kernel-backward rounding) AND the last block's "
                                  "consumer-state part is exactly 0; estimator spread vs "
                                  "cos_tol reported"),
                       passed=ok, per_tensor=gate_rows, estimators=est_rows)
    print(f"GATE additivity max_rel_err={worst:.3e} (producer {worst_p:.3e}) "
          f"fidelity={worst_fid:.3e} repro_floor={worst_repro:.3e} "
          f"estimator_spread={spread_max:.3e} split_loss_bitexact={out['loss_split_bitexact']} "
          f"dead_end={max(dead):.1e} {'PASS' if ok else 'FAIL'}", flush=True)

    rows, bykind = [], {}
    for name, kind, blk, _p, _sl, _key in specs:
        gt = tot["total"][name]
        gs, gp = part(name, "S"), part(name, "P")
        gsp, gpp = part(name, "Sprod"), part(name, "Pprod")
        ns, np_ = float(gs.norm()), float(gp.norm())
        row = dict(
            name=name, kind=kind, block=blk,
            cos_state_pred=cos(gs, gp), norm_state=ns, norm_pred=np_, norm_total=float(gt.norm()),
            ratio_pred_over_state=(np_ / ns) if ns > 0 else None,
            stream_exclusive=(ns == 0.0 or np_ == 0.0),
            control_batch_split_state=cos(part(name, "S", 0), part(name, "S", 1)),
            control_batch_split_pred=cos(part(name, "P", 0), part(name, "P", 1)),
            control_batch_split_total=cos(res["total"][0][name], res["total"][1][name]),
            control_cross_stream_cross_half=cos(part(name, "S", 0), part(name, "P", 1)),
            cos_total_minus_pred=cos(gt - gp, gp), cos_total_minus_state=cos(gs, gt - gs),
            cos_disattenuated=disattenuated_cos(part(name, "S", 0), part(name, "S", 1),
                                                part(name, "P", 0), part(name, "P", 1)),
            producer=dict(cos_state_pred=cos(gsp, gpp), norm_state=float(gsp.norm()),
                          norm_pred=float(gpp.norm()),
                          cos_disattenuated=disattenuated_cos(
                              part(name, "Sprod", 0), part(name, "Sprod", 1),
                              part(name, "Pprod", 0), part(name, "Pprod", 1))),
        )
        if part(name, "cell_ss") is not None and kind in ("attn_k", "attn_v"):
            cells = {c: part(name, f"cell_{c}") for c in _KV_CELLS}
            row["kv_cells"] = dict(
                norm={c: float(v.norm()) for c, v in cells.items()},
                note="cell xy = produced at slot x, read by a query at slot y")
        rows.append(row)
        if row["cos_state_pred"] is not None:
            bykind.setdefault(kind, []).append(row)
    out["tensors"] = rows

    def agg(key, sel, sub=None):
        v = [(r[sub] if sub else r).get(key) for r in sel]
        v = [x for x in v if x is not None]
        if not v:
            return None
        return dict(n=len(v), mean=float(np.mean(v)), absmean=float(np.mean(np.abs(v))),
                    min=float(np.min(v)), max=float(np.max(v)))

    def agg_block(sel):
        return dict(cos_state_pred=agg("cos_state_pred", sel),
                    cos_disattenuated=agg("cos_disattenuated", sel),
                    control_batch_split_state=agg("control_batch_split_state", sel),
                    control_batch_split_pred=agg("control_batch_split_pred", sel),
                    control_cross_stream_cross_half=agg("control_cross_stream_cross_half", sel),
                    ratio_pred_over_state=agg("ratio_pred_over_state", sel),
                    producer_cos_state_pred=agg("cos_state_pred", sel, "producer"),
                    producer_cos_disattenuated=agg("cos_disattenuated", sel, "producer"))
    out["aggregate"] = {k: agg_block(v) for k, v in bykind.items()}
    allrows = [r for v in bykind.values() for r in v]
    out["aggregate"]["ALL"] = agg_block(allrows)
    out["aggregate"]["ALL_BLOCKS"] = agg_block([r for r in allrows if r["block"] >= 0])
    out["aggregate"]["ATTN"] = agg_block([r for r in allrows if r["kind"].startswith("attn_")])
    suffix = "_init" if args.untrained else ""
    path = args.out or os.path.join(C.RESULTS_DIR, f"{ANALYSIS}_{run}{suffix}.json")
    C.save_json(path, out)
    print(json.dumps({k: v["cos_state_pred"] for k, v in out["aggregate"].items()}, indent=1)[:4000],
          flush=True)
    return out

# ======================================================================================
# main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--untrained", action="store_true",
                    help="CONTROL 1: same measurement on a fresh random init")
    ap.add_argument("--untied", action="store_true",
                    help="CONTROL 3: reference cosines between the untied arm's separate tensors")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gate-tol", type=float, default=GATE_TOL)
    ap.add_argument("--skip-repro", action="store_true",
                    help="skip the second identical total pass (the reproducibility floor "
                         "has already been measured as exactly 0 on this stack)")
    ap.add_argument("--cos-tol", type=float, default=COS_TOL,
                    help="max allowed spread between the three cosine estimators")
    ap.add_argument("--add-tol", type=float, default=1e-5,
                    help="joint branch: exact-additivity tolerance (same-graph, rel L2)")
    ap.add_argument("--fid-tol", type=float, default=0.05,
                    help="joint branch: same-graph vs model gradient (bf16 rounding floor)")
    ap.add_argument("--attn-backend", default=None,
                    help="override model.config.attn_backend (flex's compiled backward uses "
                         "atomics and is not bit-reproducible; 'sdpa' is quieter)")
    ap.add_argument("--fp32", action="store_true",
                    help="run the loss in fp32 instead of the bf16 autocast training uses")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    C.assert_node_local_triton()
    run = C.resolve(args.run)
    torch.manual_seed(0)

    if args.untrained:
        model, cfg, family, ckpt = C.load_untrained(run, seed=args.seed)
    else:
        model, cfg, family, ckpt = C.load(run)
        assert family in ("two_tower", "sps"), \
            f"{run}: A16 has two-tower and joint (tied-SPS) branches, got {family}"
    if family == "sps":
        return run_joint(args, model, cfg, family, ckpt, run)
    model.eval()
    if args.attn_backend:
        model.config.attn_backend = args.attn_backend
    _AUTOCAST["on"] = not args.fp32
    for p in model.parameters():
        p.requires_grad_(True)

    data = C.val_memmap(cfg)
    starts = C.seq_starts(data, args.n_seq)
    # The val set holds 3,198 non-overlapping 4096-token sequences; asking for more is
    # capped there rather than silently reusing data.  The batch-split control is what
    # says whether the batch that results actually resolves the expected gradient.
    n_seq = len(starts)
    assert n_seq >= 2, f"need >= 2 sequences, got {n_seq}"
    mb = args.micro_batch
    batches = []
    for i in range(0, len(starts), mb):
        b = starts[i:i + mb]
        X = torch.stack([torch.from_numpy(data[j:j + C.BLOCK].astype(np.int64)) for j in b]).cuda()
        Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + C.BLOCK].astype(np.int64)) for j in b]).cuda()
        batches.append((X, Y))
    n_mb = len(batches)
    half_of = lambda bi: 0 if bi < n_mb // 2 else 1  # noqa: E731

    out = dict(analysis=ANALYSIS, run=run, checkpoint=str(ckpt), untrained=bool(args.untrained),
               untied_reference=bool(args.untied), n_seq=n_seq, micro_batch=mb,
               n_micro_batches=n_mb, tokens=n_seq * C.BLOCK,
               config=dict(tie_attn_across_towers=bool(getattr(model, "tie_attn_across_towers", False)),
                           share_ffn_across_towers=bool(getattr(model, "share_ffn_across_towers", False)),
                           tie_ffn_across_towers=bool(getattr(model, "tie_ffn_across_towers", False)),
                           tie_norms_across_towers=bool(getattr(model, "tie_norms_across_towers", False)),
                           tie_lm_head=bool(model.config.tie_lm_head),
                           read_map=str(model.config.read_map)))

    # ---------------------------------------------------------------- untied reference
    if args.untied:
        pairs = untied_pair_specs(model)
        pas = {}
        for name, _k, _b, sp, ss, pp, ps in pairs:
            pas[name + "::state"] = (sp, ss)
            pas[name + "::pred"] = (pp, ps)
        acc, losses = accumulate(model, batches, pas, "total", half_of)
        tot = {k: acc[0][k] + acc[1][k] for k in acc[0]}
        rows, bykind = [], {}
        for name, kind, blk, _sp, _ss, _pp, _ps in pairs:
            gs, gp = tot[name + "::state"], tot[name + "::pred"]
            c = cos(gs, gp)
            cA = cos(acc[0][name + "::state"], acc[1][name + "::state"])
            cP = cos(acc[0][name + "::pred"], acc[1][name + "::pred"])
            rows.append(dict(name=name, kind=kind, block=blk, cos_state_pred=c,
                             cos_disattenuated=disattenuated_cos(
                                 acc[0][name + "::state"], acc[1][name + "::state"],
                                 acc[0][name + "::pred"], acc[1][name + "::pred"]),
                             norm_state=float(gs.norm()), norm_pred=float(gp.norm()),
                             ratio_pred_over_state=float(gp.norm() / gs.norm()),
                             control_batch_split_state=cA, control_batch_split_pred=cP))
            bykind.setdefault(kind, []).append(c)
        out["loss_mean"] = float(np.mean(losses))
        out["tensors"] = rows
        out["aggregate"] = {k: dict(n=len(v), cos_mean=float(np.mean(v)),
                                    cos_absmean=float(np.mean(np.abs(v))),
                                    cos_min=float(np.min(v)), cos_max=float(np.max(v)))
                            for k, v in bykind.items()}
        allc = [r["cos_state_pred"] for r in rows]
        out["aggregate"]["ALL"] = dict(n=len(allc), cos_mean=float(np.mean(allc)),
                                       cos_absmean=float(np.mean(np.abs(allc))),
                                       cos_min=float(np.min(allc)), cos_max=float(np.max(allc)))
        out["gate"] = dict(applicable=False,
                           note="no shared tensor in this arm; nothing to decompose")
    # ------------------------------------------------------------------- shared arms
    else:
        specs = shared_specs(model)
        assert specs, f"{run}: no shared parameter tensors -- use --untied for this arm"
        pas = {name: (p, sl) for name, _k, _b, p, sl in specs}
        res, losses = {}, {}
        # "total2" is a SECOND, identical total pass.  It is not a mode: it measures the
        # numerical REPRODUCIBILITY FLOOR of this whole pipeline (bf16 activations, and a
        # compiled flex_attention backward that accumulates with atomics and is therefore
        # not bit-reproducible).  The additivity identity g_state + g_pred == g_total is
        # exact in exact arithmetic, so the only honest gate is "the residual is at the
        # level of the floor", not "the residual is below an arbitrary absolute number".
        passes = [("total", "total"), ("state", "state"), ("pred", "pred")]
        if not args.skip_repro:
            passes.append(("total", "total2"))
        for mode, tag in passes:
            res[tag], losses[tag] = accumulate(model, batches, pas, mode, half_of)
            print(f"mode={tag} loss={np.mean(losses[tag]):.8f}", flush=True)
        # Detach changes the graph, never the numbers: all three modes must agree exactly.
        lm = {m: float(np.mean(v)) for m, v in losses.items()}
        out["loss_by_mode"] = lm
        out["loss_max_spread"] = max(lm.values()) - min(lm.values())
        out["loss_mean"] = lm["total"]

        tot = {m: {k: res[m][0][k] + res[m][1][k] for k in res[m][0]} for m in res}

        # -------- CORRECTNESS GATE: g_state + g_pred == g_total
        cos_estimators = []
        for name, kind, _blk, _p, _sl in specs:
            gt = tot["total"].get(name)
            if gt is None:
                continue
            zero = torch.zeros_like(gt)
            gs = tot["state"].get(name, zero)
            gp = tot["pred"].get(name, zero)
            e_split = cos(gs, gp)
            e_sub_state = cos(gt - gp, gp)   # state side defined by subtraction
            e_sub_pred = cos(gs, gt - gs)    # pred side defined by subtraction
            vals = [v for v in (e_split, e_sub_state, e_sub_pred) if v is not None]
            cos_estimators.append(dict(
                name=name, kind=kind, cos_split=e_split, cos_total_minus_pred=e_sub_state,
                cos_total_minus_state=e_sub_pred,
                spread=(max(vals) - min(vals)) if len(vals) == 3 else None))

        gate_rows, worst, worst_repro = [], 0.0, 0.0
        for name, kind, _blk, _p, _sl in specs:
            gt = tot["total"].get(name)
            gs = tot["state"].get(name)
            gp = tot["pred"].get(name)
            if gt is None:
                continue
            zero = torch.zeros_like(gt)
            gs = zero if gs is None else gs
            gp = zero if gp is None else gp
            den = float(gt.norm())
            rel = float((gs + gp - gt).norm()) / den if den > 0 else 0.0
            g2 = tot.get("total2", {}).get(name)
            repro = float((g2 - gt).norm()) / den if (g2 is not None and den > 0) else 0.0
            worst = max(worst, rel)
            worst_repro = max(worst_repro, repro)
            gate_rows.append(dict(name=name, kind=kind, rel_err=rel,
                                  repro_rel_err=repro, norm_total=den))
        # THREE INDEPENDENT ESTIMATORS of the same cosine.  The additivity identity is
        # exact in exact arithmetic, so a residual can only be round-off -- but "can only
        # be" is an argument, not a measurement.  The measurement is this: two of the
        # three estimators below impose additivity BY CONSTRUCTION (they define one side
        # by subtraction from the measured total), and the third does not.  If a path had
        # actually been lost, the measured side would be wrong and the three would
        # disagree by O(1).  Their SPREAD is therefore the gate that matters -- it bounds
        # the error in the number this analysis reports, which ``max_rel_err`` does not.
        spread_max, spread_rows = 0.0, []
        for r in cos_estimators:
            spread_rows.append(r)
            if r["spread"] is not None:
                spread_max = max(spread_max, r["spread"])
        ok = bool(spread_max <= args.cos_tol and (args.skip_repro or worst_repro == 0.0))
        out["gate"] = dict(applicable=True,
                           max_rel_err=worst,
                           max_repro_rel_err=worst_repro,
                           max_estimator_spread=spread_max,
                           tol=args.gate_tol, cos_tol=args.cos_tol,
                           criterion=("repro floor == 0 AND the three cosine estimators "
                                      "(split / total-minus-pred / total-minus-state) "
                                      "agree within cos_tol"),
                           passed=ok, per_tensor=gate_rows, estimators=spread_rows)
        print(f"GATE max_rel_err={worst:.3e} repro_floor={worst_repro:.3e} "
              f"estimator_spread={spread_max:.3e} cos_tol={args.cos_tol:.1e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

        rows, bykind = [], {}
        for name, kind, blk, _p, _sl in specs:
            gt = tot["total"].get(name)
            if gt is None:
                continue
            gs = tot["state"].get(name)
            gp = tot["pred"].get(name)
            zero = torch.zeros_like(gt)
            gs = zero if gs is None else gs
            gp = zero if gp is None else gp
            ns, np_ = float(gs.norm()), float(gp.norm())
            c = cos(gs, gp)
            row = dict(
                name=name, kind=kind, block=blk,
                cos_state_pred=c,
                norm_state=ns, norm_pred=np_,
                norm_total=float(gt.norm()),
                ratio_pred_over_state=(np_ / ns) if ns > 0 else None,
                stream_exclusive=(ns == 0.0 or np_ == 0.0),
                # CONTROL 2: same stream, disjoint halves of the same 32 sequences.
                control_batch_split_state=cos(res["state"][0].get(name, zero),
                                              res["state"][1].get(name, zero)),
                control_batch_split_pred=cos(res["pred"][0].get(name, zero),
                                             res["pred"][1].get(name, zero)),
                control_batch_split_total=cos(res["total"][0][name], res["total"][1][name]),
                # cross-stream AND cross-half: kills "the zero is just noise cancelling"
                control_cross_stream_cross_half=cos(res["state"][0].get(name, zero),
                                                    res["pred"][1].get(name, zero)),
                # additivity-imposed cross-checks of the SAME cosine (see the gate)
                cos_total_minus_pred=cos(gt - gp, gp),
                cos_total_minus_state=cos(gs, gt - gs),
                # THE headline number: the naive cosine is attenuated by minibatch noise
                # by ~sqrt(r_state * r_pred); this divides that out.  None when the
                # expected gradient is not resolved (see disattenuated_cos).
                cos_disattenuated=disattenuated_cos(
                    res["state"][0].get(name, zero), res["state"][1].get(name, zero),
                    res["pred"][0].get(name, zero), res["pred"][1].get(name, zero)),
            )
            rows.append(row)
            if c is not None:
                bykind.setdefault(kind, []).append(row)
        out["tensors"] = rows

        def agg(key, sel):
            v = [r[key] for r in sel if r.get(key) is not None]
            if not v:
                return None
            return dict(n=len(v), mean=float(np.mean(v)), absmean=float(np.mean(np.abs(v))),
                        min=float(np.min(v)), max=float(np.max(v)))

        out["aggregate"] = {k: dict(cos_state_pred=agg("cos_state_pred", v),
                                    cos_disattenuated=agg("cos_disattenuated", v),
                                    control_batch_split_state=agg("control_batch_split_state", v),
                                    control_batch_split_pred=agg("control_batch_split_pred", v),
                                    control_cross_stream_cross_half=agg("control_cross_stream_cross_half", v),
                                    ratio_pred_over_state=agg("ratio_pred_over_state", v))
                            for k, v in bykind.items()}
        allrows = [r for v in bykind.values() for r in v]
        out["aggregate"]["ALL"] = dict(
            cos_state_pred=agg("cos_state_pred", allrows),
            cos_disattenuated=agg("cos_disattenuated", allrows),
            control_batch_split_state=agg("control_batch_split_state", allrows),
            control_batch_split_pred=agg("control_batch_split_pred", allrows),
            control_cross_stream_cross_half=agg("control_cross_stream_cross_half", allrows),
            ratio_pred_over_state=agg("ratio_pred_over_state", allrows))

    suffix = "_init" if args.untrained else ("_untied" if args.untied else "")
    path = args.out or os.path.join(C.RESULTS_DIR, f"{ANALYSIS}_{run}{suffix}.json")
    C.save_json(path, out)
    print(json.dumps(out.get("aggregate", {}), indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
