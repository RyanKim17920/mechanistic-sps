"""A2 -- causal distance-band knockout of the memory read.  -> F1 (hero, with A5)

The causal test of whether the long-range attention mass is load-bearing or decoration.

Intervention.  For the readout stream's read of the memory keys, restrict which memory
keys a query may use by TOKEN DISTANCE, then re-score.  Two bands, so the two curves can
cross -- which is the figure's whole content:

    near   keep memory keys at distance <= cap   (the far band is knocked out)
    far    keep memory keys at distance >  cap   (the near band is knocked out)

and two interventions on the knocked-out band, because the choice is itself a confound:

    mask   drop those keys and renormalise the softmax.  Simple, but a renormalised
           softmax over a truncated key set is an OFF-DISTRIBUTION input to everything
           downstream, and an OOD intervention inflates measured importance.
    mean   replace the knocked-out band's contribution with the calibration-set MEAN
           value vector, keeping that band's total attention weight.  This is the
           in-distribution corruption Zhang & Nanda (arXiv:2309.16042) recommend.
           Symmetric token replacement is not available for a distance predicate -- the
           predicate is not a property of any token -- so this is the closest
           in-distribution analogue, and the caption must say so.

    Both are computed from ONE pair of partial softmaxes per layer, merged with the
    log-sum-exp identity the model's own `merge_attn_lse` implements.

How the intervention reaches the model.  Never by re-implementing a forward pass:
  * two-tower -- `TwoTowerModel._attend` is replaced, and the visibility it starts from
    is `_MaskSet.bool_mask`, i.e. THE MODEL'S OWN MASK.  Nothing is re-derived.
  * joint (tied SPS) -- the fused Triton kernel is replaced by an explicit attention with
    the interleaved-2T visibility (checked by the CONTROL below).  Projections, routing,
    RoPE and dtypes stay the model's.

CONTROL, and it is a gate, not a formality: the patched path at cap = infinity must
reproduce the UNPATCHED model's sub-sweep NLL.  If it does not, the explicit attention is
not the model's attention and every delta below is meaningless; the script says so and
exits non-zero.

Scoring uses the 512-sequence sub-sweep (2.1 M tokens).  Every delta is a PAIRED
comparison -- one checkpoint, one val slice, one deterministic order, only the
intervention differing -- so the resolvable floor is the paired floor (~0.0005), not the
0.0022 seed floor; deltas above ~0.005 are real at one seed.  The control's own full-sweep
NLL is reported alongside so the sub-sweep's offset is visible.

Usage:
  a2_knockout.py --run <run|role> [--caps 16,64,256,1024] [--sub-seqs 512]
                 [--state-queries/--no-state-queries] [--full-sweep-control]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import common as C
from modeling.models.two_tower.attention import merge_attn_lse

CHUNK = 256          # query chunk for the explicit attention; bounds score memory
CALIB_SEQ = 32       # calibration sequences for the mean value vectors


# --------------------------------------------------------------------------------------
# explicit attention with a knocked-out key band
# --------------------------------------------------------------------------------------
def attend_explicit(q, k, v, scale, vis, ablate, v_mean, mode):
    """q (b,h,Tq,d); k,v (b,h,K,d); vis/ablate (b,Tq,K) bool -> out (b,h,Tq,d) float32.

    `ablate` marks (query, key) pairs whose contribution is removed; `mode` decides what
    replaces it.  A query left with no kept keys yields an all-zero output row, which is
    what attention over an empty key set must give (the convention `merge_attn_lse` uses).
    """
    b, h, Tq, d = q.shape
    outs = []
    for s in range(0, Tq, CHUNK):
        qc = q[:, :, s:s + CHUNK].float()
        sc = torch.matmul(qc, k.float().transpose(-1, -2)) * scale       # (b,h,c,K)
        visc = vis[:, s:s + CHUNK]
        abc = ablate[:, s:s + CHUNK] & visc
        keep = visc & ~abc
        sk = sc.masked_fill(~keep.unsqueeze(1), float("-inf"))
        lse_k = torch.logsumexp(sk, dim=-1)                              # (b,h,c)
        pk = torch.nan_to_num(torch.exp(sk - lse_k.unsqueeze(-1)), nan=0.0)
        out_k = torch.matmul(pk, v.float())
        if mode == "mean":
            sa = sc.masked_fill(~abc.unsqueeze(1), float("-inf"))
            lse_a = torch.logsumexp(sa, dim=-1)
            vm = v_mean.view(1, h, 1, d).expand(b, h, out_k.shape[2], d).float()
            out, _ = merge_attn_lse(out_k, lse_k, vm, lse_a)
        else:
            out = torch.where(torch.isfinite(lse_k).unsqueeze(-1), out_k,
                              torch.zeros_like(out_k))
        outs.append(out)
        del sc, sk, pk
    return torch.cat(outs, dim=2)


class Knockout:
    """Owns the intervention state and the calibration accumulators for one model."""

    def __init__(self, family, window, n_head, head_dim, target_stream="pred"):
        self.family = family
        self.window = window
        self.n_head = n_head
        self.head_dim = head_dim
        self.target_stream = target_stream      # which QUERY stream is intervened on
        self.mode = "off"                       # off | calibrate | apply
        self.cap = None
        self.band = "near"                      # near = keep <= cap; far = keep > cap
        self.ablation = "mask"                  # mask | mean
        self.sum = {}                           # layer -> (h, d) float64 sum of v
        self.cnt = {}
        self.mean = {}                          # layer -> (h, d) float32 cuda

    def calibrate_add(self, layer, v, keysel):
        """v (b,h,K,d); keysel (K,) bool over the ablatable key population."""
        vv = v.float()[:, :, keysel]
        s = vv.sum(dim=(0, 2)).double()
        n = vv.shape[0] * vv.shape[2]
        if layer in self.sum:
            self.sum[layer] += s
            self.cnt[layer] += n
        else:
            self.sum[layer], self.cnt[layer] = s, n

    def finalise(self):
        self.mean = {l: (self.sum[l] / self.cnt[l]).float() for l in self.sum}
        return {str(l): round(float(m.abs().mean()), 6) for l, m in self.mean.items()}

    def band_ablate(self, dist, memory_key, query_in_target):
        """(query, key) pairs whose contribution is removed."""
        if self.cap is None:
            return torch.zeros_like(memory_key.expand_as(dist))
        far = dist > self.cap
        drop = far if self.band == "near" else ~far
        return drop & memory_key & query_in_target


# --------------------------------------------------------------------------------------
# patches
# --------------------------------------------------------------------------------------
def patch_two_tower(model, ko):
    """Replace `TwoTowerModel._attend`.  Visibility comes from the model's own _MaskSet."""
    layer_counter = {"i": 0, "calls": 0}

    def _attend(self, q, k, v, masks, stream):
        vis = masks.bool_mask()                                 # the model's OWN mask
        b, h, t, d = q.shape
        K = k.shape[2]
        scale = 1.0 / math.sqrt(d)
        dev = q.device
        kv = torch.arange(K, device=dev)
        is_mem = (kv < t).view(1, 1, -1)
        k_tok = torch.where(kv < t, kv, kv - t).view(1, 1, -1)
        q_idx = torch.arange(t, device=dev).view(1, -1, 1)
        dist = (q_idx - k_tok).clamp(min=0)

        idx = layer_counter["i"]          # state blocks then pred blocks, per forward
        layer_counter["i"] += 1
        if ko.mode == "apply" and stream == ko.target_stream:
            layer_counter["calls"] += 1
        if ko.mode == "calibrate":
            if stream == ko.target_stream:
                ko.calibrate_add(idx, v, (kv < t))
            return _native_attend(self, q, k, v, masks, stream)
        if ko.mode == "off" or stream != ko.target_stream:
            return _native_attend(self, q, k, v, masks, stream)
        ablate = ko.band_ablate(dist, is_mem, torch.ones_like(q_idx, dtype=torch.bool))
        out = attend_explicit(q, k, v, scale, vis, ablate.expand(b, t, K),
                              ko.mean.get(idx), ko.ablation)
        return out.to(q.dtype)

    _native_attend = type(model)._attend
    model._attend = _attend.__get__(model, type(model))
    model._knockout_reset = lambda: layer_counter.update(i=0)
    model._knockout_calls = layer_counter
    return model


def patch_joint(ko):
    """Replace the fused Triton SPS kernel with the explicit, maskable equivalent.

    Only the KERNEL is replaced: the model's own projections, RoPE, routing and dtype
    handling all still run.  The visibility rebuilt here is the interleaved-2T SPS mask
    (common._joint_views) -- state (even) keys persistent at any distance, pred (odd) keys
    only within the window, causality in SLOT space, plus the document mask.
    """
    import modeling.models.sps.core as score
    # The kernel is a MODULE GLOBAL bound at import time. A missed patch is silent -- the
    # sweep runs the real kernel and every delta comes out exactly 0.000000, which looks
    # like a null result rather than a bug. Hence the assert
    # below that the patch actually fired.
    native = score.triton_sps_sliding_attention
    assert native is not None, "patch_joint replaces the Triton SPS kernel; it needs a GPU"
    counter = {"i": 0, "calls": 0}

    def kernel(q, k, v, scale, window, warp_specialize=False,
               documents_idx_BxT=None, persistent_key_window=None):
        # q,k,v: (b, h, 2T, d)
        b, h, two_t, d = q.shape
        dev = q.device
        idx = counter["i"] % 10_000
        counter["i"] += 1
        counter["calls"] += 1
        layer = idx
        if ko.mode == "calibrate":
            ko.calibrate_add(layer, v, (torch.arange(two_t, device=dev) % 2 == 0))
            return native(q, k, v, scale, window, warp_specialize=warp_specialize,
                          documents_idx_BxT=documents_idx_BxT,
                          persistent_key_window=persistent_key_window)
        if ko.mode == "off":
            return native(q, k, v, scale, window, warp_specialize=warp_specialize,
                          documents_idx_BxT=documents_idx_BxT,
                          persistent_key_window=persistent_key_window)
        kv = torch.arange(two_t, device=dev)
        k_is_pred = (kv % 2 == 1)
        k_tok = kv // 2
        q_slot = torch.arange(two_t, device=dev)
        q_tok = q_slot // 2
        q_is_pred = (q_slot % 2 == 1)
        dist = (q_tok.view(-1, 1) - k_tok.view(1, -1)).clamp(min=0)
        vis = (kv.view(1, -1) <= q_slot.view(-1, 1))
        vis = vis & ((~k_is_pred.view(1, -1)) | (dist <= window))
        vis = vis.unsqueeze(0).expand(b, two_t, two_t)
        if documents_idx_BxT is not None:
            dm = documents_idx_BxT.unsqueeze(-1) == documents_idx_BxT.unsqueeze(1)
            vis = vis & dm
        target_is_pred = (ko.target_stream == "pred")
        q_in_target = (q_is_pred if target_is_pred else ~q_is_pred).view(1, -1, 1)
        memory_key = (~k_is_pred).view(1, 1, -1)
        ablate = ko.band_ablate(dist.unsqueeze(0), memory_key, q_in_target)
        out = attend_explicit(q, k, v, scale, vis, ablate.expand(b, two_t, two_t),
                              ko.mean.get(layer), ko.ablation)
        return out.to(q.dtype)

    def install():
        counter["i"] = 0
        score.triton_sps_sliding_attention = kernel

    def restore():
        score.triton_sps_sliding_attention = native

    return install, restore, (lambda: counter.update(i=0)), counter


# --------------------------------------------------------------------------------------
# STANDARD (single-stream Transformer): context-window knockout baseline
# --------------------------------------------------------------------------------------
def patch_standard(model, ko):
    """Replace each block's `attn.forward` (instance attribute) with the explicit,
    maskable attention.  Everything but the score/softmax is the module's own code path:
    `c_attn` -> RoPE (`apply_rotary_emb`) -> attention -> `c_proj` -> `resid_dropout`.

    Visibility is plain causal AND same-document, the document index being the model's
    own `generate_document_idx` of the input (captured by a forward pre-hook, exactly what
    the Triton/flex paths mask on).  In the single stream EVERY key is "memory", so the
    band predicate of `Knockout.band_ablate` acts on all keys; `ko.layers` (a set of block
    indices) plays the role `target_stream` plays in the two-stream families: only those
    blocks' queries are restricted, every other block runs the model's native attention.
    """
    from modeling.models.model import apply_rotary_emb
    blocks = list(model.transformer.h)
    state = {"docs": None, "calls": 0}

    def pre(mod, args, kwargs):
        idx = args[0] if args else kwargs["idx_BxT"]
        state["docs"] = mod.generate_document_idx(idx)
    handle = model.register_forward_pre_hook(pre, with_kwargs=True)

    def make(i, a):
        native = type(a).forward

        def fwd(x, freqs_cis, attn_block_mask=None, **kw):
            if ko.mode == "calibrate" or ko.mode == "off" or i not in ko.layers:
                if ko.mode == "calibrate" and i in ko.layers:
                    B, T, Cc = x.size()
                    v = a.c_attn(x).split(a.hidden_size, dim=2)[2]
                    v = v.view(B, T, a.n_head, Cc // a.n_head).transpose(1, 2)
                    ko.calibrate_add(i, v, torch.ones(T, dtype=torch.bool, device=x.device))
                return native(a, x, freqs_cis, attn_block_mask=attn_block_mask, **kw)
            state["calls"] += 1
            B, T, Cc = x.size()
            hd = Cc // a.n_head
            q, k, v = a.c_attn(x).split(a.hidden_size, dim=2)
            q = q.view(B, T, a.n_head, hd)
            k = k.view(B, T, a.n_head, hd)
            v = v.view(B, T, a.n_head, hd).transpose(1, 2)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            dev = x.device
            pos = torch.arange(T, device=dev)
            dist = (pos.view(-1, 1) - pos.view(1, -1)).clamp(min=0).unsqueeze(0)  # (1,T,T)
            vis = (pos.view(1, -1) <= pos.view(-1, 1)).unsqueeze(0).expand(B, T, T)
            docs = state["docs"]
            if docs is not None:
                vis = vis & (docs.unsqueeze(-1) == docs.unsqueeze(1))
            every = torch.ones(1, 1, T, dtype=torch.bool, device=dev)
            ablate = ko.band_ablate(dist, every, torch.ones(1, T, 1, dtype=torch.bool,
                                                            device=dev))
            y = attend_explicit(q, k, v, 1.0 / math.sqrt(hd), vis, ablate.expand(B, T, T),
                                ko.mean.get(i), ko.ablation)
            y = y.to(q.dtype).transpose(1, 2).contiguous().view(B, T, Cc)
            return a.resid_dropout(a.c_proj(y))
        return fwd

    for i, blk in enumerate(blocks):
        blk.attn.forward = make(i, blk.attn)

    def restore():
        for blk in blocks:
            blk.attn.__dict__.pop("forward", None)
        handle.remove()
    return restore, state


def run_standard(args, run, out_path, caps, model, cfg, family, ckpath):
    """The context-window knockout for the single-stream Transformer: the reference the
    separated models' read-reach numbers lack ("any LM needs long context").

    Same semantics as the two-stream A2 arms: a query at token t keeps keys at distance
    <= W (near band; far band masked-and-renormalised, or mean-replaced keeping the far
    band's total attention weight with the calibration-set mean value vector), or only
    keys at distance > W (far band, mask).  Applied (a) in ALL blocks -- nothing in the
    model can then see beyond W per layer (the receptive field still compounds across
    layers, as it does through the separated models' state tower); (b) in the LAST N
    blocks only -- the closer analogue of the separated models, whose state tower keeps
    full context while only the prediction stream's read is capped.
    """
    L = len(model.transformer.h)
    mc = cfg.model.config
    n_head = int(mc.n_head)
    head_dim = int(mc.hidden_size) // n_head
    val_path = C.val_path_of(cfg)
    data = C.val_memmap(cfg)
    last_ns = [int(x) for x in args.last_n.split(",") if x.strip()]
    assert all(1 <= n <= L for n in last_ns), f"--last-n must be in [1, {L}]"

    print(f"run={run} family={family} L={L} ckpt={ckpath}", flush=True)
    t0 = time.time()
    native_nll, ntok, nseq = C.sweep_nll(model, val_path, args.sub_seqs)
    print(f"  native sub-sweep nll={native_nll:.6f} ({ntok} tok, {nseq} seq, "
          f"{time.time()-t0:.0f}s)", flush=True)
    res = dict(analysis="a2_ctx_knockout", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               arm="trained", n_layer=L, caps=caps, last_n=last_ns,
               semantics=("single-stream context-window knockout: queries of the targeted "
                          "blocks keep keys at token distance <= cap (near) / > cap (far); "
                          "knocked-out band masked+renormalised (mask) or replaced by the "
                          "calibration mean value vector at that band's total weight "
                          "(mean); untargeted blocks run native attention"),
               scoring=dict(kind="sub-sweep", n_seq=nseq, n_tokens=ntok,
                            paired_floor=C.PAIRED_FLOOR),
               native_sub_sweep_nll=round(native_nll, 6), sweeps={})
    if args.full_sweep_control:
        fn, ft, fs = C.sweep_nll(model, val_path, None)
        res["native_full_sweep_nll"] = round(fn, 6)
        res["scoring"]["full_sweep_tokens"] = ft

    ko = Knockout(family, None, n_head, head_dim, target_stream="single")
    ko.layers = set(range(L))
    restore, state = patch_standard(model, ko)

    ko.mode = "calibrate"
    for j in C.seq_starts(data, CALIB_SEQ):
        X, Y = C.batch_of(data, j)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(X, Y)
    ko.mode = "off"
    res["calibration"] = dict(n_seq=CALIB_SEQ, mean_abs_value=ko.finalise())
    print(f"  calibrated mean value vectors over {CALIB_SEQ} sequences", flush=True)

    def scored(tag, layers, cap, band, ablation):
        ko.layers, ko.cap, ko.band, ko.ablation = set(layers), cap, band, ablation
        ko.mode = "apply"
        t = time.time()
        nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
        ko.mode = "off"
        entry = dict(layers=sorted(layers), cap=cap, band=band, ablation=ablation,
                     val_nll=round(nll, 6), seconds=round(time.time() - t, 1))
        res["sweeps"][tag] = entry
        print(f"  {tag:34s} nll={nll:.6f}  ({entry['seconds']:.0f}s)", flush=True)
        return nll

    ctrl = scored("control_uncapped", range(L), None, "near", "mask")
    if state["calls"] == 0:
        print("\nA2 (standard) FAILED: the attention patch never fired.", flush=True)
        sys.exit(1)
    res["patched_attention_calls"] = int(state["calls"])
    res["control_uncapped_nll"] = round(ctrl, 6)
    res["control_delta_vs_native"] = round(ctrl - native_nll, 6)
    passed = abs(ctrl - native_nll) <= args.gate_tol
    res["control_gate_passed"] = bool(passed)
    print(f"  GATE patched-vs-native delta = {ctrl - native_nll:+.6f} "
          f"(tolerance {args.gate_tol})", flush=True)
    if not passed:
        C.save_json(out_path, res)
        print("\nA2 (standard) CONTROL GATE FAILED.", flush=True)
        sys.exit(1)
    C.save_json(out_path, res)

    for cap in caps:
        for band, ablations in (("near", ("mask", "mean")), ("far", ("mask",))):
            for ab in ablations:
                tag = f"all_{band}{cap}_{ab}"
                nll = scored(tag, range(L), cap, band, ab)
                res["sweeps"][tag]["delta_nll"] = round(nll - ctrl, 6)
        for n in last_ns:
            for ab in ("mask", "mean"):
                tag = f"last{n}_near{cap}_{ab}"
                nll = scored(tag, range(L - n, L), cap, "near", ab)
                res["sweeps"][tag]["delta_nll"] = round(nll - ctrl, 6)
        C.save_json(out_path, res)

    restore()
    C.save_json(out_path, res)
    print("\nsingle-stream context knockout, dNLL vs uncapped control:", flush=True)
    for cap in caps:
        row = [f"all mask {res['sweeps'][f'all_near{cap}_mask']['delta_nll']:+.4f}",
               f"all mean {res['sweeps'][f'all_near{cap}_mean']['delta_nll']:+.4f}"]
        row += [f"last{n} mask {res['sweeps'][f'last{n}_near{cap}_mask']['delta_nll']:+.4f}"
                for n in last_ns]
        print(f"  cap {cap:5d}  " + "  ".join(row), flush=True)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--caps", default="16,64,256,1024")
    ap.add_argument("--sub-seqs", type=int, default=C.SUB_SWEEP_SEQS)
    ap.add_argument("--state-queries", action=argparse.BooleanOptionalAction, default=None,
                    help="also knock out the MEMORY stream's own read (joint arms only)")
    ap.add_argument("--full-sweep-control", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--gate-tol", type=float, default=0.002)
    ap.add_argument("--untrained", action="store_true",
                    help="run the identical knockout on a freshly initialised model of the "
                         "same config -- the design's null, which must be ~flat beyond the "
                         "trivial. Writes to a separate '_untrained' result file.")
    ap.add_argument("--standard-ctx", action="store_true",
                    help="single-stream Transformer: run the context-window knockout "
                         "baseline into a2_ctx_knockout_<run>.json (see run_standard)")
    ap.add_argument("--last-n", default="1,3,6,9",
                    help="--standard-ctx: also restrict only the last N blocks, per N")
    args = ap.parse_args()

    run = C.resolve(args.run)
    out_path = args.out or C.result_path(
        "a2_knockout" + ("_untrained" if args.untrained else ""), run)
    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    C.assert_node_local_triton()

    if args.untrained:
        model, cfg, family, ckpath = C.load_untrained(run, C.UNTRAINED_SEEDS[0])
    else:
        model, cfg, family, ckpath = C.load(run)
    if family == "standard" and args.standard_ctx and not args.untrained:
        # separate result name: the committed a2_knockout_<run>.json "skipped" record
        # (and every figure that globs a2_knockout_*) is left untouched.
        out = args.out or C.result_path("a2_ctx_knockout", run)
        return run_standard(args, run, out, caps, model, cfg, family, ckpath)
    if family == "standard":
        C.save_json(out_path, dict(analysis="a2_knockout", run=run, model_id=C.mid_of(run),
                                   family=family, checkpoint=ckpath, skipped=True,
                                   reason="single-stream model has no memory read to cap"))
        return
    mc = cfg.model.config
    window = 0 if family == "two_tower" else int(mc.window_size)   # two-tower: no pred keys
    n_head = int(mc.n_head)
    head_dim = int(mc.hidden_size) // n_head
    val_path = C.val_path_of(cfg)
    data = C.val_memmap(cfg)

    if args.state_queries is None:
        args.state_queries = family == "sps"

    print(f"run={run} family={family} window={window} ckpt={ckpath}", flush=True)
    t0 = time.time()
    native_nll, ntok, nseq = C.sweep_nll(model, val_path, args.sub_seqs)
    print(f"  native sub-sweep nll={native_nll:.6f} ({ntok} tok, {nseq} seq, "
          f"{time.time()-t0:.0f}s)", flush=True)

    res = dict(analysis="a2_knockout", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               arm=("untrained-init null" if args.untrained else "trained"),
               window_size=window, caps=caps,
               scoring=dict(kind="sub-sweep", n_seq=nseq, n_tokens=ntok,
                            paired_floor=C.PAIRED_FLOOR),
               native_sub_sweep_nll=round(native_nll, 6), sweeps={})
    if args.full_sweep_control:
        fn, ft, fs = C.sweep_nll(model, val_path, None)
        res["native_full_sweep_nll"] = round(fn, 6)
        res["scoring"]["full_sweep_tokens"] = ft
        print(f"  native FULL sweep nll={fn:.6f}", flush=True)

    ko = Knockout(family, window, n_head, head_dim, target_stream="pred")
    if family == "two_tower":
        patch_two_tower(model, ko)
        reset = model._knockout_reset
        restore = None
        patch_counter = model._knockout_calls
    else:
        install, restore, reset, patch_counter = patch_joint(ko)
        install()
    # The layer index is positional within ONE forward, so it must be rewound at the start
    # of every forward -- not once per sweep, which silently indexed the calibration means
    # by "layers seen since the sweep began" after the first batch.
    model.register_forward_pre_hook(lambda *a, **k: reset())

    # --- calibration: mean value vector per layer over a strided calibration set, which
    # is deliberately NOT the first 512 sequences the sub-sweep scores.
    ko.mode = "calibrate"
    for j in C.seq_starts(data, CALIB_SEQ):
        reset()
        X, Y = C.batch_of(data, j)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(X, Y)
    ko.mode = "off"
    res["calibration"] = dict(n_seq=CALIB_SEQ, mean_abs_value=ko.finalise())
    print(f"  calibrated mean value vectors over {CALIB_SEQ} sequences", flush=True)

    def scored(tag, stream, cap, band, ablation):
        ko.target_stream, ko.cap, ko.band, ko.ablation = stream, cap, band, ablation
        ko.mode = "apply"
        reset()
        t = time.time()
        nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
        ko.mode = "off"
        entry = dict(query_stream=stream, cap=cap, band=band, ablation=ablation,
                     val_nll=round(nll, 6), seconds=round(time.time() - t, 1))
        res["sweeps"][tag] = entry
        print(f"  {tag:34s} nll={nll:.6f}  ({entry['seconds']:.0f}s)", flush=True)
        return nll

    # --- GATE: the patched path with no cap must reproduce the unpatched model.
    ctrl = scored("control_uncapped", "pred", None, "near", "mask")
    # ... and it must actually BE the patched path. A patch that never fires reproduces
    # the control perfectly and then reports every knockout delta as exactly zero.
    if patch_counter["calls"] == 0:
        print("\nA2 FAILED: the attention patch never fired -- the sweep ran the model's "
              "own kernel, so every delta would be a fake zero.", flush=True)
        sys.exit(1)
    res["patched_attention_calls"] = int(patch_counter["calls"])
    res["control_uncapped_nll"] = round(ctrl, 6)
    res["control_delta_vs_native"] = round(ctrl - native_nll, 6)
    passed = abs(ctrl - native_nll) <= args.gate_tol
    res["control_gate_passed"] = bool(passed)
    print(f"  GATE patched-vs-native delta = {ctrl - native_nll:+.6f} "
          f"(tolerance {args.gate_tol})", flush=True)
    if not passed:
        C.save_json(out_path, res)
        print("\nA2 CONTROL GATE FAILED: the explicit attention is not reproducing the "
              "model's own attention, so no knockout delta below would mean anything.",
              flush=True)
        sys.exit(1)

    for cap in caps:
        for band, ablations in (("near", ("mask", "mean")), ("far", ("mask",))):
            for ab in ablations:
                nll = scored(f"pred_{band}{cap}_{ab}", "pred", cap, band, ab)
                res["sweeps"][f"pred_{band}{cap}_{ab}"]["delta_nll"] = round(nll - ctrl, 6)
        C.save_json(out_path, res)

    if args.state_queries:
        print("--- memory-stream queries", flush=True)
        sctrl = scored("control_uncapped_state", "state", None, "near", "mask")
        for cap in caps:
            nll = scored(f"state_near{cap}_mask", "state", cap, "near", "mask")
            res["sweeps"][f"state_near{cap}_mask"]["delta_nll"] = round(nll - sctrl, 6)
        res["control_uncapped_state_nll"] = round(sctrl, 6)
        C.save_json(out_path, res)

    if restore is not None:
        restore()
    C.save_json(out_path, res)

    print("\nreadout-stream memory read, dNLL vs uncapped control:", flush=True)
    for cap in caps:
        a = res["sweeps"].get(f"pred_near{cap}_mask", {}).get("delta_nll")
        b = res["sweeps"].get(f"pred_near{cap}_mean", {}).get("delta_nll")
        c = res["sweeps"].get(f"pred_far{cap}_mask", {}).get("delta_nll")
        print(f"  cap {cap:5d}  near-kept mask {a:+.4f}  near-kept mean {b:+.4f}  "
              f"far-kept mask {c:+.4f}", flush=True)


if __name__ == "__main__":
    main()
