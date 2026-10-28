"""A13 -- how DEEP into the state tower the readout actually has to read.

A12 asks which blocks' compute is load-bearing.  A13 asks the complementary question about
the READ INTERFACE, and it is the one with an architectural consequence: `read_map="post"`
has pred block ``i`` read state level ``f(i) = i + 1``, so the readout tower's last block
reads the state tower's OUTPUT.  If a shallower read is just as good, every state block
above that level is computing something nothing ever consumes, and the architecture can be
made cheaper.

Two families of rewritten read maps, both pure eval-time interventions on
``model.read_levels`` (the list `TwoTowerModel._forward_towers` indexes per pred block):

    cap    f(i) = min(i + 1, K)   -- nothing may be read above state level K.  K = L_s is
                                     the unmodified `post` map and MUST reproduce the
                                     native NLL exactly; that is this script's gate.
    floor  f(i) = max(i + 1, K)   -- nothing may be read BELOW level K, i.e. the shallow
                                     pred blocks are forced to read deep state.  The
                                     control direction: it says whether the depth ALIGNMENT
                                     matters or only the depth CEILING does.

Both are expressed with levels the trained model already produces (level j's k/v come from
state block j's own projections, or -- under ``read_source="pred_proj"`` -- from the pred
block's read projection applied to the state residual at level j), so no weight is
invented and no tensor is reshaped.

IMPORTANT, and the caption must say it: a cap applied at eval time to a model TRAINED with
the uncapped map is a LESION, not a trained architecture.  A flat cap curve is evidence
that the deep state levels carry no information the readout uses, which MOTIVATES training
an ``L_s = K`` / ``L_p = 12`` two-tower; it is not itself that result.  The FLOPs number
this script prints is therefore labelled `proposal_not_measured`.

Scoring: the same 512-sequence sub-sweep as A12/A2, paired against the same native.

Usage:
  a13_read_depth_cap.py --run <run|role> [--caps 2,4,6,8,10,12] [--sub-seqs 512]
                        [--floors/--no-floors] [--flat-tol 0.01]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

from plotting.flops import two_tower_flops_per_token  # noqa: E402
from modeling.models.two_tower.core import read_level  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--caps", default="2,4,6,8,10,12")
    ap.add_argument("--sub-seqs", type=int, default=C.SUB_SWEEP_SEQS)
    ap.add_argument("--floors", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--full-sweep-control", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--flat-tol", type=float, default=0.01,
                    help="a cap whose paired delta is below this is called 'flat' and "
                         "enters the FLOPs proposal")
    ap.add_argument("--gate-tol", type=float, default=1e-6)
    ap.add_argument("--impose-map", default=None, choices=["pre", "post", "top", "final"],
                    help="eval-time-only: force read_levels to this READ_MAP's f(i) "
                         "(computed from this checkpoint's own L_s/L_p), independent of "
                         "the checkpoint's trained read_map. Mirrors the cap/floor "
                         "machinery but sweeps a whole alternate map rather than a "
                         "depth ceiling/floor -- e.g. imposing 'post' on a read_map='final' "
                         "(fully-sequential) checkpoint is the reverse of the existing "
                         "floor-L_s lesion (imposing 'final' via floor) on a "
                         "read_map='post' checkpoint. Skipped entirely (no caps/floors "
                         "run) when passed; use a separate invocation from the cap sweep.")
    args = ap.parse_args()

    run = C.resolve(args.run)
    analysis_name = "a13_read_depth_cap"
    if args.impose_map is not None:
        # Distinct output file: an impose-map run must never clobber the standard
        # cap/floor sweep result for the same run (or vice versa).
        analysis_name = f"a13_read_depth_cap_impose_{args.impose_map}"
    out_path = args.out or C.result_path(analysis_name, run)
    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    C.assert_node_local_triton()

    model, cfg, family, ckpath = C.load(run)
    if family != "two_tower":
        C.save_json(out_path, dict(analysis="a13_read_depth_cap", run=run,
                                   model_id=C.mid_of(run), family=family,
                                   checkpoint=ckpath, skipped=True,
                                   reason="read_levels is a two-tower concept; the joint "
                                          "families read their own interleaved stack"))
        return

    val_path = C.val_path_of(cfg)
    mc = cfg.model.config
    L_s, L_p = int(model.state_n_layer), int(model.pred_n_layer)
    native_levels = list(model.read_levels)
    max_available = L_s if model.needs_final_level else L_s - 1
    print(f"run={run} L_state={L_s} L_pred={L_p} read_map={mc.read_map} "
          f"read_source={model.read_source} native_levels={native_levels} "
          f"max_available_level={max_available} ckpt={ckpath}", flush=True)

    t0 = time.time()
    native, ntok, nseq = C.sweep_nll(model, val_path, args.sub_seqs)
    print(f"  native sub-sweep nll={native:.6f} ({ntok} tok, {nseq} seq, "
          f"{time.time()-t0:.0f}s)", flush=True)

    # Baseline FLOPs/token for the trained geometry, from the project's own accountant.
    def flops_for(state_n_layer):
        return two_tower_flops_per_token(
            state_n_layer=int(state_n_layer), pred_n_layer=L_p,
            state_hidden=int(model.state_hidden), pred_hidden=int(model.pred_hidden),
            state_intermediate=int(model.state_intermediate),
            pred_intermediate=int(model.pred_intermediate),
            # `read_map` enters the FLOPs count ONLY through the level-L_s read head, which
            # is not allocated at all under read_source="pred_proj". At L_s != L_p the last
            # pred block reads level L_s, which "final" charges exactly.
            read_map="final" if int(state_n_layer) != L_p else str(mc.read_map),
            read_source=str(model.read_source),
            head_dim=int(model.head_dim),
            block_size=int(model.config.block_size),
        )

    base_flops = flops_for(L_s)

    res = dict(
        analysis=analysis_name, run=run, model_id=C.mid_of(run),
        label=C.label_of(run), family=family, checkpoint=ckpath,
        geometry=dict(state_n_layer=L_s, pred_n_layer=L_p,
                      state_hidden=int(model.state_hidden),
                      pred_hidden=int(model.pred_hidden),
                      state_intermediate=int(model.state_intermediate),
                      pred_intermediate=int(model.pred_intermediate),
                      read_map=str(mc.read_map), read_source=str(model.read_source),
                      native_read_levels=native_levels,
                      max_available_level=max_available),
        scoring=dict(kind="sub-sweep", n_seq=nseq, n_tokens=ntok, val_bin=val_path,
                     paired_floor=C.PAIRED_FLOOR),
        native_sub_sweep_nll=round(native, 6),
        baseline_flops_per_token=base_flops,
        caps={}, floors={},
    )
    if args.full_sweep_control:
        t = time.time()
        fn, ft, fs = C.sweep_nll(model, val_path, None)
        res["native_full_sweep_nll"] = round(fn, 6)
        res["scoring"]["full_sweep_tokens"] = ft
        res["scoring"]["sub_sweep_offset"] = round(native - fn, 6)
        print(f"  native FULL sweep nll={fn:.6f} ({ft} tok, {time.time()-t:.0f}s)",
              flush=True)
    C.save_json(out_path, res)

    if args.impose_map is not None:
        imposed_levels = [read_level(i, L_s, L_p, args.impose_map) for i in range(L_p)]
        assert max(imposed_levels) <= max_available, (
            f"imposed map {args.impose_map!r} needs level {max(imposed_levels)}, this "
            f"model produces at most {max_available}")
        model.read_levels = list(imposed_levels)
        t = time.time()
        nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
        model.read_levels = list(native_levels)
        d = nll - native
        res["imposed_map"] = dict(
            map=args.impose_map, read_levels=imposed_levels,
            val_nll=round(nll, 6), delta=round(d, 6), seconds=round(time.time() - t, 1))
        print(f"  imposed map={args.impose_map} levels={imposed_levels} nll={nll:.6f} "
              f"delta={d:+.6f} ({time.time()-t:.0f}s)", flush=True)
        if imposed_levels == native_levels:
            # Sanity gate: imposing the checkpoint's OWN map must exactly reproduce the
            # native sub-sweep NLL, otherwise the rewrite path itself is suspect.
            res["imposed_map_self_gate_delta"] = round(d, 8)
            res["imposed_map_self_gate_passed"] = bool(abs(d) <= args.gate_tol)
            print(f"  GATE impose-own-map delta = {d:+.8f} (tol {args.gate_tol})",
                  flush=True)
            if not res["imposed_map_self_gate_passed"]:
                C.save_json(out_path, res)
                print("\nA13 FAILED: imposing this checkpoint's own read_map changed "
                      "the loss.", flush=True)
                sys.exit(1)
        C.save_json(out_path, res)
        print(f"\nA13 (impose-map={args.impose_map}) done.", flush=True)
        return

    def scored(bucket, K, levels):
        assert max(levels) <= max_available, (
            f"level {max(levels)} is not produced by this model (max {max_available})")
        model.read_levels = list(levels)
        t = time.time()
        nll, _, _ = C.sweep_nll(model, val_path, args.sub_seqs)
        model.read_levels = list(native_levels)
        d = nll - native
        res[bucket][str(K)] = dict(K=K, read_levels=list(levels), val_nll=round(nll, 6),
                                   delta=round(d, 6), seconds=round(time.time() - t, 1))
        print(f"  {bucket[:-1]} K={K:<3d} levels={levels}  nll={nll:.6f}  "
              f"delta={d:+.6f}  ({time.time()-t:.0f}s)", flush=True)
        C.save_json(out_path, res)
        return d

    # --- GATE: K = L_s is the identity rewrite of `post`; it must be bit-for-bit native.
    gate_levels = [min(i + 1, L_s) for i in range(L_p)]
    if gate_levels == native_levels:
        d0 = scored("caps", L_s, gate_levels)
        res["identity_gate_delta"] = round(d0, 8)
        res["identity_gate_passed"] = bool(abs(d0) <= args.gate_tol)
        print(f"  GATE identity-rewrite delta = {d0:+.8f} (tol {args.gate_tol})", flush=True)
        if not res["identity_gate_passed"]:
            C.save_json(out_path, res)
            print("\nA13 FAILED: rewriting read_levels to the map the model already has "
                  "changed the loss, so the rewrite is not the intervention it claims.",
                  flush=True)
            sys.exit(1)
    else:
        res["identity_gate_passed"] = None
        res["identity_gate_note"] = (
            f"native map is {native_levels}, not min(i+1, {L_s}); the identity gate does "
            f"not apply to this read_map")
    C.save_json(out_path, res)

    for K in caps:
        if str(K) in res["caps"]:
            continue
        scored("caps", K, [min(i + 1, K) for i in range(L_p)])

    if args.floors:
        for K in caps:
            levels = [max(i + 1, K) for i in range(L_p)]
            if max(levels) > max_available:
                print(f"  floor K={K}: SKIP (needs level {max(levels)}, model produces "
                      f"at most {max_available})", flush=True)
                continue
            scored("floors", K, levels)

    # --- the proposal ------------------------------------------------------------
    flat = sorted(int(k) for k, v in res["caps"].items()
                  if v["delta"] < args.flat_tol and int(k) < L_s)
    prop = dict(flat_tol=args.flat_tol,
                flat_caps=flat,
                proposal_not_measured=True,
                note="An eval-time cap on a model trained uncapped is a LESION. A flat cap "
                     "curve motivates training an L_s=K / L_p=12 two-tower; it is not that "
                     "model's measured NLL. Such a model also needs a read_map that CAPS "
                     "(f(i)=min(i+1,K)) rather than COMPRESSES (f(i)=floor((i+1)L_s/L_p)) -- a "
                     "one-line addition to "
                     "src/modeling/models/two_tower/core.py:read_level, not made here.")
    if flat:
        K = min(flat)
        f_k = flops_for(K)
        prop.update(
            shallowest_flat_cap=K,
            proposed_state_n_layer=K, proposed_pred_n_layer=L_p,
            proposed_flops_per_token=f_k,
            flops_saving_frac=round(1.0 - f_k / base_flops, 6),
            flops_saving_pct=round(100.0 * (1.0 - f_k / base_flops), 3),
            cap_delta_at_K=res["caps"][str(K)]["delta"],
        )
    prop["flops_by_state_depth"] = {
        str(K): dict(flops_per_token=flops_for(K),
                     saving_pct=round(100.0 * (1.0 - flops_for(K) / base_flops), 3),
                     cap_delta=res["caps"].get(str(K), {}).get("delta"))
        for K in sorted(set(caps) | {L_s}) if 0 < K <= L_s}
    res["proposal"] = prop
    C.save_json(out_path, res)
    print(f"\nA13 done.  flat caps (<{args.flat_tol}): {flat or 'none'}", flush=True)


if __name__ == "__main__":
    main()
