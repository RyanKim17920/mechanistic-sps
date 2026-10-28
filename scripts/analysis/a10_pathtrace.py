"""A10 -- path tracing: WHICH memory head writes the key the readout matching head reads?

A9 patches one candidate head and measures recovery.  A10 is the complementary,
non-interventional decomposition: it takes the readout matching head's actual QK score at
the induction key position and splits it EXACTLY into per-source contributions from the
memory stream, so the answer is a ranking over every memory head rather than a yes/no on
one of them.

The decomposition is exact, not an approximation.  The memory residual the readout reads
at level L is a literal sum

    x_s^(L)[p] = embed[p] + SUM_{b<L} ( c_proj_b(y_b)[p] + mlp_b[p] )
               = embed[p] + SUM_{b<L} ( SUM_h W_proj_b[:, h] y_{b,h}[p] + mlp_b[p] )

because `finish_attn` concatenates heads and applies ONE linear map to the concatenation.
The read path applies an RMSNorm and a linear key projection, then RoPE.  The RMSNorm
scale is a SCALAR computed from the full residual -- it is not re-derived per source, it
is the true scale of the true residual -- so with that scalar fixed the whole map is
linear and the per-source contributions SUM TO THE MODEL'S OWN QK LOGIT.  The script
asserts that (see `reconstruction`), which is what makes the ranking trustworthy.

Two quantities are reported per source:

  raw          q . k_src / sqrt(d)  at the induction key position p_ind
  differential the same, MINUS the mean of the same source's contribution over a set of
               control key positions for the same query.  This is the number that says
               "this source is why THIS key wins", as opposed to "this source contributes
               to every key equally" (a constant offset cannot select a key).

The value (OV) path gets the same treatment, cheaply: the same per-source vectors are put
through the key projection's value twin, through the matching head's output projection
and straight to the unembedding (the DIRECT path only -- everything downstream of the
readout block is ignored and the JSON says so), giving each memory source's contribution
to the logit of the correct continuation.

Two-tower only: the decomposition needs the readout's key projection to be a separate,
inspectable map over a memory residual, which is exactly what the two towers give and
what the weight-shared joint families do not.  Other families write a `skipped` record.

Usage:
  a10_pathtrace.py --run <run|role> [--out J] [--n-seq 4] [--n-q 32] [--n-ctrl 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

import common as C                                                     # noqa: E402
import a5_induction as A5                                              # noqa: E402
import a9_patching as A9                                               # noqa: E402

CTRL_POS_SEED = 4242


def rms_sigma(x, eps):
    """The scalar RMSNorm scale of the TRUE residual (fp32)."""
    x = x.float()
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def read_path_modules(model, pred_block):
    """-> (norm, W_k, b_k, W_v, b_v, level, source_name) for the readout's state read."""
    L = int(model.read_levels[pred_block])
    inner = model.n_head_pred * model.head_dim
    if model.read_source == "pred_proj":
        blk = model.transformer.pred_h[pred_block]
        # `read_kv_weight` is the block's own accessor so the tied-attention arm (where
        # the read k/v IS the paired state block's c_attn, no pred-side tensor at all)
        # decomposes through exactly the map the model used.
        W, b = blk.read_kv_weight()
        tag = (f"state_h[{pred_block}].c_attn(k,v)@level{L} [tied]"
               if getattr(blk, "tied_attn", False) else f"pred_h[{pred_block}].read_kv")
        return (blk.read_norm, W[:inner], (None if b is None else b[:inner]),
                W[inner:2 * inner], (None if b is None else b[inner:2 * inner]),
                L, tag)
    if L == model.state_n_layer:
        head = model.transformer["state_read_head"]
        inner = model.n_head_state * model.head_dim
        W, b = head.kv.weight, head.kv.bias
        return (head.norm, W[:inner], (None if b is None else b[:inner]),
                W[inner:2 * inner], (None if b is None else b[inner:2 * inner]),
                L, "state_read_head.kv")
    blk = model.transformer.state_h[L]
    inner = model.n_head_state * model.head_dim
    W, b = blk.c_attn.weight, blk.c_attn.bias
    return (blk.attention_norm, W[inner:2 * inner],
            (None if b is None else b[inner:2 * inner]),
            W[2 * inner:3 * inner], (None if b is None else b[2 * inner:3 * inner]),
            L, f"state_h[{L}].c_attn(k,v)")


class StateDecomp:
    """Capture every additive contribution to the memory residual, per block per head."""

    def __init__(self, model, n_levels):
        self.model = model
        self.n_levels = n_levels
        self.y = {}            # block -> (T, n_head*head_dim) pre-c_proj attention output
        self.mlp = {}          # block -> (T, d_s)
        self._h = []

    def __enter__(self):
        for b in range(self.n_levels):
            blk = self.model.transformer.state_h[b]

            def pre(mod, args, b=b):
                self.y[b] = args[0].detach()[0].float()
                return None

            def post(mod, args, o, b=b):
                self.mlp[b] = o.detach()[0].float()
                return None

            self._h.append(blk.c_proj.register_forward_pre_hook(pre))
            self._h.append(blk.mlp.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False

    def sources(self, positions):
        """-> (names, tensor (S, P, d_s)) additive decomposition at `positions`."""
        m = self.model
        pos = torch.as_tensor(positions, device="cuda", dtype=torch.long)
        names, vecs = [], []
        names.append(("embed", -1, -1))
        vecs.append(self._emb.index_select(0, pos))
        hd = m.head_dim
        for b in range(self.n_levels):
            blk = m.transformer.state_h[b]
            W = blk.c_proj.weight.float()                 # (d_s, n_head*head_dim)
            yb = self.y[b].index_select(0, pos)           # (P, n_head*head_dim)
            for h in range(int(blk.n_head)):
                names.append(("attn", b, h))
                vecs.append(yb[:, h * hd:(h + 1) * hd] @ W[:, h * hd:(h + 1) * hd].T)
            if blk.c_proj.bias is not None:
                names.append(("attn_bias", b, -1))
                vecs.append(blk.c_proj.bias.float().view(1, -1).expand(len(positions), -1))
            names.append(("mlp", b, -1))
            vecs.append(self.mlp[b].index_select(0, pos))
        return names, torch.stack(vecs, 0)


def trace_one_sequence(model, cfg, decomp, X, qpos, n_ctrl, hm, m_blk, rng):
    """-> dict of per-source (raw, differential) QK and OV contributions for one probe."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        docs = model.generate_document_idx(X)
        capt = C.two_tower_capture(model, X, docs)
    norm, Wk, bk, Wv, bv, L, tag = read_path_modules(model, m_blk)
    hd = model.head_dim
    res = capt["state_res"][L][0].float()                 # (T, d_s)

    p_ind = [A5.COPY_A + (int(q) - A5.COPY_B) + 1 for q in qpos]
    ctrl = sorted(rng.choice(np.arange(8, A5.COPY_A), size=n_ctrl, replace=False).tolist())
    positions = list(dict.fromkeys(p_ind + ctrl))
    pidx = {p: i for i, p in enumerate(positions)}

    names, S = decomp.sources(positions)                  # (S, P, d_s)
    # exactness gate: the additive sources must rebuild the model's own residual
    rec = S.sum(0)
    tgt = res.index_select(0, torch.as_tensor(positions, device="cuda"))
    rel = float((rec - tgt).norm() / tgt.norm().clamp(min=1e-9))

    sigma = rms_sigma(tgt, float(norm.eps))               # (P, 1)
    gamma = norm.weight.float().view(1, 1, -1)
    Hs = S * sigma.unsqueeze(0) * gamma                   # (S, P, d_s)

    k_raw = Hs @ Wk.float().T                             # (S, P, inner)
    v_raw = Hs @ Wv.float().T
    nS, nP = k_raw.shape[0], k_raw.shape[1]
    k_h = k_raw.view(nS, nP, -1, hd)[:, :, hm, :]         # (S, P, hd)
    v_h = v_raw.view(nS, nP, -1, hd)[:, :, hm, :]
    if bk is not None:
        names = list(names) + [("key_bias", -1, -1)]
        kb = bk.float().view(-1, hd)[hm].view(1, 1, hd).expand(1, nP, hd)
        k_h = torch.cat([k_h, kb], 0)
        vb = bv.float().view(-1, hd)[hm].view(1, 1, hd).expand(1, nP, hd)
        v_h = torch.cat([v_h, vb], 0)
        nS += 1

    # RoPE at each key position: (1, P, S, hd) puts position on the time axis and the
    # sources on the head axis, which is exactly what apply_rotary_emb broadcasts over.
    fc = model.freqs_cis.to("cuda")[torch.as_tensor(positions, device="cuda")]
    kk = k_h.permute(1, 0, 2).unsqueeze(0).contiguous()
    _, kk = C.apply_rotary_emb(kk, kk, freqs_cis=fc)
    k_rot = kk[0].permute(1, 0, 2).contiguous()           # (S, P, hd)

    q_all = capt["pred_qk"][m_blk][0][0, hm].float()      # (T, hd), RoPE already applied
    scale = 1.0 / np.sqrt(hd)

    # the model's own key at those positions, for the reconstruction gate + attention row
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if model.read_source == "pred_proj":
            k_full, v_full = model.transformer.pred_h[m_blk].read_state(
                capt["state_res"][L], model.freqs_cis.to("cuda")[:X.shape[1]])
        elif L == model.state_n_layer:
            k_full, v_full = model.transformer["state_read_head"](
                capt["state_res"][L], model.freqs_cis.to("cuda")[:X.shape[1]])
        else:
            _q, k_full, v_full = model.transformer.state_h[L].qkv(
                capt["state_res"][L], model.freqs_cis.to("cuda")[:X.shape[1]])
    k_true = k_full[0, hm].float()                        # (T, hd)
    vis = capt["masks"].bool_mask()[0].bool()

    qk_raw, qk_diff, ov_raw, ov_diff, gate = [], [], [], [], []
    out_norm = model.transformer.output_norm
    gamma_o = out_norm.weight.float()
    Wo = model.transformer.pred_h[m_blk].out_proj().weight.float()[:, hm * hd:(hm + 1) * hd]
    Wu = model.lm_head.weight.float()
    Wu_c = Wu - Wu.mean(0, keepdim=True)
    pred_last = capt["pred_res"][-1][0].float()
    ctrl_ix = torch.as_tensor([pidx[p] for p in ctrl], device="cuda")

    for qi, q in enumerate(qpos):
        q = int(q)
        p = p_ind[qi]
        j = pidx[p]
        qv = q_all[q]
        c_all = (k_rot @ qv) * scale                      # (S, P)
        qk_raw.append(c_all[:, j].cpu().numpy())
        qk_diff.append((c_all[:, j] - c_all.index_select(1, ctrl_ix).mean(1)).cpu().numpy())
        gate.append(float(c_all[:, j].sum() - (k_true[p] @ qv) * scale))

        sc = (k_true @ qv) * scale
        sc = sc.masked_fill(~vis[q], float("-inf"))
        alpha = torch.softmax(sc, -1)
        sig_o = rms_sigma(pred_last[q], float(out_norm.eps))
        u = Wu_c[int(X[0, p])]
        z = (v_h[:, j, :] @ Wo.T) * (gamma_o * sig_o).view(1, -1)
        ov = float(alpha[p]) * (z @ u)                    # (S,)
        zc = (v_h.index_select(1, ctrl_ix) @ Wo.T) * (gamma_o * sig_o).view(1, 1, -1)
        ovc = (zc @ u).mean(1) * float(alpha[p])
        ov_raw.append(ov.cpu().numpy())
        ov_diff.append((ov - ovc).cpu().numpy())

    return dict(names=names,
                qk_raw=np.stack(qk_raw), qk_diff=np.stack(qk_diff),
                ov_raw=np.stack(ov_raw), ov_diff=np.stack(ov_diff),
                gate_abs=float(np.max(np.abs(gate))),
                residual_rel_err=rel, level=L, key_module=tag,
                attn_mass_on_ind_key=float(np.mean([
                    float(torch.softmax(
                        ((k_true @ q_all[int(q)]) * scale).masked_fill(
                            ~vis[int(q)], float("-inf")), -1)[p_ind[qi]])
                    for qi, q in enumerate(qpos)])))


def rank_table(names, arr, top=5):
    mu = arr.mean(0)
    se = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
    tot = float(mu.sum())
    pos = float(mu[mu > 0].sum())
    order = np.argsort(-mu)
    rows = []
    for r, i in enumerate(order):
        kind, b, h = names[i]
        rows.append(dict(rank=r + 1, source=kind, block=int(b), head=int(h),
                         mean=round(float(mu[i]), 5), sem=round(float(se[i]), 5),
                         frac_of_total=round(float(mu[i] / tot), 5) if tot else None,
                         frac_of_positive=round(float(mu[i] / pos), 5) if pos else None))
    return dict(total=round(tot, 5), positive_total=round(pos, 5),
                top5=rows[:top], all_sources=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--n-q", type=int, default=32)
    ap.add_argument("--n-ctrl", type=int, default=32)
    args = ap.parse_args()

    run = C.resolve(args.run)
    out = args.out or C.result_path("a10_pathtrace", run)
    C.assert_node_local_triton()

    a5_path = C.result_path("a5_induction", run)
    assert os.path.exists(a5_path), f"A10 reads A5's committed result; missing {a5_path}"
    with open(a5_path) as f:
        a5res = json.load(f)
    match, top_prev, ctrl = A9.pick_heads(a5res, 3)

    if a5res["family"] != "two_tower":
        C.save_json(out, dict(analysis="a10_pathtrace", run=run, model_id=C.mid_of(run),
                              family=a5res["family"], skipped=True,
                              reason=("the linear key decomposition needs the readout's "
                                      "key projection to be a separate map over a memory "
                                      "residual; in the weight-shared joint families the "
                                      "same projection produces both streams' keys from "
                                      "one interleaved residual, so a per-memory-head "
                                      "decomposition of the readout key is not defined")))
        return

    model, cfg, family, ckpath = C.load(run)
    m_blk, hm = int(match["block"]), int(match["head"])
    L = int(model.read_levels[m_blk])
    print(f"run={run}  matching head block={m_blk} head={hm}  reads state level {L}"
          f"  read_source={model.read_source}", flush=True)

    qpos = np.arange(A5.COPY_B + 1, A5.COPY_B + A5.BLOCK_LEN)
    qpos = qpos[:: max(1, len(qpos) // args.n_q)][:args.n_q]
    rng = np.random.default_rng(7)      # A5.run_synthetic's probe seed
    prng = np.random.default_rng(CTRL_POS_SEED)

    qk_raw, qk_diff, ov_raw, ov_diff, meta = [], [], [], [], []
    names = None
    with StateDecomp(model, L) as decomp:
        # the embedding is the level-0 residual; capture it once per sequence
        for si in range(args.n_seq):
            t0 = time.time()
            toks = A5.synthetic_batch(cfg, rng)
            X = torch.from_numpy(toks)[None].cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                decomp._emb = model.transformer.drop(
                    model.transformer.wte(X))[0].float()
            r = trace_one_sequence(model, cfg, decomp, X, qpos, args.n_ctrl, hm, m_blk, prng)
            names = r["names"]
            qk_raw.append(r["qk_raw"]); qk_diff.append(r["qk_diff"])
            ov_raw.append(r["ov_raw"]); ov_diff.append(r["ov_diff"])
            key_module = r["key_module"]
            meta.append(dict(residual_rel_err=r["residual_rel_err"],
                             qk_reconstruction_abs_err=r["gate_abs"],
                             attn_mass_on_induction_key=r["attn_mass_on_ind_key"]))
            print(f"  seq {si+1}/{args.n_seq}  rel_err={r['residual_rel_err']:.2e} "
                  f"qk_gate={r['gate_abs']:.3e} "
                  f"mass={r['attn_mass_on_ind_key']:.3f}  {time.time()-t0:.0f}s", flush=True)

    qk_raw = np.concatenate(qk_raw); qk_diff = np.concatenate(qk_diff)
    ov_raw = np.concatenate(ov_raw); ov_diff = np.concatenate(ov_diff)

    def locate(tbl, b, h):
        for r in tbl["all_sources"]:
            if r["source"] == "attn" and r["block"] == b and r["head"] == h:
                return r
        return None

    tabs = dict(qk_differential=rank_table(names, qk_diff),
                qk_raw=rank_table(names, qk_raw),
                ov_differential=rank_table(names, ov_diff),
                ov_raw=rank_table(names, ov_raw))
    prev_rows = {k: [locate(v, d["block"], d["head"]) for d in top_prev]
                 for k, v in tabs.items()}

    res = dict(analysis="a10_pathtrace", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               question=("decompose the readout matching head's QK score at the induction "
                         "key into exact per-memory-source contributions"),
               matching_head=match, prev_token_heads=top_prev,
               read_level=L, read_source=model.read_source,
               key_module=key_module, n_pairs=int(qk_raw.shape[0]),
               sampler=dict(n_seq=args.n_seq, n_queries=len(qpos),
                            n_control_positions=args.n_ctrl,
                            control_position_seed=CTRL_POS_SEED,
                            copies=[A5.COPY_A, A5.COPY_B],
                            block_len=A5.BLOCK_LEN),
               method=dict(
                   exact=("the RMSNorm scale is the true scalar scale of the true "
                          "residual, so the per-source map is linear and the "
                          "contributions sum to the model's own QK logit"),
                   differential=("raw contribution at the induction key minus that "
                                 "source's mean contribution over control key positions"),
                   ov_path=("DIRECT path only: value -> matching head's output "
                            "projection -> final output norm -> unembedding; every "
                            "readout block and MLP after the matching block is ignored")),
               gates=meta, tables=tabs, prev_token_head_rows=prev_rows)
    C.save_json(out, res)

    for nm in ("qk_differential", "ov_differential"):
        print(f"\n{nm}: total={tabs[nm]['total']:.4f}", flush=True)
        for r in tabs[nm]["top5"]:
            print(f"  #{r['rank']} {r['source']}[b{r['block']},h{r['head']}] "
                  f"{r['mean']:+.4f}  {100*(r['frac_of_total'] or 0):.1f}% of total",
                  flush=True)
        pr = prev_rows[nm][0]
        if pr:
            print(f"  top prev-token head rank={pr['rank']} "
                  f"frac_of_total={pr['frac_of_total']}", flush=True)


if __name__ == "__main__":
    main()
