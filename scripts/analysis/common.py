"""Shared plumbing for the analysis scripts in this directory.

Everything here exists so the analysis scripts in this directory agree, by construction,
on four things that are easy to get wrong:

1. WHICH MODEL IS WHICH.  paper_manifest.yaml is the one registry of runs, their arms,
   labels and roles; `resolve`, `mid_of` and `label_of` read it.  No script spells a run
   name itself.

2. WHAT THE VISIBILITY MASK IS.  An inverted mask is the worst failure mode, and it
   recurs every time the mask is re-derived by hand.  So:
     * for the TWO-TOWER family the mask is never re-derived -- `_MaskSet.bool_mask`,
       the model's OWN mask, is what these scripts read;
     * for SPS the interleaved-2T mask IS re-derived here (the Triton kernel exposes
       nothing); `_joint_views` documents the convention it mirrors.

3. WHERE THE QUERIES COME FROM.  Query token positions are sampled uniformly over
   [Q_MIN, T): sampling from position 1 confounds attention distance with absolute
   position (p99 distance then saturates near 1053).

4. HOW ATTENTION PROBABILITIES ARE OBTAINED.  Recomputed from the block's own public
   methods, never by running flex_attention eagerly (~10 GB of fp32 scores per layer at
   T=4096).  Only sampled query ROWS are ever materialised.

The unified view every analysis consumes is `AttnView`: one (block, query-stream) pair,
with per-head probabilities over a key set that is labelled by key stream and key token.
That abstraction is what makes the 2x2 of A5 (query stream x key stream) expressible for
the joint family and gracefully degenerate for two-tower, whose readout tower has no
self keys at all, so its 2x2 collapses to a single key-stream column.
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "scripts", "dualsps")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                     # noqa: E402
import yaml                                                            # noqa: E402
import torch                                                           # noqa: E402

import repo_paths                                                      # noqa: E402
from eval_runs import (  # noqa: E402
    load_model, val_path_of, full_sweep_nll, BLOCK, MB, CONF,
)
from modeling.models.model import apply_rotary_emb                     # noqa: E402

# Every script in this directory is eval-only: nothing here ever needs a graph, and an
# autograd graph built over recomputed 4096x8192 score matrices is both a memory leak and
# (measured) a hard error the moment a result is moved to numpy. Disabled process-wide at
# import so no analysis can forget the `no_grad` around a recompute path.
torch.set_grad_enabled(False)

# --------------------------------------------------------------------------------------
# model registry: scripts/analysis/paper_manifest.yaml (read without importing matplotlib)
# --------------------------------------------------------------------------------------
with open(os.path.join(REPO, "scripts", "analysis", "paper_manifest.yaml")) as _f:
    MANIFEST = yaml.safe_load(_f)
ROLES = MANIFEST["roles"]
ARM_OF_RUN = {s: k for k, a in MANIFEST["arms"].items() for s in a["seeds"]}

# Sampling constants shared by every attention analysis.  Fixed seeds everywhere: two
# analyses that sample differently cannot be put in the same table.
Q_MIN = 256            # queries sampled from [Q_MIN, T): >= 256 tokens of history
N_SEQ_ATTN = 32
N_Q_ATTN = 128
QPOS_SEED = 12345
SUB_SWEEP_SEQS = 512   # "sub-sweep" = first 512 val sequences = 2.1 M tokens
UNTRAINED_SEEDS = (1234,)
# noise floor recorded with every intervention result: paired-intervention noise on the
# same sequences.  (The seed SD of the final loss is \SeedSD in numbers.tex, computed from
# the ledger by paper_numbers.seed_sd.)
PAIRED_FLOOR = 0.0005

RESULTS_DIR = str(repo_paths.RESULTS)


def mid_of(run: str) -> str:
    """The run's arm (the primary run of its config family in paper_manifest.yaml)."""
    return ARM_OF_RUN.get(run, run)


def label_of(run: str) -> str:
    arm = ARM_OF_RUN.get(run)
    return MANIFEST["arms"][arm]["label"] if arm else run


def resolve(name: str) -> str:
    """Accept a role name (paper_manifest.yaml `roles`) or a run name; return the run name."""
    return ROLES.get(name, name)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def family_of(cfg) -> str:
    tgt = str(cfg.model._target_)
    if "two_tower" in tgt:
        return "two_tower"
    if "sps" in tgt:
        return "sps"
    return "standard"


def load(run: str):
    """-> (model, cfg, family, ckpt_path).  Trained weights, eval mode, cuda."""
    m, cfg, ck, ckpath, _ = load_model(run)
    return m, cfg, family_of(cfg), ckpath


def load_untrained(run: str, seed: int = 1234):
    """The architectural-prior null: identical config, freshly initialised weights.

    The re-initialisation is the model constructor's own (`self.apply(_init_weights)`
    plus the depth-scaled c_proj pass); the only control is the global torch seed set
    immediately before `instantiate`.  Nothing
    is hand-rolled, so the null is the architecture's prior and not this script's idea
    of one.
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(config_dir=CONF, version_base=None):
        cfg = compose("config", overrides=[f"+experiment={run}",
                                           f"system.data_root={repo_paths.data_root()}"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = instantiate(cfg.model).cuda().eval()
    return m, cfg, family_of(cfg), f"untrained-init(seed={seed})"


# --------------------------------------------------------------------------------------
# data sampling
# --------------------------------------------------------------------------------------
def val_memmap(cfg):
    return np.memmap(val_path_of(cfg), dtype=np.uint16, mode="r")


def seq_starts(data, n_seq: int):
    """Deterministic, evenly-strided, non-overlapping sequence starts."""
    starts = list(range(0, len(data) - BLOCK - 1, BLOCK))
    stride = max(1, len(starts) // max(1, n_seq))
    return starts[::stride][:n_seq]


def query_positions(n_q: int = N_Q_ATTN, seed: int = QPOS_SEED):
    """Query TOKEN positions, uniform over [Q_MIN, BLOCK).  Fixed seed, shared by every
    model so cross-model numbers are measured at the same positions."""
    rng = np.random.default_rng(seed)
    hi = BLOCK
    n = min(n_q, hi - Q_MIN)
    return np.sort(rng.choice(np.arange(Q_MIN, hi), size=n, replace=False))


def batch_of(data, j):
    X = torch.from_numpy(data[j:j + BLOCK].astype(np.int64))[None].cuda()
    Y = torch.from_numpy(data[j + 1:j + 1 + BLOCK].astype(np.int64))[None].cuda()
    return X, Y


def sweep_nll(model, val_path, n_seq=None):
    """`eval_runs.full_sweep_nll` restricted to the first `n_seq` sequences.

    n_seq=None is the full deterministic sweep (13,085,928 tokens / 3,198 seqs) --
    the metric of record.  n_seq=SUB_SWEEP_SEQS is the "sub-sweep" the design allows for
    k-way scans; any number produced that way must say so in its caption.
    """
    if n_seq is None:
        return full_sweep_nll(model, val_path)
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    starts = list(range(0, len(data) - BLOCK - 1, BLOCK))[:n_seq]
    s = c = 0.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(starts), MB):
            b = starts[i:i + MB]
            X = torch.stack([torch.from_numpy(data[j:j + BLOCK].astype(np.int64)) for j in b]).cuda()
            Y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + BLOCK].astype(np.int64)) for j in b]).cuda()
            _, _, st = model(X, Y)
            s += float(st["token_nll_sum"]); c += float(st["token_nll_count"])
    return s / c, int(c), len(starts)


# --------------------------------------------------------------------------------------
# the unified attention view
# --------------------------------------------------------------------------------------
class AttnView:
    """One (block, query-stream) attention distribution over a labelled key set.

    p        (nq, nh, K) float32 probabilities, already softmaxed over the VISIBLE keys
    q_tok    (nq,)  int64 token position of each query
    k_tok    (K,)   int64 token position of each key
    k_stream (K,)   0 = memory/state key, 1 = readout/pred key
    vis      (nq, K) bool visibility actually used for the softmax
    """

    __slots__ = ("block", "stream", "p", "q_tok", "k_tok", "k_stream", "vis", "n_head",
                 "read_level")

    def __init__(self, block, stream, p, q_tok, k_tok, k_stream, vis, read_level=None):
        self.block, self.stream = block, stream
        self.p, self.q_tok, self.k_tok, self.k_stream, self.vis = p, q_tok, k_tok, k_stream, vis
        self.n_head = int(p.shape[1])
        self.read_level = read_level

    def dist(self):
        """(nq, K) token-space distance q_tok - k_tok, clamped at 0."""
        return (self.q_tok.view(-1, 1) - self.k_tok.view(1, -1)).clamp(min=0)


def _hook_capture(modules):
    cap = {}

    def mk(i):
        def hook(mod, args, kwargs):
            cap[i] = (args[0].detach(), args[1],
                      kwargs.get("documents_idx_Bx2T",
                                 kwargs.get("documents_idx_BxT",
                                            args[2] if len(args) > 2 else None)))
        return hook
    handles = [m.register_forward_pre_hook(mk(i), with_kwargs=True)
               for i, m in enumerate(modules)]
    return cap, handles


def _joint_views(model, cfg, X, Y, qpos):
    """SPS: one interleaved 2T key set, key stream = slot parity.

    Visibility, mirrored from `attention/triton_sps_flash_attention.py` (and identical to
    the table in `two_tower/core.py`'s docstring, which is the cross-check):

        state key k visible to slot query s  <=>  2k   <= s
        pred  key k visible to slot query s  <=>  2k+1 <= s  and  (q_tok - k) <= W

    i.e. causality in SLOT space plus the temporary-key window on pred keys only; state
    keys are persistent (`persistent_key_window` is None) and carry the only long-range
    channel.  DO NOT take the mirrored `reverse_sps` reference as authority here -- that
    is the exact inverse convention.
    """
    mc = cfg.model.config
    blocks = list(model.transformer.h)
    W = int(mc.window_size)
    NH = int(mc.n_head)
    HD = int(mc.hidden_size) // NH
    cap, handles = _hook_capture([b.attn for b in blocks])
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(X, Y)
        two_t = 2 * BLOCK
        k_idx = torch.arange(two_t, device="cuda")
        k_is_pred = (k_idx % 2 == 1)
        k_tok = k_idx // 2
        qpos_t = torch.from_numpy(np.asarray(qpos)).cuda()
        for i, blk in enumerate(blocks):
            x, freqs_cis, docs = cap[i]
            a = blk.attn
            c_ = int(mc.hidden_size)
            qkv = a.c_attn(x.float()).reshape(1, two_t, 3 * c_)
            q, k, _v = qkv.split(a.hidden_size, dim=2)
            q = q.view(1, two_t, NH, HD)
            k = k.view(1, two_t, NH, HD)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q, k = q[0].float(), k[0].float()
            scale = 1.0 / math.sqrt(HD)
            for parity, pname in ((0, "state"), (1, "pred")):
                qi = qpos_t * 2 + parity
                qs = q.index_select(0, qi)
                sc = torch.einsum("qhd,khd->qhk", qs, k) * scale
                q_tok = qpos_t.view(-1, 1)
                dist = q_tok - k_tok.view(1, -1)
                vis = (k_idx.view(1, -1) <= qi.view(-1, 1))
                vis = vis & ((~k_is_pred.view(1, -1)) | (dist <= W))
                if docs is not None:
                    d = docs[0]
                    vis = vis & (d.index_select(0, qi).view(-1, 1) == d.view(1, -1))
                sc = sc.masked_fill(~vis.unsqueeze(1), float("-inf"))
                p = torch.softmax(sc, dim=-1)
                yield AttnView(i, pname, p, qpos_t, k_tok, k_is_pred.long(), vis)
            del qkv, q, k
    finally:
        for h in handles:
            h.remove()
        cap.clear()


def two_tower_capture(model, X, docs_BxT):
    """Replay `TwoTowerModel._forward_towers` keeping every q / k / residual.

    Deliberately NOT a re-implementation: every tensor is produced by the model's own
    public methods (`StateBlock.qkv`, `PredBlock.query/read_state`,
    `TwoTowerModel._attend`) with the model's own `_MaskSet`, so the residual stream this
    returns is the one the model actually computes, and the masks are the model's, not a
    hand-derived copy of them.
    """
    from modeling.models.two_tower import attention as attn_backends
    from modeling.models.two_tower.core import _MaskSet, _needs_document_mask

    b, t = X.shape
    device = X.device
    freqs_cis = model.freqs_cis.to(device)[:t]
    needs_doc = _needs_document_mask(docs_BxT)
    backend = attn_backends.resolve_backend(model.config.attn_backend,
                                            needs_document_mask=needs_doc)
    docs = docs_BxT if needs_doc else None
    masks = _MaskSet(t, docs, device, backend, b, mask_mod=model._mask_mod(t, b, device, docs),
                     flex_fns=model._flex_fns)
    x_s = model.transformer.drop(model.transformer.wte(X))
    if model.config.predict_embedding == "constant":
        x_p = model.predict_wte.weight[0].to(x_s.dtype).view(1, 1, -1).expand(b, t, -1)
    else:
        x_p = model.predict_wte(X)
    x_p = model.transformer.drop(x_p)

    state_res, pred_res = [], []
    state_qk, pred_qk = [], []
    kv_levels, res_levels = [], []
    for block in model.transformer.state_h:
        state_res.append(x_s)
        if model.read_source == "pred_proj":
            res_levels.append(x_s)
        q, k, v = block.qkv(x_s, freqs_cis)
        kv_levels.append((k, v))
        state_qk.append((q, k, v))
        y = model._attend(q, k, v, masks, "state")
        x_s = block.mlp_step(block.finish_attn(x_s, y))
    state_res.append(x_s)
    if model.needs_final_level:
        if model.read_source == "pred_proj":
            res_levels.append(x_s)
        else:
            kv_levels.append(model.transformer["state_read_head"](x_s, freqs_cis))

    for i, block in enumerate(model.transformer.pred_h):
        pred_res.append(x_p)
        lvl = model.read_levels[i]
        if model.read_source == "pred_proj":
            k_s, v_s = block.read_state(res_levels[lvl], freqs_cis)
        else:
            k_s, v_s = kv_levels[lvl]
        q = block.query(x_p, freqs_cis)
        pred_qk.append((q, k_s, v_s, lvl))
        y = model._attend(q, k_s, v_s, masks, "pred")
        x_p = block.mlp_step(block.finish_attn(x_p, y))
    pred_res.append(x_p)
    return dict(masks=masks, state_res=state_res, pred_res=pred_res,
                state_qk=state_qk, pred_qk=pred_qk, out=model.transformer.output_norm(x_p))


def _two_tower_views(model, cfg, X, Y, qpos):
    """Two-tower: state tower = plain causal self-attention over T state keys; readout
    tower = queries over the state keys at level f(i); it has no keys of its own.  The key
    set therefore has ONE key stream for pred queries, and the A5 2x2 collapses to a
    single column -- that is a property of the architecture, not a missing measurement.
    """
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        docs = model.generate_document_idx(X)
        capt = two_tower_capture(model, X, docs)
    masks = capt["masks"]
    t = X.shape[1]
    qpos_t = torch.from_numpy(np.asarray(qpos)).cuda()
    tok = torch.arange(t, device="cuda")

    def emit(block, stream, q, k, read_level=None):
        scale = 1.0 / math.sqrt(q.shape[-1])
        qs = q[0].transpose(0, 1).float().index_select(0, qpos_t)        # (nq, nh, hd)
        ks = k[0].transpose(0, 1).float()                                # (K, nh, hd)
        sc = torch.einsum("qhd,khd->qhk", qs, ks) * scale
        vis = masks.bool_mask()[0].index_select(0, qpos_t)               # model's OWN mask
        sc = sc.masked_fill(~vis.unsqueeze(1), float("-inf"))
        p = torch.softmax(sc, dim=-1)
        k_stream = torch.zeros(t, dtype=torch.long, device="cuda")
        return AttnView(block, stream, p, qpos_t, tok, k_stream, vis, read_level)

    for i, (q, k, _v) in enumerate(capt["state_qk"]):
        yield emit(i, "state", q, k)
    for i, (q, k_s, _v_s, lvl) in enumerate(capt["pred_qk"]):
        yield emit(i, "pred", q, k_s, read_level=lvl)


def _standard_views(model, cfg, X, Y, qpos):
    """Single-stream transformer: one stream, one key type, plain causal."""
    mc = cfg.model.config
    blocks = list(model.transformer.h)
    NH, HD = int(mc.n_head), int(mc.hidden_size) // int(mc.n_head)
    cap, handles = _hook_capture([b.attn for b in blocks])
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(X, Y)
        t = X.shape[1]
        tok = torch.arange(t, device="cuda")
        qpos_t = torch.from_numpy(np.asarray(qpos)).cuda()
        for i, blk in enumerate(blocks):
            x, freqs_cis, docs = cap[i]
            a = blk.attn
            q, k, _v = a.c_attn(x.float()).split(a.hidden_size, dim=2)
            q = q.view(1, t, NH, HD)
            k = k.view(1, t, NH, HD)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q, k = q[0].float(), k[0].float()
            scale = 1.0 / math.sqrt(HD)
            qs = q.index_select(0, qpos_t)
            sc = torch.einsum("qhd,khd->qhk", qs, k) * scale
            vis = tok.view(1, -1) <= qpos_t.view(-1, 1)
            if docs is not None:
                d = docs[0]
                vis = vis & (d.index_select(0, qpos_t).view(-1, 1) == d.view(1, -1))
            sc = sc.masked_fill(~vis.unsqueeze(1), float("-inf"))
            yield AttnView(i, "single", torch.softmax(sc, dim=-1), qpos_t, tok,
                           torch.zeros(t, dtype=torch.long, device="cuda"), vis)
    finally:
        for h in handles:
            h.remove()
        cap.clear()


def attention_views(model, cfg, family, X, Y, qpos):
    """Generator of `AttnView`s for one forward pass.  The only attention entry point."""
    if family == "two_tower":
        yield from _two_tower_views(model, cfg, X, Y, qpos)
    elif family == "sps":
        yield from _joint_views(model, cfg, X, Y, qpos)
    else:
        yield from _standard_views(model, cfg, X, Y, qpos)


def stream_names(family):
    """Query streams this family has, in figure order."""
    return ("single",) if family == "standard" else ("state", "pred")


def key_stream_names(family):
    return ("state",) if family in ("two_tower", "standard") else ("state", "pred")


# --------------------------------------------------------------------------------------
# forward-signature adapters -- ONE place, because the families disagree
# --------------------------------------------------------------------------------------
def forward_logits(model, X, Y):
    """-> (b, t, V) logits, whatever the family's forward signature is.

    The standard family's `forward` REQUIRES targets and always returns
    `(logits, loss, stats)`; two-tower / SPS take targets optionally and return a bare
    logits tensor when they are omitted.  So everything calls the one shape that is
    valid for all three -- WITH targets -- and unwraps the tuple here.

    `Y` is cloned because the standard forward writes IGNORE_INDEX into its targets in
    place, which would silently corrupt a caller that scores against the same tensor.
    """
    out = model(X, Y.clone())
    return out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------------------
# joint-family slot masks -- the ONE definition of "which slots are state / pred"
# --------------------------------------------------------------------------------------
def joint_slot_masks(two_t: int, device=None):
    """-> (state_mask, pred_mask), each (1, 2T, 1) bool, for the tied-SPS interleaving.

    Slot 2i is the STATE slot (it embeds the real token x_i, its key is persistent);
    slot 2i+1 is the PRED slot (it embeds <predict>, its key is windowed, and it is the
    only slot the LM head reads: `SPSModelBase.forward` takes `lm_head(x[:, 1::2])`).
    Every joint branch (a12 / a16 / a20) selects slots through this function, so the
    parity convention is written down exactly once (and pinned by a unit test).
    """
    import torch as _t
    assert two_t % 2 == 0, f"interleaved length must be even, got {two_t}"
    idx = _t.arange(two_t, device=device)
    state = (idx % 2 == 0).view(1, two_t, 1)
    return state, ~state


def joint_blocks(model):
    """The single weight-shared block stack of the tied-SPS model."""
    return list(model.transformer.h)


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def result_path(analysis: str, run: str) -> str:
    return os.path.join(RESULTS_DIR, f"{analysis}_{run}.json")


def save_json(path, obj):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    print(f"WROTE {path}", flush=True)


def assert_node_local_triton():
    """A Triton cache shared on a network filesystem silently returns WRONG KERNELS
    (measured val_nll 9.71 vs 3.11).  scripts/run/eval_shim.sh makes a fresh node-local cache and
    sets DUALSPS_NODE_LOCAL_CACHE=1; anything else (including an unset TRITON_CACHE_DIR,
    whose default ~/.triton is often on NFS) is refused."""
    tc = os.environ.get("TRITON_CACHE_DIR", "")
    assert os.environ.get("DUALSPS_NODE_LOCAL_CACHE") == "1" and tc, (
        "no node-local Triton cache: a shared cache can return wrong kernels without any "
        "error. Run this through scripts/run/eval_shim.sh.")
    print(f"TRITON_CACHE_DIR={tc}", flush=True)
