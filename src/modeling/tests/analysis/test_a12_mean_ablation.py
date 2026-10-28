"""A12 mean-ablation mode (`--ablation mean`), two-tower `DepthLesion` and joint `SlotLesion`.

Runs on CPU on tiny models.  What is pinned:

  * the CLI default is `zero`, writing the original file name;
  * mean mode with a "mean" equal to the ACTUAL per-position write reproduces the
    un-lesioned logits exactly (the substitution plumbing is exact);
  * the no-op gate holds (with and without means passed), the joint identity-mask gate
    holds through the mean code path, and the joint last-block state-slot dead end is
    still structurally zero under mean-ablation;
  * the clean-run mean capture is inert (logits unchanged) and returns the true
    per-channel means; the held-out mean sequences are disjoint from the scored ones.
"""
import pathlib
import sys

import pytest
import torch

_ANALYSIS = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

C = pytest.importorskip("common")
A12 = pytest.importorskip("a12_depth_lesion")

import modeling.models.sps.core as sps_core  # noqa: E402
from modeling.models.sps import SPSConfig, SPSModel  # noqa: E402
from modeling.models.two_tower.core import TwoTowerConfig, TwoTowerModel  # noqa: E402



@pytest.fixture(autouse=True)
def no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    yield
    torch.set_grad_enabled(prev)


# ------------------------------------------------------------------------------ models
def _tt(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(block_size=64, vocab_size=128, n_layer=3, n_head=2, hidden_size=32,
               intermediate_size=64, eos_token_id=120, pad_token_id=127,
               predict_token_id=121, state_n_layer=3, pred_n_layer=3, flex_compile=False,
               attn_dtype="keep")
    cfg.update(kw)
    m = TwoTowerModel(TwoTowerConfig(**cfg))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def _tt_batch(seed=1, b=2, t=20):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 110, (b, t), generator=g)
    X[0, 7] = 120
    Y = torch.randint(0, 110, (b, t), generator=g)
    return X, Y


def _ref_sps_attention(q, k, v, sm_scale, window, warp_specialize=False,
                       documents_idx_BxT=None, persistent_key_window=None):
    q, k, v = q.float(), k.float(), v.float()
    two_t = q.shape[2]
    qp = torch.arange(two_t).view(-1, 1)
    kp = torch.arange(two_t).view(1, -1)
    vis = (kp <= qp) & ((kp % 2 == 0) | ((qp // 2 - kp // 2) <= window))
    vis = vis.view(1, 1, two_t, two_t)
    if documents_idx_BxT is not None:
        d = documents_idx_BxT
        vis = vis & (d.unsqueeze(-1) == d.unsqueeze(-2)).unsqueeze(1)
    sc = torch.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale
    sc = sc.masked_fill(~vis, float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(sc, -1), v)


@pytest.fixture
def ref_attn(monkeypatch):
    monkeypatch.setattr(sps_core, "triton_sps_sliding_attention", _ref_sps_attention)


def _sps(seed=0):
    torch.manual_seed(seed)
    m = SPSModel(SPSConfig(block_size=16, vocab_size=64, n_layer=3, n_head=2,
                           hidden_size=32, intermediate_size=48, dropout=0.0, bias=False,
                           eos_token_id=63, pad_token_id=62, predict_token_id=61,
                           window_size=2, enable_triton_attention=True,
                           warp_specialize=False, tie_lm_head=True))
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(0.3 * torch.randn_like(p))
    return m.eval()


def _sps_batch(seed=1, b=2, t=16):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 60, (b, t), generator=g)
    X[0, 7] = 63
    Y = torch.randint(0, 60, (b, t), generator=g)
    return X, Y


def _logits(m, X, Y):
    return m(X, Y)[0]


# ---------------------------------------------------------------- default path identity
def test_cli_default_is_zero_and_writes_original_file():
    assert A12.ABLATIONS[0] == "zero"
    run = "s_two_tower_w0_equal_20b"
    assert A12.default_out_path(run) == C.result_path("a12_depth_lesion", run)
    assert A12.default_out_path(run, "zero") == C.result_path("a12_depth_lesion", run)
    mean = A12.default_out_path(run, "mean")
    assert mean.endswith(f"a12_depth_lesion_{run}_mean.json")
    assert mean != C.result_path("a12_depth_lesion", run)


# --------------------------------------------------------- mean == actual write -> no-op
def _capture_tt_writes(m, stream, idx, X, Y):
    """Per-position (B,T,C) attention and MLP writes of one two-tower block."""
    blk = (m.transformer.state_h if stream == "state" else m.transformer.pred_h)[idx]
    cls, cap = type(blk), {}

    def finish(x, y):
        w = cls.finish_attn(blk, torch.zeros_like(x), y)
        cap["attn"] = w.clone()
        return x + w

    def mlp(x):
        w = blk.mlp(blk.mlp_norm(x))
        cap["mlp"] = w.clone()
        return x + w
    blk.finish_attn, blk.mlp_step = finish, mlp
    out = _logits(m, X, Y)
    del blk.finish_attn, blk.mlp_step
    return cap, out


def test_two_tower_mean_equal_to_actual_write_is_exact_noop():
    m = _tt()
    X, Y = _tt_batch()
    base = _logits(m, X, Y)
    les = A12.DepthLesion(m)
    for stream in ("state", "pred"):
        for idx in range(3):
            cap, out = _capture_tt_writes(m, stream, idx, X, Y)
            assert torch.equal(out, base)                    # the capture is inert
            means = {(stream, idx, "attn"): cap["attn"], (stream, idx, "mlp"): cap["mlp"]}
            for kind in A12.KINDS:
                les.apply(stream, idx, kind, means=means)
                got = _logits(m, X, Y)
                les.clear()
                assert torch.equal(got, base), (stream, idx, kind)
            # ...and a genuine per-channel mean is NOT a no-op (the patch does something)
            mu = {k: v.mean(dim=(0, 1)) for k, v in means.items()}
            les.apply(stream, idx, "block", means=mu)
            assert not torch.allclose(_logits(m, X, Y), base)
            les.clear()
    assert torch.equal(_logits(m, X, Y), base)
    assert les.calls > 0


def test_joint_mean_equal_to_actual_write_is_exact_noop(ref_attn):
    m = _sps()
    X, Y = _sps_batch()
    base = _logits(m, X, Y)
    les = A12.SlotLesion(m)
    for idx, blk in enumerate(les.blocks):
        cap = {}
        h1 = blk.attn.register_forward_hook(lambda _m, _a, o: cap.__setitem__("attn", o.clone()))
        h2 = blk.mlp.register_forward_hook(lambda _m, _a, o: cap.__setitem__("mlp", o.clone()))
        assert torch.equal(_logits(m, X, Y), base)
        h1.remove(), h2.remove()
        for stream in ("state", "pred"):
            means = {(stream, idx, "attn"): cap["attn"], (stream, idx, "mlp"): cap["mlp"]}
            for kind in A12.KINDS:
                les.apply(stream, idx, kind, means=means)
                got = _logits(m, X, Y)
                les.clear()
                assert torch.equal(got, base), (idx, stream, kind)


# ------------------------------------------------------------------------------ gates
def test_noop_gate_holds_in_mean_mode():
    m = _tt()
    X, Y = _tt_batch()
    base = _logits(m, X, Y)
    les = A12.DepthLesion(m)
    means = {(s, i, k): torch.randn(32) for s in ("state", "pred") for i in range(3)
             for k in ("attn", "mlp")}
    les.apply("pred", 0, "attn", noop=True, means=means)
    assert torch.equal(_logits(m, X, Y), base)
    assert les.calls > 0
    les.clear()
    assert "finish_attn" not in les.pred_blocks[0].__dict__


def test_joint_gates_in_mean_mode(ref_attn):
    m = _sps()
    X, Y = _sps_batch()
    base = _logits(m, X, Y)
    les = A12.SlotLesion(m)
    L = len(les.blocks)
    means = {(s, i, k): torch.randn(32) for s in ("state", "pred") for i in range(L)
             for k in ("attn", "mlp")}
    les.apply("pred", 0, "block", noop=True, means=means)
    assert torch.equal(_logits(m, X, Y), base)
    les.apply("pred", 1, "block", identity_mask=True, means=means)
    assert torch.equal(_logits(m, X, Y), base)
    for kind in A12.KINDS:                        # dead end stays a dead end under means
        les.apply("state", L - 1, kind, means=means)
        assert torch.equal(_logits(m, X, Y), base), kind
    les.apply("state", 0, "attn", means=means)
    assert not torch.allclose(_logits(m, X, Y), base)
    les.clear()
    assert torch.equal(_logits(m, X, Y), base)


# ------------------------------------------------------------------- mean capture
def test_two_tower_write_means_are_inert_and_correct():
    m = _tt()
    X, Y = _tt_batch()
    base = _logits(m, X, Y)
    acc = A12.two_tower_write_means(m, [(X, Y)])
    mu = acc.means()
    assert len(mu) == 2 * 3 * 2
    assert torch.equal(_logits(m, X, Y), base)                  # patches removed
    for blk in list(m.transformer.state_h) + list(m.transformer.pred_h):
        assert "finish_attn" not in blk.__dict__ and "mlp_step" not in blk.__dict__
    cap, out = _capture_tt_writes(m, "state", 1, X, Y)
    assert torch.equal(out, base)
    ref = cap["mlp"].double().mean(dim=(0, 1)).float()
    assert torch.allclose(mu[("state", 1, "mlp")], ref, atol=1e-6)
    ref = cap["attn"].double().mean(dim=(0, 1)).float()
    assert torch.allclose(mu[("state", 1, "attn")], ref, atol=1e-6)


def test_joint_write_means_split_by_slot(ref_attn):
    m = _sps()
    X, Y = _sps_batch()
    base = _logits(m, X, Y)
    acc = A12.joint_write_means(m, [(X, Y)])
    mu = acc.means()
    assert torch.equal(_logits(m, X, Y), base)
    blk = m.transformer.h[0]
    cap = {}
    h = blk.mlp.register_forward_hook(lambda _m, _a, o: cap.__setitem__("mlp", o.clone()))
    m(X, Y)
    h.remove()
    w = cap["mlp"].double()
    assert torch.allclose(mu[("state", 0, "mlp")], w[:, 0::2].mean(dim=(0, 1)).float(), atol=1e-6)
    assert torch.allclose(mu[("pred", 0, "mlp")], w[:, 1::2].mean(dim=(0, 1)).float(), atol=1e-6)


def test_mean_sequences_disjoint_from_scored():
    block = 4096
    n_tok = 3200 * block
    s = A12.mean_seq_starts(n_tok, 512, 256, block=block)
    grid = list(range(0, n_tok - block - 1, block))
    assert len(s) == 256
    assert min(s) >= grid[512]
    assert not set(s) & set(grid[:512])
    assert len(set(s)) == len(s)
