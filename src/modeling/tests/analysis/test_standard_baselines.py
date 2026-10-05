"""Single-stream Transformer baselines: A12 `standard` branch (per-layer lesions) and the
A2 `--standard-ctx` context-window knockout.

Runs on CPU on tiny models (flex-attention `Block` path; the Triton block is CUDA-only).
What is pinned:

  * the result-file names; A2 without `--standard-ctx` still writes the skipped record
    for a standard run;
  * A12 standard: the no-op and identity-rebuild gates are bit-exact and the patch
    fires; mean == actual write is an exact no-op; zero / mean lesions change the loss;
    the write-mean capture is inert and correct; `clear()` restores the class forward;
  * A2 standard: untargeted layers run the native attention bit-for-bit (a no-op gate);
    the uncapped explicit control matches native to float tolerance; a cap >= T is the
    uncapped control; near/far bands and last-N targeting act only where they should.
"""
import pathlib
import sys

import pytest
import torch

_ANALYSIS = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

C = pytest.importorskip("common")
A12 = pytest.importorskip("a12_depth_lesion")
A2 = pytest.importorskip("a2_knockout")

import modeling.models.sps.core as sps_core  # noqa: E402
from modeling.models.full_attention_model import Model, ModelConfig  # noqa: E402
from modeling.models.sps import SPSConfig, SPSModel  # noqa: E402
from modeling.models.two_tower.core import TwoTowerConfig, TwoTowerModel  # noqa: E402


@pytest.fixture(autouse=True)
def no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    yield
    torch.set_grad_enabled(prev)


# ------------------------------------------------------------------------------ models
def _std(seed=0, n_layer=3, tie=True):
    torch.manual_seed(seed)
    m = Model(ModelConfig(block_size=32, vocab_size=64, n_layer=n_layer, n_head=2,
                          hidden_size=32, intermediate_size=48, dropout=0.0, bias=False,
                          eos_token_id=63, pad_token_id=62, use_triton_full_attention=False,
                          tie_lm_head=tie))
    with torch.no_grad():                       # bump so every path carries signal
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def _std_batch(seed=1, b=2, t=24):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 60, (b, t), generator=g)
    X[0, 9] = 63                                # a document boundary in row 0
    Y = torch.randint(0, 60, (b, t), generator=g)
    return X, Y


def _logits(m, X, Y):
    return C.forward_logits(m, X, Y)


def _tt(seed=0):
    torch.manual_seed(seed)
    m = TwoTowerModel(TwoTowerConfig(
        block_size=64, vocab_size=128, n_layer=3, n_head=2, hidden_size=32,
        intermediate_size=64, eos_token_id=120, pad_token_id=127, predict_token_id=121,
        state_n_layer=3, pred_n_layer=3, flex_compile=False, attn_dtype="keep"))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def _tt_batch(seed=1, b=2, t=20):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 110, (b, t), generator=g)
    X[0, 7] = 120
    return X, torch.randint(0, 110, (b, t), generator=g)


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
    return X, torch.randint(0, 60, (b, t), generator=g)


# ================================================================ default paths unchanged
def test_a12_result_paths_unchanged():
    run = "s_full_attention_20b_fw100"
    assert A12.default_out_path(run) == C.result_path("a12_depth_lesion", run)
    assert A12.default_out_path(run, "mean").endswith(f"a12_depth_lesion_{run}_mean.json")
    assert A12.STREAMS == ("state", "pred") and A12.KINDS == ("attn", "mlp", "block")


# ============================================================================ A12 standard
def test_a12_standard_noop_and_identity_gates_bit_exact():
    m = _std()
    X, Y = _std_batch()
    base = _logits(m, X, Y)
    les = A12.StandardLesion(m)
    for idx in range(3):
        c0 = les.calls
        les.apply("single", idx, "block", noop=True)
        assert torch.equal(_logits(m, X, Y), base)
        assert les.calls > c0
        for kind in A12.KINDS:
            c0 = les.calls
            les.apply("single", idx, kind, identity=True)
            assert torch.equal(_logits(m, X, Y), base), (idx, kind)
            assert les.calls > c0
    les.clear()
    for blk in les.blocks:
        assert "forward" not in blk.__dict__
    assert torch.equal(_logits(m, X, Y), base)


def test_a12_standard_lesions_change_loss_and_restore():
    m = _std()
    X, Y = _std_batch()
    base = _logits(m, X, Y)
    les = A12.StandardLesion(m)
    means = {("single", i, k): 0.1 * torch.randn(32) for i in range(3)
             for k in ("attn", "mlp")}
    for idx in range(3):
        for kind in A12.KINDS:
            for mu in (None, means):
                les.apply("single", idx, kind, means=mu)
                assert not torch.allclose(_logits(m, X, Y), base), (idx, kind)
                les.clear()
                assert torch.equal(_logits(m, X, Y), base)
    with pytest.raises(AssertionError):
        les.apply("state", 0, "mlp")


def test_a12_standard_block_zero_is_identity_map():
    """kind=block zero-lesion makes the block the identity on its residual."""
    m = _std()
    X, Y = _std_batch()
    blk = m.transformer.h[1]
    cap = {}
    h1 = blk.register_forward_pre_hook(lambda _m, a, kw: cap.__setitem__("in", a[0].clone()),
                                       with_kwargs=True)
    h2 = blk.register_forward_hook(lambda _m, a, o: cap.__setitem__("out", o.clone()))
    les = A12.StandardLesion(m)
    les.apply("single", 1, "block")
    _logits(m, X, Y)
    les.clear()
    h1.remove(), h2.remove()
    assert torch.equal(cap["in"], cap["out"])


def test_a12_standard_mean_equal_actual_write_is_exact_noop():
    m = _std()
    X, Y = _std_batch()
    base = _logits(m, X, Y)
    les = A12.StandardLesion(m)
    for idx, blk in enumerate(les.blocks):
        cap = {}
        h1 = blk.attn.register_forward_hook(lambda _m, _a, o: cap.__setitem__("attn", o.clone()))
        h2 = blk.mlp.register_forward_hook(lambda _m, _a, o: cap.__setitem__("mlp", o.clone()))
        assert torch.equal(_logits(m, X, Y), base)
        h1.remove(), h2.remove()
        means = {("single", idx, "attn"): cap["attn"], ("single", idx, "mlp"): cap["mlp"]}
        for kind in A12.KINDS:
            les.apply("single", idx, kind, means=means)
            assert torch.equal(_logits(m, X, Y), base), (idx, kind)
            les.clear()


def test_a12_standard_write_means_inert_and_correct():
    m = _std()
    X, Y = _std_batch()
    base = _logits(m, X, Y)
    mu = A12.standard_write_means(m, [(X, Y)]).means()
    assert set(mu) == {("single", i, k) for i in range(3) for k in ("attn", "mlp")}
    assert torch.equal(_logits(m, X, Y), base)
    for blk in m.transformer.h:
        assert "forward" not in blk.__dict__
    blk = m.transformer.h[2]
    cap = {}
    h = blk.mlp.register_forward_hook(lambda _m, _a, o: cap.__setitem__("mlp", o.clone()))
    _logits(m, X, Y)
    h.remove()
    ref = cap["mlp"].double().mean(dim=(0, 1)).float()
    assert torch.allclose(mu[("single", 2, "mlp")], ref, atol=1e-6)


# ============================================================================= A2 standard
def _ko(m, layers):
    mc = m.config
    ko = A2.Knockout("standard", None, mc.n_head, mc.hidden_size // mc.n_head,
                     target_stream="single")
    ko.layers = set(layers)
    return ko


def test_a2_standard_untargeted_is_native_and_patch_restores():
    m = _std()
    X, Y = _std_batch()
    base = _logits(m, X, Y)
    ko = _ko(m, [])
    restore, state = A2.patch_standard(m, ko)
    ko.mode, ko.cap = "apply", 2
    assert torch.equal(_logits(m, X, Y), base)       # no targeted layer -> native, exact
    assert state["calls"] == 0
    ko.mode = "off"
    ko.layers = {0, 1, 2}
    assert torch.equal(_logits(m, X, Y), base)       # mode off -> native, exact
    restore()
    for blk in m.transformer.h:
        assert "forward" not in blk.attn.__dict__
    assert not m._forward_pre_hooks
    assert torch.equal(_logits(m, X, Y), base)


def test_a2_standard_uncapped_control_matches_native():
    m = _std()
    X, Y = _std_batch()
    base = _logits(m, X, Y)
    ko = _ko(m, range(3))
    restore, state = A2.patch_standard(m, ko)
    ko.mode, ko.cap = "apply", None
    got = _logits(m, X, Y)
    assert state["calls"] == 3
    assert torch.allclose(got, base, atol=1e-5), (got - base).abs().max()
    ko.cap = X.shape[1]                              # cap >= T keeps everything
    assert torch.allclose(_logits(m, X, Y), got, atol=0)
    for ab in ("mask", "mean"):                      # nothing ablated -> mean path exact too
        ko.ablation = ab
        ko.mean = {i: torch.randn(2, 16) for i in range(3)}
        assert torch.allclose(_logits(m, X, Y), got, atol=1e-6), ab
    restore()


def _explicit_reference(m, X, Y, cap, layers):
    """Model logits with a hand-built banded document-causal mask on `layers` (flex path
    via the model's own forced window hook is not per-layer, so rebuild with SDPA)."""
    import torch.nn.functional as F
    from modeling.models.model import apply_rotary_emb
    docs = m.generate_document_idx(X)
    T = X.shape[1]
    pos = torch.arange(T)
    base_vis = (pos.view(1, -1) <= pos.view(-1, 1)) & (docs.unsqueeze(-1) == docs.unsqueeze(1))
    saved = []
    for i, blk in enumerate(m.transformer.h):
        a = blk.attn

        def fwd(x, freqs_cis, attn_block_mask=None, _a=a, _i=i):
            B, T_, Cc = x.shape
            hd = Cc // _a.n_head
            q, k, v = _a.c_attn(x).split(_a.hidden_size, dim=2)
            q, k = apply_rotary_emb(q.view(B, T_, _a.n_head, hd),
                                    k.view(B, T_, _a.n_head, hd), freqs_cis=freqs_cis)
            vis = base_vis
            if _i in layers:
                vis = vis & ((pos.view(-1, 1) - pos.view(1, -1)) <= cap)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.view(B, T_, _a.n_head, hd).transpose(1, 2),
                                               attn_mask=vis.unsqueeze(1))
            return _a.c_proj(y.transpose(1, 2).reshape(B, T_, Cc))
        a.forward = fwd
        saved.append(a)
    out = _logits(m, X, Y)
    for a in saved:
        del a.forward
    return out


@pytest.mark.parametrize("layers", [(0, 1, 2), (2,), (1, 2)])
def test_a2_standard_near_mask_matches_banded_reference(layers):
    m = _std()
    X, Y = _std_batch()
    cap = 3
    ref = _explicit_reference(m, X, Y, cap, set(layers))
    ko = _ko(m, layers)
    restore, _ = A2.patch_standard(m, ko)
    ko.mode, ko.cap, ko.band, ko.ablation = "apply", cap, "near", "mask"
    got = _logits(m, X, Y)
    restore()
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    assert not torch.allclose(got, _logits(m, X, Y), atol=1e-4)   # the cap bites


def test_a2_standard_calibration_and_far_band():
    m = _std()
    X, Y = _std_batch()
    ko = _ko(m, range(3))
    restore, _ = A2.patch_standard(m, ko)
    ko.mode = "calibrate"
    base = _logits(m, X, Y)
    ko.mode = "off"
    assert torch.equal(_logits(m, X, Y), base)       # calibration is inert
    ko.finalise()
    assert set(ko.mean) == {0, 1, 2} and ko.mean[0].shape == (2, 16)
    ko.mode, ko.cap = "apply", 3
    outs = {}
    for band, ab in (("near", "mask"), ("near", "mean"), ("far", "mask")):
        ko.band, ko.ablation = band, ab
        outs[(band, ab)] = _logits(m, X, Y)
        assert torch.isfinite(outs[(band, ab)]).all()
        assert not torch.allclose(outs[(band, ab)], base, atol=1e-4), (band, ab)
    assert not torch.allclose(outs[("near", "mask")], outs[("near", "mean")], atol=1e-4)
    restore()


def test_a2_standard_out_path_is_separate():
    run = "s_full_attention_20b_fw100"
    assert C.result_path("a2_ctx_knockout", run) != C.result_path("a2_knockout", run)
