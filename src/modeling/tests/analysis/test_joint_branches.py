"""Joint-family (tied-SPS) branches of A12 (slot lesion), A16 (per-slot gradient split)
and A20 (per-slot role probe).

Runs on CPU: the Triton SPS kernel is swapped for a dense reference that implements the
same visibility rule (slot-causal, persistent state keys at even slots, windowed pred
keys at odd slots, per-document), so every test here exercises the analysis code, not
the kernel.  What is pinned:

  * slot parity: state = even slots (real token), pred = odd slots (read by the LM head);
  * A20: the joint sites are the model's own post-block residuals at the right slots, and
    the `final` site is exactly what `lm_head` reads;
  * A12: the no-op and identity lesions are bit-exact, the last block's state-slot
    writes are a dead end (delta exactly 0), and a lesion changes only what it should;
  * A16: the leaf forward reproduces the model's loss bit-for-bit, the per-slot parts sum
    to the full gradient (both the consumer and the producer split), and the structural
    zeros hold (last block's consumer-state part; q rows of the cross cells).
"""
import pathlib
import sys

import pytest
import torch

_ANALYSIS = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

C = pytest.importorskip("common")
A12 = pytest.importorskip("a12_depth_lesion")
A16 = pytest.importorskip("a16_grad_orthogonality")
A20 = pytest.importorskip("a20_role_probe")

import modeling.models.sps.core as sps_core  # noqa: E402
from modeling.models.sps import SPSConfig, SPSModel  # noqa: E402


def _ref_sps_attention(q, k, v, sm_scale, window, warp_specialize=False,
                       documents_idx_BxT=None, persistent_key_window=None):
    assert persistent_key_window is None
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
def grad_mode():
    prev = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(prev)


@pytest.fixture
def ref_attn(monkeypatch):
    monkeypatch.setattr(sps_core, "triton_sps_sliding_attention", _ref_sps_attention)


def _model(tie=True, n_layer=3, seed=0):
    torch.manual_seed(seed)
    m = SPSModel(SPSConfig(block_size=16, vocab_size=64, n_layer=n_layer, n_head=2,
                           hidden_size=32, intermediate_size=48, dropout=0.0, bias=False,
                           eos_token_id=63, pad_token_id=62, predict_token_id=61,
                           window_size=2, enable_triton_attention=True,
                           warp_specialize=False, tie_lm_head=tie))
    # a random init has tiny weights everywhere; bump them so every path carries signal
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(0.3 * torch.randn_like(p))
    return m.eval()


def _batch(seed=1, b=2, t=16):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 60, (b, t), generator=g)
    X[0, 7] = 63                                    # one EOS: exercises the document mask
    Y = torch.randint(0, 60, (b, t), generator=g)
    return X, Y


# -------------------------------------------------------------------------------- masks
def test_joint_slot_masks_select_even_state_odd_pred():
    s, p = C.joint_slot_masks(10)
    assert s.shape == (1, 10, 1) and p.shape == (1, 10, 1)
    assert s.view(-1).tolist() == [True, False] * 5
    assert torch.equal(p, ~s)
    x = torch.arange(2 * 10 * 3, dtype=torch.float32).view(2, 10, 3)
    assert torch.equal(x[s.expand_as(x)].view(2, 5, 3), x[:, 0::2])
    assert torch.equal(x[p.expand_as(x)].view(2, 5, 3), x[:, 1::2])
    with pytest.raises(AssertionError):
        C.joint_slot_masks(7)


def test_slot_parity_matches_model_embedding(ref_attn):
    """even slot embeds the real token, odd slot embeds <predict>."""
    m = _model()
    X, _ = _batch()
    idx2 = m.add_predict_tokens(X)
    s, p = C.joint_slot_masks(idx2.shape[1])
    assert torch.equal(idx2[:, 0::2], X)
    assert bool((idx2[p.view(1, -1).expand_as(idx2)] == m.config.predict_token_id).all())


# -------------------------------------------------------------------------------- A20
def test_a20_joint_sites_are_the_models_residuals(ref_attn, grad_mode):
    torch.set_grad_enabled(False)
    m = _model()
    X, Y = _batch()
    sites, final = A20._joint_sites(m, X, Y)
    # reference: replay the stack by hand
    is_real, docs_T, docs_2T = m._expand_real_and_document_idx(X)
    idx2 = m.add_predict_tokens(X)
    t = X.shape[1]
    freqs = m.freqs_cis[torch.arange(t).repeat_interleave(2).unsqueeze(0).expand(X.shape[0], -1)]
    x = m.transformer.wte(idx2)
    for i, blk in enumerate(m.transformer.h):
        x = blk(x, freqs, documents_idx_Bx2T=docs_2T)
        assert torch.equal(sites[f"state_{i}"], x[:, 0::2])
        assert torch.equal(sites[f"pred_{i}"], x[:, 1::2])
    assert torch.equal(final, m.transformer.output_norm(x)[:, 1::2])
    logits, _, _ = m(X, Y)
    assert torch.equal(m.lm_head(final), logits)


def test_a20_capture_sites_dispatches_joint(ref_attn, grad_mode):
    torch.set_grad_enabled(False)
    m = _model()
    X, Y = _batch()
    pos = torch.tensor([1, 5, 9])
    out = A20.capture_sites(m, "sps", X, Y, pos)
    assert set(out) == {f"{s}_{i}" for s in ("state", "pred") for i in range(3)} | {"final"}
    assert out["state_0"].shape == (2, 3, 32)


# -------------------------------------------------------------------------------- A12
def _logits(m, X, Y):
    with torch.no_grad():
        return m(X, Y)[0]


def test_a12_slot_lesion_controls(ref_attn, grad_mode):
    torch.set_grad_enabled(False)
    m = _model()
    X, Y = _batch()
    base = _logits(m, X, Y)
    les = A12.SlotLesion(m)
    L = len(les.blocks)

    les.apply("pred", 0, "block", noop=True)
    assert torch.equal(_logits(m, X, Y), base)
    les.apply("pred", 1, "block", identity_mask=True)
    assert torch.equal(_logits(m, X, Y), base)
    calls = les.calls
    assert calls > 0

    # dead end: the last block's state-slot writes are read by nothing
    for kind in ("attn", "mlp", "block"):
        les.apply("state", L - 1, kind)
        assert torch.equal(_logits(m, X, Y), base), kind
    # the last block's pred-slot writes are what the head reads
    les.apply("pred", L - 1, "mlp")
    assert not torch.allclose(_logits(m, X, Y), base)
    # an early state-slot lesion reaches the loss only through later reads -- but it does
    les.apply("state", 0, "attn")
    assert not torch.allclose(_logits(m, X, Y), base)
    les.clear()
    assert torch.equal(_logits(m, X, Y), base)
    assert "forward" not in les.blocks[0].__dict__


def test_a12_slot_lesion_rows(ref_attn, grad_mode):
    """A block-0 state-slot lesion changes block 0's state rows only (pred rows of block 0
    read block-0 INPUT keys, which the lesion leaves intact)."""
    torch.set_grad_enabled(False)
    m = _model()
    X, Y = _batch()
    cap = {}
    h = m.transformer.h[0].register_forward_hook(lambda _m, _a, o: cap.__setitem__("x", o))
    m(X, Y)
    base = cap["x"].clone()
    les = A12.SlotLesion(m)
    les.apply("state", 0, "block")
    m(X, Y)
    les.clear()
    h.remove()
    assert torch.equal(cap["x"][:, 1::2], base[:, 1::2])
    assert not torch.allclose(cap["x"][:, 0::2], base[:, 0::2])


# -------------------------------------------------------------------------------- A16
def _grads_total(m, X, Y, specs):
    m.zero_grad(set_to_none=True)
    _, loss, _ = m(X, Y)
    loss.backward()
    out = {n: A16._take(p, sl) for n, _k, _b, p, sl, _key in specs}
    m.zero_grad(set_to_none=True)
    return float(loss.detach()), out


@pytest.mark.parametrize("tie", [True, False])
def test_a16_joint_split_is_exact_and_additive(ref_attn, grad_mode, tie):
    torch.set_grad_enabled(True)
    A16._AUTOCAST["on"] = False
    m = _model(tie=tie)
    for p in m.parameters():
        p.requires_grad_(True)
    X, Y = _batch()
    specs = A16.joint_specs(m)
    loss_t, gt = _grads_total(m, X, Y, specs)

    lv = A16.joint_make_leaves(m)
    loss_s = A16.joint_split_loss(m, X, Y, lv)
    assert float(loss_s.detach()) == loss_t                     # identical forward VALUES
    loss_s.backward()
    # the same graph with ONE leaf per tensor: the exact-additivity reference
    lvf = A16.joint_make_leaves(m, shared=True)
    loss_f = A16.joint_split_loss(m, X, Y, lvf)
    assert float(loss_f.detach()) == loss_t
    loss_f.backward()
    L = len(C.joint_blocks(m))
    kinds = {k for _n, k, *_ in specs}
    assert ("embed_head" in kinds) == tie and ("embed" in kinds) == (not tie)
    for name, kind, blk, _p, sl, key in specs:
        d = A16.joint_decompose(lv, key, sl)
        ref = gt[name]
        scale = float(ref.abs().max())
        assert scale > 0, name
        # The model casts q/k/v to bf16 before attention, so the backward rounds dk/dv to
        # bf16 -- once for the summed key gradient in the model, once PER CONSUMER in the
        # split.  Additivity is exact in exact arithmetic; the residual is that rounding.
        for S, P in (("S", "P"), ("Sprod", "Pprod")):
            rel = float((d[S] + d[P] - ref).norm() / ref.norm())
            assert rel <= 1e-2, (name, S, rel)
        gf = A16._leaf_grad(next(iter(lvf[key].values())), sl)
        for S, P in (("S", "P"), ("Sprod", "Pprod")):   # EXACT in the same graph
            assert float((d[S] + d[P] - gf).norm() / gf.norm()) <= 1e-6, (name, S)
        if blk == L - 1 and kind != "norm_attn":       # dead end, consumer split
            assert float(d["S"].abs().max()) == 0.0, name
        if kind == "attn_q":                            # a query is read by its own slot
            assert float(d["cell_sp"].abs().max()) == 0.0
            assert float(d["cell_ps"].abs().max()) == 0.0
            assert torch.equal(d["S"], d["Sprod"])
        if kind in ("attn_k", "attn_v") and blk < L - 1:
            # a state key IS read by pred queries: the cross cell carries gradient
            assert float(d["cell_sp"].abs().max()) > 0.0, name
        if kind == "embed":                             # untied: disjoint row support
            assert float(torch.dot(d["S"], d["P"])) == 0.0


def test_a16_joint_accumulate_halves_sum(ref_attn, grad_mode):
    torch.set_grad_enabled(True)
    A16._AUTOCAST["on"] = False
    m = _model()
    for p in m.parameters():
        p.requires_grad_(True)
    batches = [_batch(seed=s, b=1) for s in range(4)]
    specs = A16.joint_specs(m)
    lv = A16.joint_make_leaves(m)
    half_of = lambda bi: 0 if bi < 2 else 1  # noqa: E731
    acc_t, lt = A16.joint_accumulate(m, batches, specs, "total", half_of)
    acc_s, ls = A16.joint_accumulate(m, batches, specs, "split", half_of, lv=lv)
    assert lt == ls
    for name, *_ in specs:
        tot = acc_t[0][name] + acc_t[1][name]
        parts = sum(acc_s[h][f"{name}::{w}"] for h in (0, 1) for w in ("S", "P"))
        assert float((parts - tot).norm() / tot.norm()) <= 1e-2, name
