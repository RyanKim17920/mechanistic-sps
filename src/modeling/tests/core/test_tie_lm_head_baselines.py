from __future__ import annotations

"""CPU tests for ``tie_lm_head`` on the two BASELINE models: single-stream full attention
(``modeling.models.full_attention_model.Model``, hydra model ``standard``) and SPS
(``modeling.models.sps.SPSModel``, hydra model ``sps``).

Every two-tower arm has an untied head, so the baselines need an untied variant for a head-matched frontier.
Default (True) must be bit-for-bit the pre-flag behaviour; False must give the LM head its
own ``vocab_size x hidden_size`` table (at production geometry 50304*768 = 38,633,472).

SPS insists on the Triton sliding-attention kernel, so its forward/backward substitutes a
dense CPU reference of the same visibility. Full attention runs its eager
flex-attention path (``use_triton_full_attention=False``).
"""

import pytest
import torch

from modeling.models import full_attention_model as fa
from modeling.models.sps import SPSConfig, SPSModel
from modeling.models.sps import core as sps_core

HIDDEN = 32
BLOCK = 16
VOCAB = 64
COMMON = dict(
    block_size=BLOCK,
    vocab_size=VOCAB,
    n_layer=2,
    n_head=2,
    hidden_size=HIDDEN,
    intermediate_size=64,
    norm_eps=1e-6,
    dropout=0.0,
    bias=False,
    eos_token_id=63,
    pad_token_id=62,
)
SPS_EXTRA = dict(predict_token_id=61, window_size=2, enable_triton_attention=True,
                 warp_specialize=False)
FA_EXTRA = dict(use_triton_full_attention=False, warp_specialize=False)


def _reference_sps_sliding_attention(q, k, v, softmax_scale, window_size,
                                     warp_specialize=False, documents_idx_BxT=None):
    """Dense CPU stand-in for ``triton_sps_sliding_attention`` over the interleaved 2T slots:
    causal in slot space, state (even) keys at any distance, <predict> (odd) keys only
    within ``window_size`` tokens, same document only."""
    del warp_specialize
    b, h, n, _ = q.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * softmax_scale
    idx = torch.arange(n, device=q.device)
    q_idx, k_idx = idx.view(n, 1), idx.view(1, n)
    in_window = (k_idx % 2 == 0) | ((q_idx // 2 - k_idx // 2) <= window_size)
    visible = ((k_idx <= q_idx) & in_window).view(1, 1, n, n).expand(b, h, n, n)
    if documents_idx_BxT is not None:
        same_doc = documents_idx_BxT[:, None, :, None] == documents_idx_BxT[:, None, None, :]
        visible = visible & same_doc
    weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", weights, v.float()).to(v.dtype)


@pytest.fixture(autouse=True)
def _cpu_attention(monkeypatch):
    monkeypatch.setattr(sps_core, "triton_sps_sliding_attention",
                        _reference_sps_sliding_attention)


def _make(kind: str, seed: int = 0, **overrides):
    torch.manual_seed(seed)
    if kind == "sps":
        model = SPSModel(SPSConfig(**{**COMMON, **SPS_EXTRA, **overrides}))
    else:
        model = fa.Model(fa.ModelConfig(**{**COMMON, **FA_EXTRA, **overrides}))
    model.eval()
    return model


KINDS = ["sps", "full_attention"]


@pytest.mark.parametrize("kind", KINDS)
def test_default_is_tied_by_identity(kind):
    model = _make(kind)
    assert model.config.tie_lm_head is True
    assert model.transformer.wte.weight is model.lm_head.weight


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_true_matches_default_bit_for_bit(kind):
    a = _make(kind, seed=3).state_dict()
    b = _make(kind, seed=3, tie_lm_head=True).state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("kind", KINDS)
def test_untied_has_separate_storage_and_exact_param_delta(kind):
    tied = _make(kind)
    untied = _make(kind, tie_lm_head=False)
    assert untied.transformer.wte.weight is not untied.lm_head.weight
    assert (untied.transformer.wte.weight.data_ptr()
            != untied.lm_head.weight.data_ptr())
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(untied) - n(tied) == VOCAB * HIDDEN
    # Same state_dict keys either way -- only the storage sharing differs.
    assert tied.state_dict().keys() == untied.state_dict().keys()
    # The untied head goes through _init_weights (N(0, 0.02)), not nn.Linear's default.
    assert untied.lm_head.weight.std().item() == pytest.approx(0.02, abs=0.01)


@pytest.mark.parametrize("kind", KINDS)
def test_untied_forward_backward_finite_and_reaches_both_tables(kind):
    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, VOCAB - 4, (2, BLOCK), generator=g)
    tgt = torch.randint(0, VOCAB - 4, (2, BLOCK), generator=g)
    model = _make(kind, seed=13, tie_lm_head=False)
    model.train()
    out = model(idx.clone(), tgt.clone())
    loss = out[1]
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    wg, hg = model.transformer.wte.weight.grad, model.lm_head.weight.grad
    assert wg is not None and hg is not None
    assert wg.abs().sum() > 0 and hg.abs().sum() > 0
    assert not torch.equal(wg, hg)


def test_production_geometry_param_delta():
    """At s scale the untied head adds exactly 50304*768 = 38,633,472 parameters."""
    prod = dict(block_size=64, vocab_size=50304, n_layer=1, n_head=12, hidden_size=768,
                intermediate_size=2304, norm_eps=1e-6, dropout=0.0, bias=False,
                eos_token_id=50256, pad_token_id=50303)
    with torch.device("meta"):
        t = fa.Model(fa.ModelConfig(**prod, **FA_EXTRA))
        u = fa.Model(fa.ModelConfig(**prod, **FA_EXTRA, tie_lm_head=False))
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(u) - n(t) == 38_633_472
