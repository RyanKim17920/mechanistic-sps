from __future__ import annotations

import types

import pytest
import torch

from modeling.models.sps import SPSConfig, SPSModel


def _make_sps_model(*, enable_triton_attention: bool = False, hidden_size: int = 32,
                    intermediate_size: int = 96) -> SPSModel:
    config = SPSConfig(
        block_size=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dropout=0.0,
        bias=False,
        eos_token_id=63,
        pad_token_id=62,
        predict_token_id=61,
        window_size=2,
        enable_triton_attention=enable_triton_attention,
        warp_specialize=False,
    )
    return SPSModel(config).eval()


def _fake_hidden_states(model: SPSModel, x_Bx2T: torch.Tensor) -> None:
    def _forward_hidden_states(self, idx_Bx2T, *, documents_idx_Bx2T):
        del self, idx_Bx2T, documents_idx_Bx2T
        return x_Bx2T

    model.forward_hidden_states = types.MethodType(_forward_hidden_states, model)


def test_sps_forward_reads_odd_slot_logits() -> None:
    model = _make_sps_model()
    idx = torch.tensor([[4, 5]], dtype=torch.long)
    x_Bx2T = torch.randn(1, 4, model.config.hidden_size)
    _fake_hidden_states(model, x_Bx2T)

    with torch.no_grad():
        logits = model(idx)

    expected = model.lm_head(x_Bx2T[:, 1::2])
    torch.testing.assert_close(logits, expected)
    assert not torch.allclose(logits, model.lm_head(x_Bx2T[:, ::2]))


def test_sps_triton_symbol_imports_cleanly() -> None:
    from modeling.models.attention.triton_sps_flash_attention import ATTN_FWD_CONFIGS
    from modeling.models.sps.core import triton_sps_sliding_attention

    assert triton_sps_sliding_attention is not None
    assert ATTN_FWD_CONFIGS


def test_sps_stats_are_token_and_document_counts() -> None:
    model = _make_sps_model()
    idx = torch.tensor([[4, 5]], dtype=torch.long)
    _fake_hidden_states(model, torch.randn(1, 4, model.config.hidden_size))

    with torch.no_grad():
        _, _, stats = model(idx, idx.clone())

    assert {k for k in stats if not k.startswith("document_length_")} == {
        "token_nll_sum", "token_nll_count", "token_count"}


def test_add_predict_tokens_interleaves_the_predict_slot() -> None:
    model = _make_sps_model()
    idx = torch.tensor([[4, 5, 6]], dtype=torch.long)
    p = model.config.predict_token_id
    assert model.add_predict_tokens(idx).tolist() == [[4, p, 5, p, 6, p]]


@pytest.mark.cuda
def test_sps_dense_forward_reads_odd_slot_logits_on_cuda() -> None:
    from modeling.models.sps.core import triton_sps_sliding_attention

    if triton_sps_sliding_attention is None:
        pytest.skip("dense SPS forward requires CUDA + Triton")

    model = _make_sps_model(enable_triton_attention=True, hidden_size=128,
                            intermediate_size=256).cuda()
    with torch.no_grad():
        for block in model.transformer.h:
            for param in block.parameters():
                param.zero_()
        wte = model.transformer.wte.weight
        wte.zero_()
        wte[4, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        wte[5, :4] = torch.tensor([2.0, 1.0, 0.5, 3.0], device="cuda")
        wte[model.config.predict_token_id, :4] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")

    idx = torch.tensor([[4, 5]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = model(idx)
        slot = model.transformer.wte.weight[torch.full_like(idx, model.config.predict_token_id)]
        expected = model.lm_head(model.transformer.output_norm(slot))

    torch.testing.assert_close(logits, expected, atol=2e-2, rtol=2e-2)
