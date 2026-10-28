"""Shared model factories and test utilities."""

from __future__ import annotations

import torch

from modeling.models.full_attention_model import Model, ModelConfig


_BASE_DEFAULTS = dict(
    block_size=16,
    vocab_size=32,
    n_layer=2,
    n_head=2,
    hidden_size=32,
    intermediate_size=96,
    dropout=0.0,
    bias=False,
    eos_token_id=30,
    pad_token_id=31,
)


def make_full_model(**overrides) -> Model:
    model = Model(ModelConfig(**{**_BASE_DEFAULTS, **overrides}))
    model.eval()
    return model


def forward_logits(model, idx_BxT: torch.Tensor) -> torch.Tensor:
    return model(idx_BxT, idx_BxT.clone())[0]
