"""flex_attention mask_mods for the standard model."""
from typing import Callable

from torch import Tensor


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def document_mask_factory_method(documents_idx: Tensor) -> Callable:
    return lambda b, h, q_idx, kv_idx: documents_idx[b][q_idx] == documents_idx[b][kv_idx]


def left_padding_mask_factory_method(padding_offsets: Tensor) -> Callable:
    """Hide the ``padding_offsets[b]`` left-pad keys at the start of each row."""
    def mask(b, h, q_idx, kv_idx):
        return kv_idx >= padding_offsets[b]
    return mask
