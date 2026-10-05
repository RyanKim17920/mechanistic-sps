"""Diagnostic document-length statistics logged next to the loss."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def masked_per_document_count(documents_idx_BxT: Tensor, mask_BxT: Tensor) -> tuple[Tensor, Tensor]:
    """Per-(row, document) token counts over the masked-in positions, and which are non-empty."""
    B, T = documents_idx_BxT.shape
    device = documents_idx_BxT.device
    mask_BxT = mask_BxT.to(dtype=torch.bool, device=device)
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * T
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)
    doc_count = torch.zeros(B * T, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, mask_BxT.reshape(-1).to(torch.long))
    return doc_count, doc_count > 0


def add_distribution_stats(
    stats: Dict[str, Tensor],
    prefix: str,
    values: Tensor,
) -> None:
    """Add mean/var/min/max/percentile stats for a 1-D tensor into stats dict."""
    vf = values.detach().float()
    stats[f"{prefix}_mean"] = vf.mean()
    stats[f"{prefix}_var"] = vf.var(unbiased=False)
    stats[f"{prefix}_min"] = vf.min()
    stats[f"{prefix}_max"] = vf.max()
    _add_quantile_distribution_stats(stats, prefix, vf)


def add_empty_distribution_stats(
    stats: Dict[str, Tensor],
    prefix: str,
    device: torch.device | str = "cpu",
) -> None:
    """Add NaN-filled distribution stats when there are no values.

    Ensures the stats dict has the same keys as ``add_distribution_stats``
    regardless of whether data was present, which is required for DDP
    all_reduce to use identically-shaped packed tensors on every rank.
    """
    nan = torch.tensor(float("nan"), device=device)
    stats[f"{prefix}_mean"] = nan
    stats[f"{prefix}_var"] = nan
    stats[f"{prefix}_min"] = nan
    stats[f"{prefix}_max"] = nan
    for p in (25, 50, 75, 90, 99):
        stats[f"{prefix}_p{p}"] = nan


@torch._dynamo.disable
def _add_quantile_distribution_stats(
    stats: Dict[str, Tensor],
    prefix: str,
    values: Tensor,
) -> None:
    """Compute percentile stats eagerly to avoid Dynamo symbolic-shape failures."""
    for p in (25, 50, 75, 90, 99):
        stats[f"{prefix}_p{p}"] = values.quantile(p / 100.0)
