from __future__ import annotations

"""Attention backends for the two-tower model.

Each tower's queries attend the T state keys: plain causal (key k visible to query i iff
k <= i), restricted to the query's own document. Three backends, resolved in this order by
``resolve_backend("auto")``:

``flash``
    FlashAttention-3 (``flash_attn_interface``) if importable, else FlashAttention-2
    (``flash_attn``). FA cannot express the per-document mask, so this backend is only
    selected when the batch needs no document mask.
``flex``
    ``torch.nn.attention.flex_attention`` with one ``mask_mod``/``BlockMask``. This is the
    backend that runs in this repo's venv (no flash_attn wheel is installed).
``sdpa``
    ``F.scaled_dot_product_attention`` with an explicit boolean mask. Always correct,
    O(T^2) mask memory; the last-resort fallback and the reference in tests.

The callers hand in already-RoPE'd, head-shaped ``(b, n_head, seq, head_dim)`` tensors.
``merge_attn_lse`` (the log-sum-exp merge of two partial softmaxes) is used by
``scripts/analysis/a2_knockout.py``.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

try:  # FlashAttention-3
    from flash_attn_interface import flash_attn_func as _fa3_func
except Exception:  # pragma: no cover - not installed in this venv
    _fa3_func = None

try:  # FlashAttention-2
    from flash_attn import flash_attn_func as _fa2_func
except Exception:  # pragma: no cover - not installed in this venv
    _fa2_func = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover
    flex_attention = None
    create_block_mask = None


BACKENDS = ("auto", "flash", "flex", "sdpa")


def flash_available() -> bool:
    return _fa3_func is not None or _fa2_func is not None


def resolve_backend(name: str, *, needs_document_mask: bool) -> str:
    """Pick a concrete backend.

    ``needs_document_mask`` is decided per batch by the caller: FlashAttention has no way
    to express "same document" for an arbitrary segmentation, so a batch that carries
    document boundaries falls through to flex/sdpa even when a flash wheel is present.
    """
    if name not in BACKENDS:
        raise ValueError(f"attn_backend must be one of {BACKENDS}, got {name!r}")
    if name == "flash":
        if not flash_available():
            raise RuntimeError(
                "attn_backend='flash' requested but neither flash_attn_interface (FA3) "
                "nor flash_attn (FA2) is importable"
            )
        if needs_document_mask:
            raise RuntimeError(
                "attn_backend='flash' cannot express the per-document mask; use "
                "'auto', 'flex' or 'sdpa' for batches with document boundaries"
            )
        return "flash"
    if name == "flex":
        if flex_attention is None:
            raise RuntimeError("attn_backend='flex' requested but flex_attention is unavailable")
        return "flex"
    if name == "sdpa":
        return "sdpa"
    # auto
    if flash_available() and not needs_document_mask:
        return "flash"
    if flex_attention is not None:
        return "flex"
    return "sdpa"


# --------------------------------------------------------------------------------------
# log-sum-exp softmax merge
# --------------------------------------------------------------------------------------
def merge_attn_lse(
    out_a: Tensor, lse_a: Tensor, out_b: Tensor, lse_b: Tensor
) -> tuple[Tensor, Tensor]:
    """Combine two partial softmax attentions over DISJOINT key sets.

    ``out_x`` is ``(b, h, t, d)``, ``lse_x`` is ``(b, h, t)`` -- the natural-log
    normaliser ``log sum_k exp(score_k)`` over that key group. Because softmax over the
    union is the weight-averaged combination of the two partial softmaxes,

        out = (out_a * exp(lse_a - m) + out_b * exp(lse_b - m)) / (exp(lse_a-m) + exp(lse_b-m))
        lse = m + log(exp(lse_a-m) + exp(lse_b-m)),    m = max(lse_a, lse_b)

    A key group that is EMPTY for a query has ``lse = -inf``; the weight is then
    exactly 0 and its (undefined) ``out`` rows are zeroed before the combination so no
    NaN propagates.
    """
    lse_a = lse_a.float()
    lse_b = lse_b.float()
    m = torch.maximum(lse_a, lse_b)
    # both groups empty -> leave m finite so the arithmetic stays defined; the result is
    # then an all-zero output row, which is what an attention over no keys must give.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    wa = torch.nan_to_num(wa, nan=0.0, posinf=0.0, neginf=0.0)
    wb = torch.nan_to_num(wb, nan=0.0, posinf=0.0, neginf=0.0)
    denom = (wa + wb).clamp(min=torch.finfo(torch.float32).tiny)
    oa = torch.nan_to_num(out_a.float(), nan=0.0, posinf=0.0, neginf=0.0)
    ob = torch.nan_to_num(out_b.float(), nan=0.0, posinf=0.0, neginf=0.0)
    out = (oa * wa.unsqueeze(-1) + ob * wb.unsqueeze(-1)) / denom.unsqueeze(-1)
    lse = m + torch.log(denom)
    return out.to(out_a.dtype), lse


# --------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------
def sdpa_attend(q: Tensor, k: Tensor, v: Tensor, scale: float, mask_BxTxK: Tensor) -> Tensor:
    """SDPA with an explicit boolean mask, broadcast over heads."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_BxTxK.unsqueeze(1), scale=scale)


# flex_attention run eagerly materialises the full score matrix, so the compiled form is the
# only usable one at training shapes. `dynamic=False`: every batch here is a fixed shape.
#
# Dynamo keys its guard cache on the CODE object, so every two-tower model in the process
# shares one cache, and on overflow (default limit 8) it silently falls back to eager
# (measured: 856 ms/step instead of 88 ms, 19.8 GiB instead of 14.7 GiB peak). A gate script
# or sweep holding several differently-configured models would hit that, so the limits are
# raised here.
_LIMITS_RAISED = False


def _raise_dynamo_cache_limits(minimum: int = 64) -> None:
    global _LIMITS_RAISED
    if _LIMITS_RAISED:
        return
    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, minimum)
    dynamo.config.accumulated_cache_size_limit = max(
        dynamo.config.accumulated_cache_size_limit, 4 * minimum)
    _LIMITS_RAISED = True


def make_flex_callables(compile_it: bool = True):
    """-> (flex_attention, create_block_mask), compiled unless `compile_it` is False."""
    if not compile_it:
        return flex_attention, create_block_mask
    _raise_dynamo_cache_limits()
    return (torch.compile(flex_attention, dynamic=False),
            torch.compile(create_block_mask, dynamic=False))


def flex_attend(q: Tensor, k: Tensor, v: Tensor, scale: float, block_mask, fn=None) -> Tensor:
    return (fn or flex_attention)(q, k, v, block_mask=block_mask, scale=scale)


def _flash_call(q: Tensor, k: Tensor, v: Tensor, scale: float, causal: bool, window: tuple):
    """One FA call, returning ``(out_bhtd, lse_bht)``.

    FA takes ``(b, seq, n_head, head_dim)``; the callers here work in ``(b, h, t, d)``,
    so transpose in and out. FA3 returns ``(out, lse)``; FA2 returns
    ``(out, lse, S_dmask)`` when ``return_attn_probs=True``.
    """
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if _fa3_func is not None:
        res = _fa3_func(qt, kt, vt, softmax_scale=scale, causal=causal, window_size=window)
    else:
        res = _fa2_func(
            qt, kt, vt, softmax_scale=scale, causal=causal, window_size=window,
            return_attn_probs=True,
        )
    out, lse = res[0], res[1]
    return out.transpose(1, 2), lse
