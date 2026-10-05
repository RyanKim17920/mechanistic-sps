"""Components shared by the three model families (standard, SPS, two-tower).

Llama-style blocks after https://github.com/meta-llama/llama/blob/main/llama/model.py.
"""

import inspect
from dataclasses import dataclass
from typing import ClassVar, Tuple

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention

from repo_paths import CONF
from modeling.models.utils.masked_stats import (
    add_distribution_stats,
    add_empty_distribution_stats,
    masked_per_document_count,
)

IGNORE_INDEX = -100

# The token ids are defined once, in conf/model/tokens/, which the model yamls include.
# They are read here only to serve as defaults for models built directly (tests).
_TOKENS = CONF / "model" / "tokens"
GPT2_TOKENS = yaml.safe_load((_TOKENS / "gpt2.yaml").read_text())
PREDICT_TOKEN_ID = yaml.safe_load((_TOKENS / "gpt2_predict.yaml").read_text())["predict_token_id"]


def infer_is_real_tokens(idx_BxT: Tensor, pad_token_id: int) -> Tensor:
    return idx_BxT != pad_token_id


def validate_left_padded_tokens(is_real_BxT: Tensor, *, context: str) -> None:
    """Raise unless padding (if any) is contiguous at the left of each row."""
    is_real_BxT = is_real_BxT.to(dtype=torch.bool)
    ever_real = is_real_BxT.cumsum(dim=1) > 0
    has_pad_after_real = ((~is_real_BxT) & ever_real).any(dim=1)
    if bool(has_pad_after_real.any()):
        bad_rows = has_pad_after_real.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"{context} must be left padded only; found pad tokens after real tokens in rows {bad_rows}"
        )


def compute_left_padded_position_ids(is_real_BxT: Tensor) -> Tensor:
    """Consecutive positions for real tokens; pad positions get 0."""
    is_real_BxT = is_real_BxT.to(dtype=torch.bool)
    cumsum = is_real_BxT.long().cumsum(dim=1) - 1
    return torch.where(is_real_BxT, cumsum, torch.zeros_like(cumsum))


def generate_left_padded_document_idx(idx_BxT: Tensor, *, eos_token_id: int, pad_token_id: int) -> Tensor:
    """Document ids: an EOS ends the current document (the EOS itself belongs to it), and
    each left-pad token is its own fake document, disjoint from the real suffix."""
    is_real_BxT = infer_is_real_tokens(idx_BxT, pad_token_id)
    validate_left_padded_tokens(is_real_BxT, context="document index inputs")

    b, t = idx_BxT.shape
    pad_count_Bx1 = (~is_real_BxT).sum(dim=1, keepdim=True)
    is_real_eos_BxT = is_real_BxT & (idx_BxT == eos_token_id)
    document_idx = torch.zeros((b, t), dtype=idx_BxT.dtype, device=idx_BxT.device)
    if t > 1:
        document_idx[:, 1:] = torch.cumsum(is_real_eos_BxT[:, :-1], dim=1, dtype=idx_BxT.dtype)
    document_idx = document_idx + pad_count_Bx1.to(dtype=idx_BxT.dtype)

    pad_doc_idx = torch.cumsum((~is_real_BxT).long(), dim=1) - 1
    pad_doc_idx = torch.clamp(pad_doc_idx, min=0).to(dtype=idx_BxT.dtype)
    return torch.where(is_real_BxT, document_idx, pad_doc_idx)


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.norm_eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """RoPE rotations ``exp(i * t * theta^(-2k/dim))`` for t < end, as complex64 ``(end, dim/2)``."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    """View ``freqs_cis`` -- ``(T, hs)`` shared, or ``(B, T, hs)`` per sequence -- so it
    broadcasts against ``x`` of shape ``(B, T, nh, hs)``."""
    assert x.ndim == 4, f"Expected x to have 4 dimensions (B, T, nh, hs), got {x.ndim}"
    if freqs_cis.ndim == 3:
        assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1]), (
            f"Shape mismatch: freqs_cis {tuple(freqs_cis.shape)} vs x {tuple(x.shape)}")
        return freqs_cis.unsqueeze(2)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        f"Shape mismatch: freqs_cis {tuple(freqs_cis.shape)} vs expected ({x.shape[1]}, {x.shape[-1]})")
    return freqs_cis.view(1, x.shape[1], 1, x.shape[-1])


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    """Rotate ``(B, T, nh, hs)`` queries and keys by ``freqs_cis`` (computed in fp32)."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    """Standard-model attention through flex_attention with the caller's block mask."""

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        # Eager flex_attention: compiled flex has no CPU lowering (pytorch#139434).
        self.flex_attention_fn = flex_attention

    def forward(self, x, freqs_cis: Tensor, attn_block_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        y = self.flex_attention_fn(q, k, v, block_mask=attn_block_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """SwiGLU."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis: Tensor, attn_block_mask=None):
        x = x + self.attn(self.attention_norm(x), freqs_cis, attn_block_mask=attn_block_mask)
        return x + self.mlp(self.mlp_norm(x))


@dataclass(kw_only=True)
class ModelConfig:
    """Base config. ``conf/model/*.yaml`` is authoritative and sets every field; the
    defaults here equal its values (the token ids are read from it) and only serve models
    built directly, i.e. tests. ``block_size`` has no default: set it explicitly.
    """
    block_size: int
    vocab_size: int = GPT2_TOKENS["vocab_size"]
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3 * 768
    norm_eps: float = 1e-6
    dropout: float = 0.0
    bias: bool = False  # bias in Linears
    eos_token_id: int = GPT2_TOKENS["eos_token_id"]
    pad_token_id: int = GPT2_TOKENS["pad_token_id"]

    # Fields that older checkpoints' ``model_args`` still carry but this class no longer
    # has, each with the only value any run used. See ``config_args``.
    REMOVED_FIELDS: ClassVar[dict] = {}


def config_args(config_cls, model_args: dict) -> dict:
    """A checkpoint's ``model_args`` as keyword arguments for ``config_cls``: private keys
    and ``config_cls.REMOVED_FIELDS`` are dropped, after checking that each removed field
    holds its one supported value."""
    removed = config_cls.REMOVED_FIELDS
    for k, only in removed.items():
        if k in model_args and model_args[k] != only:
            raise ValueError(f"{config_cls.__name__}: {k}={model_args[k]!r} is not supported "
                             f"(only {only!r})")
    return {k: v for k, v in model_args.items() if not k.startswith("_") and k not in removed}


def masked_lm_loss(logits_BxTxV: Tensor, idx_BxT: Tensor, targets_BxT: Tensor,
                   is_real_BxT: Tensor, documents_idx_BxT: Tensor, eos_token_id: int):
    """Mean next-token NLL over real, non-EOS input positions, plus logged stats.

    The objective shared by SPS and the two-tower family: padding and every position whose
    INPUT token is EOS are dropped, and the NLL is summed in fp32 before dividing.
    """
    masked_targets_BxT = torch.where(
        is_real_BxT & (idx_BxT != eos_token_id),
        targets_BxT,
        torch.full_like(targets_BxT, IGNORE_INDEX),
    )
    token_count = (masked_targets_BxT != IGNORE_INDEX).sum()
    token_nll_sum = F.cross_entropy(
        logits_BxTxV.view(-1, logits_BxTxV.size(-1)),
        masked_targets_BxT.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    ).float()
    stats = {
        "token_nll_sum": token_nll_sum.detach(),
        "token_nll_count": token_count.detach(),
        "token_count": token_count.detach(),
    }
    with torch.no_grad():
        doc_counts, doc_count_mask = masked_per_document_count(documents_idx_BxT, is_real_BxT)
        doc_lengths = doc_counts[doc_count_mask].float()
        if doc_lengths.numel() > 0:
            add_distribution_stats(stats, "document_length", doc_lengths)
        else:
            add_empty_distribution_stats(stats, "document_length", device=documents_idx_BxT.device)
    return token_nll_sum / token_count.clamp(min=1), stats


def configure_adamw(model: nn.Module, weight_decay, learning_rate, betas, device_type):
    """AdamW over the trainable parameters: weight decay on every >=2-D tensor only."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    print(f"num decayed parameter tensors: {len(decay_params)}, with "
          f"{sum(p.numel() for p in decay_params):,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with "
          f"{sum(p.numel() for p in nodecay_params):,} parameters")
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                  **(dict(fused=True) if use_fused else {}))
    print(f"using fused AdamW: {use_fused}")
    return optimizer
