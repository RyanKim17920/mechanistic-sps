"""Standard transformer: causal self-attention over the T tokens, document-masked."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import and_masks, create_block_mask

from modeling.masking import causal_mask, document_mask_factory_method, left_padding_mask_factory_method
from modeling.models.model import (
    IGNORE_INDEX,
    MLP,
    Block,
    CausalSelfAttention,
    ModelConfig as BaseModelConfig,
    RMSNorm,
    apply_rotary_emb,
    compute_left_padded_position_ids,
    configure_adamw,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    precompute_freqs_cis,
    validate_left_padded_tokens,
)

try:
    from modeling.models.attention.triton_full_flash_attention import full_attention as triton_full_attention
except Exception:
    triton_full_attention = None


@dataclass(kw_only=True)
class ModelConfig(BaseModelConfig):
    # The Triton document-masked kernel on CUDA (CPU always takes the flex path).
    # standard.yaml sets true; the False default is what small test models rely on.
    use_triton_full_attention: bool = False
    warp_specialize: bool = False
    # Weight-tie lm_head to transformer.wte. False adds vocab_size * hidden_size parameters.
    tie_lm_head: bool = True


class TritonFullAttention(CausalSelfAttention):
    """Standard causal attention backed by the Triton document-masked causal kernel."""

    def __init__(self, config):
        super().__init__(config)
        self.warp_specialize = getattr(config, "warp_specialize", False)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        documents_idx_BxT: Optional[torch.Tensor] = None,
    ):
        if (
            triton_full_attention is None
            or not x.is_cuda
            or attn_block_mask is not None
            or documents_idx_BxT is None
        ):
            return super().forward(x, freqs_cis, attn_block_mask=attn_block_mask)

        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        y = triton_full_attention(
            q,
            k,
            v,
            1.0 / math.sqrt(q.shape[-1]),
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_BxT.contiguous(),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = y.to(self.c_proj.weight.dtype)
        return self.resid_dropout(self.c_proj(y))


class TritonFullAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = TritonFullAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        documents_idx_BxT: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attention_norm(x), freqs_cis, attn_block_mask=attn_block_mask,
                          documents_idx_BxT=documents_idx_BxT)
        return x + self.mlp(self.mlp_norm(x))

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_triton_full_attention = bool(config.use_triton_full_attention)
        block_cls = TritonFullAttentionBlock if self.use_triton_full_attention else Block

        # Module creation order fixes the init RNG stream: keep it.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.hidden_size),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([block_cls(config) for _ in range(config.n_layer)]),
            output_norm=RMSNorm(config),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_lm_head:
            self.transformer.wte.weight = self.lm_head.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.hidden_size // config.n_head, config.block_size),
            persistent=False,
        )

        self.apply(self._init_weights)
        # GPT-2's scaled init of the residual projections.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def generate_document_idx(self, idx_BxT: Tensor) -> Tensor:
        return generate_left_padded_document_idx(
            idx_BxT, eos_token_id=self.config.eos_token_id, pad_token_id=self.config.pad_token_id)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward_hidden_states(self, idx_BxT: Tensor, targets_BxT: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        device = idx_BxT.device
        b, t = idx_BxT.size()

        is_not_pad_mask = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(is_not_pad_mask, context="forward inputs")
        # The Triton kernel handles causal + document masking itself (left padding lands in
        # isolated fake documents); every other path builds a flex block mask.
        use_triton = self.use_triton_full_attention and triton_full_attention is not None and idx_BxT.is_cuda

        # RoPE is relative, so left-padded position ids agree with arange on the real tokens.
        position_ids = compute_left_padded_position_ids(is_not_pad_mask)
        assert t <= self.freqs_cis.shape[0], f"Cannot forward sequence of length {t}, block size is only {self.freqs_cis.shape[0]}"
        freqs_cis = self.freqs_cis.to(device)[position_ids]

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        attn_block_mask = None
        if not use_triton:
            mask_fn = and_masks(causal_mask, document_mask_factory_method(documents_idx_BxT))
            if not bool(is_not_pad_mask.all()):
                padding_offsets = is_not_pad_mask.long().argmax(dim=1)
                mask_fn = and_masks(mask_fn, left_padding_mask_factory_method(padding_offsets))
            attn_block_mask = create_block_mask(mask_fn, B=b, H=None, Q_LEN=t, KV_LEN=t, device=device)

        x = self.transformer.drop(self.transformer.wte(idx_BxT))
        for block in self.transformer.h:
            if self.use_triton_full_attention:
                x = block(x, freqs_cis, attn_block_mask=attn_block_mask,
                          documents_idx_BxT=documents_idx_BxT if use_triton else None)
            else:
                x = block(x, freqs_cis, attn_block_mask=attn_block_mask)
        x = self.transformer.output_norm(x)
        return x, targets_BxT, is_not_pad_mask

    def forward(self, idx_BxT: Tensor, targets_BxT: Tensor):
        """Logits and loss. NOTE: masks ``targets_BxT`` in place (callers pass a copy)."""
        x, targets_BxT, is_not_pad_mask = self.forward_hidden_states(idx_BxT, targets_BxT)
        b = x.size(0)
        logits = self.lm_head(x)
        original_logits = logits[targets_BxT != IGNORE_INDEX].view(b, idx_BxT.size(1), logits.size(-1))

        targets_BxT[~is_not_pad_mask] = IGNORE_INDEX
        targets_BxT[idx_BxT == self.config.eos_token_id] = IGNORE_INDEX
        token_count = (targets_BxT != IGNORE_INDEX).sum()
        token_nll_sum = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets_BxT.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        loss = token_nll_sum / token_count.clamp(min=1)
        stats = {
            "token_nll_sum": token_nll_sum.detach(),
            "token_nll_count": token_count.detach(),
            "token_count": token_count.detach(),
        }
        return original_logits, loss, stats

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_adamw(self, weight_decay, learning_rate, betas, device_type)
