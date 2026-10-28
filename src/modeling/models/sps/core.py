from __future__ import annotations

"""SPS: one weight-shared stack over the interleaved 2T sequence.

Token ``i`` occupies slot ``2i`` (the state slot, fed ``x_i``) and slot ``2i+1`` (the
``<predict>`` slot, fed the single ``predict_token_id`` row). The Triton kernel keeps
state keys visible at any distance and <predict> keys only within ``window_size``; the LM
head reads the odd (<predict>) slots.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modeling.models.full_attention_model import ModelConfig
from modeling.models.model import (
    PREDICT_TOKEN_ID,
    MLP,
    RMSNorm,
    apply_rotary_emb,
    configure_adamw,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    masked_lm_loss,
    precompute_freqs_cis,
    validate_left_padded_tokens,
)

try:
    from modeling.models.attention.triton_sps_flash_attention import (
        sps_sliding_attention as triton_sps_sliding_attention,
    )
except Exception:
    triton_sps_sliding_attention = None


@dataclass(kw_only=True)
class SPSConfig(ModelConfig):
    predict_token_id: int = PREDICT_TOKEN_ID
    window_size: int      # <predict> keys are visible within this many tokens
    enable_triton_attention: bool = True
    # Every <predict> slot is fed ``wte[predict_token_id]``; older checkpoints record that
    # as predict_embedding="constant".
    REMOVED_FIELDS: ClassVar[dict] = {"predict_embedding": "constant"}


class SPSFlashAttention(nn.Module):
    def __init__(self, config: SPSConfig):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        self.enable_triton_attention = config.enable_triton_attention
        self.warp_specialize = config.warp_specialize
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.window_size = int(config.window_size)

    def forward(self, x: Tensor, freqs_cis: Tensor,
                documents_idx_Bx2T: Optional[Tensor] = None) -> Tensor:
        b, two_t, c = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(b, two_t, self.n_head, c // self.n_head)
        k = k.view(b, two_t, self.n_head, c // self.n_head)
        v = v.view(b, two_t, self.n_head, c // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        use_triton = (
            self.enable_triton_attention
            and triton_sps_sliding_attention is not None
            and self.dropout == 0.0
            and q.shape[-1] in {16, 32, 64, 128, 256}
        )
        if not use_triton:
            raise RuntimeError(
                "SPS needs the Triton sliding-attention kernel. "
                "Check enable_triton_attention, dropout, and head_dim."
            )
        y = triton_sps_sliding_attention(
            q, k, v, 1.0 / math.sqrt(q.shape[-1]), self.window_size,
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_Bx2T,
        )

        y = y.transpose(1, 2).contiguous().view(b, two_t, c)
        y = y.to(self.c_proj.weight.dtype)
        return self.resid_dropout(self.c_proj(y))


class SPSBlock(nn.Module):
    def __init__(self, config: SPSConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = SPSFlashAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, freqs_cis: Tensor,
                documents_idx_Bx2T: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.attention_norm(x), freqs_cis, documents_idx_Bx2T=documents_idx_Bx2T)
        return x + self.mlp(self.mlp_norm(x))


class SPSModel(nn.Module):
    def __init__(self, config: SPSConfig):
        super().__init__()
        self.config = config

        # Module creation order fixes the init RNG stream: keep it.
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([SPSBlock(config) for _ in range(config.n_layer)]),
                output_norm=RMSNorm(config),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_lm_head:
            self.transformer.wte.weight = self.lm_head.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.hidden_size // config.n_head, config.block_size),
            persistent=False,
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def generate_document_idx(self, idx_BxT: Tensor) -> Tensor:
        return generate_left_padded_document_idx(
            idx_BxT, eos_token_id=self.config.eos_token_id, pad_token_id=self.config.pad_token_id)

    def _expand_real_and_document_idx(self, idx_BxT: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(is_real_BxT, context="sps inputs")
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        return is_real_BxT, documents_idx_BxT, documents_idx_BxT.repeat_interleave(2, dim=1)

    def add_predict_tokens(self, idx_BxT: Tensor) -> Tensor:
        idx_Bx2T = idx_BxT.repeat_interleave(2, dim=1)
        idx_Bx2T[:, 1::2] = self.config.predict_token_id
        return idx_Bx2T

    def forward_hidden_states(self, idx_Bx2T: Tensor, *, documents_idx_Bx2T: Tensor) -> Tensor:
        b, two_t = idx_Bx2T.size()
        assert two_t % 2 == 0, "Doubled sequence length must be even"
        t = two_t // 2
        assert t <= self.freqs_cis.shape[0], (
            f"Cannot forward sequence of length {two_t}, block size is only {self.freqs_cis.shape[0]}")
        position_ids_Bx2T = torch.arange(t, device=idx_Bx2T.device).repeat_interleave(2).unsqueeze(0).expand(b, -1)
        freqs_cis = self.freqs_cis.to(idx_Bx2T.device)[position_ids_Bx2T]

        x = self.transformer.drop(self.transformer.wte(idx_Bx2T))
        for block in self.transformer.h:
            x = block(x, freqs_cis, documents_idx_Bx2T=documents_idx_Bx2T)
        return self.transformer.output_norm(x)

    def forward(self, idx_BxT: Tensor, targets_BxT: Optional[Tensor] = None):
        is_real_BxT, documents_idx_BxT, documents_idx_Bx2T = self._expand_real_and_document_idx(idx_BxT)
        x_Bx2T = self.forward_hidden_states(self.add_predict_tokens(idx_BxT),
                                            documents_idx_Bx2T=documents_idx_Bx2T)
        token_logits_BxTxV = self.lm_head(x_Bx2T[:, 1::2])
        if targets_BxT is None:
            return token_logits_BxTxV
        loss, stats = masked_lm_loss(token_logits_BxTxV, idx_BxT, targets_BxT, is_real_BxT,
                                     documents_idx_BxT, self.config.eos_token_id)
        return token_logits_BxTxV, loss, stats

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_adamw(self, weight_decay, learning_rate, betas, device_type)
