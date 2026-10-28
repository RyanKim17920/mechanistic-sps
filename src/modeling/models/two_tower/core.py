from __future__ import annotations

"""Two-tower SPS: the memory (state) and readout (pred) streams as two towers.

Both towers run over the same T token positions:

* **State tower** -- own embedding and blocks; plain causal self-attention over state keys.
  It is an ordinary decoder and never reads the pred stream.
* **Pred tower** -- fed a pause embedding (``predict_embedding="constant"``: one learned row
  at every position) or its own untied token table (``"separate"``). Each pred block's
  queries come from the pred residual; its keys/values are the STATE keys at the level
  chosen by ``read_map``. The pred tower feeds the final norm and the LM head.

Read alignment. Level ``j < L_s`` is the k/v that state block ``j`` computes from the
residual entering it; level ``L_s`` is the state tower's output (read through the extra
``state_read_head`` under ``read_source="state_kv"``). Pred block ``i`` reads level ``f(i)``:

    ``pre``       f(i) = i                  -- state block L's output then feeds nothing
    ``post``      f(i) = i + 1              -- no dead state block
    ``top``       f(i) = L_s - L_p + 1 + i  -- a shorter pred tower reads the TOP L_p levels
    ``final``     f(i) = L_s                -- fully sequential: every pred block reads the
                                               finished state tower's output; L_s != L_p ok
    ``explicit``  f(i) = read_levels[i]     -- the freeze-and-retrain hybrids

Where the pred block's state keys come from (``read_source``):

    ``state_kv``   the k/v state level f(i) already computed (equal widths only).
    ``pred_proj``  the pred block's own ``RMSNorm(d_s)`` + ``Linear(d_s, 2 d_p)`` applied to
                   the state residual at level f(i).

Visibility, mirrored from the interleaved SPS Triton kernel restricted to state keys: state
query i and pred query i both see state key k iff k <= i and k is in the same document.

Cross-tower sharing, all default off:

* ``tie_attn_across_towers`` -- pred block i uses state block i's ``c_attn``/``c_proj``.
* ``share_ffn_across_towers`` -- ONE gated FFN per position pooled from both streams
  (Almost-Free SPS); forces a lockstep schedule with ``read_map="pre"``.
* ``tie_norms_across_towers`` -- pred block i's two RMSNorms are state block i's.
* ``tie_ffn_across_towers`` -- pred block i's MLP is state block i's (weights only).

Checkpoints written before three unused fields were removed still carry them in
``model_args``; ``TwoTowerConfig.REMOVED_FIELDS`` lists them with the only value they ever
took, and ``modeling.models.model.config_args`` drops them on load.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from modeling.models.model import (
    PREDICT_TOKEN_ID,
    ModelConfig,
    apply_rotary_emb,
    configure_adamw,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    masked_lm_loss,
    precompute_freqs_cis,
    validate_left_padded_tokens,
)
from modeling.models.two_tower import attention as attn_backends


READ_MAPS = ("pre", "post", "top", "final", "explicit")
READ_SOURCES = ("state_kv", "pred_proj")
PREDICT_EMBEDDINGS = ("constant", "separate")


# ======================================================================================
# config
# ======================================================================================
@dataclass(kw_only=True)
class TwoTowerConfig(ModelConfig):
    """``n_layer`` / ``hidden_size`` / ``intermediate_size`` are the defaults for both towers;
    the per-tower fields override them. ``n_head`` only fixes the head dim
    (``hidden_size // n_head``), which both towers share."""

    predict_token_id: int = PREDICT_TOKEN_ID

    # --- depth / width, per tower ---------------------------------------------------
    state_n_layer: int = 12
    pred_n_layer: int = 12
    state_hidden: Optional[int] = None          # None -> hidden_size
    pred_hidden: Optional[int] = None           # None -> hidden_size
    state_intermediate: Optional[int] = None    # None -> intermediate_size; 0 = no state MLP
    pred_intermediate: Optional[int] = None     # None -> intermediate_size
    # None -> every state block uses ``state_intermediate``; else one width per state block.
    state_intermediate_per_block: Optional[list] = None

    # --- reading --------------------------------------------------------------------
    read_map: str = "post"
    read_levels: Optional[list] = None          # read_map="explicit" only: f(i) per pred block
    read_source: str = "state_kv"
    predict_embedding: str = "constant"
    tie_lm_head: bool = True

    # --- cross-tower sharing (see the module docstring) -----------------------------
    tie_attn_across_towers: bool = False
    share_ffn_across_towers: bool = False
    tie_norms_across_towers: bool = False
    tie_ffn_across_towers: bool = False

    # --- implementation -------------------------------------------------------------
    attn_backend: str = "auto"  # auto | flash | flex | sdpa
    # Eager flex_attention materialises the whole score matrix; compiled is the only usable
    # form at T=4096. Off for CPU runs (compiled flex has no CPU lowering) and debugging.
    flex_compile: bool = True
    # q/k/v are cast to this dtype before attention, unconditionally, as SPS does; "keep"
    # leaves the incoming dtype alone.
    attn_dtype: str = "bfloat16"

    # Fields of older checkpoints' ``model_args`` that no longer exist, with the one value
    # every run used (the pred tower had no keys of its own).
    REMOVED_FIELDS: ClassVar[dict] = {"pred_window": 0, "state_pred_window": 0,
                                      "pred_self_module": "fused"}


def read_level(i: int, state_n_layer: int, pred_n_layer: int, read_map: str) -> int:
    """``f(i)``: which state LEVEL pred block ``i`` reads. See the module docstring."""
    if read_map == "pre":
        return i
    if read_map == "post":
        return i + 1
    if read_map == "top":
        return state_n_layer - pred_n_layer + 1 + i
    if read_map == "final":
        return state_n_layer
    if read_map == "explicit":
        raise ValueError("read_map='explicit' takes its levels from config.read_levels; "
                         "read_level() cannot compute them")
    raise ValueError(f"read_map must be one of {READ_MAPS}, got {read_map!r}")


# ======================================================================================
# small per-tower primitives (explicit widths; the repo's RMSNorm/MLP read a config)
# ======================================================================================
class TowerRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class TowerMLP(nn.Module):
    """SwiGLU with explicit widths. ``intermediate == 0`` is a valid, empty MLP: it holds no
    parameters and returns exact zeros, so the block is attention-only."""

    def __init__(self, hidden_size: int, intermediate: int, bias: bool, dropout: float):
        super().__init__()
        self.intermediate = int(intermediate)
        if self.intermediate > 0:
            self.gate_proj = nn.Linear(hidden_size, self.intermediate, bias=bias)
            self.up_proj = nn.Linear(hidden_size, self.intermediate, bias=bias)
            self.down_proj = nn.Linear(self.intermediate, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.intermediate == 0:
            return torch.zeros_like(x)
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SharedGatedMLP(nn.Module):
    """ONE FFN evaluated per POSITION, pooled from both streams (Almost-Free SPS §2)::

        g  = sigmoid(W_in [a_bar ; p_bar])            # (b, t, 1), one scalar per position
        f  = FFN(g * a_bar + (1 - g) * p_bar)         # ONE FFN evaluation per position
        da = sigmoid(W_a a_bar) * f ;  dp = sigmoid(W_p p_bar) * f

    ``a_bar``/``p_bar`` are the two streams' post-norm residuals. The gates are
    zero-initialised (every gate 0.5 at init); ``TwoTowerModel`` re-zeroes them after its
    generic init pass.
    """

    def __init__(self, hidden: int, intermediate: int, bias: bool, dropout: float):
        super().__init__()
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.mlp = TowerMLP(hidden, intermediate, bias, dropout)
        self.gate_in = nn.Linear(2 * hidden, 1, bias=False)
        self.gate_state = nn.Linear(hidden, 1, bias=False)
        self.gate_pred = nn.Linear(hidden, 1, bias=False)
        self.reset_gates()

    def reset_gates(self) -> None:
        for g in (self.gate_in, self.gate_state, self.gate_pred):
            nn.init.zeros_(g.weight)

    def forward(self, h_state_BxTxC: Tensor, h_pred_BxTxC: Tensor):
        g = torch.sigmoid(self.gate_in(torch.cat([h_state_BxTxC, h_pred_BxTxC], dim=-1)))
        f = self.mlp(g * h_state_BxTxC + (1.0 - g) * h_pred_BxTxC)
        return (torch.sigmoid(self.gate_state(h_state_BxTxC)) * f,
                torch.sigmoid(self.gate_pred(h_pred_BxTxC)) * f)


def _split_heads(x_BxTxC: Tensor, n_head: int, head_dim: int) -> Tensor:
    b, t, _ = x_BxTxC.shape
    return x_BxTxC.view(b, t, n_head, head_dim)


# ======================================================================================
# blocks
# ======================================================================================
class StateBlock(nn.Module):
    """An ordinary pre-norm causal decoder block, split into steps the forward (and the
    analysis scripts) can interleave with the pred tower."""

    def __init__(self, hidden: int, intermediate: int, n_head: int, head_dim: int,
                 norm_eps: float, bias: bool, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.hidden = hidden
        self.attention_norm = TowerRMSNorm(hidden, norm_eps)
        self.c_attn = nn.Linear(hidden, 3 * n_head * head_dim, bias=bias)
        self.c_proj = nn.Linear(n_head * head_dim, hidden, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.mlp_norm = TowerRMSNorm(hidden, norm_eps)
        self.mlp = TowerMLP(hidden, intermediate, bias, dropout)

    def qkv(self, x_BxTxC: Tensor, freqs_cis: Tensor):
        """-> (q, k, v), each ``(b, n_head, t, head_dim)``, RoPE applied to q and k."""
        h = self.attention_norm(x_BxTxC)
        q, k, v = self.c_attn(h).split(self.n_head * self.head_dim, dim=2)
        q = _split_heads(q, self.n_head, self.head_dim)
        k = _split_heads(k, self.n_head, self.head_dim)
        v = _split_heads(v, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        return q.transpose(1, 2), k.transpose(1, 2), v

    def finish_attn(self, x_BxTxC: Tensor, y_BxHxTxD: Tensor) -> Tensor:
        b, _, t, _ = y_BxHxTxD.shape
        y = y_BxHxTxD.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return x_BxTxC + self.resid_dropout(self.c_proj(y.to(self.c_proj.weight.dtype)))

    def mlp_step(self, x_BxTxC: Tensor) -> Tensor:
        return x_BxTxC + self.mlp(self.mlp_norm(x_BxTxC))


class StateReadHead(nn.Module):
    """k/v for state LEVEL ``L_s`` -- the state tower's OUTPUT residual. Built only under
    ``read_source="state_kv"`` when some ``f(i) == L_s``."""

    def __init__(self, hidden: int, n_head: int, head_dim: int, norm_eps: float, bias: bool):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.norm = TowerRMSNorm(hidden, norm_eps)
        self.kv = nn.Linear(hidden, 2 * n_head * head_dim, bias=bias)

    def forward(self, x_BxTxC: Tensor, freqs_cis: Tensor):
        h = self.norm(x_BxTxC)
        k, v = self.kv(h).split(self.n_head * self.head_dim, dim=2)
        k = _split_heads(k, self.n_head, self.head_dim)
        v = _split_heads(v, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE needs a (q, k) pair; the query side is discarded.
        _, k = apply_rotary_emb(k, k, freqs_cis=freqs_cis)
        return k.transpose(1, 2), v


class PredBlock(nn.Module):
    """Pred-tower block: queries from the pred residual, keys/values from the state read.

    Owns ``q_proj`` and ``c_proj``, plus ``read_norm`` + ``read_kv`` under
    ``read_source="pred_proj"``. With ``tied_attn`` the block owns no attention projection:
    q, read-k/v and the output map are the paired state block's ``c_attn``/``c_proj``.
    """

    def __init__(self, pred_hidden: int, state_hidden: int, intermediate: int,
                 n_head: int, head_dim: int, norm_eps: float, bias: bool, dropout: float,
                 read_source: str, tied_attn: bool = False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.hidden = pred_hidden
        self.read_source = read_source
        self.tied_attn = bool(tied_attn)
        inner = n_head * head_dim
        # Creation order fixes the init RNG stream: keep it.
        self.attention_norm = TowerRMSNorm(pred_hidden, norm_eps)
        if not self.tied_attn:
            self.q_proj = nn.Linear(pred_hidden, inner, bias=bias)
        if read_source == "pred_proj":
            self.read_norm = TowerRMSNorm(state_hidden, norm_eps)
            if not self.tied_attn:
                self.read_kv = nn.Linear(state_hidden, 2 * inner, bias=bias)
        if not self.tied_attn:
            self.c_proj = nn.Linear(inner, pred_hidden, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.mlp_norm = TowerRMSNorm(pred_hidden, norm_eps)
        self.mlp = TowerMLP(pred_hidden, intermediate, bias, dropout)

    def bind_tied(self, state_block: "StateBlock") -> None:
        """Point this block's attention projections at ``state_block``'s.

        ``object.__setattr__`` keeps ``nn.Module`` from registering the state block as a
        child here too, which would duplicate its tensors in ``state_dict``. The parameters
        are the same objects either way: one tensor, gradient from both towers.
        """
        assert self.tied_attn
        object.__setattr__(self, "_tied", state_block)

    def out_proj(self) -> nn.Linear:
        """The output projection this block actually uses (the state block's when tied)."""
        return self._tied.c_proj if self.tied_attn else self.c_proj

    def read_kv_weight(self):
        """-> (weight, bias) of the k/v map applied to the state residual. When tied, the
        k and v slices of the state block's fused ``c_attn`` (output order q, k, v)."""
        inner = self.n_head * self.head_dim
        if self.tied_attn:
            W, b = self._tied.c_attn.weight, self._tied.c_attn.bias
            return W[inner:3 * inner], (None if b is None else b[inner:3 * inner])
        return self.read_kv.weight, self.read_kv.bias

    def query(self, x_BxTxC: Tensor, freqs_cis: Tensor) -> Tensor:
        h = self.attention_norm(x_BxTxC)
        if self.tied_attn:
            inner = self.n_head * self.head_dim
            W, b = self._tied.c_attn.weight, self._tied.c_attn.bias
            qh = F.linear(h, W[:inner], None if b is None else b[:inner])
        else:
            qh = self.q_proj(h)
        q = _split_heads(qh, self.n_head, self.head_dim)
        q, _ = apply_rotary_emb(q, q, freqs_cis=freqs_cis)
        return q.transpose(1, 2)

    def read_state(self, state_residual_BxTxC: Tensor, freqs_cis: Tensor):
        assert self.read_source == "pred_proj"
        h = self.read_norm(state_residual_BxTxC)
        W, b = self.read_kv_weight()
        k, v = F.linear(h, W, b).split(self.n_head * self.head_dim, dim=2)
        k = _split_heads(k, self.n_head, self.head_dim)
        v = _split_heads(v, self.n_head, self.head_dim).transpose(1, 2)
        _, k = apply_rotary_emb(k, k, freqs_cis=freqs_cis)
        return k.transpose(1, 2), v

    def finish_attn(self, x_BxTxC: Tensor, y_BxHxTxD: Tensor) -> Tensor:
        b, _, t, _ = y_BxHxTxD.shape
        proj = self.out_proj()
        y = y_BxHxTxD.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return x_BxTxC + self.resid_dropout(proj(y.to(proj.weight.dtype)))

    def mlp_step(self, x_BxTxC: Tensor) -> Tensor:
        return x_BxTxC + self.mlp(self.mlp_norm(x_BxTxC))


# ======================================================================================
# masks
# ======================================================================================
def _needs_document_mask(documents_idx_BxT: Optional[Tensor]) -> bool:
    if documents_idx_BxT is None:
        return False
    return bool((documents_idx_BxT != documents_idx_BxT[:, :1]).any())


class _MaskSet:
    """The causal + same-document mask of one batch, built once per forward and reused by
    every layer of both towers (both query streams see the same T state keys)."""

    def __init__(self, t: int, documents_idx_BxT: Optional[Tensor], device, backend: str,
                 batch: int, mask_mod=None, flex_fns=(None, None)):
        self.t = t
        self.docs = documents_idx_BxT
        self.device = device
        self.backend = backend
        self.batch = batch
        self.mask_mod = mask_mod
        self.flex_fn, self.block_mask_fn = flex_fns
        self._cache: dict = {}

    def bool_mask(self) -> Tensor:
        """``(b, t, t)`` boolean visibility, for the sdpa backend and as the reference."""
        if "bool" not in self._cache:
            t = self.t
            q_idx = torch.arange(t, device=self.device).view(1, t, 1)
            kv_idx = torch.arange(t, device=self.device).view(1, 1, t)
            vis = kv_idx <= q_idx
            if self.docs is not None:
                vis = vis & (self.docs.unsqueeze(1) == self.docs.unsqueeze(-1))
            else:
                vis = vis.expand(self.batch, t, t)
            self._cache["bool"] = vis.contiguous()
        return self._cache["bool"]

    def block_mask(self):
        """The flex BlockMask. ``mask_mod`` must be a STABLE function object across
        forwards: torch.compile guards on it, so the owning model builds it once and refills
        the document buffer it captured (see ``TwoTowerModel._mask_mod``)."""
        if "block" not in self._cache:
            self._cache["block"] = self.block_mask_fn(
                self.mask_mod,
                B=(None if self.docs is None else self.batch), H=None,
                Q_LEN=self.t, KV_LEN=self.t, device=self.device,
            )
        return self._cache["block"]


def make_mask_mod(docs_buf: Optional[Tensor]):
    """Causal + same-document ``mask_mod``. ``docs_buf`` is a fixed tensor the model
    refills each step, so the compiled block-mask/attention graphs are reused."""

    def mask_mod(b, h, q_idx, kv_idx):
        vis = kv_idx <= q_idx
        if docs_buf is not None:
            vis = vis & (docs_buf[b, kv_idx] == docs_buf[b, q_idx])
        return vis

    return mask_mod


# ======================================================================================
# model
# ======================================================================================
class TwoTowerModel(nn.Module):
    def __init__(self, config: TwoTowerConfig):
        super().__init__()
        assert config.read_map in READ_MAPS, f"read_map must be one of {READ_MAPS}"
        assert config.read_source in READ_SOURCES, f"read_source must be one of {READ_SOURCES}"
        assert config.predict_embedding in PREDICT_EMBEDDINGS
        assert config.state_n_layer > 0 and config.pred_n_layer > 0

        self.config = config
        d_s = int(config.state_hidden or config.hidden_size)
        d_p = int(config.pred_hidden or config.hidden_size)
        i_s = config.intermediate_size if config.state_intermediate is None else int(config.state_intermediate)
        i_p = config.intermediate_size if config.pred_intermediate is None else int(config.pred_intermediate)
        assert config.hidden_size % config.n_head == 0
        head_dim = config.hidden_size // config.n_head
        assert d_s % head_dim == 0 and d_p % head_dim == 0, (
            f"state_hidden/pred_hidden must be multiples of the head dim {head_dim}")
        self.state_hidden, self.pred_hidden = d_s, d_p
        self.state_intermediate, self.pred_intermediate = int(i_s), int(i_p)
        self.head_dim = head_dim
        self.n_head_state = d_s // head_dim
        self.n_head_pred = d_p // head_dim

        L_s, L_p = int(config.state_n_layer), int(config.pred_n_layer)
        if config.read_map in ("pre", "post"):
            assert L_s == L_p, (
                f"read_map={config.read_map!r} requires state_n_layer == pred_n_layer "
                f"(got {L_s} != {L_p})")
        if config.read_map == "top":
            assert L_p <= L_s, f"read_map='top' needs pred_n_layer <= state_n_layer (got {L_p} > {L_s})"
        self.state_n_layer, self.pred_n_layer = L_s, L_p
        same_shape = d_s == d_p and L_s == L_p

        if config.state_intermediate_per_block is None:
            self.state_intermediates = [self.state_intermediate] * L_s
        else:
            per_block = [int(v) for v in config.state_intermediate_per_block]
            assert len(per_block) == L_s, (
                f"state_intermediate_per_block must have state_n_layer={L_s} entries, "
                f"got {len(per_block)}")
            assert all(v >= 0 for v in per_block), "MLP widths must be >= 0"
            self.state_intermediates = per_block

        read_source = config.read_source
        if read_source == "state_kv":
            assert d_s == d_p, "read_source='state_kv' reuses the state k/v, so widths must match"
        self.read_source = read_source

        self.tie_attn_across_towers = bool(config.tie_attn_across_towers)
        if self.tie_attn_across_towers:
            assert same_shape, "tie_attn_across_towers needs equal widths and depths"

        self.share_ffn_across_towers = bool(config.share_ffn_across_towers)
        if self.share_ffn_across_towers:
            assert same_shape, "share_ffn_across_towers needs equal widths and depths"
            assert self.state_intermediate == self.pred_intermediate, (
                "share_ffn_across_towers needs ONE FFN width, but state_intermediate="
                f"{self.state_intermediate} != pred_intermediate={self.pred_intermediate}")
            assert config.state_intermediate_per_block is None, (
                "share_ffn_across_towers is incompatible with state_intermediate_per_block")
            assert config.read_map == "pre", (
                "share_ffn_across_towers pools both streams at the SAME layer, so pred block i "
                f"can only read level i: read_map='pre' (got {config.read_map!r})")
            # The FFN lives in transformer.shared_mlp; the per-tower MLPs are built empty.
            self.shared_intermediate = int(self.state_intermediate)
            self.state_intermediates = [0] * L_s
            self.pred_intermediate = 0
        else:
            self.shared_intermediate = 0

        self.tie_norms_across_towers = bool(config.tie_norms_across_towers)
        if self.tie_norms_across_towers:
            assert same_shape, "tie_norms_across_towers needs equal widths and depths"

        self.tie_ffn_across_towers = bool(config.tie_ffn_across_towers)
        if self.tie_ffn_across_towers:
            assert not self.share_ffn_across_towers, (
                "tie_ffn_across_towers and share_ffn_across_towers are exclusive")
            assert same_shape, "tie_ffn_across_towers needs equal widths and depths"
            assert self.state_intermediate == self.pred_intermediate, (
                "tie_ffn_across_towers shares ONE FFN weight set per layer, so "
                f"state_intermediate={self.state_intermediate} must equal "
                f"pred_intermediate={self.pred_intermediate}")
            assert config.state_intermediate_per_block is None, (
                "tie_ffn_across_towers is incompatible with state_intermediate_per_block")

        if config.read_map == "explicit":
            assert config.read_levels is not None and len(config.read_levels) == L_p, (
                f"read_map='explicit' needs read_levels with pred_n_layer={L_p} entries, "
                f"got {config.read_levels!r}")
            self.read_levels = [int(v) for v in config.read_levels]
        else:
            assert config.read_levels is None, (
                f"read_levels is only meaningful with read_map='explicit' "
                f"(got read_map={config.read_map!r})")
            self.read_levels = [read_level(i, L_s, L_p, config.read_map) for i in range(L_p)]
        assert min(self.read_levels) >= 0 and max(self.read_levels) <= L_s
        self.needs_final_level = max(self.read_levels) == L_s

        if config.tie_lm_head:
            assert d_s == d_p, "tie_lm_head requires state_hidden == pred_hidden"

        # Module creation order fixes the init RNG stream: keep it.
        common = dict(norm_eps=config.norm_eps, bias=config.bias, dropout=config.dropout)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, d_s),
                drop=nn.Dropout(config.dropout),
                state_h=nn.ModuleList([
                    StateBlock(d_s, self.state_intermediates[j], self.n_head_state, head_dim,
                               **common)
                    for j in range(L_s)
                ]),
                pred_h=nn.ModuleList([
                    PredBlock(d_p, d_s, self.pred_intermediate, self.n_head_pred, head_dim,
                              read_source=read_source, tied_attn=self.tie_attn_across_towers,
                              **common)
                    for _ in range(L_p)
                ]),
                output_norm=TowerRMSNorm(d_p, config.norm_eps),
            )
        )
        if self.share_ffn_across_towers:
            self.transformer["shared_mlp"] = nn.ModuleList([
                SharedGatedMLP(d_p, self.shared_intermediate, config.bias, config.dropout)
                for _ in range(L_p)
            ])
        if self.tie_attn_across_towers:
            for i, blk in enumerate(self.transformer.pred_h):
                blk.bind_tied(self.transformer.state_h[i])
        # Tied norms / FFN are plain attribute assignments: the pred block's module IS the
        # state block's. state_dict then carries aliased entries under pred_h, and
        # named_parameters() (which de-duplicates by identity) counts each tensor once.
        if self.tie_norms_across_towers:
            for i, blk in enumerate(self.transformer.pred_h):
                blk.attention_norm = self.transformer.state_h[i].attention_norm
                blk.mlp_norm = self.transformer.state_h[i].mlp_norm
        if self.tie_ffn_across_towers:
            for i, blk in enumerate(self.transformer.pred_h):
                blk.mlp = self.transformer.state_h[i].mlp
        if self.needs_final_level and read_source == "state_kv":
            self.transformer["state_read_head"] = StateReadHead(
                d_s, self.n_head_state, head_dim, config.norm_eps, config.bias)
        if config.predict_embedding == "constant":
            self.predict_wte = nn.Embedding(1, d_p)   # the single pause row
        else:
            self.predict_wte = nn.Embedding(config.vocab_size, d_p)

        self.lm_head = nn.Linear(d_p, config.vocab_size, bias=False)
        if config.tie_lm_head:
            self.transformer.wte.weight = self.lm_head.weight

        # Per-model compiled flex callables; see `attention.make_flex_callables`.
        self._flex_fns = (attn_backends.make_flex_callables(bool(config.flex_compile))
                          if attn_backends.flex_attention is not None else (None, None))
        self.attn_dtype = None if config.attn_dtype in (None, "keep") else getattr(torch, config.attn_dtype)

        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(head_dim, config.block_size), persistent=False
        )

        self.apply(self._init_weights)
        depth = max(L_s, L_p)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * depth))
        # The generic pass above also hit the shared FFN's gates, which must start at zero.
        for m in self.modules():
            if isinstance(m, SharedGatedMLP):
                m.reset_gates()

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
            idx_BxT, eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_adamw(self, weight_decay, learning_rate, betas, device_type)

    def _mask_mod(self, t, b, device, documents_idx_BxT):
        """The stable ``mask_mod``, built once and reused for the model's lifetime.

        torch.compile guards on the closure object, so rebuilding it per step would
        recompile ``create_block_mask`` and ``flex_attention`` on every forward. The
        document ids live in a fixed buffer the closure captured, refilled in place; the
        buffer (and closure) is rebuilt only when the batch shape changes.
        """
        shape = (b, t) if documents_idx_BxT is not None else None
        if getattr(self, "_mask_mod_shape", "unset") != shape:
            self._doc_buf = None if shape is None else torch.zeros(shape, dtype=torch.long, device=device)
            self._mask_mod_cached = make_mask_mod(self._doc_buf)
            self._mask_mod_shape = shape
        if self._doc_buf is not None:
            self._doc_buf.copy_(documents_idx_BxT)
        return self._mask_mod_cached

    def _masks(self, idx_BxT: Tensor, documents_idx_BxT: Optional[Tensor]) -> _MaskSet:
        b, t = idx_BxT.shape
        device = idx_BxT.device
        needs_doc = _needs_document_mask(documents_idx_BxT)
        docs = documents_idx_BxT if needs_doc else None
        backend = attn_backends.resolve_backend(self.config.attn_backend, needs_document_mask=needs_doc)
        return _MaskSet(t, docs, device, backend, b, mask_mod=self._mask_mod(t, b, device, docs),
                        flex_fns=self._flex_fns)

    # -- attention dispatch ---------------------------------------------------------
    def _attend(self, q, k_state, v_state, masks: _MaskSet, stream: str):
        """One attention of one tower (``stream`` is "state" or "pred") over the state keys.
        The analysis scripts wrap this method to intervene on attention."""
        scale = 1.0 / math.sqrt(q.shape[-1])
        if self.attn_dtype is not None:
            q, k_state, v_state = q.to(self.attn_dtype), k_state.to(self.attn_dtype), v_state.to(self.attn_dtype)
        if masks.backend == "flash":
            out, _ = attn_backends._flash_call(q, k_state, v_state, scale, causal=True,
                                               window=(-1, -1))
            return out
        if masks.backend == "flex":
            return attn_backends.flex_attend(q, k_state, v_state, scale, masks.block_mask(),
                                             fn=masks.flex_fn)
        return attn_backends.sdpa_attend(q, k_state, v_state, scale, masks.bool_mask())

    # -- forward --------------------------------------------------------------------
    def forward_hidden_states(self, idx_BxT: Tensor, *, documents_idx_BxT: Optional[Tensor]):
        """Run both towers; return the pred tower's normed output ``(b, t, d_p)``."""
        b, t = idx_BxT.shape
        assert t <= self.freqs_cis.shape[0], (
            f"Cannot forward sequence of length {t}, block size is only "
            f"{self.freqs_cis.shape[0]}")
        freqs_cis = self.freqs_cis.to(idx_BxT.device)[:t]
        masks = self._masks(idx_BxT, documents_idx_BxT)

        x_s = self.transformer.drop(self.transformer.wte(idx_BxT))
        if self.config.predict_embedding == "constant":
            x_p = self.predict_wte.weight[0].to(x_s.dtype).view(1, 1, -1).expand(b, t, -1)
        else:
            x_p = self.predict_wte(idx_BxT)
        x_p = self.transformer.drop(x_p)

        if self.share_ffn_across_towers:
            return self._forward_shared_ffn(x_s, x_p, freqs_cis, masks)
        return self._forward_towers(x_s, x_p, freqs_cis, masks)

    def _state_tower(self, x_s, freqs_cis, masks):
        """Run the (independent) state tower to completion.

        -> ``(kv_levels, res_levels, x_s_out)``: the per-level k/v (``state_kv``) and the
        per-level residuals (``pred_proj``; level j = the residual ENTERING state block j,
        level L_s = the tower's output), exactly as the pred tower consumes them.
        """
        kv_levels: list = []
        res_levels: list = []
        for block in self.transformer.state_h:
            if self.read_source == "pred_proj":
                res_levels.append(x_s)
            q, k, v = block.qkv(x_s, freqs_cis)
            kv_levels.append((k, v))
            y = self._attend(q, k, v, masks, "state")
            x_s = block.finish_attn(x_s, y)
            x_s = block.mlp_step(x_s)
        if self.needs_final_level:
            if self.read_source == "pred_proj":
                res_levels.append(x_s)
            else:
                kv_levels.append(self.transformer["state_read_head"](x_s, freqs_cis))
        return kv_levels, res_levels, x_s

    def _forward_towers(self, x_s, x_p, freqs_cis, masks):
        """The state tower is independent, so run it to completion, then the pred tower."""
        kv_levels, res_levels, _ = self._state_tower(x_s, freqs_cis, masks)
        for i, block in enumerate(self.transformer.pred_h):
            lvl = self.read_levels[i]
            if self.read_source == "pred_proj":
                k_s, v_s = block.read_state(res_levels[lvl], freqs_cis)
            else:
                k_s, v_s = kv_levels[lvl]
            y = self._attend(block.query(x_p, freqs_cis), k_s, v_s, masks, "pred")
            x_p = block.finish_attn(x_p, y)
            x_p = block.mlp_step(x_p)
        return self.transformer.output_norm(x_p)

    def _forward_shared_ffn(self, x_s, x_p, freqs_cis, masks):
        """Almost-Free-SPS schedule: lockstep towers with ONE FFN evaluation per position.

        Per layer i the pred block reads level i (the residual ENTERING state block i, taken
        before the state block updates it), both attentions run, then the shared gated FFN
        pools the two post-norm residuals and each stream adds back its gated share. The
        state tower still never attends the pred stream.
        """
        for i, (s_block, p_block) in enumerate(
                zip(self.transformer.state_h, self.transformer.pred_h)):
            # State qkv first, so read_source='state_kv' reuses state block i's own (k, v).
            q_s, k_s, v_s = s_block.qkv(x_s, freqs_cis)
            if self.read_source == "state_kv":
                k_r, v_r = k_s, v_s
            else:
                k_r, v_r = p_block.read_state(x_s, freqs_cis)
            q_p = p_block.query(x_p, freqs_cis)
            y_s = self._attend(q_s, k_s, v_s, masks, "state")
            y_p = self._attend(q_p, k_r, v_r, masks, "pred")
            x_s = s_block.finish_attn(x_s, y_s)
            x_p = p_block.finish_attn(x_p, y_p)
            d_state, d_pred = self.transformer["shared_mlp"][i](
                s_block.mlp_norm(x_s), p_block.mlp_norm(x_p))
            x_s = x_s + d_state
            x_p = x_p + d_pred
        return self.transformer.output_norm(x_p)

    # -- freeze-and-retrain support ------------------------------------------------------
    STATE_TOWER_PREFIXES = ("transformer.wte.", "transformer.state_h.",
                            "transformer.state_read_head.")

    @torch.no_grad()
    def state_levels(self, idx_BxT: Tensor):
        """Diagnostic: the state tower's per-level outputs for one batch, exactly as the
        pred tower consumes them. -> ``(kv_levels, res_levels)``."""
        assert not self.share_ffn_across_towers
        freqs_cis = self.freqs_cis.to(idx_BxT.device)[:idx_BxT.shape[1]]
        masks = self._masks(idx_BxT, self.generate_document_idx(idx_BxT))
        x_s = self.transformer.drop(self.transformer.wte(idx_BxT))
        kv_levels, res_levels, _ = self._state_tower(x_s, freqs_cis, masks)
        return kv_levels, res_levels

    def state_tower_param_names(self) -> list:
        """Names of every STATE-tower parameter (embedding, blocks, level-L_s read head).
        Only defined for fully untied towers: otherwise state tensors are also pred/readout
        tensors and "freeze the state tower" has no meaning."""
        assert not (self.config.tie_lm_head or self.tie_attn_across_towers
                    or self.tie_ffn_across_towers or self.tie_norms_across_towers
                    or self.share_ffn_across_towers), (
            "freezing the state tower requires fully untied towers (tie_lm_head=false, no "
            "tie_*/share_ffn knob)")
        return [n for n, _ in self.named_parameters()
                if n.startswith(self.STATE_TOWER_PREFIXES)]

    @torch.no_grad()
    def freeze_state_tower(self, source_state_dict: dict, verify_existing: bool = False) -> str:
        """Load the state tower from ``source_state_dict`` (a trained checkpoint's model
        state) and freeze it (``requires_grad=False``, hence no optimizer state or decay).

        Every state-tower tensor must be present in the source with the same shape. With
        ``verify_existing`` (a resumed run) the tensors already in the model must equal the
        source bit for bit before they are overwritten: the frozen tower never moved.
        """
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
              for k, v in source_state_dict.items()}
        names = self.state_tower_param_names()
        params = dict(self.named_parameters())
        n_el = 0
        for n in names:
            assert n in sd, f"source checkpoint has no state-tower tensor {n!r}"
            p, src = params[n], sd[n]
            assert tuple(p.shape) == tuple(src.shape), (
                f"{n}: model {tuple(p.shape)} vs source {tuple(src.shape)}")
            src = src.to(device=p.device, dtype=p.dtype)
            if verify_existing:
                assert torch.equal(p.data, src), (
                    f"{n}: resumed tensor differs from the frozen source -- the state tower "
                    f"moved during training")
            p.data.copy_(src)
            p.requires_grad_(False)
            n_el += p.numel()
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"freeze_state_tower: loaded+froze {len(names)} tensors ({n_el:,} params); "
                f"trainable params {n_train:,}; verify_existing={verify_existing}")

    def forward(self, idx_BxT: Tensor, targets_BxT: Optional[Tensor] = None):
        """The SPS objective (``masked_lm_loss``) on the pred tower's output, so NLLs are
        directly comparable across families."""
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(is_real_BxT, context="two_tower inputs")
        documents_idx_BxT = self.generate_document_idx(idx_BxT)

        x_BxTxC = self.forward_hidden_states(idx_BxT, documents_idx_BxT=documents_idx_BxT)
        token_logits_BxTxV = self.lm_head(x_BxTxC)
        if targets_BxT is None:
            return token_logits_BxTxV
        loss, stats = masked_lm_loss(token_logits_BxTxV, idx_BxT, targets_BxT, is_real_BxT,
                                     documents_idx_BxT, self.config.eos_token_id)
        return token_logits_BxTxV, loss, stats
