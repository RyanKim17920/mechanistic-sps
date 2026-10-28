"""Term-by-term forward-FLOP accounting for the SPS architecture/size grid.

Shared source of truth so every figure that puts training FLOPs on an axis agrees exactly
(the loss-vs-compute panel and the FLOPs validation-history plot). Component-level accounting
in the DeepMind convention (see A. Casson, "Transformer FLOPs",
adamcasson.com/posts/transformer-flops) -- the full attention-score breakdown (QK-logits +
softmax + attention-value reduction), NOT the 6ND shortcut and NOT the lumped single-term
attention of the Kaplan/OpenAI convention.

Every FLOP is 1 multiply or 1 add, so a matmul with M output elements over a contraction of
length K costs 2 * M * K. The forward pass has three groups, summed over all layers:

  1. PARAMETER MATMULS (2 FLOP / parameter):
       attn_qkv     = 2 * n_layer * d * (3d)      # c_attn  d -> 3d   (Q,K,V)
       attn_project = 2 * n_layer * d * d          # c_proj  d -> d
       ff           = 2 * n_layer * 3 * (d * 3d)   # SwiGLU gate+up+down, each d<->3d
     These sum to 2 * (13 * n_layer * d^2) = 2N, where the 13 = 4 (attention: 3d^2 for QKV
     + 1d^2 for the output projection) + 9 (SwiGLU: 3d^2 each for gate, up, down).

  2. ATTENTION SCORE (does not touch parameters; scales with the attended context, per query):
       qk_logits = 2 * n_layer * n_ctx * d        # Q . K^T
       softmax   = 3 * n_layer * n_head * n_ctx    # ~1.2% of the score work
       reduce    = 2 * n_layer * n_ctx * d        # softmax_weights . V

  3. LM HEAD (2 FLOP / parameter over the vocabulary), charged ONCE:
       logits    = 2 * d * n_vocab

The input embedding is an nn.Embedding lookup (a gather, ~0 FLOP), so vocabulary is charged
only at the head -- we do NOT include the DeepMind one-hot `embeddings` term.

Two multipliers distinguish SPS from Standard. They run ONE forward pass over an interleaved length-2T sequence (even slot = the
input/state token, odd slot = the `<predict>` token), so:
  * pass_mult = 2 on the PARAMETER matmuls -- every parameter matmul runs twice per real token.
  * score_mult on the ATTENTION-SCORE terms -- 2 queries per real token, and every query attends
    the full persistent context (~T keys) together with the recent window (~W keys), i.e. ~(T+W)
    keys per query, so over the two queries the factor is 2(T+W)/T = 2 + 2W/T.
The LM head is charged once for every architecture (only real tokens predict).

The two-tower family is accounted per tower instead; see `two_tower_flops_per_token`.
"""
from __future__ import annotations

# Kaplan vocab used for the LM-head term (task-specified; the real padded vocab 50304 is used
# only for parameter sanity checks elsewhere and is not part of the FLOP count).
V_FLOP = 50257
T_CTX = 4096
W_WINDOW = 64
D_HEAD = 64  # head dim is 64 at every size, so n_head = d_model // 64


def _forward_flops(n_layer: int, d_model: int, pass_mult: float, score_mult: float,
                   intermediate_size: int | None = None) -> float:
    """Forward FLOPs per real token, summing the named components above. `pass_mult` scales the
    parameter matmuls; `score_mult` scales the attention-score terms; the head is charged once.
    `intermediate_size` is the SwiGLU FFN intermediate width (3*d_model when not given)."""
    d = d_model
    ff_intermediate = 3 * d if intermediate_size is None else intermediate_size
    ff_width_sum = n_layer * ff_intermediate
    n_head = d_model // D_HEAD
    # (1) parameter matmuls -- sum to 2N; run pass_mult times per real token
    attn_qkv = pass_mult * (2 * n_layer * d * (3 * d))     # c_attn  d -> 3d
    attn_project = pass_mult * (2 * n_layer * d * d)        # c_proj  d -> d
    ff = pass_mult * (2 * 3 * d * ff_width_sum)  # SwiGLU gate+up+down, summed over layers
    # (2) attention score -- QK-logits + softmax + attention-value reduction, per query
    qk_logits = score_mult * (2 * n_layer * T_CTX * d)     # Q . K^T
    softmax = score_mult * (3 * n_layer * n_head * T_CTX)
    reduce = score_mult * (2 * n_layer * T_CTX * d)        # softmax . V
    # (3) LM head -- charged once (only real tokens predict)
    head = 2 * d * V_FLOP
    return attn_qkv + attn_project + ff + qk_logits + softmax + reduce + head


# NOTE: the ledger's fwd_flops_per_token for s_two_tower_afsps_20b (750,773,760) is the
# UNSHARED value (share_ffn_across_towers=False) and overstates by ~26% -- the correct
# value per this function, with the run's actual share_ffn_across_towers=True config, is
# 595,060,224. The ledger entry itself is left as-is (not edited here).
def two_tower_flops_per_token(
    state_n_layer: int,
    pred_n_layer: int,
    state_hidden: int,
    pred_hidden: int,
    state_intermediate: int,
    pred_intermediate: int,
    read_map: str = "post",
    read_source: str = "state_kv",
    head_dim: int = D_HEAD,
    block_size: int = T_CTX,
    share_ffn_across_towers: bool = False,
    state_intermediate_per_block: list | None = None,
    read_levels: list | None = None,
) -> float:
    """Forward FLOPs per real token for the two-tower architecture, accounted PER TOWER.

    The interleaved-2T shortcut (`pass_mult=2`, `score_mult=2 + 2W/T`) cannot describe
    this family: the two towers may differ in depth, width and FFN width, and -- the whole
    point of the design -- they no longer attend the same key set. So each tower is
    charged separately, in the same DeepMind component convention as `_forward_flops`:

      PARAMETER MATMULS, per tower, per layer (2 FLOP/parameter, ONE query per real token
      per tower -- there is no `pass_mult` here because the towers are counted separately
      rather than as two passes of one stack):
        state: c_attn d_s -> 3*d_s, c_proj d_s -> d_s, SwiGLU 3 * d_s * I_s
        pred:  q_proj d_p -> d_p, c_proj d_p -> d_p, SwiGLU 3 * d_p * I_p,
               + read_kv d_s -> 2*d_p   ONLY when read_source == "pred_proj". Under the
                 default `read_source="state_kv"` the pred block attends the k/v the state
                 tower ALREADY computed, so the read costs no parameters and no matmul.

      ATTENTION SCORE, per tower, per layer: both towers' queries attend the T state keys
      (the pred tower has no keys of its own).

      STATE READ HEAD: when some pred block reads state
      LEVEL L_s, whose k/v come from an extra `d_s -> 2*d_s` projection charged once per
      token (not per layer). It is absent for `read_map="pre"`.

      LM HEAD: 2 * d_p * V_FLOP, charged once.

    WEIGHT TYING CHANGES NO FLOP. `tie_attn_across_towers`, `tie_norms_across_towers` and
    `tie_ffn_across_towers` make pred block i reuse state block i's TENSORS; both towers
    still run every matmul on their own residual, so none of them appears in this
    signature. `share_ffn_across_towers` is the one sharing knob that IS a FLOP change
    (one pooled FFN EVALUATION per position instead of two), which is why it alone is a
    parameter here. The read projection is likewise charged by its computation, not its
    weights: under `read_source="pred_proj"` each of the L_p pred blocks runs its own
    d_s -> 2*d_p read matmul, while under `read_source="state_kv"` with `read_map="final"`
    every pred block reads the SAME level-L_s k/v, computed ONCE by the state read head --
    i.e. state_kv IS the read-projection-tied-across-pred-blocks configuration, and it is
    what makes the fully-tied sequential arm cheaper than the pred_proj one by
    (L_p - 1) * 2 * d_s * 2 * d_p FLOPs/token.
    """
    if state_n_layer <= 0 or pred_n_layer <= 0:
        raise ValueError("both towers need at least one layer")
    if state_hidden % head_dim or pred_hidden % head_dim:
        raise ValueError("tower widths must be multiples of the head dim")
    # read_map only changes WHICH state level is read, never a matmul shape; it matters
    # only through whether some pred block reads level L_s (the read head below).
    if read_map not in ("pre", "post", "top", "final", "explicit"):
        raise ValueError(f"unknown read_map {read_map!r}")
    if read_source not in ("state_kv", "pred_proj"):
        raise ValueError(f"unknown read_source {read_source!r}")

    d_s, d_p = int(state_hidden), int(pred_hidden)
    L_s, L_p = int(state_n_layer), int(pred_n_layer)
    I_s, I_p = int(state_intermediate), int(pred_intermediate)
    nh_s, nh_p = d_s // head_dim, d_p // head_dim

    # (1) parameter matmuls
    # per-block state MLP widths (None => uniform I_s); sum replaces L_s * I_s
    if state_intermediate_per_block is not None:
        if len(state_intermediate_per_block) != L_s:
            raise ValueError("state_intermediate_per_block must have state_n_layer entries")
        sum_I_s = sum(int(x) for x in state_intermediate_per_block)
    else:
        sum_I_s = L_s * I_s
    state_params = (L_s * (2 * d_s * (3 * d_s) + 2 * d_s * d_s) + 2 * 3 * d_s * sum_I_s)
    pred_attn_in = 2 * d_p * d_p                       # q_proj
    if read_source == "pred_proj":
        pred_attn_in += 2 * d_s * (2 * d_p)            # read_kv
    # shared gated FFN: one FFN evaluation per position serves both towers, so the
    # prediction tower's own FFN term disappears (the gates are negligible).
    pred_ffn = 0 if share_ffn_across_towers else 2 * 3 * d_p * I_p
    pred_params = L_p * (pred_attn_in + 2 * d_p * d_p + pred_ffn)

    # (2) attention score: QK logits + attention-value reduction + softmax, over T keys
    T = block_size
    state_score = L_s * (2 * 2 * T * d_s + 3 * nh_s * T)
    pred_score = L_p * (2 * 2 * T * d_p + 3 * nh_p * T)

    # (3) the extra level-L_s read projection, once per token
    if read_map == "pre":
        levels = [i for i in range(L_p)]
    elif read_map == "post":
        levels = [i + 1 for i in range(L_p)]
    elif read_map == "final":
        # FULLY SEQUENTIAL: every pred block reads level L_s (the state tower's output).
        levels = [L_s for _ in range(L_p)]
    elif read_map == "top":
        # TOP-ALIGNED: the L_p pred blocks read the TOP L_p state levels, one each --
        # f(i) = L_s - L_p + 1 + i (see read_level() in two_tower/core.py).
        levels = [L_s - L_p + 1 + i for i in range(L_p)]
    else:  # "explicit"
        if read_levels is None or len(read_levels) != L_p:
            raise ValueError("read_map='explicit' needs read_levels with pred_n_layer entries")
        levels = [int(v) for v in read_levels]
    read_head = (2 * d_s * (2 * d_s)
                 if (max(levels) == L_s and read_source == "state_kv") else 0)

    # (4) LM head, once
    head = 2 * d_p * V_FLOP
    return float(state_params + pred_params + state_score + pred_score + read_head + head)


def forward_flops_per_token(arch: str, n_layer: int, d_model: int,
                            intermediate_size: int | None = None,
                            two_tower: dict | None = None) -> float:
    """Forward FLOPs per *real* token, in the DeepMind component convention (see module docstring).

    Standard runs one pass with full single-query attention. SPS runs the interleaved
    length-2T sequence, so its parameter matmuls double (pass_mult=2) and its attention score
    carries 2 queries, each attending the full context (~T keys) plus the window (~W keys),
    giving score_mult = 2 + 2W/T. Pass the config's real `intermediate_size`.
    """
    if arch == "two_tower":
        # The two towers no longer share a key set or even a shape, so there is no single
        # (pass_mult, score_mult) pair that describes them; the whole accounting lives in
        # `two_tower_flops_per_token`, which takes the per-tower geometry as a dict.
        if two_tower is None:
            raise ValueError("arch='two_tower' needs the per-tower geometry in `two_tower=`")
        return two_tower_flops_per_token(**two_tower)
    if arch == "standard":
        return _forward_flops(n_layer, d_model, pass_mult=1, score_mult=1,
                              intermediate_size=intermediate_size)
    if arch == "sps":
        return _forward_flops(n_layer, d_model, pass_mult=2, score_mult=2 + 2 * W_WINDOW / T_CTX,
                              intermediate_size=intermediate_size)
    raise ValueError(f"unknown arch {arch!r}")
