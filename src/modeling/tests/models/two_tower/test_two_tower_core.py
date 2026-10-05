"""Two-tower model: mask semantics, read alignment, dead blocks, backend agreement.

The load-bearing claim of `modeling/models/two_tower` is that its attention visibility is
the interleaved SPS Triton kernel's, restricted to the two-tower key sets. These tests
re-derive that visibility from the kernel's own rules (2T causal mask plus the parity
bias in `attention/triton_sps_flash_attention.py` :69-107) and check the model against it,
rather than against a restatement of the model's own mask code.
"""
import math

import pytest
import torch

from modeling.models.two_tower.core import (
    TwoTowerConfig,
    TwoTowerModel,
    _MaskSet,
    read_level,
)

T = 24
SMALL = dict(
    block_size=128, vocab_size=512, n_layer=3, n_head=2, hidden_size=32,
    intermediate_size=64, eos_token_id=500, pad_token_id=511, predict_token_id=501,
    state_n_layer=3, pred_n_layer=3, flex_compile=False,
    # These tests compare SEMANTICS across backends, so run attention in the incoming
    # dtype instead of the SPS family's unconditional bf16 cast (checked separately in
    # test_attention_dtype_defaults_to_the_sps_family_bf16_cast).
    attn_dtype="keep",
)


def build(**kw):
    torch.manual_seed(0)
    return TwoTowerModel(TwoTowerConfig(**{**SMALL, **kw}))


def batch():
    torch.manual_seed(1)
    x = torch.randint(0, 400, (2, T))
    x[0, 7] = 500  # EOS -> a document boundary, so the document mask is exercised
    x[1, 13] = 500
    y = torch.randint(0, 400, (2, T))
    return x, y


# ======================================================================================
# visibility, re-derived from the interleaved kernel
# ======================================================================================
def kernel_sees_state_key(stream: str, i: int, k: int) -> bool:
    """Is state key token `k` visible to query token `i`, per the interleaved SPS kernel?

    A state key sits at slot 2k, a state query at 2i and a pred query at 2i+1; the kernel's
    2T causal mask is `q_slot >= k_slot`, and state (persistent) keys carry no window.
    """
    q_slot = 2 * i if stream == "state" else 2 * i + 1
    return q_slot >= 2 * k


@pytest.mark.parametrize("stream", ["state", "pred"])
def test_mask_matches_the_interleaved_kernel_rules(stream):
    got = _MaskSet(T, None, torch.device("cpu"), "sdpa", 1).bool_mask()[0]
    for i in range(T):
        for k in range(T):
            assert bool(got[i, k]) == kernel_sees_state_key(stream, i, k), (stream, i, k)


def test_document_mask_blocks_across_the_eos_boundary():
    x, _ = batch()
    m = build(attn_backend="sdpa")
    masks = _MaskSet(T, m.generate_document_idx(x), torch.device("cpu"), "sdpa", 2)
    vis = masks.bool_mask()
    # row 0's EOS is at position 7, so position 8 must not see position 6
    assert bool(vis[0, 8, 6]) is False
    assert bool(vis[0, 8, 8]) is True


@pytest.mark.parametrize("bad", [
    dict(read_map="scaled"), dict(read_source="auto"), dict(predict_embedding="state_final"),
])
def test_unimplemented_knob_values_are_rejected(bad):
    with pytest.raises(AssertionError):
        build(**bad)


def test_checkpoint_args_with_removed_fields_still_load():
    """Older checkpoints' model_args carry pred_window / state_pred_window /
    pred_self_module; config_args drops them at their one value and rejects any other."""
    from modeling.models.model import config_args
    old = {**SMALL, "pred_window": 0, "state_pred_window": 0, "pred_self_module": "fused"}
    assert TwoTowerConfig(**config_args(TwoTowerConfig, old)) == TwoTowerConfig(**SMALL)
    with pytest.raises(ValueError):
        config_args(TwoTowerConfig, {**old, "pred_window": 64})


# ======================================================================================
# read alignment
# ======================================================================================
def test_read_level_definitions():
    assert [read_level(i, 12, 12, "pre") for i in range(12)] == list(range(12))
    assert [read_level(i, 12, 12, "post") for i in range(12)] == list(range(1, 13))
    with pytest.raises(ValueError):
        read_level(0, 12, 12, "scaled")


def test_read_level_top_is_top_aligned():
    """``top`` gives the SHORTER pred tower the TOP L_p state levels, one each."""
    assert [read_level(i, 12, 6, "top") for i in range(6)] == [7, 8, 9, 10, 11, 12]
    # the last pred block still reads the state tower's output, as ``post`` does
    assert read_level(5, 12, 6, "top") == 12
    # at equal depth ``top`` IS ``post``, so no existing config changes meaning
    assert ([read_level(i, 12, 12, "top") for i in range(12)]
            == [read_level(i, 12, 12, "post") for i in range(12)])


def test_top_read_map_needs_a_pred_tower_no_deeper_than_the_state_tower():
    with pytest.raises(AssertionError):
        build(state_n_layer=2, pred_n_layer=3, read_map="top")


def test_read_level_final_is_the_state_towers_output_for_every_pred_block():
    """``final`` = FULLY SEQUENTIAL: f(i) = L_s, so the state tower has finished writing
    before any pred block reads."""
    assert [read_level(i, 6, 6, "final") for i in range(6)] == [6] * 6
    # it is NOT ``post`` except at the last block
    assert [read_level(i, 12, 12, "final") for i in range(12)] == [12] * 12
    assert read_level(11, 12, 12, "final") == read_level(11, 12, 12, "post")
    assert [read_level(i, 12, 6, "final") for i in range(6)] != \
        [read_level(i, 12, 6, "top") for i in range(6)]
    # every level is L_s, so the level-L_s read applies and nothing reads below it
    m = build(read_map="final", attn_backend="sdpa")
    assert m.read_levels == [m.state_n_layer] * m.pred_n_layer
    assert m.needs_final_level
    x, y = batch()
    _, loss, _ = m(x, y)
    assert torch.isfinite(loss)


def test_final_read_map_decouples_the_two_depths():
    """Unlike ``pre``/``post`` (which pin f(i) to i) ``final`` ties no pred block to a
    state level of its own index, so L_s != L_p is allowed in both directions and the
    read levels stay [L_s] * L_p."""
    with pytest.raises(AssertionError):                       # the pinned maps refuse
        build(state_n_layer=2, pred_n_layer=3, read_map="post")
    shallow = build(state_n_layer=2, pred_n_layer=3, read_map="final")
    assert shallow.read_levels == [2, 2, 2]
    deep = build(state_n_layer=3, pred_n_layer=2, read_map="final")
    assert deep.read_levels == [3, 3]
    # ``top`` would refuse the first of these (it needs L_p <= L_s); ``final`` does not
    x, y = batch()
    for m in (shallow, deep):
        _, loss, _ = m(x, y)
        assert torch.isfinite(loss)


# ======================================================================================
# per-block state MLP widths
# ======================================================================================
def test_state_intermediate_per_block_defaults_to_the_uniform_width():
    """None (the default) must reproduce the uniform build EXACTLY -- same widths, same
    parameter names, same shapes, same values under the same seed. This is what keeps
    every committed config and checkpoint bit-identical."""
    a = build()
    b = build(state_intermediate_per_block=None)
    assert a.state_intermediates == [a.state_intermediate] * a.state_n_layer
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb)
    # spelling the uniform width out explicitly is also the same model
    c = build(state_intermediate_per_block=[SMALL["intermediate_size"]] * 3)
    for (_, pa), (_, pc) in zip(a.named_parameters(), c.named_parameters()):
        assert torch.equal(pa, pc)


def test_state_intermediate_per_block_sets_each_block_independently():
    m = build(state_intermediate_per_block=[128, 64, 0])
    assert m.state_intermediates == [128, 64, 0]
    h = m.transformer.state_h
    assert h[0].mlp.gate_proj.weight.shape == (128, SMALL["hidden_size"])
    assert h[1].mlp.gate_proj.weight.shape == (64, SMALL["hidden_size"])
    assert h[2].mlp.intermediate == 0                      # attention-only block
    assert not any(p.requires_grad for p in h[2].mlp.parameters())
    x, y = batch()
    _, loss, _ = m(x, y)
    assert torch.isfinite(loss)


def test_state_intermediate_per_block_length_must_match_state_depth():
    with pytest.raises(AssertionError):
        build(state_intermediate_per_block=[128, 64])


def test_attention_dtype_defaults_to_the_sps_family_bf16_cast():
    """SPS casts q/k/v to bf16 before the kernel regardless of autocast."""
    m = TwoTowerModel(TwoTowerConfig(**{**SMALL, "attn_dtype": "bfloat16"}))
    assert m.attn_dtype is torch.bfloat16
    assert TwoTowerConfig(**{k: v for k, v in SMALL.items() if k != "attn_dtype"}).attn_dtype == "bfloat16"
    assert TwoTowerModel(TwoTowerConfig(**SMALL)).attn_dtype is None  # "keep"


def test_asymmetric_depth_needs_a_decoupled_read_map():
    with pytest.raises(AssertionError):
        build(state_n_layer=2, pred_n_layer=3, read_map="post")
    build(state_n_layer=3, pred_n_layer=2, read_map="top")


# ======================================================================================
# the dead final state block, and its fix
# ======================================================================================
def _state_grad_norms(model, x, y):
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(x, y)
    loss.backward()
    out = {}
    for name, p in model.named_parameters():
        if name.startswith("transformer.state_h") or "state_read_head" in name:
            out[name] = 0.0 if p.grad is None else float(p.grad.norm())
    return out


def test_read_map_pre_leaves_the_last_state_block_output_gradient_inert():
    """`pre` reproduces the measured bug, and localises it.

    Under f(i)=i the last state block's k/v ARE read (by the last pred block), so its
    `attention_norm`/`c_attn` still train. What feeds nothing is everything AFTER that
    attention -- `c_proj`, `mlp_norm`, `mlp` -- which is precisely the "state block L's
    output feeds nothing / d12 truncation delta = 0.000" finding the design fixes.
    """
    x, y = batch()
    m = build(read_map="pre", attn_backend="sdpa")
    grads = _state_grad_norms(m, x, y)
    dead, alive = {}, {}
    for name, g in grads.items():
        if not name.startswith("transformer.state_h.2."):
            continue
        tail = name[len("transformer.state_h.2."):]
        (alive if tail.startswith(("attention_norm", "c_attn")) else dead)[tail] = g
    assert alive and all(g > 0.0 for g in alive.values()), alive
    assert dead and all(g == 0.0 for g in dead.values()), dead


def test_read_map_post_gives_every_state_parameter_a_gradient():
    """The dead-block fix: with `post` even the last block and the read head train."""
    x, y = batch()
    m = build(read_map="post", attn_backend="sdpa")
    grads = _state_grad_norms(m, x, y)
    assert grads, "no state parameters found"
    zero = [k for k, v in grads.items() if v == 0.0]
    assert zero == [], f"gradient-inert state parameters under read_map=post: {zero}"
    assert any("state_read_head" in k for k in grads)


def test_zero_width_state_mlp_is_valid_and_finite():
    """An attention-only state path must be a legal configuration."""
    x, y = batch()
    m = build(state_intermediate=0, read_map="post", attn_backend="sdpa")
    assert not hasattr(m.transformer.state_h[0].mlp, "gate_proj")
    _, loss, _ = m(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ======================================================================================
# backend agreement
# ======================================================================================
@pytest.mark.parametrize("kw", [
    dict(read_map="post"), dict(read_map="pre"), dict(read_map="final"),
    dict(read_map="post", read_source="pred_proj", predict_embedding="separate"),
])
def test_flex_and_sdpa_backends_agree(kw):
    x, y = batch()
    a = build(attn_backend="sdpa", **kw)
    b = build(attn_backend="flex", **kw)
    with torch.no_grad():
        la = a(x, y)[1]
        lb = b(x, y)[1]
    assert torch.allclose(la, lb, atol=1e-5), (float(la), float(lb))


def test_merge_attn_lse_reproduces_a_single_softmax():
    """The log-sum-exp merge, checked against one softmax: split a key set in two, attend each half, merge, and
    require the result to equal attention over the union -- including a query whose second
    half is empty. (Used by scripts/analysis/a2_knockout.py.)
    """
    from modeling.models.two_tower.attention import merge_attn_lse

    def math_attend_lse(q, k, v, scale, mask_BxTxK):
        """Explicit softmax attention returning ``(out, lse)``."""
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask_BxTxK.unsqueeze(1), float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.nan_to_num(torch.exp(scores - lse.unsqueeze(-1)), nan=0.0)
        return torch.matmul(probs, v.float()).to(q.dtype), lse

    torch.manual_seed(3)
    b, h, t, d, k2 = 2, 2, 6, 8, 6
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t + k2, d)
    v = torch.randn(b, h, t + k2, d)
    scale = 1.0 / math.sqrt(d)
    full_mask = torch.ones(b, t, t + k2, dtype=torch.bool)
    full_mask[:, :, t:] = torch.tril(torch.ones(t, k2, dtype=torch.bool), diagonal=-1)
    ref, _ = math_attend_lse(q, k, v, scale, full_mask)

    out_a, lse_a = math_attend_lse(q, k[:, :, :t], v[:, :, :t], scale, full_mask[:, :, :t])
    out_b, lse_b = math_attend_lse(q, k[:, :, t:], v[:, :, t:], scale, full_mask[:, :, t:])
    assert torch.isinf(lse_b[:, :, 0]).all()  # query 0's second group is empty
    merged, lse = merge_attn_lse(out_a, lse_a, out_b, lse_b)
    assert torch.allclose(merged, ref, atol=1e-5)


# ======================================================================================
# tie_attn_across_towers (the Almost-Free-SPS attention-sharing knob)
# ======================================================================================
TIED = dict(read_map="post", read_source="pred_proj",
            predict_embedding="separate", tie_lm_head=False)


def test_tie_attn_is_off_by_default_and_changes_nothing_when_off():
    """The knob is additive: the untied model's parameter inventory is untouched."""
    assert TwoTowerConfig(**SMALL).tie_attn_across_towers is False
    a = build(**TIED)
    b = build(**TIED, tie_attn_across_towers=False)
    assert sorted(a.state_dict()) == sorted(b.state_dict())
    x, y = batch()
    torch.manual_seed(0)
    assert torch.equal(a(x, y)[0], b(x, y)[0])


def test_tie_attn_removes_exactly_the_pred_towers_four_attention_matrices():
    """Saving is L_p * 4 * d^2 -- q, the two read halves and the output projection."""
    untied = build(**TIED)
    tied = build(**TIED, tie_attn_across_towers=True)
    d, L = SMALL["hidden_size"], SMALL["pred_n_layer"]
    n_un = sum(p.numel() for p in untied.parameters())
    n_ti = sum(p.numel() for p in tied.parameters())
    assert n_un - n_ti == L * 4 * d * d
    # no pred-side attention projection is even allocated
    assert not [k for k in tied.state_dict() if k.startswith("transformer.pred_h.")
                and k.split(".")[-2] in ("q_proj", "kv_self", "read_kv", "c_proj")]
    # ... and the state tower is NOT duplicated into the pred subtree by the binding
    assert len(list(tied.parameters())) == len(set(id(p) for p in tied.parameters()))


def test_tied_read_path_is_the_state_blocks_own_key_projection():
    """Pred block i's read k/v is state block i's c_attn k/v slices, on the state read."""
    m = build(**TIED, tie_attn_across_towers=True)
    for i, blk in enumerate(m.transformer.pred_h):
        W, b = blk.read_kv_weight()
        inner = blk.n_head * blk.head_dim
        assert torch.equal(W, m.transformer.state_h[i].c_attn.weight[inner:3 * inner])
        assert b is None
        assert blk.out_proj() is m.transformer.state_h[i].c_proj


def test_tied_attn_gradient_reaches_the_shared_tensors_from_both_towers():
    m = build(**TIED, tie_attn_across_towers=True)
    x, y = batch()
    m(x, y)[1].backward()
    for blk in m.transformer.state_h:
        assert blk.c_attn.weight.grad is not None
        assert torch.isfinite(blk.c_attn.weight.grad).all()
        assert blk.c_attn.weight.grad.abs().sum() > 0
        assert blk.c_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bad,err", [
    (dict(pred_hidden=64), "widths and depths"),
])
def test_tie_attn_rejects_configurations_where_sharing_is_undefined(bad, err):
    with pytest.raises(AssertionError, match=err):
        build(**{**TIED, **bad}, tie_attn_across_towers=True)


def test_tie_attn_accepts_state_kv_and_leaves_the_pred_block_no_read_path():
    """read_source='state_kv' is legal under tie_attn.

    It is the paper's own read path (arXiv:2609.03807: the prediction stream "forms only a
    query over the state's keys and values"), and it composes with the tying: W_q and
    W_out stay shared while the pred block owns NO read projection at all.
    """
    m = build(**{**TIED, "read_source": "state_kv"}, tie_attn_across_towers=True)
    names = [n for n, _ in m.named_parameters() if n.startswith("transformer.pred_h")]
    assert not [n for n in names if "read_norm" in n or "read_kv" in n]
    x, y = batch()
    loss = m(x, y)[1]
    assert torch.isfinite(loss)


# ======================================================================================
# share_ffn_across_towers -- ONE gated FFN evaluated per POSITION (Almost-Free SPS §2)
# ======================================================================================
# The fully-shared setting: layer i reads layer i, constant pause row, tied
# head.  `tie_attn_across_towers` is deliberately NOT set here -- the FFN knob has to stand
# on its own, and one test below turns both on to check they compose.
SHARED = dict(read_map="pre", read_source="pred_proj",
              predict_embedding="constant", tie_lm_head=True)


def test_share_ffn_is_off_by_default_and_changes_nothing_when_off():
    """The knob is additive: with it off the model is the one that already existed."""
    assert TwoTowerConfig(**SMALL).share_ffn_across_towers is False
    a = build(**SHARED)
    b = build(**SHARED, share_ffn_across_towers=False)
    assert sorted(a.state_dict()) == sorted(b.state_dict())
    assert not any(k.startswith("transformer.shared_mlp") for k in a.state_dict())
    x, y = batch()
    assert torch.equal(a(x, y)[0], b(x, y)[0])


def test_share_ffn_evaluates_one_ffn_per_position_with_zero_init_gates():
    """One FFN module per layer, no per-tower MLP parameters, all three gates at 0.5."""
    d, I, L = SMALL["hidden_size"], SMALL["intermediate_size"], SMALL["state_n_layer"]
    unshared = build(**SHARED)
    shared = build(**SHARED, share_ffn_across_towers=True)

    # the per-tower MLPs are realised EMPTY, so the saving is compute, not just weights
    assert [b.mlp.intermediate for b in shared.transformer.state_h] == [0] * L
    assert [b.mlp.intermediate for b in shared.transformer.pred_h] == [0] * L
    assert sum(p.numel() for b in shared.transformer.state_h for p in b.mlp.parameters()) == 0
    assert sum(p.numel() for b in shared.transformer.pred_h for p in b.mlp.parameters()) == 0

    # one distinct shared module per layer, each holding the single FFN
    mods = list(shared.transformer["shared_mlp"])
    assert len(mods) == L and len({id(m) for m in mods}) == L
    assert [m.mlp.intermediate for m in mods] == [I] * L

    # exactly one tower's FFN removed, three bias-free gates per layer added back
    n_un = sum(p.numel() for p in unshared.parameters())
    n_sh = sum(p.numel() for p in shared.parameters())
    assert n_un - n_sh == L * 3 * d * I - L * 4 * d

    # zero-init gates => every gate is EXACTLY 0.5 at init, so neither stream is
    # privileged and the pooled FFN input is the plain mean.  `self.apply(_init_weights)`
    # runs after construction and would otherwise have filled these with N(0, 0.02).
    gates = [p for n, p in shared.named_parameters()
             if ".gate_in." in n or ".gate_state." in n or ".gate_pred." in n]
    assert len(gates) == 3 * L
    assert all(float(p.abs().max()) == 0.0 for p in gates)
    m = mods[0]
    a_bar, p_bar = torch.randn(2, 5, d), torch.randn(2, 5, d)
    da, dp = m(a_bar, p_bar)
    f = m.mlp(0.5 * (a_bar + p_bar))
    assert torch.allclose(da, 0.5 * f) and torch.allclose(dp, 0.5 * f)


def test_share_ffn_reads_layer_i_at_layer_i_and_trains_both_streams():
    """Read alignment is the paper's (f(i) = i) and gradient reaches the shared FFN."""
    m = build(**SHARED, share_ffn_across_towers=True, tie_attn_across_towers=True)
    L = SMALL["state_n_layer"]
    assert m.read_levels == list(range(L))
    x, y = batch()
    loss = m(x, y)[1]
    assert torch.isfinite(loss)
    loss.backward()
    for i, mod in enumerate(m.transformer["shared_mlp"]):
        assert mod.mlp.gate_proj.weight.grad.abs().sum() > 0
        assert mod.gate_in.weight.grad.abs().sum() > 0
        assert mod.gate_pred.weight.grad.abs().sum() > 0
        # read_map="pre" means nothing reads what the LAST state block writes, so the
        # last layer's STATE-side output gate is the one expected gradient-inert tensor.
        if i < L - 1:
            assert mod.gate_state.weight.grad.abs().sum() > 0
        else:
            assert mod.gate_state.weight.grad is None
    # fully shared: the prediction tower owns nothing but RMSNorm weights
    kinds = {n.split(".")[-2] for n, _ in m.named_parameters()
             if n.startswith("transformer.pred_h")}
    assert kinds == {"attention_norm", "read_norm", "mlp_norm"}
    assert m.predict_wte.weight.shape[0] == 1


@pytest.mark.parametrize("bad,err", [
    (dict(read_map="post"), "read_map='pre'"),
    (dict(pred_intermediate=16), "ONE FFN width"),
    (dict(state_intermediate_per_block=[64, 64, 64]), "state_intermediate_per_block"),
    (dict(pred_n_layer=2, read_map="final"), "widths and depths"),
])
def test_share_ffn_rejects_configurations_where_sharing_is_undefined(bad, err):
    with pytest.raises(AssertionError, match=err):
        build(**{**SHARED, **bad}, share_ffn_across_towers=True)


# ======================================================================================
# the PAPER-FAITHFUL arm: state_kv read + tied norms under the shared FFN
# ======================================================================================
FAITHFUL = dict(SHARED, read_source="state_kv")


def test_shared_ffn_state_kv_reuses_the_state_blocks_own_kv_and_owns_no_read_path():
    """CONTRACT CHANGE: 'state_kv' is legal under share_ffn_across_towers.

    The old assert refused it because "the state tower has not been run to completion, so
    there is no cache of per-level k/v". Under read_map='pre' no cache is needed: the
    lockstep loop computes state block i's (k, v) from exactly the residual level i that
    pred block i reads, in the same iteration.
    """
    m = build(**FAITHFUL, share_ffn_across_towers=True, tie_attn_across_towers=True)
    names = [n for n, _ in m.named_parameters() if n.startswith("transformer.pred_h")]
    assert not [n for n in names if "read_norm" in n or "read_kv" in n]
    x, y = batch()
    loss = m(x, y)[1]
    assert torch.isfinite(loss)
    loss.backward()
    assert m.predict_wte.weight.grad.abs().sum() > 0
    assert m.transformer.state_h[0].c_attn.weight.grad.abs().sum() > 0


def test_tie_norms_shares_the_modules_and_removes_two_norms_per_layer():
    """Pred block i's norms ARE state block i's; the count falls by 2 * L_p * d."""
    assert TwoTowerConfig(**SMALL).tie_norms_across_towers is False
    untied = build(**FAITHFUL, share_ffn_across_towers=True, tie_attn_across_towers=True)
    tied = build(**FAITHFUL, share_ffn_across_towers=True, tie_attn_across_towers=True,
                 tie_norms_across_towers=True)
    L, d = SMALL["pred_n_layer"], SMALL["hidden_size"]
    n = lambda m: sum(p.numel() for p in set(m.parameters()))
    assert n(untied) - n(tied) == 2 * L * d
    for i in range(L):
        assert tied.transformer.pred_h[i].attention_norm is tied.transformer.state_h[i].attention_norm
        assert tied.transformer.pred_h[i].mlp_norm is tied.transformer.state_h[i].mlp_norm
    # with everything shared the pred tower owns NO parameter of its own
    assert not [n_ for n_, _ in tied.named_parameters()
                if n_.startswith("transformer.pred_h")
                and not n_.startswith("transformer.state_h")]
    x, y = batch()
    loss = tied(x, y)[1]
    assert torch.isfinite(loss)
    loss.backward()
    assert tied.transformer.state_h[0].attention_norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [dict(pred_hidden=64), dict(pred_n_layer=2, read_map="final")])
def test_tie_norms_rejects_mismatched_towers(bad):
    with pytest.raises(AssertionError, match="widths and depths"):
        build(**{**SHARED, **bad}, tie_norms_across_towers=True)


# ======================================================================================
# tie_ffn_across_towers -- PLAIN FFN weight sharing (the sequential arm's sharing knob)
# ======================================================================================
SEQ_TIED = dict(read_map="final",
                read_source="state_kv", predict_embedding="constant", tie_lm_head=True)


def test_tie_ffn_is_off_by_default_and_changes_nothing_when_off():
    """The knob is additive: with it off the model is the one that already existed."""
    assert TwoTowerConfig(**SMALL).tie_ffn_across_towers is False
    a, b = build(), build(tie_ffn_across_towers=False)
    x, y = batch()
    assert torch.equal(a(x, y)[1], b(x, y)[1])


def test_tie_ffn_shares_the_module_and_removes_one_ffn_per_layer():
    """Pred block i's MLP IS state block i's; the count falls by L_p * 3 * d * I."""
    untied = build(**SEQ_TIED, tie_attn_across_towers=True)
    tied = build(**SEQ_TIED, tie_attn_across_towers=True, tie_ffn_across_towers=True)
    L, d, I = SMALL["pred_n_layer"], SMALL["hidden_size"], SMALL["intermediate_size"]
    n = lambda m: sum(p.numel() for p in set(m.parameters()))
    assert n(untied) - n(tied) == L * 3 * d * I
    for i in range(L):
        assert tied.transformer.pred_h[i].mlp is tied.transformer.state_h[i].mlp
    # It is NOT the pooled AFSPS FFN: each tower still EVALUATES its own FFN.
    assert "shared_mlp" not in tied.transformer
    assert all(b.mlp.intermediate == I for b in tied.transformer.state_h)
    assert all(b.mlp.intermediate == I for b in tied.transformer.pred_h)


def test_fully_tied_sequential_pred_tower_owns_no_parameters():
    """attn + FFN + norms tied, constant pause, tied head -> pred_h contributes nothing."""
    m = build(**SEQ_TIED, tie_attn_across_towers=True, tie_ffn_across_towers=True,
              tie_norms_across_towers=True)
    assert not [n_ for n_, _ in m.named_parameters()
                if n_.startswith("transformer.pred_h")]
    # read_map='final' + read_source='state_kv': ONE read projection for every pred block.
    assert m.read_levels == [SMALL["state_n_layer"]] * SMALL["pred_n_layer"]
    assert "state_read_head" in m.transformer
    x, y = batch()
    loss = m(x, y)[1]
    assert torch.isfinite(loss)
    loss.backward()
    # Nothing is gradient-inert: level L_s is read, so every state block is live.
    assert not [n_ for n_, p in m.named_parameters() if p.grad is None]


def test_tie_ffn_rejects_configurations_where_sharing_is_undefined():
    # read_map='final' decouples the depths, so an unequal-depth pair reaches the knob.
    with pytest.raises(AssertionError, match="widths and depths"):
        build(**SEQ_TIED, pred_n_layer=2, tie_ffn_across_towers=True)
    with pytest.raises(AssertionError, match="ONE FFN weight set"):
        build(**SEQ_TIED, pred_intermediate=128, tie_ffn_across_towers=True)
    with pytest.raises(AssertionError, match="exclusive"):
        build(**SHARED, share_ffn_across_towers=True, tie_ffn_across_towers=True)
