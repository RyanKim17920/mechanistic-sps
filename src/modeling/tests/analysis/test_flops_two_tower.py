"""FLOP accounting for the two-tower architecture (src/plotting/flops.py).

The anchor is the 12+12 two-tower at 768/2304 with `read_map="pre"` and
`read_source="state_kv"`: 722,462,208 FLOPs/token. It is the SPS count at the same geometry
(755,547,648, `forward_flops_per_token("sps", ...)`) minus what the two-tower does not run:
the pred-side k/v projection (12 * 2 * 768 * 1536) and the 64 window keys each stream's
queries attend in SPS (2 * 12 * (2*2*64*768 + 3*12*64)).
"""
import pytest

from plotting.flops import (
    forward_flops_per_token,
    two_tower_flops_per_token,
)

BASE = dict(
    state_n_layer=12,
    pred_n_layer=12,
    state_hidden=768,
    pred_hidden=768,
    state_intermediate=2304,
    pred_intermediate=2304,
    read_map="pre",
    read_source="state_kv",
)

SPS_12L_FLOPS = 755547648.0
BASE_FLOPS = 722462208.0


def test_base_config_is_sps_minus_the_pred_keys():
    assert forward_flops_per_token("sps", 12, 768, intermediate_size=2304) == SPS_12L_FLOPS
    kv_self = 12 * (2 * 768 * (2 * 768))
    window_keys = 2 * 12 * (2 * 2 * 64 * 768 + 3 * 12 * 64)
    assert two_tower_flops_per_token(**BASE) == BASE_FLOPS == SPS_12L_FLOPS - kv_self - window_keys
    assert forward_flops_per_token("two_tower", 12, 768, two_tower=BASE) == BASE_FLOPS


def test_read_map_post_charges_the_level_Ls_read_head_once():
    """`post` reads state level L_s, which needs one extra d_s -> 2*d_s projection."""
    post = two_tower_flops_per_token(**{**BASE, "read_map": "post"})
    assert post - BASE_FLOPS == 2 * 768 * (2 * 768)
    assert two_tower_flops_per_token(**{**BASE, "read_map": "final"}) == post


def test_pred_proj_read_source_costs_one_projection_per_pred_block():
    projected = two_tower_flops_per_token(**{**BASE, "read_source": "pred_proj"})
    assert projected - BASE_FLOPS == 12 * (2 * 768 * (2 * 768))


def test_asymmetric_geometry_is_accounted_per_tower():
    """Narrowing only the state MLP changes only the state tower's FFN term."""
    narrow = two_tower_flops_per_token(**{**BASE, "state_intermediate": 1152})
    assert BASE_FLOPS - narrow == 12 * (2 * 3 * 768 * (2304 - 1152))


def test_zero_width_state_mlp_is_a_legal_configuration():
    attn_only = two_tower_flops_per_token(**{**BASE, "state_intermediate": 0})
    assert BASE_FLOPS - attn_only == 12 * (2 * 3 * 768 * 2304)


def test_rejects_unknown_settings():
    with pytest.raises(ValueError):
        two_tower_flops_per_token(**{**BASE, "read_map": "sideways"})
    with pytest.raises(ValueError):
        two_tower_flops_per_token(**{**BASE, "read_source": "telepathy"})
    with pytest.raises(ValueError):
        forward_flops_per_token("two_tower", 12, 768)
