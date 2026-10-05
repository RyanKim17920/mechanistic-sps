"""Control-head selection shared by A8 (`a8_headpatch`) and A9 (`a9_patching`).

The specificity claim ("silencing the previous-token heads costs more than silencing
unrelated heads") used ONE control draw.  `select_controls` adds a distribution of draws
in three modes; this pins down that

  1. the DEFAULT arguments still return exactly the historical heads (the rule and the
     `np.random.default_rng(CTRL_SEED)` draw, frozen below, and the heads recorded in
     every committed a8/a9 result JSON), so existing results reproduce bit-for-bit;
  2. `random_any` never returns a treated (previous-token) head;
  3. `pattern_matched` returns distinct, non-treated heads.

The analysis scripts are not an installed package, so their directory is put on sys.path.
"""
import glob
import json
import os
import pathlib
import sys

import numpy as np
import pytest

_ANALYSIS = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

A8 = pytest.importorskip("a8_headpatch")
A9 = pytest.importorskip("a9_patching")

_RESULTS = _ANALYSIS / "results"


# ------------------------------------------------------------------------------------
# frozen copy of the historical rule (a8_headpatch.pick_heads before the change)
# ------------------------------------------------------------------------------------
def _historical_ctrl(top, per_block, seed=20260919):
    top_keys = {(d["block"], d["head"]) for d in top}
    rng = np.random.default_rng(seed)
    ctrl = []
    for d in top:
        b = d["block"]
        masses = per_block[b]
        med = float(np.median(masses))
        elig = [h for h in range(len(masses))
                if (b, h) not in top_keys and masses[h] <= med]
        if not elig:
            elig = [h for h in range(len(masses)) if (b, h) not in top_keys]
        h = int(rng.choice(elig))
        ctrl.append(dict(block=b, head=h, prev_token_mass=float(masses[h])))
    return ctrl


def _fake_a5(n_block=12, n_head=12, seed=0, family="two_tower"):
    """A minimal A5 result with the fields pick_heads reads."""
    rng = np.random.default_rng(seed)
    mem = "single" if family == "standard" else "state"
    rd = "single" if family == "standard" else "pred"
    syn, nat = [], []
    for b in range(n_block):
        syn.append(dict(block=b, query_stream=rd, key_stream="state",
                        induction_lift=list(rng.gamma(2.0, 5.0, n_head))))
        pm = rng.beta(0.5, 8.0, n_head)
        nat.append(dict(block=b, query_stream=mem, key_stream="state",
                        prev_token_mass=list(pm)))
    return dict(family=family, synthetic=dict(per_head=syn), natural=dict(per_head=nat))


def _per_block(a5res, mem_q="state"):
    pb = {}
    for r in a5res["natural"]["per_head"]:
        if r["query_stream"] == mem_q and "prev_token_mass" in r:
            pb[int(r["block"])] = list(r["prev_token_mass"])
    return pb


def _keys(heads):
    return [(int(d["block"]), int(d["head"])) for d in heads]


def _fake_stats(per_block, seed=3):
    rng = np.random.default_rng(seed)
    return {(b, h): dict(entropy=float(rng.normal(3, 1)), log_distance=float(rng.normal(4, 1)),
                         sink_mass=float(rng.uniform()), out_norm=float(rng.gamma(2, 1)),
                         mean_distance=0.0)
            for b, ms in per_block.items() for h in range(len(ms))}


# ------------------------------------------------------------------------------------
# 1. default == historical, bit-for-bit
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_default_matches_frozen_historical_rule(seed):
    a5 = _fake_a5(seed=seed)
    match, top, ctrl = A8.pick_heads(a5)
    assert ctrl == _historical_ctrl(top, _per_block(a5))
    _m9, top9, ctrl9 = A9.pick_heads(a5)          # also asserts A9 == A8 internally
    assert _keys(top9) == _keys(top) and ctrl9 == ctrl


def test_default_seed_is_unchanged():
    assert A8.CTRL_SEED == 20260919
    assert A8.draw_seed() == A8.CTRL_SEED
    assert A8.DEFAULT_CTRL_MODE == "same_block_lowprev"
    assert A8.ctrl_suffix(A8.DEFAULT_CTRL_MODE, 1) == ""
    assert A8.ctrl_suffix("random_any", 3) == "_ctrl-random_any-n3"
    assert A8.ctrl_suffix(A8.DEFAULT_CTRL_MODE, 10) != ""
    assert A8.ctrl_suffix("pattern_matched", 10) == "_ctrl-pattern_matched-n10"
    assert A8.ctrl_suffix("pattern_matched", 100, match_k=5) == \
        "_ctrl-pattern_matched-n100-k5"


def test_p_plus1():
    s = A8.summarise_draws([0.1, 0.5, 0.2, 0.3], 0.4)
    assert s["frac_ctrl_ge_prev"] == 0.25
    assert s["p_plus1"] == (1 + 1) / (4 + 1)
    assert A8.summarise_draws([0.0] * 9, 1.0)["p_plus1"] == 0.1


_A5_FILES = sorted(glob.glob(str(_RESULTS / "a5_induction_*.json")))


@pytest.mark.skipif(not _A5_FILES, reason="no committed a5 results")
@pytest.mark.parametrize("path", _A5_FILES, ids=lambda p: os.path.basename(p))
def test_default_reproduces_committed_results(path):
    run = os.path.basename(path)[len("a5_induction_"):-len(".json")]
    with open(path) as f:
        a5 = json.load(f)
    match, top, ctrl = A9.pick_heads(a5)
    for name in ("a8_headpatch", "a9_patching"):
        rp = _RESULTS / f"{name}_{run}.json"
        if not rp.exists():
            continue
        with open(rp) as f:
            res = json.load(f)
        assert _keys(res["prev_token_heads"]) == _keys(top), name
        assert _keys(res["control_heads"]) == _keys(ctrl), name
        assert [d["prev_token_mass"] for d in res["control_heads"]] == \
            [d["prev_token_mass"] for d in ctrl], name
        assert (res["matching_head"]["block"], res["matching_head"]["head"]) == \
            (match["block"], match["head"]), name


# ------------------------------------------------------------------------------------
# 2. random_any never includes a treated head
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["two_tower", "standard"])
def test_random_any_excludes_treated(family):
    for seed in range(5):
        a5 = _fake_a5(seed=seed, family=family)
        for draw in range(20):
            match, top, ctrl = A9.pick_heads(a5, 3, ctrl_mode="random_any", draw=draw)
            k = _keys(ctrl)
            assert len(k) == len(top) == 3
            assert not set(k) & set(_keys(top))
            assert len(set(k)) == len(k)
            if family == "standard":
                assert (match["block"], match["head"]) not in set(k)


def test_random_any_draws_differ_and_span_blocks():
    a5 = _fake_a5(seed=1)
    sets = {tuple(_keys(A8.pick_heads(a5, 3, ctrl_mode="random_any", draw=d)[2]))
            for d in range(10)}
    assert len(sets) > 1
    top_blocks = {d["block"] for d in A8.pick_heads(a5)[1]}
    blocks = {b for s in sets for b, _ in s}
    assert blocks - top_blocks, "random_any should reach blocks outside the treated ones"


# ------------------------------------------------------------------------------------
# 3. pattern_matched returns distinct heads
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3])
def test_pattern_matched_distinct(k):
    for seed in range(5):
        a5 = _fake_a5(seed=seed)
        stats = _fake_stats(_per_block(a5), seed=seed)
        for draw in range(10):
            match, top, ctrl = A8.pick_heads(a5, 3, ctrl_mode="pattern_matched", draw=draw,
                                             head_stats=stats, match_k=k)
            kk = _keys(ctrl)
            assert len(kk) == len(set(kk)) == 3
            assert not set(kk) & set(_keys(top))
            thr = A8.PM_PREV_FRAC * min(d["prev_token_mass"] for d in top)
            assert all(d["prev_token_mass"] < thr for d in ctrl)
            assert [(d["matched_to"]["block"], d["matched_to"]["head"]) for d in ctrl] \
                == _keys(top)


def test_pattern_matched_k1_is_nearest_neighbour():
    top = [dict(block=0, head=0, prev_token_mass=0.9)]
    per_block = {0: [0.9, 0.01, 0.02, 0.03]}
    stats = {(0, h): dict(entropy=float(e), log_distance=0.0, sink_mass=0.0, out_norm=0.0)
             for h, e in enumerate([1.0, 5.0, 1.2, 3.0])}
    for s in range(5):
        c = A8.select_controls(top, per_block, mode="pattern_matched",
                               rng=np.random.default_rng(s), head_stats=stats, match_k=1)
        assert _keys(c) == [(0, 2)]


# ------------------------------------------------------------------------------------
# 4. --ctrl-exclude-layers: default unchanged; excluded layers never supply a control
# ------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["random_any", "pattern_matched"])
def test_exclude_layers_default_is_bit_identical(mode):
    """exclude_layers=() (the default) must return exactly what omitting it returns."""
    for seed in range(4):
        a5 = _fake_a5(seed=seed)
        stats = _fake_stats(_per_block(a5), seed=seed)
        for draw in range(10):
            kw = dict(ctrl_mode=mode, draw=draw, head_stats=stats, match_k=5)
            assert A8.pick_heads(a5, 3, **kw) == A8.pick_heads(a5, 3, exclude_layers=(), **kw)
            assert A9.pick_heads(a5, 3, **kw) == A9.pick_heads(a5, 3, exclude_layers=[], **kw)
    assert A8.ctrl_suffix("random_any", 100, exclude_layers=()) == "_ctrl-random_any-n100"
    assert A8.ctrl_suffix("random_any", 100, exclude_layers=[1]) == \
        "_ctrl-random_any-n100-exL1"
    assert A8.ctrl_suffix("pattern_matched", 100, match_k=5, exclude_layers=[1]) == \
        "_ctrl-pattern_matched-n100-k5-exL1"
    assert A8.exclude_layers_record([], ()) == {}


@pytest.mark.parametrize("family", ["two_tower", "standard"])
@pytest.mark.parametrize("mode", ["random_any", "pattern_matched"])
def test_exclude_layers_removes_layer_from_pool(family, mode):
    for seed in range(4):
        a5 = _fake_a5(seed=seed, family=family)
        pb = _per_block(a5, "single" if family == "standard" else "state")
        stats = _fake_stats(pb, seed=seed)
        blocks = set()
        for draw in range(30):
            match, top, ctrl = A9.pick_heads(a5, 3, ctrl_mode=mode, draw=draw,
                                             head_stats=stats, match_k=5,
                                             exclude_layers=[1, 3])
            k = _keys(ctrl)
            assert len(k) == len(set(k)) == 3
            assert not {b for b, _ in k} & {0, 2}, "control drawn from an excluded layer"
            assert not set(k) & set(_keys(top))
            blocks |= {b for b, _ in k}
        assert blocks, "no controls drawn"


def test_exclude_layers_keeps_treated_head_and_reports_it():
    per_block = {0: [0.9, 0.01, 0.02, 0.01], 1: [0.02, 0.8, 0.01, 0.03],
                 2: [0.01, 0.02, 0.03, 0.01]}
    top = [dict(block=0, head=0, prev_token_mass=0.9),
           dict(block=1, head=1, prev_token_mass=0.8)]
    stats = _fake_stats(per_block, seed=1)
    for mode in ("random_any", "pattern_matched"):
        for s in range(10):
            c = A8.select_controls(top, per_block, mode=mode, rng=np.random.default_rng(s),
                                   head_stats=stats, match_k=3, exclude_layers=[1])
            assert all(d["block"] != 0 for d in c)
    rec = A8.exclude_layers_record(top, [1])
    assert rec["ctrl_exclude_layers"] == [1]
    assert [(d["block"], d["head"], d["layer"]) for d in rec["treated_in_excluded_layers"]] \
        == [(0, 0, 1)]
    assert A8.exclude_layers_record(top, [3])["treated_in_excluded_layers"] == []


def test_exclude_layers_rejected_for_default_mode():
    a5 = _fake_a5(seed=0)
    with pytest.raises(AssertionError):
        A8.pick_heads(a5, 3, exclude_layers=[1])
