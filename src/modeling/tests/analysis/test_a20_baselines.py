"""A20 baseline floors (--untrained / --embed-only / token floors).

What is pinned:
  * the default output path and JSON field set, and the parser defaults;
  * the layer-0 site is literally the tensor block 0 reads (wte(x) -- RoPE models add no
    positional vector), per family, and `--embed-only` returns only those sites;
  * the new modes write new files (`_init`, `_init_s<S>`, `_embed`), never the default one;
  * the token floors are exact on a deterministic toy stream.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parents[4]
_ANALYSIS = _REPO / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

C = pytest.importorskip("common")
A20 = pytest.importorskip("a20_role_probe")

import modeling.models.sps.core as sps_core  # noqa: E402
from modeling.models.sps import SPSConfig, SPSModel  # noqa: E402
from modeling.models.two_tower.core import TwoTowerConfig, TwoTowerModel  # noqa: E402

# ------------------------------------------------------------------------------ models
def _ref_sps_attention(q, k, v, sm_scale, window, warp_specialize=False,
                       documents_idx_BxT=None, persistent_key_window=None):
    q, k, v = q.float(), k.float(), v.float()
    two_t = q.shape[2]
    qp = torch.arange(two_t).view(-1, 1)
    kp = torch.arange(two_t).view(1, -1)
    vis = (kp <= qp) & ((kp % 2 == 0) | ((qp // 2 - kp // 2) <= window))
    vis = vis.view(1, 1, two_t, two_t)
    if documents_idx_BxT is not None:
        d = documents_idx_BxT
        vis = vis & (d.unsqueeze(-1) == d.unsqueeze(-2)).unsqueeze(1)
    sc = torch.einsum("bhqd,bhkd->bhqk", q, k) * sm_scale
    sc = sc.masked_fill(~vis, float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(sc, -1), v)


@pytest.fixture
def ref_attn(monkeypatch):
    monkeypatch.setattr(sps_core, "triton_sps_sliding_attention", _ref_sps_attention)


def _bump(m):
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(0.3 * torch.randn_like(p))
    return m.eval()


def _sps():
    torch.manual_seed(0)
    return _bump(SPSModel(SPSConfig(
        block_size=16, vocab_size=64, n_layer=3, n_head=2, hidden_size=32,
        intermediate_size=48, dropout=0.0, bias=False, eos_token_id=63, pad_token_id=62,
        predict_token_id=61, window_size=2, enable_triton_attention=True,
        warp_specialize=False, tie_lm_head=True)))


def _two_tower():
    torch.manual_seed(0)
    return _bump(TwoTowerModel(TwoTowerConfig(
        block_size=16, vocab_size=64, n_layer=3, n_head=2, hidden_size=32,
        intermediate_size=48, eos_token_id=63, pad_token_id=62, predict_token_id=61,
        state_n_layer=3, pred_n_layer=3, flex_compile=False, attn_dtype="keep")))


def _standard():
    from modeling.models.full_attention_model import Model, ModelConfig
    torch.manual_seed(0)
    return _bump(Model(ModelConfig(
        block_size=16, vocab_size=64, n_layer=3, n_head=2, hidden_size=32,
        intermediate_size=48, dropout=0.0, bias=False, eos_token_id=63, pad_token_id=62)))


def _batch(fam="sps"):
    g = torch.Generator().manual_seed(1)
    X = torch.randint(0, 60, (2, 16), generator=g)
    if fam != "standard":      # the tiny standard model's CPU path has no EOS handling
        X[0, 7] = 63
    Y = torch.randint(0, 60, (2, 16), generator=g)
    return X, Y


FAMILIES = {"sps": _sps, "two_tower": _two_tower, "standard": _standard}


def _build(fam):
    try:
        return FAMILIES[fam]()
    except Exception as e:  # pragma: no cover - config drift should fail loudly elsewhere
        pytest.skip(f"cannot build a tiny {fam} model on CPU: {e}")


def _args(argv):
    old_argv = sys.argv
    sys.argv = ["a20"] + argv
    try:
        ap_args = {}
        orig = A20.run_measurement
        A20.run_measurement = lambda a: ap_args.setdefault("a", a)
        try:
            A20.main()
        finally:
            A20.run_measurement = orig
        return ap_args["a"]
    finally:
        sys.argv = old_argv


def test_default_output_path():
    new_args = _args(["--run", "r"])
    assert A20._mode(new_args) == ("", {})
    assert A20.out_path(new_args, "r").endswith("/a20_role_probe_r.json")


def test_new_modes_write_new_files():
    assert A20.out_path(_args(["--run", "r", "--untrained"]), "r").endswith("_r_init.json")
    assert A20.out_path(_args(["--run", "r", "--untrained", "--init-seed", "4321"]),
                        "r").endswith("_r_init_s4321.json")
    assert A20.out_path(_args(["--run", "r", "--embed-only"]), "r").endswith("_r_embed.json")
    s, extra = A20._mode(_args(["--run", "r", "--untrained"]))
    assert extra["untrained"] and extra["embed_site"] and extra["init_seed"] == 1234
    with pytest.raises(AssertionError):
        A20._mode(_args(["--run", "r", "--untrained", "--embed-only"]))


# ---------------------------------------------------------------------- layer-0 site
def test_embed_site_is_block0_input(ref_attn):
    pos = torch.tensor([1, 5, 9, 15])
    for fam in FAMILIES:
        X, Y = _batch(fam)
        m = _build(fam)
        with torch.no_grad():
            full = A20.capture_sites(m, fam, X, Y, pos, embed=True)
            only = A20.capture_sites(m, fam, X, Y, pos, embed_only=True)
            dflt = A20.capture_sites(m, fam, X, Y, pos)
            wte = m.transformer.wte(X).index_select(1, pos).to(torch.float16)
        emb = [k for k in full if k.endswith("_emb")]
        assert set(only) == set(emb)
        for k in dflt:                       # adding the site perturbs nothing else
            assert torch.equal(full[k], dflt[k]), (fam, k)
        key = "single_emb" if fam == "standard" else "state_emb"
        assert torch.equal(full[key], wte), fam
        assert torch.equal(only[key], wte), fam
        if fam == "sps":
            pw = m.transformer.wte(torch.full_like(X, m.config.predict_token_id))
            assert torch.equal(full["pred_emb"], pw.index_select(1, pos).to(torch.float16))


def test_parse_site():
    assert A20._parse_site("final") == ("final", -1)
    assert A20._parse_site("state_emb") == ("state", -1)
    assert A20._parse_site("pred_11") == ("pred", 11)
    assert A20._parse_site("single_emb") == ("single", -1)


# ---------------------------------------------------------------------- token floors
def test_token_floors_deterministic_cycle():
    """Stream 0,1,2,...,9,0,1,...: next/prev are functions of cur, so the bigram floors
    go to ~0 and the unigram floor is ln(10)."""
    block = 20
    data = np.tile(np.arange(10, dtype=np.uint16), 200)
    starts = list(range(0, 8 * block, block))
    V = 16
    lut = torch.arange(V)
    pos = np.arange(1, block, 2)
    X = np.stack([data[j:j + block + 1].astype(np.int64) for j in range(8 * block, 12 * block, block)])
    cur = torch.from_numpy(X[:, pos].reshape(-1))
    nxt = torch.from_numpy(X[:, pos + 1].reshape(-1))
    prv = torch.from_numpy(X[:, pos - 1].reshape(-1))
    fl = A20.token_floors(data, starts, block, pos, lut, V, 15, cur, nxt, prv)
    assert fl["bigram_next_given_cur"] < 0.02
    assert fl["bigram_prev_given_cur"] < 0.02
    assert abs(fl["unigram_P1"] - np.log(10)) < 0.05   # add-one over 16 labels, 160 pairs
    assert fl["lambda_next"] >= 0.99 and fl["lambda_prev"] >= 0.99


def test_token_floors_iid_stream_bigram_equals_unigram():
    """i.i.d. tokens: the current token carries no information about its neighbours, so
    the tuned bigram cannot beat the unigram by more than estimation noise."""
    rng = np.random.default_rng(0)
    block = 64
    data = rng.integers(0, 8, size=200 * block, dtype=np.uint16)
    starts = list(range(0, 100 * block, block))
    lut = torch.arange(8)
    pos = np.arange(1, block, 4)
    X = np.stack([data[j:j + block + 1].astype(np.int64) for j in range(100 * block, 190 * block, block)])
    cur = torch.from_numpy(X[:, pos].reshape(-1))
    nxt = torch.from_numpy(X[:, pos + 1].reshape(-1))
    prv = torch.from_numpy(X[:, pos - 1].reshape(-1))
    fl = A20.token_floors(data, starts, block, pos, lut, 8, 99, cur, nxt, prv)
    assert abs(fl["bigram_next_given_cur"] - fl["unigram_P1"]) < 0.02
    assert abs(fl["unigram_P1"] - np.log(8)) < 0.02
