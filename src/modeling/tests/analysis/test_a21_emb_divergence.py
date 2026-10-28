"""Unit tests for scripts/analysis/a21_emb_divergence.py's metrics, on tiny random tables.

Pinned: CKA / Procrustes / kNN are invariant to an orthogonal change of basis (the whole
point of replacing a7's raw same-row cosine), they sit near their null for independent
tables, the CV Procrustes removes the in-sample fit's upward bias, and the spectrum /
drift / bin helpers do what their names say.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

_ANALYSIS = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

A21 = pytest.importorskip("a21_emb_divergence")

N, D = 600, 32


def _tbl(seed, n=N, d=D):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * 0.02


def _rot(seed, d=D):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q.float()


def _structured(seed, n=N, d=D, rank=6):
    """Anisotropic table with low-rank structure (like a trained embedding)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, rank, generator=g)
    return z @ torch.randn(rank, d, generator=g) + 0.05 * torch.randn(n, d, generator=g)


def test_cka_rotation_and_scale_invariant():
    x = _structured(0)
    assert A21.linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)
    assert A21.linear_cka(x, 3.7 * x @ _rot(1)) == pytest.approx(1.0, abs=1e-6)


def test_cka_independent_is_small():
    assert A21.linear_cka(_tbl(0), _tbl(1)) < 0.1


def test_raw_cosine_blind_but_procrustes_recovers_rotation():
    x = _structured(0)
    y = x @ _rot(2)
    # a7's statistic: same-row cosine is ~0 for a rotated copy -- uninformative
    assert abs(float(A21.row_cosine(x, y).mean())) < 0.2
    ins, cv = A21.procrustes_cos(x, y)
    assert float(ins.min()) > 0.999999
    assert float(cv.min()) > 0.999


def test_procrustes_cv_removes_insample_bias():
    ins, cv = A21.procrustes_cos(_tbl(0), _tbl(1))
    assert float(ins.mean()) > 0.1               # 32^2-parameter fit overfits 600 rows
    assert abs(float(cv.mean())) < 0.05          # held-out null is ~0


def test_knn_overlap():
    x = _structured(0)
    assert float(A21.knn_overlap(x, x @ _rot(3), k=5).mean()) == pytest.approx(1.0)
    assert float(A21.knn_overlap(x, 2.0 * x, k=5).mean()) == pytest.approx(1.0)
    ov = float(A21.knn_overlap(_tbl(0), _tbl(1), k=5, chunk=97).mean())
    assert ov < 0.05                             # chance = k/n ~ 0.008


def test_knn_chunking_matches_unchunked():
    x = _structured(4)
    assert torch.equal(A21.knn_indices(x, 5, chunk=64), A21.knn_indices(x, 5, chunk=10_000))


def test_spectrum_stats():
    g = torch.Generator().manual_seed(0)
    low = torch.randn(N, 3, generator=g) @ torch.randn(3, D, generator=g)
    s = A21.spectrum_stats(low)
    assert s["eff_rank"] < 3.5
    iso = A21.spectrum_stats(_tbl(0))
    assert iso["eff_rank"] > 0.8 * D
    one = torch.randn(N, 1, generator=g) @ torch.randn(1, D, generator=g)
    assert A21.spectrum_stats(one)["top1_var_frac"] == pytest.approx(1.0, abs=1e-9)
    shifted = _tbl(0) + 1.0
    assert A21.spectrum_stats(shifted)["mean_row_energy_frac"] > 0.99


def test_drift():
    x = _tbl(0)
    assert A21.rel_drift(x, x) == 0.0
    assert A21.rel_drift(2 * x, x) == pytest.approx(1.0)


def test_freq_bins_partition_and_order():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 1000, size=5000)
    active = np.nonzero(counts > 0)[0]
    bins = A21.freq_bins(counts, active)
    parts = np.concatenate([v for k, v in bins.items() if k != "all"])
    assert sorted(parts.tolist()) == list(range(len(active)))
    top, bot = bins["rank0-1pct"], bins["rank50-100pct"]
    assert counts[active[top]].min() >= counts[active[bot]].max()


def test_similarity_bundle_and_perm_null():
    x = _structured(0, n=900)
    counts = np.arange(900, 0, -1)
    bins = A21.freq_bins(counts, np.arange(900))
    same = A21.similarity(x, x @ _rot(5), bins)
    for b in same.values():
        assert b["cka"] > 0.999 and b["procrustes_cos_cv"] > 0.99
        assert b["knn10_overlap"] > 0.999
    g = torch.Generator().manual_seed(0)
    xp = A21.permute_within_bins(x, bins, g)
    for k, idx in bins.items():                  # per-bin row multiset preserved
        if k == "all":
            continue
        a = x[torch.as_tensor(idx)].sum(0)
        b = xp[torch.as_tensor(idx)].sum(0)
        assert torch.allclose(a, b, atol=1e-4)
    perm = A21.similarity(xp, x, bins)
    assert perm["all"]["knn10_overlap"] < 0.1
    assert perm["all"]["procrustes_cos_cv"] < same["all"]["procrustes_cos_cv"] - 0.5


def test_verify_init_detects_weight_decay_scaling_and_wrong_seed():
    ids = list(A21.UNSEEN_IDS)
    V = max(ids) + 1
    init = {k: _tbl(i, n=V) for i, k in enumerate(("P", "S", "H"))}
    wrong = {k: _tbl(10 + i, n=V) for i, k in enumerate(("P", "S", "H"))}
    first = {k: v.clone() for k, v in init.items()}
    for k in first:                              # trained rows move, unseen rows only decay
        first[k][: ids[0]] += 0.01 * torch.randn(ids[0], D)
        first[k][ids] *= 0.97
    ok = A21.verify_init(init, {k: v.clone() for k, v in init.items()}, wrong, first)
    assert ok["passed"]
    bad = A21.verify_init(wrong, {k: v.clone() for k, v in wrong.items()}, init, first)
    assert not bad["passed"]


def test_emb_ladder_file_stands_in_for_the_checkpoints(tmp_path):
    """With only the final on disk, the ladder and tables come from emb_ladder.pt, with the
    same (tokens, tag, path) items and the same tensors as the checkpoints themselves."""
    names = {k: v for k, v in A21.KEYS.items()}
    full, ext = tmp_path / "full", tmp_path / "ext"
    full.mkdir(), ext.mkdir()
    ckpts, tables, meta = {}, {}, {}
    for i, fn in enumerate(("ckpt_tokens_1000.pt", "ckpt_tokens_1800_pre_decay.pt",
                            "ckpt_tokens_2000_final.pt")):
        t = {k: _tbl(10 * i + j) for j, k in enumerate(names)}
        ck = dict(model={"_orig_mod." + names[k]: v for k, v in t.items()}, iter_num=i,
                  config=dict(training=dict(seed=1337)))
        torch.save(ck, full / fn)
        stem = fn[:-3]
        tables[stem] = t
        meta[stem] = dict(file=fn, tokens=int(fn.split("_")[2].split(".")[0]), tag="x",
                          iter_num=i, seed=1337, emb_probe_init=False)
    torch.save(dict(checkpoints=meta, tables=tables), ext / A21.EMB_LADDER)
    torch.save(torch.load(full / "ckpt_tokens_2000_final.pt"), ext / "ckpt_tokens_2000_final.pt")
    lf, le = A21.ladder(str(full)), A21.ladder(str(ext))
    assert [(t, g, p.replace(str(full), "")) for t, g, p in lf] == \
           [(t, g, p.replace(str(ext), "")) for t, g, p in le]
    assert [g for _, g, _ in le] == ["", "_pre_decay", "_final"]
    for (_, _, pf), (_, _, pe) in zip(lf, le):
        tf, mf = A21.load_tables(pf)
        te, me = A21.load_tables(pe)
        assert all(torch.equal(tf[k], te[k]) for k in names)
        assert (mf["iter_num"], mf["seed"], mf["emb_probe_init"]) == \
               (me["iter_num"], me["seed"], me["emb_probe_init"])
    # a full ladder on disk wins over the extracted file
    torch.save(dict(checkpoints={}, tables={}), full / A21.EMB_LADDER)
    assert A21.ladder(str(full)) == lf
