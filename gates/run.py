#!/usr/bin/env python3
"""Golden gates: the refactor must not move a single number.

    python gates/run.py all            # every CPU gate, in parallel (= make check)
    python gates/run.py g2 g3          # a subset
    python gates/run.py g4 --update    # regenerate a golden (only with a written reason)

Goldens live in gates/golden/.  They were generated from the research code the paper's
runs used (see gates/README.md, "Goldens", for what each gate covers and where each golden
came from).
"""
from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "gates" / "golden"
PY = sys.executable
# Put this checkout's src/ first, so its own code is what gets imported and tested even
# when the venv has an editable install of another checkout.
PYTHONPATH = os.pathsep.join([str(REPO / "src"), str(REPO)])
for p in (REPO / "scripts" / "analysis", REPO / "scripts", REPO, REPO / "src"):
    sys.path.insert(0, str(p))

# The 39 configs of the paper (30) and the freeze-and-retrain experiments (9): every
# experiment config except the ones that are only analysis inputs.
NOT_PAPER_ARMS = {"s_two_tower_afsps_20b"}   # a17's afsps ladder (README, "Training")
PAPER_CONFIGS = sorted(p.stem for p in (REPO / "conf" / "experiment").glob("*.yaml")
                       if p.stem not in NOT_PAPER_ARMS)
assert len(PAPER_CONFIGS) == 39, len(PAPER_CONFIGS)

# G4: one config per arm family (plus the distinct two-tower code paths). SPS is absent:
# its attention is a Triton kernel with no CPU path, so SPS is covered on GPU by G8.
G4_ARMS = {
    "transformer": "s_full_attention_20b_fw100",
    "two_tower": "s_two_tower_w0_equal_20b",
    "tied_attn": "s_two_tower_w0_equal_tiedattn_20b",
    "sequential": "s_two_tower_seq12_20b",
    "sequential_pause": "s_two_tower_seq6_cpause_20b",
    "shared": "s_two_tower_w0_shared_20b",
    "shared_sequential": "s_two_tower_seq12_tied_20b",
    "afsps": "s_two_tower_afsps_faithful_20b",
    "asym_12p6": "s_two_tower_asym_20b",
    "frz": "s_two_tower_frz_seq6src_early_20b",
}
# 3 steps of 2 micro-batches: step 0-1 in LR warmup, step 2 inside the linear decay.
G4_BLOCK, G4_MICRO, G4_GLOBAL, G4_STEPS = 128, 2, 4, 3
G4_THREADS = "4"   # CPU matmul reduction order depends on the thread count: pin it
G4_PARALLEL = 4    # arms run concurrently

# Freeze-and-retrain configs (G3, G4 arm `frz`) load a frozen state tower from a source
# run's final checkpoint. `synthetic` (the default) writes that checkpoint on the fly: the
# source config's own model, initialised with a seed derived from SYNTH_SEED and the source
# run's name (not the target's seed, so a missing load would show, and different per
# source), state-tower tensors only. `real` reads the trained checkpoints
# under <DUALSPS_OUT_ROOT>/out/ (needs them on disk). Goldens: `<name>@synthetic` / `<name>`.
FRZ_SOURCES = ("synthetic", "real")
SYNTH_SEED = 0

# G5: the shipped corpus (its train.bin token count, data/MANIFEST.json) and the paper's
# world size.
TRAIN_TOKENS = json.loads((REPO / "data" / "MANIFEST.json").read_text())["files"]["train.bin"]["tokens"]
WORLD = 8
FIRST_N = 64
RESUME_AT = (1_000, 300_000)   # samples seen per rank; the second crosses a chunk boundary



# ---------------------------------------------------------------------------------------
# host fingerprint: G4 is exact only on the golden host's CPU type, and G1 depends on its
# matplotlib and font installs.  A host-dependent FAIL prints what differs.
# ---------------------------------------------------------------------------------------
HOST_GATES = ("g1", "g4")


def _first_line(cmd) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return ((p.stdout or p.stderr).strip().splitlines() or [None])[0]


def host_fingerprint() -> dict:
    import plotting.mpl_cache  # noqa: F401  (the MPLCONFIGDIR the figures use)
    import matplotlib
    import torch
    from matplotlib import font_manager
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), None)
    serif = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]   # paper_style.py
    font = Path(font_manager.findfont(font_manager.FontProperties(family=serif),
                                      fallback_to_default=False)).name
    return {"cpu": cpu, "g4_threads": G4_THREADS, "torch": torch.__version__,
            "matplotlib": matplotlib.__version__, "serif_font": font,
            "pdftex": _first_line(["pdftex", "--version"]),
            "pdftotext": _first_line(["pdftotext", "-v"])}


def host_note() -> list[str]:
    if not (GOLDEN / "host.json").exists():
        return []
    gold = load_json(GOLDEN / "host.json")
    now = host_fingerprint()
    return [f"{k}: golden {gold.get(k)!r}, here {now.get(k)!r}" for k in sorted(gold)
            if gold.get(k) != now.get(k)]


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_imports_from_repo():
    import modeling
    import plotting
    for m in (modeling, plotting):
        f = Path(m.__file__).resolve()
        assert REPO in f.parents, f"{m.__name__} imported from {f}, not this checkout {REPO}"


def compose(name: str, overrides=()):
    from hydra import compose as hcompose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        return hcompose("config", overrides=[f"+experiment={name}",
                                             "system.data_root=/DATA_ROOT", *overrides])


def freeze_source(name: str):
    """The source run of a freeze-and-retrain config (training.freeze_state_from); None for
    other configs."""
    return compose(name).training.get("freeze_state_from")


def frz_source_ckpt(src_run: str, mode: str, root: Path) -> Path:
    """Put `src_run`'s final checkpoint at <root>/out/<src_run>/ (where train.py looks with
    system.out_root=<root>) and return its path: written from a seeded init (`synthetic`) or
    linked to the trained one under <DUALSPS_OUT_ROOT>/out/ (`real`)."""
    d = root / "out" / src_run
    have = sorted(d.glob("ckpt_tokens_*_final.pt"))
    if have:
        return have[0]
    d.mkdir(parents=True, exist_ok=True)
    if mode == "real":
        import repo_paths
        from training import final_checkpoint_path
        real = final_checkpoint_path(repo_paths.run_dir(src_run))
        (d / real.name).symlink_to(real)
        return d / real.name
    import torch
    from hydra.utils import instantiate
    torch.manual_seed(SYNTH_SEED + int(hashlib.sha256(src_run.encode()).hexdigest()[:8], 16))
    model = instantiate(compose(src_run).model)
    sd = {k: v for k, v in model.state_dict().items() if k.startswith(model.STATE_TOWER_PREFIXES)}
    path = d / "ckpt_tokens_0_final.pt"
    torch.save({"model": sd}, path)
    return path


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
        if not d and prefix:
            out[prefix[:-1]] = {}
    else:
        out[prefix[:-1]] = d
    return out


def allowed_config_diffs():
    """Parse the ```allowed block of gates/ALLOWED_CONFIG_DIFFS.md.
    Lines:  drop [<config-glob>] <key>     the key may disappear (every config by default)
            set <config-glob> <key> <yaml> the key must have this value (a pin or a new key)"""
    import yaml
    text = (REPO / "gates" / "ALLOWED_CONFIG_DIFFS.md").read_text()
    m = re.search(r"```allowed\n(.*?)```", text, re.S)
    drops, sets = [], []
    for line in (m.group(1).splitlines() if m else []):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        verb, rest = line.split(None, 1)
        if verb == "drop":
            parts = rest.split()
            drops.append(tuple(parts) if len(parts) == 2 else ("*", parts[0]))
        elif verb == "set":
            glob, key, val = rest.split(None, 2)
            sets.append((glob, key, yaml.safe_load(val)))
        else:
            raise ValueError(f"ALLOWED_CONFIG_DIFFS.md: unknown verb {verb!r}")
    return drops, sets


def compare_flat(name, gold: dict, new: dict, drops, sets) -> list[str]:
    errs = []
    pinned = {k: v for g, k, v in sets if fnmatch.fnmatch(name, g)}
    dropped = lambda k: any(fnmatch.fnmatch(name, g) and (k == d or k.startswith(d + "."))  # noqa: E731
                            for g, d in drops)
    for k in sorted(set(gold) | set(new) | set(pinned)):
        if k in pinned:
            if new.get(k, "<missing>") != pinned[k]:
                errs.append(f"{k}: pinned {pinned[k]!r}, got {new.get(k, '<missing>')!r}")
        elif k not in new:
            if not dropped(k):
                errs.append(f"{k}: removed (golden {gold[k]!r})")
        elif k not in gold:
            errs.append(f"{k}: new key = {new[k]!r}")
        elif gold[k] != new[k] or type(gold[k]) is not type(new[k]):
            errs.append(f"{k}: {gold[k]!r} -> {new[k]!r}")
    return errs


def diff_text(a: str, b: str, name: str, n=40) -> str:
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), f"golden/{name}", f"new/{name}",
                                  lineterm=""))
    return "\n".join(d[:n]) + (f"\n... ({len(d) - n} more diff lines)" if len(d) > n else "")


def load_json(p: Path):
    return json.loads(p.read_text())


def dump_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def compare_json(gold, new, label) -> list[str]:
    if gold == new:
        return []
    expand = lambda d: {f"{k}[{i}]" if isinstance(v, list) else k: x  # noqa: E731
                        for k, v in flatten(d).items()
                        for i, x in (enumerate(v) if isinstance(v, list) else [(0, v)])}
    g, n = expand(gold), expand(new)
    errs = [f"{label}: {k}: {g.get(k, '<missing>')!r} -> {n.get(k, '<missing>')!r}"
            for k in sorted(set(g) | set(n)) if g.get(k, "<missing>") != n.get(k, "<missing>")]
    return errs[:60] + ([f"... {len(errs) - 60} more"] if len(errs) > 60 else [])


def state_sha(tensors: dict) -> str:
    """sha256 over (name, dtype, shape, raw bytes) of every tensor, in sorted key order."""
    import torch
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k]
        if not torch.is_tensor(t):
            h.update(f"{k}={t!r};".encode())
            continue
        t = t.detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)};".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16
                 else t.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# G1  paper pipeline: every figure and table, byte-identical, plus the printed numbers
# ---------------------------------------------------------------------------------------
NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?")


def g1_build(out: Path) -> str:
    """Regenerate the paper's figures and tables into `out`; return the pipeline's stdout."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from plotting import paper_style as PS
        PS.OUT_DIR = out / "figs"
        import paper_figures as PF
        for make in PF.FIGS.values():      # every figure of the paper, appendix included
            make()
        import make_tables as MT
        argv, sys.argv = sys.argv, ["make_tables.py", "--outdir", str(out / "tables")]
        try:
            MT.main()
        finally:
            sys.argv = argv
    return buf.getvalue()


def g1_numbers(out: Path, stdout: str) -> dict:
    """Every number in every table (in order) and every number the pipeline prints (as a
    sorted multiset, so rewording or reordering a print is free but changing a value is not)."""
    tables = {p.name: NUM.findall(p.read_text()) for p in sorted((out / "tables").glob("*.tex"))}
    printed = sorted(n for line in stdout.splitlines() if not line.startswith("WROTE ")
                     for n in NUM.findall(line))
    return {"tables": tables, "printed": printed}


def g1(update: bool) -> list[str]:
    check_imports_from_repo()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        stdout = g1_build(out)
        files = {str(p.relative_to(out)): sha256_file(p)
                 for p in sorted(out.rglob("*")) if p.is_file()}
        # the snapshotted inputs too: a change below the tables' printed precision is still a change
        import repo_paths
        files.update({f"inputs/{p.name}": sha256_file(p)
                      for p in (repo_paths.LEDGER, repo_paths.WALLCLOCK)})
        numbers = g1_numbers(out, stdout)
        if update:
            dump_json(files, GOLDEN / "g1" / "sha256.json")
            dump_json(numbers, GOLDEN / "g1" / "numbers.json")
            (GOLDEN / "g1" / "printed.txt").write_text(stdout)   # for reading, not compared
            return []
    from collections import Counter
    gold_numbers = load_json(GOLDEN / "g1" / "numbers.json")
    errs = compare_json(gold_numbers["tables"], numbers["tables"], "table numbers")
    gone = Counter(gold_numbers["printed"]) - Counter(numbers["printed"])
    new = Counter(numbers["printed"]) - Counter(gold_numbers["printed"])
    if gone or new:
        errs.append(f"printed numbers: missing {sorted(gone.elements())[:40]}, "
                    f"new {sorted(new.elements())[:40]}")
    gold = load_json(GOLDEN / "g1" / "sha256.json")
    for k in sorted(set(gold) | set(files)):
        if gold.get(k) != files.get(k):
            errs.append(f"{k}: sha256 {gold.get(k, '<missing>')[:12]} -> "
                        f"{files.get(k, '<missing>')[:12]}")
    # the committed copies in paper/ must be what the pipeline writes (`make paper`)
    committed = {"figs": REPO / "paper" / "figures", "tables": REPO / "paper" / "tables"}
    for k, h in files.items():
        kind, _, name = k.partition("/")
        c = committed.get(kind, Path("/nonexistent")) / name
        if kind in committed and not (c.is_file() and sha256_file(c) == h):
            errs.append(f"{k}: committed paper/ copy differs from the regenerated file")
    return errs


# ---------------------------------------------------------------------------------------
# G2  resolved Hydra configs of all 39 paper configs
# ---------------------------------------------------------------------------------------
def g2(update: bool) -> list[str]:
    import yaml
    from omegaconf import OmegaConf
    drops, sets = allowed_config_diffs()
    errs = []
    for name in PAPER_CONFIGS:
        text = OmegaConf.to_yaml(compose(name), resolve=True, sort_keys=True)
        path = GOLDEN / "g2" / f"{name}.yaml"
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            continue
        gold = path.read_text()
        if gold == text:
            continue
        bad = compare_flat(name, flatten(yaml.safe_load(gold)), flatten(yaml.safe_load(text)),
                           drops, sets)
        errs += [f"{name}: {e}" for e in bad]
    return errs


# ---------------------------------------------------------------------------------------
# G3  init state_dict of every paper config (full size, CPU, the config's seed)
# ---------------------------------------------------------------------------------------
def g3_one(name: str, frz: str, root: Path) -> dict:
    """Mirror scripts/train.py's scratch init on rank 0: seed, instantiate, freeze."""
    import random
    from dataclasses import asdict
    import torch
    from hydra.utils import instantiate
    cfg = compose(name)
    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    model = instantiate(cfg.model)
    src = freeze_source(name)
    if src:
        path = frz_source_ckpt(src, frz, root)
        sd = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)["model"]
        model.freeze_state_tower(sd)
    sd = model.state_dict()
    return {"sha256": state_sha(sd),
            "n_tensors": len(sd),
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "type": type(model).__name__,
            "model_args": json.loads(json.dumps(asdict(model.config), default=str))}


def golden_key(name: str, frz: str) -> str:
    return f"{name}@synthetic" if frz == "synthetic" and freeze_source(name) else name


def g3(update: bool, frz: str = "synthetic") -> list[str]:
    check_imports_from_repo()
    import torch
    torch.set_num_threads(int(G4_THREADS))
    with tempfile.TemporaryDirectory() as td:
        res = {golden_key(name, frz): g3_one(name, frz, Path(td)) for name in PAPER_CONFIGS}
    gold = load_json(GOLDEN / "g3.json") if (GOLDEN / "g3.json").exists() else {}
    if update:
        dump_json({**gold, **res}, GOLDEN / "g3.json")
        return []
    drops, sets = allowed_config_diffs()
    sets = [s for s in sets if s[1].startswith("model.config.")]   # model_args holds only these
    errs = []
    for key in res:
        name = key.split("@")[0]
        g, n = gold[key], res[key]
        for k in ("sha256", "n_tensors", "n_params", "n_trainable", "type"):
            if g[k] != n[k]:
                errs.append(f"{name}: {k} {g[k]} -> {n[k]}")
        pre = lambda d: {f"model.config.{k}": v for k, v in d.items()}  # noqa: E731
        errs += [f"{name}: model_args {e}" for e in
                 compare_flat(name, pre(g["model_args"]), pre(n["model_args"]), drops, sets)]
    return errs


# ---------------------------------------------------------------------------------------
# G4  two optimizer steps of scripts/train.py per arm family (CPU, fp32, fixed batch)
# ---------------------------------------------------------------------------------------
def g4_data(root: Path):
    """Synthetic corpus, fixed forever: uint16 tokens < 50257 from a seeded generator."""
    import numpy as np
    d = root / "data" / compose(PAPER_CONFIGS[0]).data.dataset
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 64 * 1024), ("val", 16 * 1024)):
        rng.integers(0, 50257, n, dtype=np.uint16).tofile(d / f"{split}.bin")


LOG_KEEP = re.compile(r"^(train \| iter\s+\d+ \| total: \S+(?: \| nll: \S+)? \| ppl\s+\S+ \| lr: \S+"
                      r"|eval \| iter\s+\d+ \| val nll: \S+ \(ppl \S+\)"
                      r"|final \| iter\s+\d+ \| val nll: \S+ \(ppl \S+\)"
                      r"|freeze_state_tower: .*?trainable params [\d,]+"
                      r"|G4 grad_norm \S+)")


def g4_one(arm: str, name: str, root: Path) -> dict:
    import torch
    tokens = G4_BLOCK * G4_GLOBAL * G4_STEPS
    cmd = [PY, str(REPO / "gates" / "cpu_train.py"), f"+experiment={name}",
           f"system.data_root={root}", "system.device=cpu", "system.dtype=float32",
           "system.compile=false", "system.backend=gloo", "logging.wandb_log=false",
           "training.init_from=scratch", "training.world_size=null", f"model.config.block_size={G4_BLOCK}",
           f"training.micro_batch_size={G4_MICRO}", f"training.global_batch_size={G4_GLOBAL}",
           f"training.max_tokens={tokens}", f"scheduler.warmup_tokens={tokens // 2}",
           f"scheduler.lr_decay_tokens={tokens // 2}", f"training.eval_total_tokens={G4_BLOCK * G4_GLOBAL}",
           f"training.eval_interval_tokens={tokens}", f"training.save_every={10 * tokens}",
           f"training.rolling_save_every={10 * tokens}"]
    if "two_tower" in name:   # compiled flex_attention has no CPU lowering; run it eagerly
        cmd.append("++model.config.flex_compile=false")
    env = dict(os.environ, PYTHONPATH=PYTHONPATH, OMP_NUM_THREADS=G4_THREADS,
               MKL_NUM_THREADS=G4_THREADS, CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled")
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(k, None)
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{arm}: train.py failed\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    log = [m.group(1) for line in p.stdout.splitlines() if (m := LOG_KEEP.match(line))]
    ckpts = sorted((root / "out").glob(f"{name}/*final*.pt"))
    assert len(ckpts) == 1, f"{arm}: expected one final checkpoint, got {ckpts}"
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    opt = {f"{i}.{k}": v for i, s in ck["optimizer"]["state"].items() for k, v in s.items()}
    shutil.rmtree(ckpts[0].parent)
    return {"config": name, "log": log, "model_sha256": state_sha(ck["model"]),
            "optimizer_sha256": state_sha(opt), "cpu_rng_sha256": state_sha({"rng": ck["cpu_rng_state"]}),
            "best_val_loss": float(ck["best_val_loss"]).hex(),
            **{k: ck.get(k) for k in ("iter_num", "last_save_tokens", "sampler_offset",
                                      "sampler_samples_seen_per_rank", "next_eval_tokens")}}


def g4(update: bool, only=None, frz: str = "synthetic") -> list[str]:
    arms = {a: n for a, n in G4_ARMS.items() if not only or a in only}
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        g4_data(root)
        # frozen sources first, in this thread: Hydra's global state is not thread-safe
        for name in arms.values():
            if src := freeze_source(name):
                frz_source_ckpt(src, frz, root)
        with ThreadPoolExecutor(G4_PARALLEL) as ex:
            futs = {arm + golden_key(name, frz)[len(name):]: ex.submit(g4_one, arm, name, root)
                    for arm, name in arms.items()}
            res = {key: f.result() for key, f in futs.items()}
    gold = load_json(GOLDEN / "g4.json") if (GOLDEN / "g4.json").exists() else {}
    if update:
        dump_json({**gold, **res}, GOLDEN / "g4.json")
        return []
    errs = []
    for key in res:
        errs += compare_json(gold[key], res[key], key)
    return errs


# ---------------------------------------------------------------------------------------
# G5  sampler stream: first indices per rank at world size 8, and the resume path
# ---------------------------------------------------------------------------------------
def g5_stream(cfg) -> dict:
    """Mirror scripts/train.py's DDP sampler setup for each rank of an 8-GPU run."""
    from training import FixedRandomChunkDistributedSampler, fresh_start_offset
    t = cfg.training
    block = int(cfg.model.config.block_size)
    n = TRAIN_TOKENS - block                      # TokenDataset.__len__
    offsets = [fresh_start_offset(int(t.seed), rank, n, int(t.sampler_max_start_offset))
               for rank in range(WORLD)]
    ranks = {}
    for rank in range(WORLD):
        mk = lambda off, seen: FixedRandomChunkDistributedSampler(  # noqa: E731
            dataset_len=n, num_replicas=WORLD, rank=rank, block_size=block,
            chunk_size_units=int(t.chunk_size_units), seed=int(t.chunk_shuffle_seed),
            start_offset=off, resume_samples_seen_per_rank=seen, balanced=bool(t.sampler_fix))
        s = mk(offsets[rank], 0)
        it = iter(s)
        r = {"start_offset": offsets[rank], "len": len(s), "first": [next(it) for _ in range(FIRST_N)]}
        # train.py resume: every rank restores RANK 0's saved sampler_offset (legacy behaviour).
        for seen in RESUME_AT:
            it = iter(mk(offsets[0], seen))
            r[f"resume_{seen}"] = [next(it) for _ in range(16)]
        ranks[str(rank)] = r
    return ranks


def g5(update: bool) -> list[str]:
    check_imports_from_repo()
    res = {}
    for name in PAPER_CONFIGS:
        t = compose(name).training
        key = (f"seed={t.seed},chunk_shuffle_seed={t.chunk_shuffle_seed},"
               f"chunk_size_units={t.chunk_size_units},sampler_fix={t.sampler_fix}")
        res.setdefault("configs", {})[name] = key
        if key not in res:
            res[key] = g5_stream(compose(name))
    if update:
        dump_json(res, GOLDEN / "g5.json")
        return []
    return compare_json(load_json(GOLDEN / "g5.json"), res, "g5")


# ---------------------------------------------------------------------------------------
# G7  pytest: nothing that passed in the golden run may fail; no new failures
# ---------------------------------------------------------------------------------------
def g7(update: bool) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        env = dict(os.environ, PYTHONPATH=PYTHONPATH, CUDA_VISIBLE_DEVICES="")
        subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"],
                       cwd=REPO, env=env, capture_output=True, text=True)
        import xml.etree.ElementTree as ET
        res = {}
        for tc in ET.parse(xml).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            kids = {c.tag for c in tc}
            res[tid] = ("failed" if kids & {"failure", "error"} else
                        "skipped" if "skipped" in kids else "passed")
    if update:
        dump_json(res, GOLDEN / "g7.json")
        return []
    gold = load_json(GOLDEN / "g7.json")
    failed = [t for t, s in sorted(res.items()) if s == "failed"]
    known = [t for t in failed if gold.get(t) == "failed"]
    if known:   # failed in the golden run too
        print(f"  g7: {len(known)} tests fail as in the golden run: "
              f"{sorted({t.split('::')[0] for t in known})}")
    errs = [f"{t}: {gold.get(t, 'new')} -> failed" for t in failed if t not in known]
    gone = [t for t, s in gold.items() if s == "passed" and t not in res]
    if gone:
        print(f"  g7: {len(gone)} baseline-passing tests no longer collected (deleted?)")
    counts = {s: sum(v == s for v in res.values()) for s in ("passed", "failed", "skipped")}
    print(f"  g7: {counts}")
    return errs


GATES = {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "g7": g7}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gates", nargs="+", help="g1..g7 or all")
    ap.add_argument("--update", action="store_true", help="rewrite the golden outputs")
    ap.add_argument("--arms", default="", help="g4 only: comma-separated subset of " + ",".join(G4_ARMS))
    ap.add_argument("--frz-source", choices=FRZ_SOURCES, default="synthetic",
                    help="g3/g4: source checkpoint of the freeze-and-retrain configs: a seeded "
                         "synthetic one (default, self-contained) or the trained ones under "
                         "<DUALSPS_OUT_ROOT>/out/")
    a = ap.parse_args()
    names = list(GATES) if a.gates == ["all"] else a.gates
    if len(names) > 1:   # one subprocess per gate, all at once
        passthru = (["--update"] if a.update else []) + [f"--frz-source={a.frz_source}"] + (
            [f"--arms={a.arms}"] if a.arms else [])
        procs = {n: subprocess.Popen([PY, __file__, n, *passthru], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True) for n in names}
        ok = True
        for p in procs.values():
            out = p.communicate()[0]
            ok &= p.returncode == 0
            print(out, end="", flush=True)
        sys.exit(0 if ok else 1)
    n = names[0]
    t0 = time.time()
    kw = {"only": a.arms.split(",")} if n == "g4" and a.arms else {}
    if n in ("g3", "g4"):
        kw["frz"] = a.frz_source
    errs = GATES[n](a.update, **kw)
    if a.update:
        dump_json(host_fingerprint(), GOLDEN / "host.json")
    status = "UPDATED" if a.update else ("PASS" if not errs else "FAIL")
    print(f"{n}: {status} ({time.time() - t0:.0f}s)")
    for e in errs:
        print("   ", e)
    if errs and n in HOST_GATES and (diff := host_note()):
        print(f"    NOTE: this host differs from the one that made the goldens, so a {n} "
              f"FAIL here may be the host, not the code (gates/README.md, 'Host'):")
        for d in diff:
            print("     ", d)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
