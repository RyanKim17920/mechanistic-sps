"""Tokenize a FineWeb-Edu sample into <data_root>/data/<dataset>/{train,val}.bin (uint16 GPT-2 tokens).

    python src/data/prepare.py data=fineweb-edu-100bt system.data_root=<dir>

The whole recipe is cfg.data.prepare. conf/data/fineweb-edu-100bt.yaml holds the one that
produced the paper corpus; data/MANIFEST.json records its shards, token counts and sha256.

1. Shards: list the sample's parquet files on the Hub at the pinned revision (network
   access needed), keep and shuffle a subset (select_shards), and download only the kept
   shards that are not cached yet. HF_HOME is respected; if unset it is <data_root>/.hf_cache.
2. Count: each of num_proc workers streams shards[rank::num_proc] and counts val/train tokens.
3. Write: the same workers write both splits in one pass, each at its own offset, streaming
   documents through a seeded shuffle buffer. A document goes to val by a hash of its id.
   Because of this layout, train.bin depends on num_proc (it is the same for every num_proc
   at least as large as the number of shards).

Finally MANIFEST.json (recipe, shard order, token counts, sha256) is written next to the files.
"""

import hashlib
import json
import multiprocessing as mp
import os
import random
import threading
import time
from pathlib import Path

import hydra
import numpy as np
import tiktoken
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def select_shards(names, max_files, file_select, file_offset, shuffle_seed):
    """The shards a recipe processes, in processing order.

    From the sorted list keep max_files (all if 0): consecutive ones ('head') or evenly spaced
    ones ('spread'), starting at file_offset. Then shuffle with random.Random(shuffle_seed), so
    that neighbouring workers write unrelated shards next to each other in train.bin.
    """
    names = sorted(names)
    n = len(names)
    if file_select not in ("head", "spread"):
        raise ValueError(f"file_select must be 'head' or 'spread', got {file_select!r}")
    if 0 < max_files < n:
        step = 1 if file_select == "head" else max(1, n // max_files)
        names = [names[(file_offset + i * step) % n] for i in range(max_files)]
    random.Random(shuffle_seed).shuffle(names)
    return names


def _is_val(doc_id: str, val_seed: int, val_fraction: float) -> bool:
    h = hashlib.md5(f"{val_seed}:{doc_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") < int(val_fraction * (2 ** 64))


def resolve_shards(p) -> list[str]:
    """Local paths of the recipe's shards in processing order, downloading only missing ones.

    The shard names always come from the Hub listing at the pinned revision: a local cache
    that holds only some shards would otherwise silently change which ones are selected."""
    from huggingface_hub import HfApi, snapshot_download

    kw = dict(repo_id=p.hf_repo, repo_type="dataset", revision=p.hf_revision)
    prefix = f"sample/{p.sample}/"
    names = [f for f in HfApi().list_repo_files(**kw) if f.startswith(prefix) and f.endswith(".parquet")]
    print(f"      {len(names)} shards on the Hub", flush=True)
    selected = select_shards(names, p.max_files, p.file_select, p.file_offset, p.shuffle_seed)
    if p.max_files and len(selected) != p.max_files:
        raise RuntimeError(f"recipe wants {p.max_files} shards, found {len(selected)}")
    print(f"      Fetching {len(selected)} shards of {p.hf_repo}@{p.hf_revision} (cached ones are reused)...",
          flush=True)
    root = Path(snapshot_download(**kw, allow_patterns=selected))
    return [str(root / n) for n in selected]


def _worker(rank, p, files, progress, results, out):
    """Tokenize files[rank::num_proc] and put (rank, token counts per split) on `results`.
    Counting only when out is None; otherwise also write split s at out[s] = (path, offset)."""
    import datasets
    from datasets import load_dataset

    rank_files = files[rank::p.num_proc]
    counts = {"val": 0, "train": 0}
    if not rank_files:
        results.put((rank, counts))
        return
    datasets.disable_progress_bars()
    ds = load_dataset("parquet", data_files=rank_files, split="train", streaming=True)
    if out is not None:
        ds = ds.shuffle(buffer_size=p.shuffle_buffer_size, seed=p.shuffle_seed)
        mmaps = {s: np.memmap(path, dtype=np.uint16, mode="r+") for s, (path, _) in out.items()}
        pos = {s: offset for s, (_, offset) in out.items()}
    enc = tiktoken.get_encoding(p.tokenizer)
    batches = {"val": [], "train": []}

    def flush(split):
        for tokens in enc.encode_ordinary_batch(batches[split]):
            n = len(tokens) + 1
            if out is not None:
                mm, at = mmaps[split], pos[split]
                mm[at:at + n - 1] = tokens
                mm[at + n - 1] = enc.eot_token
                pos[split] = at + n
            counts[split] += n
        batches[split].clear()

    pending = 0
    for ex in ds:
        split = "val" if _is_val(ex["id"], p.val_split_seed, p.val_fraction) else "train"
        batches[split].append(ex["text"])
        if len(batches[split]) >= p.batch_size:
            flush(split)
        pending += 1
        if pending >= p.batch_size:
            with progress.get_lock():
                progress.value += pending
            pending = 0
    flush("val")
    flush("train")
    with progress.get_lock():
        progress.value += pending
    if out is not None:
        for mm in mmaps.values():
            mm.flush()
    results.put((rank, counts))


def _run_workers(p, files, total_docs, desc, out_for_rank):
    """Run _worker on every rank with one progress bar; return the per-rank counts."""
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    progress = ctx.Value("l", 0)
    done = threading.Event()

    def monitor():
        bar = tqdm(total=total_docs, desc=desc, unit="docs", unit_scale=True, dynamic_ncols=True)
        while not done.is_set():
            bar.n = progress.value
            bar.refresh()
            time.sleep(0.25)
        bar.n = progress.value
        bar.close()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    procs = [ctx.Process(target=_worker, args=(r, p, files, progress, results, out_for_rank(r)))
             for r in range(p.num_proc)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    done.set()
    thread.join()
    failed = [proc for proc in procs if proc.exitcode != 0]
    if failed:
        raise RuntimeError(f"{desc}: {len(failed)}/{len(procs)} worker(s) failed "
                           f"({', '.join(f'pid={x.pid} exit={x.exitcode}' for x in failed[:8])})")
    counts = {}
    while len(counts) < p.num_proc:
        rank, c = results.get(timeout=5.0)
        counts[rank] = c
    return [counts[r] for r in range(p.num_proc)]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    if not cfg.system.data_root:
        raise ValueError("cfg.system.data_root is required")
    p = cfg.data.prepare
    if p.num_proc < 1:
        raise ValueError("cfg.data.prepare.num_proc must be >= 1")
    data_root = Path(cfg.system.data_root)
    os.environ.setdefault("HF_HOME", str(data_root / ".hf_cache"))  # before any HF import
    dataset_dir = data_root / "data" / cfg.data.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"FineWeb-Edu preparation: {p.hf_repo}@{p.hf_revision} sample-{p.sample} -> {dataset_dir}"
          f" (HF_HOME={os.environ['HF_HOME']})", flush=True)

    print("\n[1/3] SHARDS", flush=True)
    files = resolve_shards(p)
    import pyarrow.parquet as pq
    total_docs = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"      {len(files)} shards, {total_docs:,} documents", flush=True)
    # Workers read local parquet files only.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    print(f"\n[2/3] COUNT TOKENS (VAL+TRAIN)\n      Workers: {p.num_proc}", flush=True)
    counts = _run_workers(p, files, total_docs, "count", lambda r: None)
    totals = {s: sum(c[s] for c in counts) for s in ("val", "train")}
    for s in ("val", "train"):
        print(f"      {s:5s} total: {totals[s]:,} tokens ({totals[s] * 2 / 1e9:.2f} GB)", flush=True)

    print(f"\n[3/3] WRITE TOKENS (VAL+TRAIN)\n      Workers: {p.num_proc}", flush=True)
    paths = {s: dataset_dir / f"{s}.bin" for s in ("val", "train")}
    offsets = {s: np.cumsum([0] + [c[s] for c in counts]).tolist() for s in ("val", "train")}
    for s, path in paths.items():
        np.memmap(path, dtype=np.uint16, mode="w+", shape=(totals[s],)).flush()
    written = _run_workers(p, files, total_docs, "write",
                           lambda r: {s: (str(paths[s]), offsets[s][r]) for s in paths})
    if written != counts:
        raise RuntimeError("the write pass produced different per-worker token counts than the count pass")

    manifest = {
        "dataset": cfg.data.dataset,
        "recipe": OmegaConf.to_container(p, resolve=True),
        "shards": [str(Path(f).relative_to(Path(f).parents[2])) for f in files],
        "files": {f"{s}.bin": {"tokens": totals[s], "bytes": paths[s].stat().st_size,
                               "sha256": file_sha256(paths[s])} for s in ("val", "train")},
    }
    (dataset_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nCOMPLETE\n{json.dumps(manifest['files'], indent=1)}", flush=True)


if __name__ == "__main__":
    main()
