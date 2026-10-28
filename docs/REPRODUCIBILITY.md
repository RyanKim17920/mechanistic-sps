# Reproducibility: corpus and data stream

What decides which tokens a paper run saw: the corpus, the per-rank data stream, and the one
known deviation, which happens when a run resumes.

## Corpus

Every 20B-token paper run trains on `fineweb-edu-100bt`. `src/data/prepare.py` builds it
from the recipe in `conf/data/fineweb-edu-100bt.yaml` (`data.prepare`):

- **Shards:** the first 36, in sorted order, of the 140 `sample/100BT` shards of
  `HuggingFaceFW/fineweb-edu` at revision `87f09149`, processed in the order
  `random.Random(2357)` shuffles them into.
- **Workers:** 86. Worker `r` streams `shards[r::86]` through a shuffle buffer of 200,000
  documents with seed 2357.
- **Split:** a document goes to `val.bin` when an md5 of `42:<id>` falls in the lowest
  0.0005 of the hash range.

`data/MANIFEST.json` records the shard order, each shard's LFS sha256, and the token counts
and sha256 of both output files:

| File | Tokens | sha256 |
|---|---|---|
| val.bin | 13,099,931 | `b2b137c1…afbf98` |
| train.bin | 27,089,110,623 | `dc647122…d113c20` |

To check a local copy, run `sha256sum $DUALSPS_DATA_ROOT/data/fineweb-edu-100bt/{val,train}.bin`
(or the `sha256sum -c` recipe in the top-level README).

**Worker count matters.** The layout of `train.bin` depends on `num_proc`. It is identical
for any `num_proc >= 36`, the number of shards, because each worker then holds at most one
shard. Keep 86.

**Revision.** The original build did not record a Hub revision. The recipe pins the `main`
revision of 2025-07: at that revision, the LFS sha256 of the 36 selected shards match the
shards the paper corpus was built from.

## Per-rank data stream

`FixedRandomChunkDistributedSampler` builds each rank's stream as follows:

1. It splits the corpus into chunks of 262,144 sequence starts, about 1.07B tokens each.
2. It shuffles the chunks once with `chunk_shuffle_seed`.
3. Rank `r` takes every 8th chunk of that order.

Each rank of a fresh run reads from its own start offset,
`random.Random(seed + r).randint(0, min(1e6, len // 2))` (`training.sampler.fresh_start_offset`,
with 1e6 = `training.sampler_max_start_offset`); gate G5 pins it for every paper seed.

The stream therefore depends on the seed, the chunk seed and the **world size**. All paper
runs used 8 GPUs, with micro-batch 6 and global batch 96. `training.world_size: 8` makes
`train.py` refuse any other world size.

Every paper run shares one known imbalance. The rank that holds the partial tail chunk runs
out first and replays the head of its own stream: 4.28% of rank 2's stream in the 8-GPU 20B
runs. `training.sampler_fix` removes it but changes the stream, so it is off for the paper
configs.

## Resuming changes the data stream

**What happens.** A checkpoint stores only rank 0's start offset (`sampler_offset`) and the
count of samples seen per rank. On resume, every rank restores **rank 0's** offset. After a
resume, rank `r` still reads its own chunks in the same order, but every read is displaced
by Δr = offset₀ − offsetᵣ tokens. Rank 0 is unaffected.

**Size of the effect.** For each chunk it visits after the resume, a displaced rank skips |Δr|
tokens that it would have read, and reads |Δr| tokens it would not have read. Sequence
boundaries move too, so no sequence of ranks 1–7 after the resume is bit-identical to the
uninterrupted run. The only other change is the length of the partial tail chunk, which
moves by a few sequences.

Δr for the seeds of the resumed runs, in tokens (one sequence is 4,096 tokens):

| Seed | Δ1 | Δ2 | Δ3 | Δ4 | Δ5 | Δ6 | Δ7 |
|---|---|---|---|---|---|---|---|
| 1337 | −261,405 | −10,443 | 278,344 | 172,772 | −234,786 | 297,311 | 210,653 |
| 1338 | 250,962 | 539,749 | 434,177 | 26,619 | 558,716 | 472,058 | 805,111 |
| 2 | 655,512 | 657,521 | 251,876 | 73,158 | 565,472 | 667,317 | 419,537 |

The largest displacement is 805,111 tokens, about 197 sequences: under 0.1% of a chunk, and
each rank visits about three chunks in a 20B-token run.

**The fix, off by default.** `training.sampler_resume=per_rank` gives each rank its own
start offset on resume, recomputed from the seed, and rank 0 checks it against the saved
offset. The resumed stream then continues exactly where the uninterrupted one would have
been, also when resuming the paper checkpoints. The default is `legacy`, which is what every
paper run did, so a legacy resume reproduces them. `per_rank` only matters under DDP, so the
CPU gates do not exercise it.

### Paper runs that resumed

A resume is a `Resume state:` line in a run's training log. Eight runs resumed at least
once; all were 8-GPU runs with `sampler_resume=legacy`. The last resume of each:

| Run | Seed | Last resume |
|---|---|---|
| `s_sps_w64_20b_fw100` | 1337 | iter 5,865, 2.31B tokens, `ckpt.pt` |
| `s_sps_w64_20b_fw100_seed2` | 2 | iter 44,115, 17.35B tokens, `ckpt.pt` |
| `s_sps_w64_20b_fw100_untied` | 1337 | iter 38,760, 15.24B tokens, `ckpt.pt` |
| `s_sps_w64_20b_fw100_untied_seed2` | 2 | iter 45,135, 17.75B tokens, `ckpt.pt` |
| `s_full_attention_20b_fw100_untied` | 1337 | iter 30,600, 12.03B tokens, `ckpt.pt` |
| `s_two_tower_afsps_faithful_20b` | 1337 | iter 45,798, 18.01B tokens, `ckpt_tokens_18008506368.pt` (decay phase) |
| `s_two_tower_seq6_slim_20b_seed2` | 1338 | iter 1,020, 0.40B tokens, `ckpt.pt` |
| `s_two_tower_seq9p3_20b` | 1337 | iter 19,125, 7.52B tokens, `ckpt.pt` |

Every other run in the ledger trained in one uninterrupted segment.
