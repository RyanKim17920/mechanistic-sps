"""Gate G5: FixedRandomChunkDistributedSampler per-rank stream-length balance.

Documents (and pins) the replay bug that `training.sampler_fix` removes.

MECHANISM
---------
The sampler cuts the corpus into fixed chunks of `chunk_size_units` block-sized
positions, shuffles the chunk ids with a fixed seed, and hands rank r the chunks
at shuffled positions r, r+W, r+2W, ... Two things make the per-rank streams
UNEQUAL in length:

  1. `num_chunks` is essentially never a multiple of the world size, so the
     first `num_chunks % W` ranks get one chunk more than the rest;
  2. the LAST chunk in unit space is a PARTIAL chunk -- it holds
     `num_units % chunk_size_units` units instead of `chunk_size_units` -- and
     the shuffle drops it on an arbitrary rank.

A rank that both misses the extra chunk and inherits the partial one gets a
materially shorter stream. DDP ranks step in lockstep, so that rank reaches
StopIteration first; `scripts/train.py`'s `get_batch()` then rebuilds
`iter(train_loader)` from an unchanged sampler and the rank REPLAYS the head of
its own stream while every other rank keeps consuming fresh tokens.

For the production 20B configuration this test pins the number: rank 2 replays
4.28% of its stream.
"""

import pytest

from training.sampler import FixedRandomChunkDistributedSampler


# Production 20B setting: fineweb-edu-100bt train.bin is 54,178,221,246 bytes of
# uint16 => 27,089,110,623 tokens; TokenDataset length is tokens - block_size.
PROD_TOKENS = 54_178_221_246 // 2
PROD_BLOCK = 4096
PROD_DATASET_LEN = PROD_TOKENS - PROD_BLOCK
PROD_CHUNK_UNITS = 262_144
PROD_SEED = 1337
PROD_WORLD = 8
# 20B tokens / block_size / world_size samples each rank must yield.
PROD_SAMPLES_PER_RANK = (20_000_000_000 // PROD_BLOCK) // PROD_WORLD


def _lengths(world, balanced, dataset_len=PROD_DATASET_LEN, chunk_units=PROD_CHUNK_UNITS,
             block=PROD_BLOCK, seed=PROD_SEED, start_offset=0):
    return [
        len(
            FixedRandomChunkDistributedSampler(
                dataset_len=dataset_len,
                num_replicas=world,
                rank=r,
                block_size=block,
                chunk_size_units=chunk_units,
                seed=seed,
                start_offset=start_offset,
            )
            if not balanced
            else FixedRandomChunkDistributedSampler(
                dataset_len=dataset_len,
                num_replicas=world,
                rank=r,
                block_size=block,
                chunk_size_units=chunk_units,
                seed=seed,
                start_offset=start_offset,
                balanced=True,
            )
        )
        for r in range(world)
    ]


def test_unbalanced_production_config_starves_one_rank():
    """With sampler_fix=false (the paper setting) one rank cannot supply a full 20B run without replaying."""
    lengths = _lengths(PROD_WORLD, balanced=False)
    short_rank = min(range(PROD_WORLD), key=lambda r: lengths[r])
    short_len = lengths[short_rank]

    # Exactly one rank falls short of what the run needs.
    starved = [r for r, n in enumerate(lengths) if n < PROD_SAMPLES_PER_RANK]
    assert starved == [short_rank], (lengths, PROD_SAMPLES_PER_RANK)

    replay = PROD_SAMPLES_PER_RANK - short_len
    replay_frac = replay / PROD_SAMPLES_PER_RANK
    # Pins the documented number: rank 2 replays 4.28% of its stream.
    assert short_rank == 2, lengths
    assert replay_frac == pytest.approx(0.0428, abs=5e-4), replay_frac


def test_balanced_gives_every_rank_an_identical_stream_length():
    lengths = _lengths(PROD_WORLD, balanced=True)
    assert len(set(lengths)) == 1, lengths
    # And the equal length still covers the whole 20B run, so nobody replays.
    assert lengths[0] >= PROD_SAMPLES_PER_RANK, (lengths[0], PROD_SAMPLES_PER_RANK)


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8, 16])
def test_balanced_is_equal_length_for_many_world_sizes(world):
    lengths = _lengths(world, balanced=True)
    assert len(set(lengths)) == 1, (world, lengths)


def test_balanced_partitions_are_still_disjoint_and_in_range():
    """Balancing must not hand the same position to two ranks."""
    world = 4
    dataset_len = 4096 * 1000 + 17
    kwargs = dict(
        dataset_len=dataset_len,
        num_replicas=world,
        rank=0,
        block_size=4096,
        chunk_size_units=37,
        seed=99,
        balanced=True,
    )
    seen = set()
    for r in range(world):
        kwargs["rank"] = r
        positions = list(FixedRandomChunkDistributedSampler(**kwargs))
        assert len(positions) == len(set(positions))
        assert seen.isdisjoint(positions)
        seen.update(positions)
        for p in positions:
            assert 0 <= p < dataset_len


def test_balanced_drops_only_the_tail():
    """Balanced total == world * floor(full_chunks / world) * chunk_size_units."""
    world = 8
    s = FixedRandomChunkDistributedSampler(
        dataset_len=PROD_DATASET_LEN,
        num_replicas=world,
        rank=0,
        block_size=PROD_BLOCK,
        chunk_size_units=PROD_CHUNK_UNITS,
        seed=PROD_SEED,
        balanced=True,
    )
    num_units = (PROD_DATASET_LEN + PROD_BLOCK - 1) // PROD_BLOCK
    full_chunks = num_units // PROD_CHUNK_UNITS
    expected_per_rank = (full_chunks // world) * PROD_CHUNK_UNITS
    assert len(s) == expected_per_rank
    # The tail we give up is a small fraction of the corpus a single-epoch run
    # never reaches anyway.
    kept = world * expected_per_rank
    assert kept / num_units > 0.9


def test_sampler_fix_off_is_bit_identical_to_head():
    """balanced defaults to False and reproduces the historical stream exactly."""
    common = dict(
        dataset_len=4096 * 5000 + 3,
        num_replicas=3,
        rank=1,
        block_size=4096,
        chunk_size_units=13,
        seed=7,
        start_offset=1234,
    )
    default = list(FixedRandomChunkDistributedSampler(**common))
    explicit_off = list(FixedRandomChunkDistributedSampler(**common, balanced=False))
    assert default == explicit_off
