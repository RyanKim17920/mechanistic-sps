"""The training data sampler."""

import bisect
import math
import random

from torch.utils.data import Sampler


def fresh_start_offset(seed: int, rank: int, dataset_len: int, max_start_offset: int) -> int:
    """Where rank `rank` of a fresh DDP run starts its stream (scripts/train.py):
    Random(seed + rank).randint(0, min(max_start_offset, dataset_len // 2))."""
    return random.Random(int(seed) + int(rank)).randint(0, min(int(max_start_offset), int(dataset_len) // 2))


class FixedRandomChunkDistributedSampler(Sampler):
    """Distributed sampler with a fixed random chunk order.

    The sampler:
    - advances over sequence starts spaced by ``block_size`` (non-overlapping sequence starts),
    - groups those starts into fixed-size chunks,
    - shuffles chunk order once deterministically via ``seed``,
    - assigns shuffled chunks to DDP ranks by striding over shuffled chunk order.

    This gives reproducible, memory-efficient shuffling without materializing all indices.
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: int,
        rank: int,
        block_size: int,
        chunk_size_units: int = 262_144,
        seed: int = 1337,
        start_offset: int = 0,
        resume_samples_seen_per_rank: int = 0,
        balanced: bool = False,
    ) -> None:
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.chunk_size_units = max(1, int(chunk_size_units))
        self.seed = int(seed)
        self.balanced = bool(balanced)
        self.start_offset = int(start_offset % self.dataset_len) if self.dataset_len > 0 else 0

        if self.dataset_len <= self.start_offset:
            self._num_units = 0
        else:
            self._num_units = math.ceil((self.dataset_len - self.start_offset) / self.block_size)

        num_chunks = math.ceil(self._num_units / self.chunk_size_units) if self._num_units > 0 else 0
        shuffled_chunk_ids = list(range(num_chunks))
        rng = random.Random(self.seed)
        rng.shuffle(shuffled_chunk_ids)

        # ------------------------------------------------------------------
        # balanced=False reproduces the historical (buggy) stream layout.
        #
        # Two sources of per-rank LENGTH SKEW exist in the unbalanced layout:
        #   1. num_chunks is rarely a multiple of num_replicas, so ranks
        #      0..(num_chunks % num_replicas - 1) get one chunk more than the rest;
        #   2. the FINAL chunk is a partial one (num_units % chunk_size_units
        #      units instead of chunk_size_units), and the shuffle drops it on an
        #      arbitrary rank.
        # A rank that both misses the extra chunk AND inherits the partial one
        # gets a materially shorter stream than its peers. Because DDP ranks step
        # in lockstep, that rank hits StopIteration first; scripts/train.py then
        # rebuilds iter(train_loader) from the same sampler state and the rank
        # REPLAYS the head of its own stream while every other rank carries on
        # with fresh tokens.
        #
        # balanced=True removes both skews: drop the partial tail chunk, then
        # truncate the shuffled chunk list to a multiple of num_replicas, so every
        # rank owns exactly floor(full_chunks / num_replicas) FULL chunks and no
        # rank can exhaust before another. The discarded tail is corpus that a
        # single-epoch run never reaches anyway.
        # ------------------------------------------------------------------
        self.dropped_chunks = 0
        if self.balanced and num_chunks > 0 and self.num_replicas > 0:
            full_chunks = self._num_units // self.chunk_size_units  # excludes the partial tail
            usable = (full_chunks // self.num_replicas) * self.num_replicas
            if usable > 0:
                shuffled_chunk_ids = [c for c in shuffled_chunk_ids if c < full_chunks][:usable]
                self.dropped_chunks = num_chunks - usable

        # Each rank takes every N-th chunk from the same shuffled order.
        rank_chunk_ids = shuffled_chunk_ids[self.rank::self.num_replicas]
        self._rank_chunks = []
        self._prefix_chunk_lengths = [0]
        for chunk_id in rank_chunk_ids:
            unit_start = chunk_id * self.chunk_size_units
            unit_end = min(unit_start + self.chunk_size_units, self._num_units)
            if unit_start >= unit_end:
                continue
            self._rank_chunks.append((unit_start, unit_end))
            self._prefix_chunk_lengths.append(self._prefix_chunk_lengths[-1] + (unit_end - unit_start))

        total_rank_units = self._prefix_chunk_lengths[-1]
        self.resume_samples_seen_per_rank = max(0, min(int(resume_samples_seen_per_rank), total_rank_units))
        self._length = total_rank_units - self.resume_samples_seen_per_rank

    def __iter__(self):
        if self._length <= 0:
            return iter(())

        skip = self.resume_samples_seen_per_rank
        chunk_idx = bisect.bisect_right(self._prefix_chunk_lengths, skip) - 1
        in_chunk_offset = skip - self._prefix_chunk_lengths[chunk_idx]

        def _iter_positions():
            for i in range(chunk_idx, len(self._rank_chunks)):
                unit_start, unit_end = self._rank_chunks[i]
                start_u = unit_start + (in_chunk_offset if i == chunk_idx else 0)
                for unit in range(start_u, unit_end):
                    yield self.start_offset + unit * self.block_size

        return _iter_positions()

    def __len__(self) -> int:
        return self._length
