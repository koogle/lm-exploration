"""N-gram hints: the next tokens that followed a position's context in earlier text.

After the NanoGPT speedrun's exact-match retrieval
(https://brunborg.com/n-gram-hinted-lm, KellerJordan/modded-nanogpt PR 367).
Position t's context is its last `n` input tokens, x_(t-n+1) .. x_t, for n in
LENGTHS; a hint lists the tokens that followed the same context in an indexed
corpus, with their counts. The deepest length that matched decides the hint's
candidates, and the cell (length bucket, total-count bin, top-token-share bin)
tells the model how far to trust it.

Causality: a hint at position t reads only x_(<=t) of its own row (never its
labels) and the tables it is looked up in. Training looks rows up in a `HintIndex` holding the
rows of earlier optimizer steps only; evaluation indexes training-split text,
never the evaluated rows. Contexts never cross a document boundary or the start
of a row; tables skip ignored targets (the BOS after an EOS, padding).

Keys are 48-bit hashes of the context with the 16-bit next token packed below
them, so one sorted uint64 array holds every (context, next token) pair and a
context's candidates are one contiguous range.
"""

import threading

import numpy as np

LENGTHS = (6, 10, 20)
COUNT_EDGES = (2, 3, 5, 9, 17, 33)  # bins of a match's total count
PURITY_EDGES = (0.3, 0.5, 0.7, 0.85, 0.95)  # bins of its top token's share
# Cell 0 is "no match"; buckets 1..len(LENGTHS) are the deepest matched length.
CELLS = (len(LENGTHS) + 1) * (len(COUNT_EDGES) + 1) * (len(PURITY_EDGES) + 1)
ENTRIES = 8  # candidates kept per position, the most frequent
IGNORE_INDEX = -100

_TOKEN_BITS = 16
_MULTIPLIER = np.uint64(0x9E3779B185EBCA87)


def _mix(h):
    """splitmix64 finalizer, so the kept high bits depend on every token."""
    h = h ^ (h >> np.uint64(30))
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27))
    h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def context_keys(inputs, eos_id):
    """{n: ([B, T] uint64 keys, [B, T] bool valid)} for the context ending at each position.

    A key holds the context hash in its high 48 bits and zeros below. The context is
    valid when its n tokens lie in one document of the row: no EOS among its first
    n - 1 tokens (EOS ends a document; the next document starts after it).
    """
    inputs = np.asarray(inputs)
    if inputs.ndim != 2 or inputs.min(initial=0) < 0 or inputs.max(initial=0) >> _TOKEN_BITS:
        raise ValueError("inputs must be [B, T] token ids below 2**16")
    tokens = inputs.astype(np.uint64) + np.uint64(1)
    eos = inputs == eos_id
    batch, length = inputs.shape
    keys = {}
    h = np.zeros((batch, length), np.uint64)
    crossing = np.zeros((batch, length), bool)  # an EOS strictly inside the context
    with np.errstate(over="ignore"):
        for back in range(max(LENGTHS)):
            shifted = np.zeros_like(tokens)
            inside = np.zeros_like(eos)
            if back < length:  # rows shorter than the context have no valid one
                shifted[:, back:] = tokens[:, : length - back]
                inside[:, back:] = eos[:, : length - back]
            h = h * _MULTIPLIER + shifted
            if back:
                crossing |= inside
            n = back + 1
            if n in LENGTHS:
                valid = ~crossing
                valid[:, : n - 1] = False
                mixed = _mix(h + np.uint64(n))
                keys[n] = ((mixed >> np.uint64(_TOKEN_BITS)) << np.uint64(_TOKEN_BITS), valid)
    return keys


def _pairs(inputs, labels, eos_id):
    """{n: sorted uint64 (context key | next token)} for every valid context with a target."""
    labels = np.asarray(labels)
    target = labels != IGNORE_INDEX
    pairs = {}
    for n, (keys, valid) in context_keys(inputs, eos_id).items():
        keep = valid & target
        pairs[n] = np.sort(keys[keep] | labels[keep].astype(np.uint64))
    return pairs


class Table:
    """Sorted unique (context key | next token) pairs and their counts."""

    def __init__(self, pairs, counts):
        self.pairs, self.counts = pairs, counts

    @classmethod
    def from_sorted(cls, pairs, counts=None):
        if not len(pairs):
            return cls(pairs, np.zeros(0, np.int64))
        starts = np.flatnonzero(np.r_[True, pairs[1:] != pairs[:-1]])
        if counts is None:
            totals = np.diff(np.r_[starts, len(pairs)])
        else:
            totals = np.add.reduceat(counts, starts)
        return cls(pairs[starts], totals.astype(np.int64))

    @classmethod
    def merge(cls, tables):
        pairs = np.concatenate([t.pairs for t in tables])
        counts = np.concatenate([t.counts for t in tables])
        order = np.argsort(pairs, kind="stable")
        return cls.from_sorted(pairs[order], counts[order])

    def __len__(self):
        return len(self.pairs)

    def ranges(self, keys):
        """[lo, hi) of each key's candidates."""
        low = np.searchsorted(self.pairs, keys)
        high = np.searchsorted(self.pairs, keys | np.uint64((1 << _TOKEN_BITS) - 1), "right")
        return low, high

    def filtered(self, keys):
        """The rows of contexts in `keys` (sorted unique uint64), dropping the rest."""
        if not len(keys):
            return Table(self.pairs[:0], self.counts[:0])
        context = self.pairs >> np.uint64(_TOKEN_BITS) << np.uint64(_TOKEN_BITS)
        keep = keys[np.searchsorted(keys, context).clip(max=len(keys) - 1)] == context
        return Table(self.pairs[keep], self.counts[keep])


class Hints:
    """One batch's hints: a cell per position and its candidates as flat entries.

    `cells` is [B, T] int64 (0: no hint). `position`, `token` and `weight` list the
    candidates: flat position b * T + t, next token, count share of the kept candidates.
    """

    def __init__(self, cells, position, token, weight):
        self.cells, self.position, self.token, self.weight = cells, position, token, weight

    def __eq__(self, other):
        return all(
            np.array_equal(getattr(self, k), getattr(other, k))
            for k in ("cells", "position", "token", "weight")
        )

    def to(self, device):
        import torch

        return {
            "cells": torch.from_numpy(self.cells).to(device, non_blocking=True),
            "position": torch.from_numpy(self.position).to(device, non_blocking=True),
            "token": torch.from_numpy(self.token).to(device, non_blocking=True),
            "weight": torch.from_numpy(self.weight).to(device, non_blocking=True),
        }

    @property
    def coverage(self):
        """Share of positions with a hint, per length bucket (1-based) and in total."""
        bucket = self.cells // ((len(COUNT_EDGES) + 1) * (len(PURITY_EDGES) + 1))
        shares = {f"bucket_{n}": float((bucket == i + 1).mean()) for i, n in enumerate(LENGTHS)}
        return {"any": float((self.cells > 0).mean()), **shares}


def lookup(tables, inputs, eos_id):
    """Hints for a [B, T] batch from `tables` ({n: [Table, ...]}), which must not hold it."""
    inputs = np.asarray(inputs)
    batch, length = inputs.shape
    cells = np.zeros(batch * length, np.int64)
    chosen = np.zeros(batch * length, bool)
    found_position, found_token, found_count = [], [], []
    keys = context_keys(inputs, eos_id)
    # Deepest length first; a position takes the first length that matches.
    for bucket, n in reversed(list(enumerate(LENGTHS, 1))):
        context, valid = keys[n]
        query = np.flatnonzero(valid.ravel() & ~chosen)
        positions, tokens, counts = [], [], []
        for table in tables.get(n, ()):
            low, high = table.ranges(context.ravel()[query])
            sizes = high - low
            if not sizes.sum():
                continue
            at = np.repeat(query, sizes)
            rows = np.repeat(low - np.cumsum(sizes) + sizes, sizes) + np.arange(sizes.sum())
            positions.append(at)
            tokens.append((table.pairs[rows] & np.uint64(0xFFFF)).astype(np.int64))
            counts.append(table.counts[rows])
        if not positions:
            continue
        at = np.concatenate(positions)
        token = np.concatenate(tokens)
        count = np.concatenate(counts)
        # Sum one token's counts across tables; sort by position, then count descending.
        combined, inverse = np.unique(at * (1 << _TOKEN_BITS) + token, return_inverse=True)
        count = np.bincount(inverse, weights=count).astype(np.int64)
        at, token = combined >> _TOKEN_BITS, combined & 0xFFFF
        order = np.lexsort((token, -count, at))
        at, token, count = at[order], token[order], count[order]
        matched, first = np.unique(at, return_index=True)
        total = np.add.reduceat(count, first)
        top = count[first]
        count_bin = np.searchsorted(COUNT_EDGES, total, "right")
        purity_bin = np.searchsorted(PURITY_EDGES, top / total, "right")
        cells[matched] = (bucket * (len(COUNT_EDGES) + 1) + count_bin) * (
            len(PURITY_EDGES) + 1
        ) + purity_bin
        chosen[matched] = True
        rank = np.arange(len(at)) - np.repeat(first, np.diff(np.r_[first, len(at)]))
        keep = rank < ENTRIES
        found_position.append(at[keep])
        found_token.append(token[keep])
        found_count.append(count[keep])
    if found_position:
        position = np.concatenate(found_position)
        token = np.concatenate(found_token)
        count = np.concatenate(found_count).astype(np.float64)
        order = np.lexsort((token, position))
        position, token, count = position[order], token[order], count[order]
        kept = np.bincount(position, weights=count, minlength=batch * length)
        weight = (count / kept[position]).astype(np.float32)
    else:
        position = token = np.zeros(0, np.int64)
        weight = np.zeros(0, np.float32)
    return Hints(cells.reshape(batch, length), position, token, weight)


def _empty_tables():
    return {n: [] for n in LENGTHS}


class HintIndex:
    """Training rows indexed after their step, for hints to later steps only.

    `add(step, inputs, labels)` records a trained step. `hints(step, ...)` reads
    tables built from steps strictly before `step`. Small tables merge in the
    background, log-structured, so a lookup reads a handful of sorted arrays.
    """

    def __init__(self, eos_id, flush_steps=16):
        self.eos_id, self.flush_steps = eos_id, flush_steps
        self.tables = _empty_tables()
        self.indexed_through = -1  # last step every table includes
        self.pending, self.pending_steps = [], []
        self.lock = threading.Lock()
        self.merging = None

    def add(self, step, inputs, labels):
        if step <= max([self.indexed_through, *self.pending_steps]):
            raise ValueError("steps must be added in increasing order")
        self.pending.append(_pairs(inputs, labels, self.eos_id))
        self.pending_steps.append(step)
        if len(self.pending) >= self.flush_steps:
            self.flush(wait=False)

    def flush(self, wait=True):
        """Index the pending steps; with wait=False, in a background thread."""
        if self.merging is not None:
            self.merging.join()
            self.merging = None
        if not self.pending:
            return
        pending, through = self.pending, self.pending_steps[-1]
        self.pending, self.pending_steps = [], []
        worker = threading.Thread(target=self._index, args=(pending, through))
        worker.start()
        if wait:
            worker.join()
        else:
            self.merging = worker

    def _index(self, pending, through):
        tables = {}
        for n in LENGTHS:
            current = list(self.tables[n])
            current.append(Table.from_sorted(np.sort(np.concatenate([p[n] for p in pending]))))
            # Merge while the newest table is at least half its predecessor's size.
            while len(current) > 1 and 2 * len(current[-1]) >= len(current[-2]):
                current[-2:] = [Table.merge(current[-2:])]
            tables[n] = current
        with self.lock:
            self.tables, self.indexed_through = tables, through

    def hints(self, step, inputs):
        with self.lock:
            tables, through = self.tables, self.indexed_through
        if through >= step:
            raise RuntimeError(f"index holds step {through}; refusing hints for step {step}")
        return lookup(tables, inputs, self.eos_id), through

    @property
    def size(self):
        with self.lock:
            return {n: int(sum(len(t) for t in self.tables[n])) for n in LENGTHS}


def query_keys(batches, eos_id):
    """{n: sorted unique context keys} of every valid context in `batches` ([B, T] inputs)."""
    keys = {n: [] for n in LENGTHS}
    for inputs in batches:
        for n, (context, valid) in context_keys(inputs, eos_id).items():
            keys[n].append(context[valid])
    return {n: np.unique(np.concatenate(k)) for n, k in keys.items()}


def save_tables(path, tables, **metadata):
    arrays = {}
    for n, table in tables.items():
        arrays[f"pairs_{n}"], arrays[f"counts_{n}"] = table.pairs, table.counts
    np.savez(path, **arrays, **{k: np.asarray(v) for k, v in metadata.items()})


def load_tables(path):
    with np.load(path) as data:
        return {n: [Table(data[f"pairs_{n}"], data[f"counts_{n}"])] for n in LENGTHS}
