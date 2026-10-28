"""Token shards for the hybrid family: preparation, contamination screen, loader.

Each upstream parquet file of a recipe source becomes one prepared shard:
`<root>/<source>/<index>.bin` (uint32 GLM token ids, every document followed by
EOS), `<index>.docs.npy` (int64 [documents, 2]: tokens including EOS, UTF-8 bytes)
and `<index>.json`, written last, whose presence marks the shard complete.
Documents are shuffled within a shard with a seed from the recipe. The last
shard of each source is held out for evaluation.

Preparation drops documents that share any token n-gram (n = 13) with the
recipe's benchmark test sets, and code outside `PERMISSIVE_LICENSES` when a source sets
`permissive_licenses_only`.

`PreparedMixture` packs each source's shards, in a seeded order, into windows of
`context + 1` tokens; the next window starts at the previous window's last
token. Each window comes from the source furthest below its weight. It can run
while shards are still being prepared: it waits for the next shard's `.json`.
"""

import json
import time
from pathlib import Path

import numpy as np
from training_data.hub_files import download as hub_download
from training_data.licenses import PERMISSIVE_LICENSES

HASH_BASE = np.uint64(0x9E3779B97F4A7C15)


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    total = sum(source["weight"] for source in recipe["sources"].values())
    if abs(total - 1) > 1e-9:
        raise ValueError("Source weights must sum to 1")
    for name, source in recipe["sources"].items():
        if source["max_passes"] != 1:
            raise ValueError(f"{name}: only single-pass sources are supported")
    return recipe


def heldout_index(source):
    return source["shards"] - 1


def training_order(recipe, name):
    """Seeded order in which training reads a source's shards (held-out shard excluded)."""
    source = recipe["sources"][name]
    seed = recipe["seed"] + sum(map(ord, name))
    order = np.random.default_rng(seed).permutation(heldout_index(source))
    return [int(index) for index in order]


def ngram_hashes(ids, n):
    """uint64 rolling hashes of every length-n window of `ids` (empty if shorter)."""
    ids = np.asarray(ids, dtype=np.uint64)
    if ids.size < n:
        return np.empty(0, dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(ids, n)
    powers = HASH_BASE ** np.arange(n - 1, -1, -1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return (windows * powers).sum(axis=1, dtype=np.uint64)


def benchmark_texts(benchmark, table):
    for column in benchmark["columns"]:
        for value in table.column(column).to_pylist():
            if isinstance(value, dict):
                value = value.get("text", [])
            values = value if isinstance(value, list) else [value]
            yield from (str(v) for v in values if v)


def build_screen(recipe, tokenizer, download=hub_download):
    """Sorted unique n-gram hashes of every benchmark text, with per-benchmark counts."""
    import pyarrow.parquet as pq

    n = recipe["contamination"]["ngram"]
    hashes, report = [], []
    for benchmark in recipe["contamination"]["benchmarks"]:
        texts = []
        for file in benchmark["files"]:
            path = download(benchmark["repo"], benchmark["revision"], file)
            texts.extend(benchmark_texts(benchmark, pq.read_table(path)))
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        found = [ngram_hashes(e.ids, n) for e in encodings]
        hashes.extend(found)
        report.append(
            {
                "repo": benchmark["repo"],
                "revision": benchmark["revision"],
                "texts": len(texts),
                "ngrams": int(sum(len(h) for h in found)),
            }
        )
    return np.unique(np.concatenate(hashes)), report


def contaminated(ids, screen, n):
    hashes = ngram_hashes(ids, n)
    if not hashes.size or not screen.size:
        return False
    positions = np.searchsorted(screen, hashes).clip(max=screen.size - 1)
    return bool((screen[positions] == hashes).any())


def shard_paths(root, name, index):
    folder = Path(root) / name
    return (
        folder / f"{index:05d}.bin",
        folder / f"{index:05d}.docs.npy",
        folder / f"{index:05d}.json",
    )


def prepare_shard(recipe, name, index, root, tokenizer, screen, eos_id, download=hub_download):
    """Tokenize one upstream file into a prepared shard; returns its metadata."""
    import pyarrow.parquet as pq

    source = recipe["sources"][name]
    file = source["pattern"].format(index=index)
    path = download(source["repo"], source["revision"], file)
    parquet = pq.ParquetFile(path)
    n = recipe["contamination"]["ngram"]
    licenses = source.get("permissive_licenses_only", False)
    documents, dropped = [], {"license": 0, "empty": 0, "contaminated": 0}
    columns = [source["column"], *(["license"] if licenses else [])]
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=columns)
        texts = table.column(source["column"]).to_pylist()
        if licenses:
            allowed = [
                lic.lower() in PERMISSIVE_LICENSES for lic in table.column("license").to_pylist()
            ]
            dropped["license"] += allowed.count(False)
            texts = [t for t, keep in zip(texts, allowed, strict=True) if keep]
        empty = [not t or not t.strip() for t in texts]
        dropped["empty"] += sum(empty)
        texts = [t for t, e in zip(texts, empty, strict=True) if not e]
        for text, encoding in zip(
            texts, tokenizer.encode_batch(texts, add_special_tokens=False), strict=True
        ):
            ids = np.asarray(encoding.ids, dtype=np.uint32)
            if contaminated(ids, screen, n):
                dropped["contaminated"] += 1
                continue
            documents.append((ids, len(text.encode())))
    order = np.random.default_rng([recipe["seed"], sum(map(ord, name)), index]).permutation(
        len(documents)
    )
    lengths = np.array(
        [[len(documents[i][0]) + 1, documents[i][1]] for i in order], np.int64
    ).reshape(-1, 2)
    tokens = np.empty(int(lengths[:, 0].sum()), dtype=np.uint32)
    offset = 0
    for i in order:
        ids = documents[i][0]
        tokens[offset : offset + len(ids)] = ids
        tokens[offset + len(ids)] = eos_id
        offset += len(ids) + 1
    bin_path, docs_path, meta_path = shard_paths(root, name, index)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(bin_path.with_suffix(".tmp"))
    bin_path.with_suffix(".tmp").rename(bin_path)
    np.save(docs_path, lengths)
    meta = {
        "source": name,
        "index": index,
        "upstream": {"repo": source["repo"], "revision": source["revision"], "file": file},
        "documents": len(lengths),
        "tokens": len(tokens),
        "bytes": int(lengths[:, 1].sum()),
        "dropped": dropped,
        "heldout": index == heldout_index(source),
    }
    meta_path.with_suffix(".tmp").write_text(json.dumps(meta, indent=2) + "\n")
    meta_path.with_suffix(".tmp").rename(meta_path)
    return meta


class PreparedMixture:
    """Weighted, resumable window stream over prepared shards.

    `refresh` is called while waiting for a shard (on Modal: the volume's reload);
    waiting longer than `wait_seconds` raises, since preparation fell behind.
    """

    def __init__(self, recipe, root, *, refresh=None, wait_seconds=1800):
        self.recipe, self.root = recipe, Path(root)
        self.context = recipe["context"]
        self.refresh, self.wait_seconds = refresh, wait_seconds
        self.names = sorted(recipe["sources"])
        self.orders = {name: training_order(recipe, name) for name in self.names}
        self.state = {name: {"shard": 0, "offset": 0, "tokens": 0} for name in self.names}
        self.loaded = {}

    def _tokens(self, name, position):
        if position >= len(self.orders[name]):
            raise StopIteration(f"{name} has no shards left")
        index = self.orders[name][position]
        key = (name, index)
        if key not in self.loaded:
            bin_path, _, meta_path = shard_paths(self.root, name, index)
            start = time.monotonic()
            while not meta_path.exists():
                if time.monotonic() - start > self.wait_seconds:
                    raise TimeoutError(f"Shard {name}/{index} was not prepared in time")
                time.sleep(15)
                if self.refresh:
                    self.refresh()
            # Read fully so no volume file stays open (volume reloads need that).
            current = self.state[name]["shard"]
            keep = {
                self.orders[name][p] for p in (current, current + 1) if p < len(self.orders[name])
            }
            self.loaded = {k: v for k, v in self.loaded.items() if k[0] != name or k[1] in keep}
            self.loaded[key] = np.fromfile(bin_path, dtype=np.uint32)
        return self.loaded[key]

    def _take(self, name, count):
        """`count` tokens of a source's stream from its cursor, without moving it."""
        state = self.state[name]
        position, offset, parts = state["shard"], state["offset"], []
        while count:
            tokens = self._tokens(name, position)
            part = tokens[offset : offset + count]
            parts.append(part)
            count -= len(part)
            position, offset = (position + 1, 0) if count else (position, offset + len(part))
        return np.concatenate(parts)

    def _advance(self, name, count):
        state = self.state[name]
        while count:
            tokens = self._tokens(name, state["shard"])
            step = min(count, len(tokens) - state["offset"])
            state["offset"] += step
            count -= step
            if state["offset"] == len(tokens):
                state["shard"], state["offset"] = state["shard"] + 1, 0
        state["tokens"] += self.context

    def next_window(self):
        weights = self.recipe["sources"]
        name = min(self.names, key=lambda n: (self.state[n]["tokens"] / weights[n]["weight"], n))
        window = self._take(name, self.context + 1)
        self._advance(name, self.context)
        return name, window

    def next_batch(self, rows):
        """int64 inputs and labels [rows, context] and the source of each row."""
        windows, names = [], []
        for _ in range(rows):
            name, window = self.next_window()
            windows.append(window)
            names.append(name)
        stacked = np.stack(windows).astype(np.int64)
        return stacked[:, :-1], stacked[:, 1:], names

    def state_dict(self):
        return {
            "recipe": self.recipe["name"],
            "sources": {k: dict(v) for k, v in self.state.items()},
        }

    def load_state_dict(self, state):
        if state["recipe"] != self.recipe["name"] or set(state["sources"]) != set(self.names):
            raise ValueError("Data state belongs to another recipe")
        self.state = {k: dict(v) for k, v in state["sources"].items()}
        self.loaded = {}


def heldout_windows(recipe, root, name, context, max_documents):
    """Held-out windows of one source or chat subset: (inputs, labels, total UTF-8 bytes).

    Pretraining sources concatenate whole held-out documents into one stream (labels
    -100 only past its end). Chat recipes pack whole conversations and train only the
    assistant tokens (`sft_data.chat_heldout_windows`; None for an empty subset).
    """
    if recipe.get("format") == "chat-sft":
        from .sft_data import chat_heldout_windows

        return chat_heldout_windows(recipe, root, name, context, max_documents)
    source = recipe["sources"][name]
    bin_path, docs_path, _ = shard_paths(root, name, heldout_index(source))
    lengths = np.load(docs_path)[:max_documents]
    count = int(lengths[:, 0].sum())
    tokens = np.fromfile(bin_path, dtype=np.uint32, count=count).astype(np.int64)
    rows = -(-(count - 1) // context)
    padded = np.full(rows * context + 1, -100, dtype=np.int64)
    padded[:count] = tokens
    inputs = padded[:-1].reshape(rows, context).copy()
    labels = padded[1:].reshape(rows, context).copy()
    inputs[inputs < 0] = 0
    return inputs, labels, int(lengths[:, 1].sum())
