"""Packed uint32 token shards: preparation, contamination screen, loader.

Each upstream parquet file of a recipe source becomes one prepared shard:
`<root>/<source>/<index>.bin` (uint32 GLM token ids, every document followed by
EOS), `<index>.docs.npy` (int64 [documents, 2]: tokens including EOS, UTF-8 bytes)
and `<index>.json`, written last, whose presence marks the shard complete.
Documents are shuffled within a shard with a seed from the recipe. The last
shard of each source is held out for evaluation.

A source may group rows into longer documents (`group_by`, e.g. one document per
code repository) and keep only documents of at least `min_tokens`; `file_offset`
shifts which upstream files its shards read. A source names its files by `pattern`
(with `{index}`) or lists them in `files`. A source with `prepared_in` reuses the
shards another recipe prepared under the same source name (`source_root`).

Preparation drops documents that share any token n-gram (n = 13) with the
recipe's benchmark test sets, and code outside `PERMISSIVE_LICENSES` when a source sets
`permissive_licenses_only`. A code source can also select rows by upstream columns:
`languages` (keep only these) or `exclude_languages` on `language`, `exclude_path`
(a regular expression searched in `path`, for vendored and built files) and
`max_bytes` (UTF-8 size of the document, for generated and data files). With a
recipe-level `stars` table (repository names with at least `minimum` stars,
`build_stars`), `starred: true` keeps only rows whose `split_column` repository is
in the table and `starred: false` only the others. `exclude_repositories` drops
rows of the listed repositories. Upstream files are parquet or gzipped JSON lines.

With `hash_split`, a training shard keeps a document only when its content and its
`split_column` group are both outside the held-out partition, and the held-out
shard only when both are inside it. `heldout_split: "group"` makes the held-out
shard keep every document of a held-out group: a larger evaluation set that no
training shard of any source with the same split contains.

A source with `max_passes` above 1 is read again, in the same shard order, once its
shards are used up, so a small curated source can carry a larger weight.

`PreparedMixture` packs each source's shards, in a seeded order, into windows of
`context + 1` tokens; the next window starts at the previous window's last
token. Each window comes from the source furthest below its weight. It can run
while shards are still being prepared: it waits for the next shard's `.json`.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .hub_files import download as hub_download
from .licenses import PERMISSIVE_LICENSES
from .mixture import deficit_source

HASH_BASE = np.uint64(0x9E3779B97F4A7C15)


def document_key(text):
    text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def partition(key):
    """Content-stable 10% holdout, shared by sampled and production corpora."""
    return int(key[:16], 16) % 10 == 0


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    total = sum(source["weight"] for source in recipe["sources"].values())
    if abs(total - 1) > 1e-9:
        raise ValueError("Source weights must sum to 1")
    for name, source in recipe["sources"].items():
        if type(source["max_passes"]) is not int or source["max_passes"] < 1:
            raise ValueError(f"{name}: max_passes must be a positive integer")
        if "starred" in source and not ("stars" in recipe and source.get("split_column")):
            raise ValueError(f"{name}: starred needs a recipe stars table and a split_column")
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
            if Path(path).suffix == ".jsonl":
                # LiveCodeBench releases are JSONL; only inspect declared fields.
                # Test payloads can be large and are not training inputs.
                import pyarrow as pa

                with Path(path).open() as handle:
                    rows = [
                        {key: row[key] for key in benchmark["columns"]}
                        for line in handle
                        if line.strip()
                        for row in [json.loads(line)]
                    ]
                table = pa.Table.from_pylist(rows)
            else:
                table = pq.read_table(path, columns=benchmark["columns"])
            texts.extend(benchmark_texts(benchmark, table))
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


def repository_hash(name):
    return int(document_key(str(name).lower())[:16], 16)


def build_stars(recipe, download=hub_download):
    """Sorted uint64 hashes of repositories with at least `minimum` stars, and a report."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    spec = recipe["stars"]
    names = set()
    rows = 0
    for file in spec["files"]:
        table = pq.read_table(
            download(spec["repo"], spec["revision"], file),
            columns=[spec["name_column"], spec["count_column"]],
        )
        rows += table.num_rows
        counts = pc.fill_null(table.column(spec["count_column"]), 0)
        kept = table.filter(pc.greater_equal(counts, spec["minimum"]))
        names.update(str(n).lower() for n in kept.column(spec["name_column"]).to_pylist())
    hashes = np.unique(np.array([repository_hash(n) for n in names], dtype=np.uint64))
    return hashes, {"minimum": spec["minimum"], "rows": rows, "repositories": len(names)}


class JsonLines:
    """A gzipped JSON-lines upstream file read like a one-row-group parquet file."""

    num_row_groups = 1

    def __init__(self, path):
        self.path = path

    def iter_batches(self, batch_size, columns, row_groups=None):
        import gzip

        import pyarrow as pa

        rows = []
        with gzip.open(self.path, "rt", encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                rows.append({column: row.get(column) for column in columns})
                if len(rows) == batch_size:
                    yield pa.Table.from_pylist(rows)
                    rows = []
        if rows:
            yield pa.Table.from_pylist(rows)


def contaminated(ids, screen, n):
    hashes = ngram_hashes(ids, n)
    if not hashes.size or not screen.size:
        return False
    positions = np.searchsorted(screen, hashes).clip(max=screen.size - 1)
    return bool((screen[positions] == hashes).any())


def source_root(recipe, root, name):
    """Folder holding a source's shards: the recipe's own, or that of the recipe named in
    the source's `prepared_in` (a sibling folder), whose shards it reuses."""
    other = recipe["sources"][name].get("prepared_in")
    return Path(root).parent / other if other else Path(root)


def shard_paths(root, name, index):
    folder = Path(root) / name
    return (
        folder / f"{index:05d}.bin",
        folder / f"{index:05d}.docs.npy",
        folder / f"{index:05d}.json",
    )


def row_filter(source, starred=None):
    """Why a source drops an upstream row ("language", "path", "size" or "stars"), or
    None to keep it.

    Returns None when the source sets none of `languages`, `exclude_languages`,
    `exclude_path`, `max_bytes`, `exclude_repositories` and `starred`. `columns` lists the upstream columns it
    reads. `starred` is the set of `repository_hash` values of the recipe's stars table.
    """
    keep = frozenset(source.get("languages", ()))
    skip = frozenset(source.get("exclude_languages", ()))
    path = re.compile(source["exclude_path"]) if source.get("exclude_path") else None
    max_bytes = source.get("max_bytes")
    wanted = source.get("starred")
    excluded = frozenset(name.lower() for name in source.get("exclude_repositories", ()))
    if wanted is not None and starred is None:
        raise ValueError("A starred source needs the recipe's stars table")
    if not (keep or skip or path or max_bytes or excluded or wanted is not None):
        return None

    def reason(row, size):
        if row.get("language") in skip or keep and row.get("language") not in keep:
            return "language"
        if path and path.search(row["path"]):
            return "path"
        if max_bytes and size > max_bytes:
            return "size"
        if excluded and str(row[source["split_column"]]).lower() in excluded:
            return "repository"
        if wanted is not None and (
            (repository_hash(row[source["split_column"]]) in starred) != wanted
        ):
            return "stars"
        return None

    reason.columns = (["language"] if keep or skip else []) + (["path"] if path else [])
    return reason


def _text_batches(
    source, parquet, dropped, render=None, heldout=None, row_groups=None, starred=None
):
    """Document texts of one upstream file, one list per row group.

    With `group_by` (e.g. `repo_name`), the file's rows are grouped first and each
    group becomes one document: its rows in `order_by` order, each under a
    `### <order_by value>` header line (repository-level code, one file after another).
    """
    licenses = source.get("permissive_licenses_only", False)
    group_by, order_by = source.get("group_by"), source.get("order_by")
    columns = [source["column"], *(["license"] if licenses else [])]
    select = row_filter(source, starred)
    columns += [c for c in ("language", "path") if select and c in select.columns]
    columns += [c for c in (group_by, order_by) if c]
    if source.get("split_column") and source["split_column"] not in columns:
        columns.append(source["split_column"])
    groups = {}
    seen = set()
    import pyarrow as pa
    import pyarrow.compute as pc

    others = [c for c in columns if c != source["column"]]
    for batch in parquet.iter_batches(batch_size=1024, columns=columns, row_groups=row_groups):
        # Filter on the small columns and the text's byte length first; only the kept
        # rows' texts become Python strings.
        text = batch.column(source["column"])
        fields = batch.select(others).to_pylist() if others else [{}] * batch.num_rows
        kept = list(range(batch.num_rows))
        if licenses:
            allowed = [i for i in kept if fields[i]["license"].lower() in PERMISSIVE_LICENSES]
            dropped["license"] += len(kept) - len(allowed)
            kept = allowed
        if select:
            sizes = [0] * batch.num_rows
            if not pa.types.is_null(text.type):
                sizes = pc.binary_length(text).fill_null(0).to_pylist()
            reasons = [select(fields[i], sizes[i]) for i in kept]
            for reason in filter(None, reasons):
                dropped[reason] = dropped.get(reason, 0) + 1
            kept = [i for i, reason in zip(kept, reasons, strict=True) if not reason]
        texts = text.take(pa.array(kept, pa.int64())).to_pylist()
        rows = [{**fields[i], source["column"]: t} for i, t in zip(kept, texts, strict=True)]
        if render:
            rows = [{**row, source["column"]: render(row[source["column"]])} for row in rows]
        empty = [not row[source["column"]] or not row[source["column"]].strip() for row in rows]
        dropped["empty"] += sum(empty)
        rows = [row for row, e in zip(rows, empty, strict=True) if not e]
        if source.get("hash_split"):
            if heldout is None or group_by:
                raise ValueError("Hash splitting requires a split and ungrouped documents")
            selected = []
            for row in rows:
                key = document_key(row[source["column"]])
                keys = [key]
                if source.get("split_column"):
                    group = row.get(source["split_column"])
                    if not group:
                        dropped["missing_split_group"] = dropped.get("missing_split_group", 0) + 1
                        continue
                    keys.append(document_key(str(group).lower()))
                    if heldout and source.get("heldout_split") == "group":
                        keys = keys[1:]
                # Both content and repository must belong to this partition.
                # This excludes exact cross-source train/validation overlap too.
                if any(partition(k) != heldout for k in keys):
                    dropped["split"] = dropped.get("split", 0) + 1
                    continue
                if key in seen:
                    dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                    continue
                seen.add(key)
                selected.append(row)
            rows = selected
        if not group_by:
            yield [row[source["column"]] for row in rows]
            continue
        for row in rows:
            groups.setdefault(row[group_by], []).append(row)
    if group_by:
        texts = [
            "".join(
                f"### {row[order_by]}\n{row[source['column']]}\n\n"
                for row in sorted(rows, key=lambda r: r[order_by])
            )
            for _, rows in sorted(groups.items())
        ]
        for start in range(0, len(texts), 256):
            yield texts[start : start + 256]


def prepare_shard(
    recipe,
    name,
    index,
    root,
    tokenizer,
    screen,
    eos_id,
    download=hub_download,
    render=None,
    starred=None,
):
    """Tokenize one upstream file into a prepared shard; returns its metadata.

    `render` turns a row's value into text for sources that name a renderer (`render`,
    e.g. chat messages rendered in the model's chat format).
    """
    import pyarrow.parquet as pq

    source = recipe["sources"][name]
    parts = source.get("parts_per_file", 1)
    file_index = index // parts
    if "files" in source:
        file = source["files"][file_index]
    else:
        file = source["pattern"].format(index=file_index + source.get("file_offset", 0))
    path = download(source["repo"], source["revision"], file)
    parquet = JsonLines(path) if str(file).endswith(".jsonl.gz") else pq.ParquetFile(path)
    if parts > parquet.num_row_groups:
        raise ValueError("parts_per_file exceeds the number of Parquet row groups")
    row_groups = [
        int(g) for g in np.array_split(np.arange(parquet.num_row_groups), parts)[index % parts]
    ]
    n = recipe["contamination"]["ngram"]
    min_tokens = source.get("min_tokens", 1)
    documents = []
    dropped = {"license": 0, "empty": 0, "short": 0, "contaminated": 0}
    if source.get("render") and render is None:
        raise ValueError(f"{name} needs its {source['render']} renderer")
    for texts in _text_batches(
        source, parquet, dropped, render, index == heldout_index(source), row_groups, starred
    ):
        for text, encoding in zip(
            texts, tokenizer.encode_batch(texts, add_special_tokens=False), strict=True
        ):
            ids = np.asarray(encoding.ids, dtype=np.uint32)
            if len(ids) < min_tokens:
                dropped["short"] += 1
            elif contaminated(ids, screen, n):
                dropped["contaminated"] += 1
            else:
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
        "upstream": {
            "repo": source["repo"],
            "revision": source["revision"],
            "file": file,
            "row_groups": row_groups,
        },
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
        self.weights = {name: source["weight"] for name, source in recipe["sources"].items()}
        self.loaded = {}

    def _load(self, path):
        return np.fromfile(path, dtype=np.uint32)

    def _tokens(self, name, position):
        order = self.orders[name]
        if position >= len(order) * self.recipe["sources"][name]["max_passes"]:
            raise StopIteration(f"{name} has no shards left")
        index = order[position % len(order)]
        key = (name, index)
        if key not in self.loaded:
            bin_path, _, meta_path = shard_paths(
                source_root(self.recipe, self.root, name), name, index
            )
            start = time.monotonic()
            while not meta_path.exists():
                if time.monotonic() - start > self.wait_seconds:
                    raise TimeoutError(f"Shard {name}/{index} was not prepared in time")
                time.sleep(15)
                if self.refresh:
                    self.refresh()
            # Read fully so no volume file stays open (volume reloads need that).
            current = self.state[name]["shard"]
            keep = {order[p % len(order)] for p in (current, current + 1)}
            self.loaded = {k: v for k, v in self.loaded.items() if k[0] != name or k[1] in keep}
            self.loaded[key] = self._load(bin_path)
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
        name = deficit_source({n: self.state[n]["tokens"] for n in self.names}, self.weights)
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


class ReplayMixture:
    """Batches whose last `replay_rows` rows come from a replay mixture.

    Replay keeps an earlier training objective (for example the pretraining mix) in
    every step, so later training does not trade away what it learned. The primary
    mixture's exhaustion ends the run.
    """

    def __init__(self, primary, replay, replay_rows):
        self.primary, self.replay, self.replay_rows = primary, replay, replay_rows
        self.wait_seconds = replay.wait_seconds

    def next_batch(self, rows):
        if not 0 < self.replay_rows < rows:
            raise ValueError("replay must take some but not all rows of a batch")
        inputs, labels, names = self.primary.next_batch(rows - self.replay_rows)
        replay_inputs, replay_labels, replay_names = self.replay.next_batch(self.replay_rows)
        return (
            np.concatenate((inputs, replay_inputs)),
            np.concatenate((labels, replay_labels)),
            names + replay_names,
        )

    def state_dict(self):
        return {"primary": self.primary.state_dict(), "replay": self.replay.state_dict()}

    def load_state_dict(self, state):
        self.primary.load_state_dict(state["primary"])
        self.replay.load_state_dict(state["replay"])


def heldout_windows(recipe, root, name, context, max_documents):
    """One source's whole held-out documents in one stream of windows (labels -100 only
    past its end): (inputs, labels, total UTF-8 bytes)."""
    source = recipe["sources"][name]
    bin_path, docs_path, _ = shard_paths(
        source_root(recipe, root, name), name, heldout_index(source)
    )
    lengths = np.load(docs_path)[:max_documents]
    count = int(lengths[:, 0].sum())
    tokens = np.fromfile(bin_path, dtype=np.uint32, count=count).astype(np.int64)
    return (*pack_windows(tokens, context), int(lengths[:, 1].sum()))


def pack_windows(stream, context):
    """int64 inputs and labels [rows, context] covering `stream`; padding targets are -100."""
    count = len(stream)
    rows = -(-(count - 1) // context)
    padded = np.full(rows * context + 1, -100, dtype=np.int64)
    padded[:count] = stream
    inputs = padded[:-1].reshape(rows, context).copy()
    labels = padded[1:].reshape(rows, context).copy()
    inputs[inputs < 0] = 0
    return inputs, labels


def wait_for_heldout(recipe, mixture):
    """Wait (up to the mixture's wait_seconds) until every source's held-out shard exists."""
    for name, source in recipe["sources"].items():
        folder = source_root(recipe, mixture.root, name)
        meta = shard_paths(folder, name, heldout_index(source))[2]
        waited = time.monotonic()
        while not meta.exists():
            if time.monotonic() - waited > mixture.wait_seconds:
                raise TimeoutError(f"Held-out shard for {name} was not prepared")
            time.sleep(15)
