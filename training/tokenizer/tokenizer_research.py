"""Reproducible corpus and measurements for the tokenizer learning curve."""

import hashlib
import itertools
import json
import math
import random
import resource
import sqlite3
import time
from collections import Counter
from pathlib import Path

from training_data.simple_llm import adapt, canonical, stream_rows
from training_data.subword import digest


def prefix(text, byte_limit):
    return text.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")


def quotas(total, sources):
    result = {s["name"]: total * s["weight_percent"] // 100 for s in sources}
    for name in itertools.islice(itertools.cycle(result), total - sum(result.values())):
        result[name] += 1
    return result


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec["sizes_mib"] != sorted(set(spec["sizes_mib"])) or min(spec["sizes_mib"]) <= 0:
        raise ValueError("Sizes must be positive, unique and increasing")
    if min(spec["vocab_sizes"]) < 260 or spec["evaluation_mib"] <= 0:
        raise ValueError("Invalid vocabulary or evaluation size")
    if not 0 < spec["chunk_bytes"] <= spec["max_document_bytes"] <= spec["max_parent_bytes"]:
        raise ValueError("Invalid document/parent bounds")
    return spec


class CorpusBuilder:
    """Single-writer durable collector; no raw corpus leaves its data volume."""

    def __init__(self, root, spec, recipe):
        self.root, self.spec, self.recipe = Path(root), spec, recipe
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "corpus.sqlite")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS cursors (source TEXT PRIMARY KEY, shard INTEGER, row INTEGER);
            CREATE TABLE IF NOT EXISTS totals (
                source TEXT, split TEXT, bytes INTEGER, PRIMARY KEY(source,split));
            CREATE TABLE IF NOT EXISTS chunks (
                hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT, text TEXT,
                bytes INTEGER, provenance TEXT, conflict INTEGER DEFAULT 0);
            CREATE TRIGGER IF NOT EXISTS chunk_added AFTER INSERT ON chunks
                WHEN NEW.conflict=0 BEGIN
                INSERT INTO totals VALUES (NEW.source,NEW.split,NEW.bytes)
                ON CONFLICT(source,split) DO UPDATE SET bytes=bytes+NEW.bytes;
            END;
            CREATE TRIGGER IF NOT EXISTS chunk_quarantined AFTER UPDATE OF conflict ON chunks
                WHEN OLD.conflict=0 AND NEW.conflict=1 BEGIN
                UPDATE totals SET bytes=bytes-OLD.bytes
                WHERE source=OLD.source AND split=OLD.split;
            END;
            CREATE INDEX IF NOT EXISTS source_split ON chunks(source,split,conflict);
            CREATE INDEX IF NOT EXISTS parent_idx ON chunks(parent);
            CREATE TABLE IF NOT EXISTS decisions (
                source TEXT, reason TEXT, count INTEGER, PRIMARY KEY(source,reason));
        """)
        identity = canonical(
            {
                "spec": spec,
                "recipe": recipe,
                "implementation": digest(__file__),
                "adapters": digest(adapt.__code__.co_filename),
            }
        )
        old = self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()
        if old and old[0] != identity:
            raise ValueError("Corpus preparation changed; use a fresh version")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES (1,?)", (identity,))
        self.db.commit()
        self.identity = json.loads(identity)
        self.targets = {
            "train": quotas(int(max(spec["sizes_mib"]) * 1024**2), recipe["sources"]),
            "validation": quotas(int(spec["evaluation_mib"] * 1024**2), recipe["sources"]),
        }

    def close(self):
        self.db.close()

    def counts(self, name):
        rows = self.db.execute("SELECT split,bytes FROM totals WHERE source=?", (name,))
        return {"train": 0, "validation": 0, **dict(rows)}

    def complete(self, name):
        counts = self.counts(name)
        return all(counts[s] >= q[name] for s, q in self.targets.items())

    def decision(self, source, reason):
        self.db.execute(
            "INSERT INTO decisions VALUES (?,?,1) ON CONFLICT(source,reason) "
            "DO UPDATE SET count=count+1",
            (source, reason),
        )

    def collect(self, name, max_records=2_000_000, reader=stream_rows, checkpoint=lambda: None):
        source = next(s for s in self.recipe["sources"] if s["name"] == name)
        shards = list(source["files"])
        random.Random(f"{self.spec['seed']}:{name}").shuffle(shards)
        shard_start, row_start = self.db.execute(
            "SELECT shard,row FROM cursors WHERE source=?", (name,)
        ).fetchone() or (0, 0)
        scanned = 0
        for shard_index in range(shard_start, len(shards)):
            rows = reader(
                source, shards[shard_index], row_start if shard_index == shard_start else 0
            )
            try:
                for row_index, row in rows:
                    if self.complete(name):
                        self.db.commit()
                        return
                    doc, reason = adapt(source, row, (shards[shard_index], row_index))
                    if doc is not None:
                        if doc["split"] == "test":
                            reason = "excluded_test"
                        elif any(f"<|{s}|>" in doc["text"] for s in ("pad", "bos", "eos", "unk")):
                            reason = "special_token_literal"
                        else:
                            # Whole-document identity is useful for provenance. Chunk identity
                            # also catches repeated passages and whitespace-only variations.
                            parent_bytes = self.db.execute(
                                "SELECT COALESCE(SUM(bytes),0) FROM chunks WHERE parent=?",
                                (doc["parent"],),
                            ).fetchone()[0]
                            text = prefix(
                                doc["text"],
                                min(
                                    self.spec["max_document_bytes"],
                                    max(0, self.spec["max_parent_bytes"] - parent_bytes),
                                ),
                            )
                            reason = "parent_cap" if not text else "sampled"
                            doc_hash = hashlib.sha256(doc["text"].encode()).hexdigest()
                            provenance = {
                                "repo": source["repo"],
                                "revision": source["revision"],
                                "shard": shards[shard_index],
                                "row": row_index,
                                "document_id": doc["document"],
                                "document_sha256": doc_hash,
                                "metadata": doc["provenance"],
                            }
                            while text:
                                chunk = prefix(text, self.spec["chunk_bytes"])
                                text = text[len(chunk) :]
                                key = hashlib.sha256(" ".join(chunk.split()).encode()).hexdigest()
                                old = self.db.execute(
                                    "SELECT split FROM chunks WHERE hash=?", (key,)
                                ).fetchone()
                                if old:
                                    self.decision(name, "duplicate_chunk")
                                    if old[0] != doc["split"]:
                                        self.db.execute(
                                            "UPDATE chunks SET conflict=1 WHERE hash=?", (key,)
                                        )
                                        self.decision(name, "cross_split_quarantine")
                                elif (
                                    self.counts(name)[doc["split"]]
                                    < self.targets[doc["split"]][name]
                                ):
                                    self.db.execute(
                                        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,0)",
                                        (
                                            key,
                                            name,
                                            doc["split"],
                                            doc["parent"],
                                            chunk,
                                            len(chunk.encode()),
                                            canonical(provenance),
                                        ),
                                    )
                    self.decision(name, reason)
                    self.db.execute(
                        "INSERT OR REPLACE INTO cursors VALUES (?,?,?)",
                        (name, shard_index, row_index + 1),
                    )
                    scanned += 1
                    if scanned % 256 == 0:
                        self.db.commit()
                        checkpoint()
                    if scanned % 4096 == 0:
                        print(f"{name}: {scanned} scanned, {self.counts(name)}", flush=True)
                    if scanned >= max_records:
                        self.db.commit()
                        return
            except BaseException:
                self.db.rollback()
                raise
            finally:
                if hasattr(rows, "close"):
                    rows.close()
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO cursors VALUES (?,?,0)", (name, shard_index + 1)
                )
        if not self.complete(name):
            raise ValueError(
                f"{name}: pinned inventory exhausted before quotas; no repetition allowed"
            )

    def freeze(self):
        if not all(self.complete(s["name"]) for s in self.recipe["sources"]):
            raise ValueError("Corpus quotas incomplete; no candidate may train")
        destination = self.root / "frozen"
        destination.mkdir(exist_ok=False)
        manifest = {
            "identity": self.identity,
            "sources": {},
            "sampling_unit": "UTF-8 bytes, not model-training token proportions",
            "limitations": [
                "Seeded shard order with sequential records, not a global reservoir",
                "Parent identity not reconciled across different datasets",
                "Normalized exact-chunk deduplication; no fuzzy near-duplicate guarantee",
                "One sampling seed; candidate differences do not measure seed variance",
                "Fresh candidates share a held-out split; frozen v1 split overlap unknown",
            ],
        }
        for source in self.recipe["sources"]:
            name = source["name"]
            manifest["sources"][name] = {"weight_percent": source["weight_percent"]}
            for split in self.targets:
                path = destination / f"{name}-{split}.jsonl"
                rows = self.db.execute(
                    "SELECT rowid,hash,parent,text,provenance FROM chunks "
                    "WHERE source=? AND split=? AND conflict=0 ORDER BY rowid",
                    (name, split),
                )
                # Encounter order is frozen, so increasing per-source byte limits are nested.
                written, count, parents = 0, 0, set()
                target = self.targets[split][name]
                with path.open("w") as output:
                    for _, key, parent, text, provenance in rows:
                        text = prefix(text, target - written)
                        if not text:
                            break
                        output.write(
                            canonical(
                                {
                                    "text": text,
                                    "hash": key,
                                    "parent": parent,
                                    "provenance": json.loads(provenance),
                                }
                            )
                            + "\n"
                        )
                        written += len(text.encode())
                        count += 1
                        parents.add(parent)
                        if written >= target:
                            break
                if target - written > 3:
                    raise ValueError("Unexpected UTF-8 quota deficit")
                manifest["sources"][name][split] = {
                    "path": path.name,
                    "sha256": digest(path),
                    "bytes": written,
                    "chunks": count,
                    "parents": len(parents),
                    "target_bytes": target,
                }
        manifest["decisions"] = list(
            self.db.execute("SELECT * FROM decisions ORDER BY source,reason")
        )
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def records(folder, manifest, split, byte_limits=None):
    for name, source in manifest["sources"].items():
        remaining = byte_limits[name] if byte_limits else source[split]["bytes"]
        with (Path(folder) / source[split]["path"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                text = prefix(row["text"], remaining)
                if not text:
                    break
                remaining -= len(text.encode())
                yield name, text, row["parent"]
        if remaining > 3:
            raise ValueError(f"{name}: corpus is shorter than the requested sample")


def verify_corpus(folder):
    path = Path(folder) / "manifest.json"
    manifest = json.loads(path.read_text())
    for source in manifest["sources"].values():
        for split in ("train", "validation"):
            entry = source[split]
            if (
                Path(entry["path"]).name != entry["path"]
                or digest(Path(folder) / entry["path"]) != entry["sha256"]
            ):
                raise ValueError("Corpus checksum/path mismatch")
    return manifest


def new_tokenizer(config):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(config["pretokenization_regex"]), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def evaluate(tokenizer, folder, manifest):
    counts, per_source, per_parent = Counter(), {}, {}
    start = time.perf_counter()
    for name, text, parent in records(folder, manifest, "validation"):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        row = per_source.setdefault(
            name,
            {
                "bytes": 0,
                "tokens": 0,
                "chunks": 0,
                "roundtrip_failures": 0,
                "unk_tokens": 0,
                "tokens_per_chunk": [],
            },
        )
        row["bytes"] += len(text.encode())
        row["tokens"] += len(ids)
        row["chunks"] += 1
        row["roundtrip_failures"] += tokenizer.decode(ids, skip_special_tokens=False) != text
        row["unk_tokens"] += ids.count(3)
        row["tokens_per_chunk"].append(len(ids))
        counts.update(ids)
        key = (name, parent)
        totals = per_parent.setdefault(key, [0, 0])
        totals[0] += len(text.encode())
        totals[1] += len(ids)
    elapsed = time.perf_counter() - start
    for row in per_source.values():
        distribution = sorted(row.pop("tokens_per_chunk"))
        row["bytes_per_token"] = row["bytes"] / row["tokens"]
        row["chunk_tokens_p95"] = distribution[math.ceil(len(distribution) * 0.95) - 1]
    total_bytes = sum(r["bytes"] for r in per_source.values())
    total_tokens = sum(r["tokens"] for r in per_source.values())
    probes = [
        "def square(x):\r\n\treturn x ** 2\r\n",
        "∫₀¹ x² dx = ⅓",
        "12345678901234567890",
        "naïve café 中文 العربية 😀",
        "\x00\t \r\n",
        "HTTPRequestHandler snake_case_identifier",
        "e\u0301 ≠ é",
    ]
    probe_results = [
        {
            "text": t,
            "tokens": len(tokenizer.encode(t).ids),
            "roundtrip": tokenizer.decode(tokenizer.encode(t).ids, skip_special_tokens=False) == t,
        }
        for t in probes
    ]
    # Diagnostic fragmentation measured on fixed terms rather than domain-dependent word splitting.
    terms = [
        "electromagnetism",
        "deoxyribonucleic",
        "Schrödinger",
        "HTTPRequestHandler",
        "get_user_configuration",
        "np.linalg.eigh",
        "123456789",
        "0.000001",
        "\\frac{a}{b}",
    ]
    fragmentation = {t: len(tokenizer.encode(t, add_special_tokens=False).ids) for t in terms}
    vocab = tokenizer.get_vocab_size()
    return {
        "sources": per_source,
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "evaluation_seconds": elapsed,
        "evaluation_mib_per_second": total_bytes / 1024**2 / elapsed,
        "vocabulary_used": len(counts),
        "vocabulary_use_fraction": len(counts) / vocab,
        "vocabulary_unseen": vocab - len(counts),
        "vocabulary_used_at_most_10_times": sum(v <= 10 for v in counts.values()),
        "roundtrip_failures": sum(r["roundtrip_failures"] for r in per_source.values()),
        "unk_tokens": sum(r["unk_tokens"] for r in per_source.values()),
        "probes": probe_results,
        "fragmentation": fragmentation,
        "parent_totals": [
            {
                "source": n,
                "parent_sha256": hashlib.sha256(p.encode()).hexdigest(),
                "bytes": v[0],
                "tokens": v[1],
            }
            for (n, p), v in per_parent.items()
        ],
    }


def train_candidate(folder, output, size_mib, vocab_size, config_path, execution="local-test"):
    import tokenizers
    from tokenizers import pre_tokenizers, trainers

    folder, output = Path(folder), Path(output)
    manifest = verify_corpus(folder)
    spec = manifest["identity"]["spec"]
    if size_mib not in spec["sizes_mib"] or vocab_size not in spec["vocab_sizes"]:
        raise ValueError("Candidate is not in the frozen experiment matrix")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config_path).read_text())
    tokenizer = new_tokenizer(config)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=spec["min_frequency"],
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=spec["max_token_length"],
        show_progress=False,
    )
    sources = manifest["identity"]["recipe"]["sources"]
    limits = quotas(int(size_mib * 1024**2), sources)
    actual = Counter()

    def text_iterator():
        for name, text, _ in records(folder, manifest, "train", limits):
            actual[name] += len(text.encode())
            yield text

    start = time.perf_counter()
    tokenizer.train_from_iterator(text_iterator(), trainer)
    duration = time.perf_counter() - start
    tokenizer.save(str(output / "tokenizer.json"))
    metrics = evaluate(tokenizer, folder, manifest)
    result = {
        "size_mib": size_mib,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "actual_training_bytes": dict(actual),
        "training_seconds": duration,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "tokenizer_sha256": digest(output / "tokenizer.json"),
        "corpus_manifest_sha256": digest(folder / "manifest.json"),
        "implementation_sha256": digest(__file__),
        "config_sha256": digest(config_path),
        "tokenizers_version": tokenizers.__version__,
        "execution": execution,
        "metrics": metrics,
    }
    result["correctness_passed"] = (
        metrics["roundtrip_failures"] == 0
        and metrics["unk_tokens"] == 0
        and all(p["roundtrip"] for p in metrics["probes"])
    )
    result["requested_vocabulary_reached"] = tokenizer.get_vocab_size() == vocab_size
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
