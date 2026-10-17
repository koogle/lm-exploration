"""Broaden evaluation parent coverage without changing frozen fitting text."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from tokenizers import Tokenizer
from training_data.scholarly import adapt, stream_rows
from training_data.subword import digest

from tokenizer_research import CorpusBuilder, evaluate


class EvaluationBuilder(CorpusBuilder):
    def __init__(self, root, training_root):
        training_root = Path(training_root)
        original = json.loads((training_root / "frozen/manifest.json").read_text())
        spec = copy.deepcopy(original["identity"]["spec"])
        spec.update(
            purpose="evaluation_only",
            training_bytes_override=0,
            max_document_bytes=16384,
            max_parent_bytes=16384,
            chunk_bytes=8192,
            training_manifest_sha256=digest(training_root / "frozen/manifest.json"),
            evaluation_builder_sha256=digest(__file__),
        )
        super().__init__(root, spec, original["identity"]["recipe"])
        self.targets["train"] = dict.fromkeys(self.targets["train"], 0)
        # Reserve every selected fitting-chunk hash in the dedup table. These
        # zero-byte sentinels are never exported as text or counted as capacity.
        with sqlite3.connect(f"file:{training_root / 'corpus.sqlite'}?mode=ro", uri=True) as old:
            rows = old.execute(
                'SELECT hash,source,parent FROM chunks WHERE split="train" AND conflict=0'
            )
            with self.db:
                self.db.executemany(
                    'INSERT OR IGNORE INTO chunks VALUES (?, ?, "train", ?, "", 0, "{}", 0)', rows
                )


def validation_rows(source, shard, start):
    rows = stream_rows(source, shard, start)
    try:
        for index, row in rows:
            doc, _ = adapt(source, row)
            if doc is not None and doc["split"] == "validation":
                yield index, row
    finally:
        rows.close()


def rescore(candidate_folder, evaluation_folder):
    candidate_folder, evaluation_folder = Path(candidate_folder), Path(evaluation_folder)
    original_path = candidate_folder / "result.json"
    original = json.loads(original_path.read_text())
    tokenizer_path = candidate_folder / "tokenizer.json"
    if digest(tokenizer_path) != original["tokenizer_sha256"]:
        raise ValueError("Candidate tokenizer hash mismatch")
    manifest_path = evaluation_folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["identity"]["spec"]["training_manifest_sha256"]
        != original["corpus_manifest_sha256"]
    ):
        raise ValueError("Evaluation corpus was built against a different fitting corpus")
    for source in manifest["sources"].values():
        entry = source["validation"]
        if (
            Path(entry["path"]).name != entry["path"]
            or digest(evaluation_folder / entry["path"]) != entry["sha256"]
        ):
            raise ValueError("Evaluation corpus checksum/path mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    metrics = evaluate(tokenizer, evaluation_folder, manifest)
    from tokenizers.pre_tokenizers import ByteLevel

    checks = {
        "byte_alphabet_complete": all(
            tokenizer.token_to_id(c) is not None for c in ByteLevel.alphabet()
        ),
        "special_ids_correct": all(
            tokenizer.token_to_id(f"<|{name}|>") == i
            for i, name in enumerate(("pad", "bos", "eos", "unk"))
        ),
        "roundtrips_passed": metrics["roundtrip_failures"] == 0
        and all(p["roundtrip"] for p in metrics["probes"]),
        "zero_unk": metrics["unk_tokens"] == 0,
    }
    result = {
        **original,
        "metrics": metrics,
        "original_result_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "evaluation_manifest_sha256": digest(manifest_path),
        "evaluation_implementation_sha256": digest(__file__),
        "evaluation_resource_requests": {"cpu": 2, "memory_mib": 8192},
        "evaluation_checks": checks,
        "correctness_passed": original["correctness_passed"] and all(checks.values()),
    }
    (candidate_folder / "result-balanced-evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
