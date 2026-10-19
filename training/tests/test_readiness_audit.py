import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from training_data.overnight_corpus import SCHEMA
from training_data.readiness_audit import audit, normalized_hash, shingles


class Encoding:
    def __init__(self, text):
        self.ids = [4] * len(text.split())


class Tokenizer:
    def encode(self, text, **kwargs):
        return Encoding(text)


class ReadinessTests(unittest.TestCase):
    def test_dedup_parent_leakage_benchmarks_capacity_and_heldouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite"
            db = sqlite3.connect(source)
            db.executescript(SCHEMA)

            def add(text, split, parent, count):
                db.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                    (
                        hashlib.sha256(text.encode()).hexdigest(),
                        "code_python",
                        split,
                        parent,
                        json.dumps({"language": "python"}),
                        zlib.compress(text.encode()),
                        count,
                    ),
                )

            add("a = 123", "train", "code_python:Owner/repo", 10)
            add("a=123", "train", "code_python:owner/clone", 10)
            add("validation body", "validation", "code_python:HELD/repo", 20)
            add("leaked train body", "train", "code_python:held/repo", 30)
            benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
            add(benchmark, "train", "code_python:benchmark/repo", 40)
            add("a second document", "train", "code_python:benchmark/repo", 50)
            add("clean validation example", "validation", "code_python:clean/held", 60)
            db.commit()
            db.close()
            with patch("training_data.readiness_audit.load_frozen", return_value=Tokenizer()):
                result = audit(
                    [("prior", source, False)], root / "audit", set(shingles(benchmark)), []
                )
            self.assertEqual(result["capacity"][0]["unique_train_tokens"], 10)
            self.assertEqual(result["benchmark_hit_documents"], 1)
            self.assertEqual(result["blocked_parents"], 2)
            self.assertEqual(result["heldout"]["code/python/unclassified"]["parents"], 1)
            self.assertTrue((root / "audit/heldout-windows.jsonl.gz").exists())
            self.assertEqual(normalized_hash("a = 123"), normalized_hash("a=123"))
