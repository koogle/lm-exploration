"""Splitting and contamination guarantees for the bounded coding-data audit."""

import json
import tempfile
import unittest
from pathlib import Path

from training_data.audit_coding_mix import audit, document_key, partition, select_rows
from training_data.packed_shards import build_screen, contaminated

from tests.test_packed_shards import CharTokenizer


class CodingMixAuditTests(unittest.TestCase):
    def test_pending_source_blocks_preparation_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "plan.json"
            plan.write_text(json.dumps({"sources": {"cc_code": {"access_status": "pending"}}}))
            out = Path(folder) / "pending"
            with self.assertRaisesRegex(ValueError, "Source access pending: cc_code"):
                audit(plan, out, 512, 100)
            self.assertFalse(out.exists())

    def test_target_counts_all_parameters_below_one_billion(self):
        import torch
        from model_hybrid.model import HybridConfig, HybridLM

        folder = Path(__file__).resolve().parents[1] / "hybrid" / "configs"
        config = json.loads((folder / "hybrid-960m-model.json").read_text())
        with torch.device("meta"):
            model = HybridLM(HybridConfig(**config))
        total = sum(parameter.numel() for parameter in model.parameters())
        self.assertTrue(900_000_000 < total < 1_000_000_000)
        self.assertIs(model.output_weight, model.token_embedding.weight)
        self.assertIsNotNone(model.mtp)

    def test_normalized_duplicates_cannot_cross_sources_or_splits(self):
        source = {"repo": "source", "revision": "pinned", "file": "part.parquet", "column": "text"}
        seen = set()
        a, _ = select_rows([{"text": "def solve():\r\n    return 1  \r\n"}], source, seen)
        b, dropped = select_rows([{"text": "def solve():\n    return 1\n"}], source, seen)
        self.assertEqual(sum(map(len, a)), 1)
        self.assertEqual(sum(map(len, b)), 0)
        self.assertEqual(dropped["duplicate"], 1)
        key = document_key("def solve():\n    return 1")
        self.assertEqual(a[partition(key)][0]["document_sha256"], key)

    def test_source_license_filter_is_preserved(self):
        source = {
            "repo": "source",
            "revision": "pin",
            "file": "part",
            "column": "code",
            "permissive_licenses_only": True,
        }
        rows = [
            {"code": "x = 1", "license": "mit"},
            {"code": "x = 2", "license": "gpl-3.0"},
            {"code": "x = 3"},
            {"code": "   ", "license": "mit"},
        ]
        splits, dropped = select_rows(rows, source, set())
        self.assertEqual(sum(map(len, splits)), 1)
        self.assertEqual(dropped, {"license": 2, "empty": 1})

    def test_livecodebench_jsonl_screens_prompt_not_test_payload(self):
        tokenizer = CharTokenizer()
        question = "A unique programming question about sorting distinct numbers."
        hidden = "hidden test payload should not enter the text screen"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl"
            path.write_text(
                json.dumps(
                    {"question_content": question, "starter_code": "", "private_test_cases": hidden}
                )
                + "\n"
            )
            recipe = {
                "contamination": {
                    "ngram": 13,
                    "benchmarks": [
                        {
                            "repo": "livecodebench/code_generation_lite",
                            "revision": "pin",
                            "files": ["test.jsonl"],
                            "columns": ["question_content", "starter_code"],
                        }
                    ],
                }
            }
            screen, report = build_screen(recipe, tokenizer, lambda *args: path)
        self.assertTrue(contaminated(tokenizer.encode_batch([question])[0].ids, screen, 13))
        self.assertFalse(contaminated(tokenizer.encode_batch([hidden])[0].ids, screen, 13))
        self.assertEqual(report[0]["texts"], 1)


if __name__ == "__main__":
    unittest.main()
