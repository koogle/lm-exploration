import json
import tempfile
import unittest
from pathlib import Path

from test_tokenizer_research import ROOT, fixture
from training_data.scholarly import adapt
from training_data.subword import digest

from tokenizer_evaluation import EvaluationBuilder, rescore
from tokenizer_research import records, train_candidate


class EvaluationTests(unittest.TestCase):
    def test_fitting_chunk_is_rejected_from_new_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, reader, rows = fixture(root / "original")
            original.collect("fineweb_edu", reader=reader)
            original.freeze()
            key, text = original.db.execute(
                'SELECT hash,text FROM chunks WHERE split="train" AND conflict=0 LIMIT 1'
            ).fetchone()
            source = original.recipe["sources"][0]
            row = dict(next(r for r in rows if adapt(source, r)[0]["split"] == "validation"))
            row["text"] = text
            original.close()
            before = digest(root / "original/corpus.sqlite")
            evaluation = EvaluationBuilder(root / "evaluation", root / "original")
            try:
                evaluation.collect(
                    "fineweb_edu", max_records=1, reader=lambda s, f, start: iter([(0, row)])
                )
                self.assertEqual(evaluation.counts("fineweb_edu")["validation"], 0)
                self.assertEqual(
                    evaluation.db.execute(
                        "SELECT conflict FROM chunks WHERE hash=?", (key,)
                    ).fetchone()[0],
                    1,
                )
            finally:
                evaluation.close()
            self.assertEqual(before, digest(root / "original/corpus.sqlite"))

    def test_fitting_corpus_preserved_and_rescore_provenance_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, reader, rows = fixture(root / "original")
            try:
                original.collect("fineweb_edu", reader=reader)
                manifest = original.freeze()
            finally:
                original.close()
            frozen = root / "original/frozen"
            hashes = {p.name: digest(p) for p in frozen.iterdir()}
            candidate = root / "candidate"
            fitted = train_candidate(
                frozen,
                candidate,
                1 / 128,
                512,
                ROOT / "tokenizer/scholarly-code-v1/config.json",
            )
            evaluation = EvaluationBuilder(root / "evaluation", root / "original")
            try:
                source = evaluation.recipe["sources"][0]
                held_out = [r for r in rows if adapt(source, r)[0]["split"] == "validation"]
                evaluation.collect(
                    "fineweb_edu", reader=lambda s, f, start: enumerate(held_out[start:], start)
                )
                balanced = evaluation.freeze()
                self.assertEqual(balanced["sources"]["fineweb_edu"]["train"]["bytes"], 0)
                self.assertLessEqual(
                    evaluation.db.execute(
                        'SELECT MAX(n) FROM (SELECT SUM(bytes) n FROM chunks WHERE split="validation" GROUP BY parent)'
                    ).fetchone()[0],
                    16384,
                )
            finally:
                evaluation.close()
            self.assertEqual(hashes, {p.name: digest(p) for p in frozen.iterdir()})
            self.assertFalse(
                {p for _, _, p in records(frozen, manifest, "train")}
                & {p for _, _, p in records(root / "evaluation/frozen", balanced, "validation")}
            )
            original_result_hash = digest(candidate / "result.json")
            rescored = rescore(candidate, root / "evaluation/frozen")
            self.assertTrue(rescored["correctness_passed"])
            self.assertEqual(rescored["training_seconds"], fitted["training_seconds"])
            self.assertEqual(rescored["original_result_sha256"], original_result_hash)
            self.assertEqual(digest(candidate / "result.json"), original_result_hash)
            balanced["identity"]["spec"]["training_manifest_sha256"] = "wrong"
            (root / "evaluation/frozen/manifest.json").write_text(json.dumps(balanced))
            with self.assertRaisesRegex(ValueError, "different fitting corpus"):
                rescore(candidate, root / "evaluation/frozen")


if __name__ == "__main__":
    unittest.main()
