import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tokenizer.structured_tokenizer import build, examples, fit, score


class StructuredTests(unittest.TestCase):
    def test_json_and_patches_are_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            for i in range(16):
                text = "# comment\ndef square(x):\n    return x * x\n"
                row = {"parent": str(i), "text": text}
                generated = examples(row)
                json.loads(generated["json"])
                json.loads(generated["tool_call"])
                (root / "src/square.py").write_text(text)
                subprocess.run(
                    ["git", "apply", "--check", "-"],
                    input=generated["git_diff"],
                    text=True,
                    cwd=root,
                    check=True,
                    capture_output=True,
                )

    def test_disjoint_corpus_fit_and_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split, prefix in [("train", "fit"), ("eval", "held")]:
                rows = [
                    {
                        "parent": f"{prefix}{i}",
                        "text": f"def function_{i}(value):\n    return value + {i}\n",
                    }
                    for i in range(120)
                ]
                (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = build(
                root / "train.jsonl",
                root / "eval.jsonl",
                root / "corpus",
                target_bytes=8192,
                eval_parents=8,
            )
            self.assertEqual(manifest["evaluation_parents_per_category"], 8)
            self.assertGreaterEqual(sum(manifest["actual_bytes"].values()), 8192)
            config = Path(__file__).resolve().parents[1] / "tokenizer/simple-llm-v1/config.json"
            fit(root / "corpus", root / "candidate", 512, config)
            result = score(
                root / "candidate/tokenizer.json", root / "corpus/evaluation.jsonl", "test"
            )
            self.assertTrue(result["correctness_passed"])
            self.assertEqual(len(result["metrics"]), 4)
            for metric in result["metrics"].values():
                self.assertEqual(metric["examples"], 8)
            with self.assertRaisesRegex(ValueError, "parent overlap"):
                build(
                    root / "eval.jsonl",
                    root / "eval.jsonl",
                    root / "bad",
                    target_bytes=1000000,
                    eval_parents=120,
                )
