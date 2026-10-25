import json
import tempfile
import unittest
from pathlib import Path

from training_data.math_sft import build, gsm8k_solution, metamath_solution, prepare, prompt
from training_data.subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    load_frozen,
)

TEST = "A farmer has twelve cows and buys seven more cows at the market on a sunny Tuesday morning"


def gsm8k(question, answer):
    return {"question": question, "answer": answer}


def metamath(query, original, response, kind="GSM_Rephrased"):
    return {"query": query, "original_question": original, "response": response, "type": kind}


class MathSftTests(unittest.TestCase):
    def setUp(self):
        self.gsm8k = [
            gsm8k("Ann has 2 pens and gets 3 more. How many?", "2+3=<<2+3=5>>5\n#### 5"),
            gsm8k("Bo reads 4 pages a day for 2 days. Pages?", "4*2=<<4*2=8>>8\n#### 8"),
            gsm8k(TEST + ". How many cows?", "12+7=19\n#### 19"),
        ]
        self.metamath = [
            # A rewrite of the validation problem: dropped from training.
            metamath(
                "Ann got 3 pens after having 2. Total?",
                self.gsm8k[0]["question"],
                "5\nThe answer is: 5",
            ),
            metamath(
                "Bo read 4 pages daily over 2 days; total?",
                self.gsm8k[1]["question"],
                "8\n#### 8\nThe answer is: 8",
            ),
            metamath("Solve x^2=4.", "Solve x^2=4.", "2\nThe answer is: 2", kind="MATH_AnsAug"),
            metamath("No final answer here?", "Q", "just text"),
            # Answer augmentation: the same question with another solution is kept once.
            metamath(
                self.gsm8k[1]["question"],
                self.gsm8k[1]["question"],
                "2*4=8\nThe answer is: 8",
                kind="GSM_AnsAug",
            ),
            metamath(
                self.gsm8k[1]["question"],
                self.gsm8k[1]["question"],
                "2*4=8\nThe answer is: 8",
                kind="GSM_AnsAug",
            ),
        ]

    def test_build_holds_out_validation_and_decontaminates(self):
        with tempfile.TemporaryDirectory() as folder:
            report = build(self.gsm8k, self.metamath, [TEST], Path(folder) / "math", 1)
            rows = [
                json.loads(line)
                for line in (Path(folder) / "math/examples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(report["splits"], {"train": 3, "validation": 1})
            self.assertEqual(
                report["kept"],
                {
                    "gsm8k": 2,
                    "metamath": 2,
                    "contaminated": 1,
                    "validation_rewrites": 1,
                    "duplicates": 1,
                },
            )
            self.assertEqual(report["skipped"]["metamath_not_gsm"], 1)
            by_id = {r["id"]: r for r in rows}
            self.assertEqual(by_id["gsm8k-train-1"]["answer"], "4*2=8\n#### 8")
            self.assertEqual(by_id["metamath-1"]["answer"], "8\n#### 8")
            # A rewrite shares its original problem's group.
            self.assertEqual(by_id["metamath-1"]["group"], by_id["gsm8k-train-1"]["group"])
            splits, digest = prepare(Path(folder) / "math/examples.jsonl", load_frozen(), 256)
            self.assertEqual(digest, report["dataset_sha256"])
            self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 3, "validation": 1})

    def test_only_the_answer_and_eos_are_supervised(self):
        row = {
            "id": "a",
            "group": "g",
            "split": "train",
            "bucket": "gsm8k",
            "question": "Two plus two?",
            "answer": "2+2=4\n#### 4",
            "source": "s",
            "license": "MIT",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text(
                json.dumps(row)
                + "\n"
                + json.dumps(
                    {**row, "id": "b", "group": "h", "split": "validation", "question": "Three?"}
                )
                + "\n"
            )
            tokenizer = load_frozen()
            splits, _ = prepare(path, tokenizer, 256)
            encoded = splits["train"][0]
            prefix = [BOS_TOKEN_ID] + tokenizer.encode(prompt(row), add_special_tokens=False).ids
            self.assertEqual(encoded["x"][: len(prefix)], prefix)
            answer = [t for t in encoded["y"] if t != LOSS_IGNORE_INDEX]
            self.assertEqual(answer[-1], EOS_TOKEN_ID)
            self.assertEqual(tokenizer.decode(answer[:-1]).strip(), "2+2=4\n#### 4")
            self.assertEqual(
                encoded["y"][: len(prefix) - 1], [LOSS_IGNORE_INDEX] * (len(prefix) - 1)
            )
            with self.assertRaisesRegex(ValueError, "exceeds context"):
                prepare(path, tokenizer, 8)

    def test_leaks_and_answers_are_rejected(self):
        row = {
            "id": "a",
            "group": "g",
            "split": "train",
            "bucket": "gsm8k",
            "question": "Same question?",
            "answer": "#### 1",
            "source": "s",
            "license": "MIT",
        }
        cases = [
            ({**row, "id": "b", "split": "validation"}, "Group crosses"),
            ({**row, "id": "b", "group": "h", "split": "validation"}, "Question duplicate"),
            ({**row, "id": "b", "group": "h", "question": "Other?", "answer": "1"}, "####"),
        ]
        for other, message in cases:
            with tempfile.TemporaryDirectory() as folder, self.subTest(message):
                path = Path(folder) / "rows.jsonl"
                path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
                with self.assertRaisesRegex(ValueError, message):
                    prepare(path, load_frozen(), 256)

    def test_solution_formats(self):
        self.assertEqual(gsm8k_solution("1+1=<<1+1=2>>2\n#### 2"), "1+1=2\n#### 2")
        self.assertEqual(metamath_solution("a\n#### 7\nThe answer is: 7"), "a\n#### 7")
        self.assertEqual(metamath_solution("a\nThe answer is: 7"), "a\n#### 7")
        self.assertIsNone(metamath_solution("a"))


if __name__ == "__main__":
    unittest.main()
