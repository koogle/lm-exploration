import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "training")]
sys.modules.setdefault("model_10m", __import__("importlib").import_module("10m"))
sys.modules.setdefault(
    "model_10m.model", __import__("importlib").import_module("10m.model")
)

import evalsets
import grader
import jobs
import prepare
import student
from common import QWEN_VOCAB, TOPK


def chunk_file(directory, records):
    path = Path(directory) / "chunk.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def record(tokens, accepted=True, finish="stop"):
    rng = np.random.default_rng(len(tokens))
    ids = [[t] + rng.integers(0, 50, 63).tolist() for t in tokens]
    logprobs = [
        sorted(np.log(rng.dirichlet(np.ones(64)) * 0.99).tolist(), reverse=True)
    ]
    return {
        "id": f"r{tokens[0]}",
        "prompt": "p",
        "text": json.dumps(
            {"problem": "x", "code": "def solve():\n    return 1", "explanation": "e"}
        ),
        "accepted": accepted,
        "finish_reason": finish,
        "token_ids": tokens,
        "targets": {"topk_ids": ids, "topk_logprobs": logprobs * len(tokens)},
    }


class PrepareTest(unittest.TestCase):
    def test_shard_keeps_accepted_stop_records_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = chunk_file(
                tmp,
                [
                    record([5, 6, 7]),
                    record([8, 9], accepted=False),
                    record([10, 11], finish="length"),
                    record([12, 13]),
                ],
            )
            summary = prepare.write_shard(
                [path], Path(tmp) / "shard", keep_records=True
            )
            shard = Path(tmp) / "shard"
            self.assertEqual(summary["documents"], 2)
            np.testing.assert_array_equal(
                np.load(shard / "tokens.npy"), [5, 6, 7, 12, 13]
            )
            np.testing.assert_array_equal(np.load(shard / "offsets.npy"), [0, 3, 5])
            self.assertEqual(np.load(shard / "topk_ids.npy").shape, (5, TOPK))
            np.testing.assert_array_equal(
                np.load(shard / "topk_ids.npy")[:, 0], [5, 6, 7, 12, 13]
            )
            self.assertEqual(np.load(shard / "counts.npy")[12], 1)
            self.assertEqual(len((shard / "records.jsonl").read_text().splitlines()), 2)

    def test_trimmed_vocabulary_maps_rare_ids_to_unknown(self):
        counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
        counts[[10, 20, 30]] = [5, 9, 1]
        table, kept = prepare.build_vocab(counts, 5)
        np.testing.assert_array_equal(kept, [20, 10])
        self.assertEqual(table[20], 3)
        self.assertEqual(table[10], 4)
        self.assertEqual(table[30], student.UNK)


class LossTest(unittest.TestCase):
    def test_soft_cross_entropy_matches_explicit_partition(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 11)
        ids = torch.tensor([[1, 4, -1], [2, 2, 7], [0, 5, 9]])
        lp = torch.log(
            torch.tensor([[0.5, 0.3, 0.0], [0.2, 0.3, 0.1], [0.6, 0.2, 0.15]])
        )
        soft, entropy = student.soft_cross_entropy(logits, ids, lp)
        q = logits.softmax(-1)
        for row in range(3):
            p = {}
            for i, value in zip(ids[row].tolist(), lp[row].exp().tolist()):
                if i >= 0:
                    p[i] = p.get(i, 0) + value
            rest = 1 - sum(p.values())
            q_rest = 1 - sum(q[row, i] for i in p)
            expected = -sum(
                v * math.log(q[row, i]) for i, v in p.items()
            ) - rest * math.log(q_rest)
            h = -sum(v * math.log(v) for v in p.values()) - rest * math.log(rest)
            self.assertAlmostEqual(float(soft[row]), expected, places=5)
            self.assertAlmostEqual(float(entropy[row]), h, places=5)

    def test_kl_is_zero_when_student_matches_teacher(self):
        logits = torch.log(torch.tensor([[0.5, 0.25, 0.125, 0.125]]))
        soft, entropy = student.soft_cross_entropy(
            logits, torch.tensor([[0, 1]]), torch.log(torch.tensor([[0.5, 0.25]]))
        )
        self.assertAlmostEqual(float(soft - entropy), 0.0, places=6)


def tiny_shard(directory, documents):
    rng = np.random.default_rng(1)
    tokens = np.concatenate(documents).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in documents])])
    ids = np.stack(
        [tokens] + [rng.integers(0, 40, tokens.size) for _ in range(TOPK - 1)], 1
    )
    lp = np.full((tokens.size, TOPK), math.log(0.9 / TOPK), dtype=np.float16)
    lp[:, 0] = math.log(0.1)
    shard = Path(directory)
    shard.mkdir(parents=True, exist_ok=True)
    for name, array in {
        "tokens": tokens,
        "offsets": offsets,
        "topk_ids": ids.astype(np.int32),
        "topk_logprobs": lp,
    }.items():
        np.save(shard / f"{name}.npy", array)
    return shard


class StreamTest(unittest.TestCase):
    def test_packing_inserts_bos_and_ignores_next_document_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = tiny_shard(tmp, [np.array([5, 6, 7]), np.array([8, 9])])
            vocab = student.Vocabulary.full(64, bos=1)
            inputs, labels, ids, lps, native = student.pack_shard(shard, vocab)
            np.testing.assert_array_equal(inputs, [1, 5, 6, 7, 1, 8, 9])
            np.testing.assert_array_equal(labels, [5, 6, 7, -100, 8, 9, -100])
            np.testing.assert_array_equal(ids[:, 0], [5, 6, 7, -1, 8, 9, -1])
            self.assertTrue(np.isneginf(lps[3]).all())
            np.testing.assert_array_equal(native, [5, 6, 7, -1, 8, 9, -1])

    def test_stream_windows_cross_shards_without_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = tiny_shard(Path(tmp) / "a", [np.arange(10, 15)])
            b = tiny_shard(Path(tmp) / "b", [np.arange(20, 24)])
            windows = list(
                student.Stream([a, b], student.Vocabulary.full(64, bos=1), 4)
            )
            flat = np.concatenate([w[0] for w in windows])
            np.testing.assert_array_equal(
                flat, [1, 10, 11, 12, 13, 14, 1, 20]
            )  # partial tail dropped


class ModelTest(unittest.TestCase):
    def model(self, vocab=40):
        torch.manual_seed(0)
        config = student.ModelConfig(
            vocab_size=vocab, block_size=64, n_layer=2, n_head=2, n_embd=32
        )
        return student.GPT(config).eval()

    def test_cached_greedy_matches_full_forward_with_left_padding(self):
        model = self.model()
        prompts = [[1, 5, 6, 7, 8], [1, 9]]
        out = student.generate(
            model, prompts, 6, eos=-1, finished=lambda ids: False, device="cpu"
        )
        for prompt, generated in zip(prompts, out):
            sequence = list(prompt)
            for _ in range(6):
                sequence.append(int(model(torch.tensor([sequence]))[0, -1].argmax()))
            self.assertEqual(generated, sequence[len(prompt) :])

    def test_generation_stops_at_eos_and_callback(self):
        model = self.model()
        first = student.generate(
            model, [[1, 5]], 3, eos=-1, finished=lambda ids: False, device="cpu"
        )[0]
        stop = student.generate(
            model, [[1, 5]], 3, eos=first[1], finished=lambda ids: False, device="cpu"
        )[0]
        self.assertEqual(stop, first[: first.index(first[1]) + 1])

    def test_training_reduces_loss_on_repeated_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = tiny_shard(
                Path(tmp) / "s", [np.array([5, 6, 7, 8, 9, 10, 11])] * 600
            )
            vocab = student.Vocabulary.full(40, bos=1)
            settings = {
                "seed": 0,
                "window": 32,
                "size": "small",
                "lr": 3e-3,
                "batch_windows": 4,
                "micro_windows": 2,
                "positions": 4 * 32 * 30,
                "kd_weight": 0.5,
                "log_every": 10,
            }
            student.SIZES["tiny"] = {"n_layer": 1, "n_head": 2, "n_embd": 32}
            settings["size"] = "tiny"
            logs = []
            model, summary = student.train(
                settings, [shard], vocab, tmp, logs.append, device="cpu"
            )
            steps = [entry for entry in logs if entry["event"] == "step"]
            self.assertLess(steps[-1]["ce"], steps[0]["ce"] / 2)
            self.assertEqual(summary["steps"], 30)
            metrics = student.validate(
                model,
                [shard],
                vocab,
                np.ones(64, dtype=np.int64),
                256,
                window=32,
                device="cpu",
            )
            self.assertGreater(metrics["labels"], 0)
            self.assertAlmostEqual(
                metrics["ce_per_byte"] * metrics["bytes"],
                metrics["ce_per_token"] * metrics["labels"],
                places=3,
            )


class EvalSetTest(unittest.TestCase):
    def test_code_string_parsing(self):
        self.assertEqual(
            evalsets.parse_code_string(
                'def solve(x):\\n    return \\"a\\"",\n  "explanation"'
            ),
            'def solve(x):\n    return "a"',
        )
        self.assertIsNone(evalsets.parse_code_string("def solve(x):\\n    return 1"))

    def test_student_prompt_round_trips_through_json(self):
        prompt = (
            evalsets.student_prompt('Say "hi"')
            + 'def solve():\\n    pass", "explanation": "x"}'
        )
        self.assertEqual(json.loads(prompt)["code"], "def solve():\n    pass")

    def test_mbpp_items_rename_tested_function(self):
        row = {
            "task_id": 2,
            "prompt": "Write a function to find the shared elements.",
            "code": "def helper(x):\n    return x\n\ndef similar_elements(a, b):\n    return tuple(set(a) & set(b))",
            "test_imports": [],
            "test_list": ["assert set(similar_elements((3, 4), (5, 4))) == set((4,))"],
        }
        item = evalsets.mbpp_items([row])[0]
        self.assertIn("`def solve(a, b):`", item["problem"])
        self.assertEqual(
            item["tests"], ["assert set(solve((3, 4), (5, 4))) == set((4,))"]
        )
        result = grader.run_case(
            {"key": "k", "code": item["reference"], "setup": "", "tests": item["tests"]}
        )
        self.assertTrue(result["passed"], result)

    def test_valid_tests_keeps_single_line_asserts(self):
        tests = [
            "assert solve(1) == 2",
            "x = 1",
            "assert other(1)",
            "assert solve(\n1)",
            3,
        ]
        self.assertEqual(evalsets.valid_tests(tests), ["assert solve(1) == 2"])

    def test_teacher_json_answer_with_fence(self):
        text = 'Sure\n```json\n{"code": "def solve():\\n    return 1"}\n```'
        self.assertEqual(
            evalsets.parse_json_answer(text, "code"), "def solve():\n    return 1"
        )


class ScoreTest(unittest.TestCase):
    def test_score_counts_passes_per_set_with_interval(self):
        items = [
            {"id": "a", "set": "mbpp"},
            {"id": "b", "set": "mbpp"},
            {"id": "c", "set": "curriculum"},
        ]
        completions = {"a": {"code": "x"}, "b": {"code": None}, "c": {"code": "y"}}
        grades = {"a": {"passed": True}, "c": {"passed": False}}
        scores = jobs.score(items, completions, grades)
        self.assertEqual(scores["mbpp"]["passed"], 1)
        self.assertEqual(scores["mbpp"]["parsed"], 1)
        self.assertEqual(scores["curriculum"]["pass_rate"], 0.0)
        low, high = jobs.wilson(3, 32)
        self.assertAlmostEqual(low, 0.0324, places=3)
        self.assertAlmostEqual(high, 0.2422, places=3)


class GraderTest(unittest.TestCase):
    # Fixed snippets only; model outputs are graded in the network-blocked sandbox.
    def test_pass_fail_error_and_timeout(self):
        cases = [
            {
                "key": "pass",
                "code": "def solve(x):\n    return x + 1",
                "setup": "",
                "tests": ["assert solve(1) == 2"],
            },
            {
                "key": "fail",
                "code": "def solve(x):\n    return x",
                "setup": "",
                "tests": ["assert solve(1) == 2"],
            },
            {
                "key": "error",
                "code": "def solve(x):\n    return y",
                "setup": "",
                "tests": ["assert solve(1) == 2"],
            },
            {
                "key": "exit",
                "code": "import sys\nsys.exit(0)",
                "setup": "",
                "tests": [],
            },
            {"key": "loop", "code": "while True:\n    pass", "setup": "", "tests": []},
        ]
        results = grader.grade(cases)
        self.assertTrue(results["pass"]["passed"])
        for key in ("fail", "error", "exit", "loop"):
            self.assertFalse(results[key]["passed"], key)
        self.assertIn("NameError", results["error"]["error"])


class PollTest(unittest.TestCase):
    def test_unfinished_call_is_pending_not_failed(self):
        import modal
        import modal_app

        class Call:
            def __init__(self, outcome):
                self.outcome = outcome

            def get(self, timeout=None):
                if isinstance(self.outcome, BaseException):
                    raise self.outcome
                return self.outcome

        self.assertEqual(modal_app.poll(Call(TimeoutError())), ("pending", None))
        self.assertEqual(
            modal_app.poll(Call(modal.exception.TimeoutError())), ("pending", None)
        )
        self.assertEqual(modal_app.poll(Call({"ok": 1})), ("done", {"ok": 1}))
        state, _ = modal_app.poll(
            Call(modal.exception.FunctionTimeoutError("child timed out"))
        )
        self.assertEqual(state, "error")
        state, _ = modal_app.poll(Call(modal.exception.InputCancellation("cancelled")))
        self.assertEqual(state, "error")
        state, _ = modal_app.poll(Call(ValueError("boom")))
        self.assertEqual(state, "error")


if __name__ == "__main__":
    unittest.main()
