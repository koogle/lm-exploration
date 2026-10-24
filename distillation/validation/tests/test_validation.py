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

    def test_chunk_cross_entropy_matches_torch_with_ignored_labels(self):
        torch.manual_seed(1)

        class Head:
            lm_head = torch.nn.Linear(8, 13, bias=False)

        features = torch.randn(6, 8)
        labels = torch.tensor([3, -100, 0, 12, -100, 7])
        ids = torch.randint(0, 13, (6, 4))
        lps = torch.log(torch.full((6, 4), 0.2))
        ce, soft, _ = student.chunk_losses(Head, features, labels, ids, lps)
        expected = torch.nn.functional.cross_entropy(
            Head.lm_head(features), labels, ignore_index=-100, reduction="sum"
        )
        self.assertAlmostEqual(float(ce), float(expected), places=4)
        rows, _ = student.soft_cross_entropy(Head.lm_head(features), ids, lps)
        self.assertAlmostEqual(float(soft), float(rows[labels != -100].sum()), places=4)

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


class AdoptionTest(unittest.TestCase):
    def test_restart_adopts_only_its_own_unfailed_calls(self):
        import modal_app

        previous = {
            "supervisor_call_id": "fc-1",
            "calls": {"a": {"id": "fc-a"}, "b": {"id": "fc-b", "error": "boom"}},
        }
        self.assertEqual(modal_app.adoptable(previous, "fc-1"), {"a": {"id": "fc-a"}})
        self.assertEqual(modal_app.adoptable(previous, "fc-2"), {})
        self.assertEqual(modal_app.adoptable({}, "fc-1"), {})


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


class CoveringVocabularyTest(unittest.TestCase):
    def fake_tokenizer(self):
        decoder = prepare.byte_decoder()
        vocab = {char: byte for char, byte in decoder.items()}
        for index, token in enumerate(["ab", "abc", "Ġsolve", "xyz"], start=256):
            vocab[token] = index
        return {
            "model": {"vocab": vocab},
            "added_tokens": [{"id": 300, "content": "<|end|>"}],
        }

    def test_dropped_tokens_are_spelled_from_kept_pieces(self):
        tokenizer = self.fake_tokenizer()
        counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
        counts[[256, 258, 300]] = [9, 5, 7]  # "ab", "Ġsolve", "<|end|>" are frequent
        kept, pieces, offsets, lengths = prepare.build_covering_vocab(
            counts, 3 + 256 + 3, tokenizer
        )
        self.assertEqual(kept.size, 259)
        vocab = student.Vocabulary.covering(kept, pieces, offsets, lengths)
        strings = {i: s for s, i in tokenizer["model"]["vocab"].items()}
        strings[300] = "<|end|>"
        for native in (257, 259, 300):  # "abc" and "xyz" were dropped
            expanded, _ = vocab.expand([native])
            spelled = "".join(strings[i] for i in vocab.decode_native(expanded))
            self.assertEqual(spelled, strings[native])
        self.assertEqual(lengths[257], 2)  # "ab" + "c"
        self.assertEqual(lengths[256], 1)

    def test_packing_expands_targets_and_document_boundaries(self):
        tokenizer = self.fake_tokenizer()
        counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
        counts[[256, 300]] = [9, 7]
        kept, pieces, offsets, lengths = prepare.build_covering_vocab(
            counts, 3 + 256 + 2, tokenizer
        )
        vocab = student.Vocabulary.covering(kept, pieces, offsets, lengths)
        with tempfile.TemporaryDirectory() as tmp:
            # document 1: "abc" (2 pieces), "ab"; document 2: "xyz" (3 pieces)
            shard = tiny_shard(tmp, [np.array([257, 256]), np.array([259])])
            inputs, labels, ids, lps, native = student.pack_shard(Path(shard), vocab)
        abc, _ = vocab.expand([257])
        xyz, _ = vocab.expand([259])
        ab = vocab.encode([256])[0]
        self.assertEqual(inputs.tolist(), [student.BOS, *abc, ab, student.BOS, *xyz])
        self.assertEqual(labels.tolist(), [*abc, ab, -100, *xyz, -100])
        self.assertEqual(native.tolist(), [257, -1, 256, -1, 259, -1, -1, -1])
        self.assertEqual(ids[1, 0], abc[1])  # continuation piece: one-hot on itself
        self.assertEqual(float(lps[1, 0]), 0.0)
        self.assertTrue((ids[1, 1:] == -1).all())
        self.assertEqual(
            ids[0, 0], abc[0]
        )  # teacher's top-1 (the token) maps to its first piece


def llama_reference(tensors, config, tokens):
    """Plain Llama forward (rotate_half RoPE) used only to check the export."""
    from torch.nn import functional as F

    def norm(x, weight):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight

    heads, width = config["num_attention_heads"], config["hidden_size"]
    head_dim = width // heads
    x = tensors["model.embed_tokens.weight"][tokens]
    length = tokens.size(-1)
    frequencies = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(length).float(), frequencies)
    cos, sin = torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)

    def rotate(t):
        half = t[..., : head_dim // 2], t[..., head_dim // 2 :]
        return t * cos + torch.cat([-half[1], half[0]], -1) * sin

    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        h = norm(x, tensors[p + "input_layernorm.weight"])
        q, k, v = (
            F.linear(
                h,
                tensors[p + f"self_attn.{n}_proj.weight"],
                tensors[p + f"self_attn.{n}_proj.bias"],
            )
            .view(length, heads, head_dim)
            .transpose(0, 1)
            for n in "qkv"
        )
        a = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        x = x + F.linear(
            a.transpose(0, 1).reshape(length, width),
            tensors[p + "self_attn.o_proj.weight"],
            tensors[p + "self_attn.o_proj.bias"],
        )
        h = norm(x, tensors[p + "post_attention_layernorm.weight"])
        mlp = {
            n: (tensors[p + f"mlp.{n}_proj.weight"], tensors[p + f"mlp.{n}_proj.bias"])
            for n in ("gate", "up", "down")
        }
        x = x + F.linear(
            F.silu(F.linear(h, *mlp["gate"])) * F.linear(h, *mlp["up"]), *mlp["down"]
        )
    x = norm(x, tensors["model.norm.weight"])
    return x @ tensors["model.embed_tokens.weight"].T


class ExportTest(unittest.TestCase):
    def test_llama_export_reproduces_student_logits(self):
        import export

        torch.manual_seed(0)
        config = student.ModelConfig(
            vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32
        )
        model = student.GPT(config).eval()
        for parameter in model.parameters():  # non-trivial biases and norms
            torch.nn.init.normal_(parameter, std=0.1)
        tokens = torch.randint(0, 50, (12,))
        with torch.no_grad():
            expected = model(tokens[None])[0]
            tensors = export.llama_state(model.state_dict(), config.__dict__)
            got = llama_reference(
                tensors, export.llama_config(config.__dict__, 1, 2), tokens
            )
        self.assertLess(float((got - expected).abs().max()), 1e-4)

    def test_export_writes_a_vllm_checkpoint(self):
        import export

        config = student.ModelConfig(
            vocab_size=40, block_size=32, n_layer=1, n_head=2, n_embd=16
        )
        model = student.GPT(config)
        vocab = student.Vocabulary.full(40, bos=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"model": model.state_dict(), "config": config.__dict__}, path)
            out = export.export(path, vocab, 2, Path(tmp) / "hf")
            written = json.loads((out / "config.json").read_text())
            self.assertEqual(written["architectures"], ["LlamaForCausalLM"])
            self.assertTrue((out / "model.safetensors").exists())
            self.assertEqual(int(np.load(out / "vocab.npz")["bos"]), 1)


if __name__ == "__main__":
    unittest.main()
