import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from model_100m.model import GPT, ModelConfig
from sampling import report
from sampling.metrics import normalize, repetition, sample_metrics
from sampling.recipe import MODE_DEFAULTS, load_recipe, validate
from sampling.sampler import generate, next_token, run_recipe, sample_model

ROOT = Path(__file__).resolve().parents[1]
FAR = time.time() + 3600
GREEDY = {"name": "greedy", "temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}


def recipe(**changes):
    return {
        "name": "tiny",
        "precision": "fp32",
        "max_new_tokens": 6,
        "prompts": [{"text": "ab", "group": "prose"}, {"text": "ba", "group": "code"}],
        "modes": [
            {"name": "greedy", "temperature": 0.0},
            {"name": "sampled", "temperature": 0.9, "top_k": 4, "seed_offsets": [1000, 2000]},
        ],
        "models": [{"kind": "transformers", "name": "tiny", "repo": "x/y", "revision": "a" * 40}],
        **changes,
    }


def tiny_model():
    torch.manual_seed(0)
    model = GPT(ModelConfig(11, 16, 1, 2, 16)).eval()
    return SimpleNamespace(
        forward=model,
        encode=lambda text: [1, *(3 + ord(c) - ord("a") for c in text)],
        decode=lambda ids: "".join(chr(ord("a") + i % 26) for i in ids),
        eos_token_id=2,
        device="cpu",
    )


class MetricTests(unittest.TestCase):
    def test_repetition(self):
        self.assertEqual(repetition([1, 2, 3], 4), 0.0)
        self.assertEqual(repetition([1, 1, 1, 1, 1], 4), 0.5)
        self.assertEqual(repetition("abcabc", 3), 0.25)
        self.assertEqual(normalize("  A\n\tB  c "), "a b c")

    def test_prefix_metrics_need_full_window(self):
        metrics = sample_metrics("word " * 40, list(range(10)))
        self.assertEqual(metrics["token_rep4"], 0.0)
        self.assertGreater(metrics["word_rep4"], 0.9)
        self.assertIsNotNone(metrics["char_rep16_first128"])
        self.assertIsNone(metrics["char_rep16_first256"])
        self.assertIsNotNone(metrics["word_rep4_first32"])


class DecodingTests(unittest.TestCase):
    def test_penalty_is_sign_aware_and_once_per_identity(self):
        logits = torch.tensor([2.0, -2.0, 1.5])
        token = next_token(logits, torch.tensor([0, 0, 1]), {**GREEDY, "repetition_penalty": 2.0})
        self.assertEqual(token, 2)
        self.assertEqual(logits.tolist(), [1.0, -4.0, 1.5])

    def test_nucleus_keeps_threshold_crossing_token_and_top_k_limits_support(self):
        logits = torch.tensor([0.5, 0.3, 0.2]).log()
        nucleus = {**GREEDY, "temperature": 1.0, "top_p": 0.6}
        top_k = {**GREEDY, "temperature": 1.0, "top_k": 1}
        draws = set()
        for seed in range(200):
            torch.manual_seed(seed)
            draws.add(next_token(logits.clone(), torch.tensor([0]), nucleus))
            self.assertEqual(next_token(logits.clone(), torch.tensor([0]), top_k), 0)
        self.assertEqual(draws, {0, 1})

    def test_tiny_model_generation_is_deterministic_and_bounded(self):
        model = tiny_model()
        sampled = {**GREEDY, "temperature": 1.0, "top_k": 5}
        run = lambda mode, seed: generate(model.forward, [1, 3, 4], mode, seed, 7, None, "cpu")
        self.assertEqual(run(GREEDY, 1), run(GREEDY, 2))
        self.assertEqual(run(sampled, 5), run(sampled, 5))
        self.assertEqual(len(run(sampled, 5)), 7)
        first = run(GREEDY, 1)[0]
        self.assertEqual(generate(model.forward, [1, 3, 4], GREEDY, 1, 7, first, "cpu"), [first])


class RecipeTests(unittest.TestCase):
    def test_study_recipes_are_valid(self):
        paths = [*ROOT.glob("analysis/*/*.json"), *ROOT.glob("analysis/sampling/recipes/*.json")]
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            with self.subTest(recipe=path.name):
                spec = load_recipe(path)
                self.assertEqual(spec["name"], path.stem)

    def test_rejects_unpinned_models_and_unbounded_generation(self):
        model = {"kind": "transformers", "name": "m", "repo": "x/y", "revision": "main"}
        for bad in (
            recipe(models=[model]),
            recipe(max_new_tokens=100000),
            recipe(precision="fp16"),
        ):
            with self.assertRaises(ValueError):
                validate(bad)

    def test_run_records_samples_resumes_and_respects_deadline(self):
        spec = recipe()
        validate(spec)
        opened = []

        def open_runs(model_spec, done):
            if model_spec["name"] not in done:
                opened.append(model_spec["name"])
                yield {"name": model_spec["name"], "revision": model_spec["revision"]}, tiny_model()

        saved = []
        result = run_recipe(spec, open_runs, FAR, save=saved.append)
        samples = result["models"][0]["samples"]
        self.assertEqual(len(samples), 2 * 3)
        self.assertEqual([s["seed"] for s in samples[:3]], [1000, 1000, 2000])
        self.assertEqual(samples[3]["prompt_group"], "code")
        self.assertTrue(all(1 <= s["generated_tokens"] <= 6 for s in samples))
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            run_recipe(spec, open_runs, FAR, existing=result)["models"], result["models"]
        )
        self.assertEqual(opened, ["tiny"])
        with self.assertRaises(TimeoutError):
            sample_model(spec, tiny_model(), time.time() - 1)


class ReportTests(unittest.TestCase):
    def test_report_groups_metrics_and_reference_identity(self):
        def sample(mode, seed, ids, **settings):
            return {
                "prompt": "p",
                "mode": mode,
                "seed": seed,
                "generation_ids": ids,
                "generated_tokens": len(ids),
                "ended_with_eos": ids[-1] == 2,
                "completion": "x " * len(ids),
                "prompt_group": "",
                **MODE_DEFAULTS,
                **settings,
            }

        current = {
            "models": [
                {
                    "name": "new",
                    "revision": "r",
                    "samples": [
                        sample("greedy", 1, [5, 5, 5, 5, 5], temperature=0.0),
                        sample("hot", 1, [5, 6, 7, 8, 2], temperature=0.8, top_k=50),
                        sample("hot", 2, [5, 6, 7, 8, 9], temperature=0.8, top_k=50),
                    ],
                }
            ]
        }
        # Greedy matches regardless of its recorded top-k; sampled modes match by settings.
        earlier = {
            "models": [
                {
                    "name": "old",
                    "revision": "r",
                    "samples": [
                        sample("greedy", 1, [5, 5, 5, 5, 5], top_k=50),
                        sample("sampled_08", 1, [5, 6, 7, 8, 2], temperature=0.8, top_k=50),
                        sample("sampled_08", 2, [5, 6, 0, 0, 0], temperature=0.8, top_k=50),
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "current.json").write_text(json.dumps(current))
            (folder / "earlier.json").write_text(json.dumps(earlier))
            report.main(
                [
                    str(folder / "current.json"),
                    "--reference",
                    str(folder / "earlier.json"),
                    "--group-by",
                    "decoding",
                    "--output-dir",
                    str(folder / "out"),
                ]
            )
            summary = json.loads((folder / "out/summary.json").read_text())
            rows = {row["temperature"]: row for row in summary["tables"][0]["rows"]}
            self.assertEqual(rows[0.0]["token_rep4"], 0.5)
            self.assertEqual(rows[0.8]["n"], 2)
            self.assertEqual(rows[0.8]["eos"], 1)
            self.assertEqual(rows[0.8]["token_identical"], 1)
            self.assertEqual(rows[0.0]["token_identical"], 1)
            divergence = [c["first_divergent_position"] for c in summary["comparisons"]]
            self.assertEqual(divergence, [None, None, 2])
            self.assertIn("| new | 0.8 | 50 |", (folder / "out/metrics-table.md").read_text())
            self.assertEqual((folder / "out/completions.md").read_text().count("## new /"), 3)


if __name__ == "__main__":
    unittest.main()
