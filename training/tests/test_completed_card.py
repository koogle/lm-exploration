"""Model cards use editable Markdown and verified evidence, independent of run shape."""

import json
import tempfile
import unittest
from pathlib import Path

from publication.build_completed_card import DEFAULT_DOCUMENT, build_card, load_document


class CompletedCardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.document = self.root / "card.md"
        self.document.write_text("""+++
audit = "audit.json"
validation = "validation.json"
segments = ["segment.json"]
baseline = "control"
baseline_label = "Untrained control"
displayed_samples = 1
[historical_training]
adam_beta1 = 0.9
+++
# Editable research notes
${parameters} params; ${layers} layers, ${width} wide, ${heads} heads, head dim ${head_dimension}.
${trained_tokens} targets; ${updates} updates; ${segment_count} segments.
${accumulation_rounds} accumulation rounds; ${unused_tokens} unused targets.
Beta: ${adam_beta1}; ${baseline_label}: ${baseline_loss}; final: ${trained_loss}.
${evaluation_policy}
${source_rows}
${sample_text}
""")
        row = {"tokens": 7, "parents": 2, "loss": 2.0, "perplexity": 7.389}
        self.validation = {
            "strict_load": True,
            "per_source_two_pass_limit_verified": True,
            "finite_parameters": True,
            "token_accounting_verified": True,
            "trained_tokens": 80,
            "step": 4,
            "parameters": 200,
            "data_manifest_sha256": "fixture",
            "trained": {"per_source": {"source": row}, "mixture_weighted_loss": 2.0},
            "control": {"per_source": {"source": row}, "mixture_weighted_loss": 4.0},
            "source_exposures": {"source": {"unique_train_tokens": 50}},
            "evaluation": "A documented evaluation policy.",
            "evaluation_device": "cpu",
            "torch_version": "test",
            "pending_data_audits": ["review pending"],
            "samples": [
                {
                    "prompt": "Explain",
                    "completion": "```python\npass\n```",
                    "mode": "greedy",
                    "seed": 17,
                }
            ],
        }
        self.audit = {
            "tokens": 80,
            "updates": 4,
            "total_session_seconds": 10,
            "source_exposures": {"source": 80},
            "data_manifest_sha256": "fixture",
            **dict.fromkeys(
                (
                    "all_metrics_finite",
                    "contiguous_steps",
                    "optimizer_states_finite",
                    "optimizer_step_counters_verified",
                    "session_resume_validation_boundaries_exact",
                ),
                True,
            ),
        }
        self.segment = {
            "total_loss_tokens": 80,
            "steps": 4,
            "parameters": 200,
            "data_identity": {"manifest_sha256": "fixture"},
            "gpu": "test GPU",
            "torch_version": "test",
            "final_checkpoint": "/runs/test/step.pt",
            "final_validation": {"weighted_loss": 2.1},
            "config": {
                "world_size": 1,
                "microbatch": 2,
                "global_windows": 8,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 16,
                "context": 32,
                "target_tokens": 100,
                "seed": 17,
                "weight_decay": 0.2,
                "warmup_tokens": 10,
                "peak_lr": 0.002,
                "min_lr_ratio": 0.2,
                "adam_beta1": 0.75,
            },
        }

    def build(self):
        for name, value in (
            ("audit.json", self.audit),
            ("validation.json", self.validation),
            ("segment.json", self.segment),
        ):
            (self.root / name).write_text(json.dumps(value))
        return build_card(self.root, self.output, self.document)

    def test_markdown_and_measured_settings_drive_card_and_evidence_copy(self):
        self.build()
        card = (self.output / "README.md").read_text()
        for expected in (
            "# Editable research notes",
            "200 params; 2 layers, 16 wide, 2 heads",
            "80 targets; 4 updates; 1 segments",
            "4 accumulation rounds; 20 unused",
            "Beta: 0.75; Untrained control: 4.0000; final: 2.0000",
            "A documented evaluation policy.",
            "````text\n```python",
        ):
            self.assertIn(expected, card)
        self.assertEqual(
            (self.output / "training-segment-00.json").read_bytes(),
            (self.root / "segment.json").read_bytes(),
        )
        self.assertFalse((self.output / "training-segment-01.json").exists())
        self.assertEqual(
            (self.output / "validation.json").read_bytes(),
            (self.root / "validation.json").read_bytes(),
        )

    def test_historical_settings_come_from_markdown_when_report_omits_them(self):
        del self.segment["config"]["adam_beta1"]
        self.document.write_text(self.document.read_text().replace("0.9", "0.85"))
        self.build()
        self.assertIn("Beta: 0.85", (self.output / "README.md").read_text())

    def test_failed_verification_and_inconsistent_evidence_write_nothing(self):
        self.validation["strict_load"] = False
        with self.assertRaisesRegex(ValueError, "strict_load"):
            self.build()
        self.validation["strict_load"] = True
        self.segment["total_loss_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "token counts disagree"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_unknown_placeholder_fails_before_writing(self):
        with self.document.open("a") as handle:
            handle.write("${misspelled_field}\n")
        with self.assertRaises(KeyError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_shipped_document_is_markdown_with_explicit_evidence_references(self):
        settings, template = load_document(DEFAULT_DOCUMENT)
        self.assertTrue(settings["segments"])
        self.assertIn("## Architecture and tokenizer", template.template)
        self.assertIn("${layers}", template.template)
