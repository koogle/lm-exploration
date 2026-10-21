"""Continuation must preserve optimization state and isolate held-out families."""

import copy
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from training_data.subword import TOKENIZER_SHA256, digest, load_frozen

from distributed.config import DistributedConfig, validate_resume
from trainer.eu_continuation import family, migrate_checkpoint, normalize, split_for


class ContinuationTests(unittest.TestCase):
    def test_one_pass_final_partial_batch_and_resume(self):
        from training_data.raw_token_stream import RawTokenMixture

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = load_frozen()
            db = sqlite3.connect(root / "raw.sqlite")
            db.execute(
                "CREATE TABLE documents (hash TEXT, source TEXT, split TEXT, "
                "text BLOB, loss_tokens INTEGER, conflict INTEGER)"
            )
            total = 0
            for i in range(3):
                text = f"A distinct training document numbered {i}. " * 3
                count = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                total += count
                db.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,0)",
                    (str(i), "eu", "train", zlib.compress(text.encode()), count),
                )
            db.commit()
            db.close()
            manifest = {
                "format": "scholarly-raw-v1",
                "training_ready": False,
                "max_passes": 2,
                "continuation_passes": 1,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "database": {"path": "raw.sqlite", "sha256": digest(root / "raw.sqlite")},
                "sources": {
                    "eu": {"weight_percent": 100, "splits": {"train": {"loss_tokens": total}}}
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            a = RawTokenMixture(root / "manifest.json", context=8, allow_unreviewed=True)
            _x, y = a.next_batch(2)
            first = int((y != -100).sum())
            cursor = a.state_dict()
            b = RawTokenMixture(root / "manifest.json", context=8, allow_unreviewed=True)
            b.load_state_dict(cursor)
            expected = a.next_batch(64)
            actual = b.next_batch(64)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(first + int((actual[1] != -100).sum()), total)
            self.assertEqual(b.exposures["eu"], total)
            committed = b.state_dict()
            with self.assertRaises(StopIteration):
                b.next_batch(32)
            self.assertEqual(committed, b.state_dict())
            self.assertTrue((actual[1][-1] == -100).all())
            a.close()
            b.close()
            from training_data.prefetch import PrefetchedBatches

            def factory():
                return RawTokenMixture(root / "manifest.json", context=8, allow_unreviewed=True)

            prefetched = PrefetchedBatches(factory, 64, depth=2)
            try:
                _x, y = prefetched.next_batch(64)
                self.assertEqual(int((y != -100).sum()), total)
                self.assertEqual(prefetched.state_dict()["loss_token_exposures"]["eu"], 0)
                prefetched.acknowledge()
                self.assertEqual(prefetched.state_dict()["loss_token_exposures"]["eu"], total)
                with self.assertRaises(StopIteration):
                    prefetched.next_batch(64)
            finally:
                prefetched.close()

    def test_language_and_format_families(self):
        row = {
            "source_id": "ecb",
            "document_id": "url-hash",
            "url": "https://example/a",
            "title": "Annual Report 2021",
            "collection_id": "annual_reports",
            "provenance_json": "{}",
        }
        translated = {
            **row,
            "document_id": "different-hash",
            "title": "Informe anual 2021",
            "collection_id": "annual_report_translations",
        }
        self.assertEqual(family(row), family(translated))
        a = {
            **row,
            "source_id": "ema",
            "collection_id": "epar_product_information",
            "url": "https://www.ema.europa.eu/en/documents/product-information/drug-epar-product-information_en.pdf",
        }
        b = {
            **a,
            "url": "https://www.ema.europa.eu/fr/documents/assessment-report/drug-epar-public-assessment-report_fr.pdf",
        }
        self.assertEqual(family(a), family(b))
        self.assertEqual(split_for(family(a)), split_for(family(b)))
        self.assertEqual(normalize(" e\u0301\r\nx\r\n "), "é\nx")

    def test_optimizer_preserving_migration_and_strict_resume(self):
        torch.set_num_threads(1)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.95))
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        config = DistributedConfig(
            world_size=8,
            global_windows=32,
            microbatch=4,
            target_tokens=1000,
            warmup_tokens=1,
            context=8192,
            crop_padding=True,
            cache_rope=True,
            peak_lr=3e-5,
        )
        identity = {"tokenizer_sha256": TOKENIZER_SHA256, "manifest_sha256": "new-data"}
        state = {"positions": {"eu": 0}, "loss_token_exposures": {"eu": 0}}
        loader = SimpleNamespace(
            identity=identity,
            sources={"eu": {"splits": {"train": {"loss_tokens": 1000}}}},
            state_dict=lambda: copy.deepcopy(state),
            close=lambda: None,
        )
        parent = {
            "format": "scholarly-ddp-v1",
            "config": config.as_dict(),
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "tokens": 5000000000,
            "step": 64000,
            "rng": list(range(8)),
            "data_identity": {"tokenizer_sha256": TOKENIZER_SHA256},
            "source_sha256": {"old": "source"},
            "torch_version": str(torch.__version__),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(parent, root / "parent.pt")
            original = (root / "parent.pt").read_bytes()
            with (
                patch("model_100m.model.GPT", return_value=torch.nn.Linear(3, 2)),
                patch("training_data.raw_token_stream.RawTokenMixture", return_value=loader),
            ):
                lineage = migrate_checkpoint(
                    root / "parent.pt", root / "seed.pt", root / "manifest.json", config
                )
            seed = torch.load(root / "seed.pt", weights_only=False)
            self.assertEqual((root / "parent.pt").read_bytes(), original)
            self.assertEqual(seed["tokens"], 0)
            self.assertEqual(seed["step"], 0)
            self.assertEqual(seed["data"], state)
            self.assertEqual(seed["rng"], parent["rng"])
            self.assertEqual(lineage["parent_tokens"], 5000000000)
            for key in parent["model"]:
                torch.testing.assert_close(seed["model"][key], parent["model"][key], rtol=0, atol=0)
            for key, value in parent["optimizer"]["state"].items():
                for field in value:
                    torch.testing.assert_close(
                        seed["optimizer"]["state"][key][field], value[field], rtol=0, atol=0
                    )
            validate_resume(seed, config, identity, seed["source_sha256"], str(torch.__version__))
            with self.assertRaisesRegex(ValueError, "dataset/tokenizer"):
                validate_resume(
                    seed, config, {"wrong": "data"}, seed["source_sha256"], str(torch.__version__)
                )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                migrate_checkpoint(
                    root / "parent.pt", root / "seed.pt", root / "manifest.json", config
                )

    def test_preparation_selects_latest_and_reuses_completed_snapshot(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from trainer.eu_continuation import prepare_snapshot

        parents = {
            split: next(str(i) for i in range(10000) if split_for("ecb:" + str(i)) == split)
            for split in ("train", "validation", "test")
        }

        def row(parent, text, retrieved="2026-10-18", document_id=None):
            return {
                "document_id": document_id or parent,
                "source_id": "ecb",
                "collection_id": "speeches",
                "url": "https://example/" + parent,
                "text": text,
                "retrieved_at": retrieved,
                "provenance_json": json.dumps({"inventory_record": {"source_document_id": parent}}),
            }

        rows = [
            row(parents["train"], "Old discarded revision.", "2026-10-17"),
            row(parents["train"], "The new training document discusses economics."),
            row(parents["validation"], "A separate validation document discusses inflation."),
            row(parents["test"], "The test document discusses public health."),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pq.write_table(pa.Table.from_pylist(rows), root / "part.parquet")
            api = SimpleNamespace(
                list_repo_files=lambda *a, **kw: ["data/part.parquet", "raw/skip.gz"]
            )
            specs = [{"repo": "owner/private-eu", "revision": "immutable"}]
            with (
                patch("huggingface_hub.HfApi", return_value=api),
                patch(
                    "huggingface_hub.hf_hub_download", return_value=str(root / "part.parquet")
                ) as fetch,
            ):
                manifest = prepare_snapshot(root / "snapshot", specs)
                self.assertEqual(manifest["latest_rows"], 3)
                self.assertEqual(manifest["sources"]["eu"]["splits"]["train"]["documents"], 1)
                self.assertEqual(manifest["continuation_passes"], 1)
                self.assertEqual(prepare_snapshot(root / "snapshot", specs), manifest)
                self.assertEqual(fetch.call_count, 1)
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    prepare_snapshot(root / "snapshot", [{**specs[0], "revision": "changed"}])


if __name__ == "__main__":
    unittest.main()
