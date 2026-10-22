import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_data.export_raw import export_raw
from training_data.fresh_raw_corpus import FreshRawCorpus, initialize_fresh
from training_data.raw_corpus import RawCorpus
from training_data.raw_token_stream import RawTokenMixture
from training_data.subword import digest

from tests.test_fast_raw_corpus import fixture


class FreshCorpusTests(unittest.TestCase):
    def test_cursor_resume_disjoint_export_and_exact_capacity(self):
        recipe, reader = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = RawCorpus(root / "old", recipe)
            for source in recipe["sources"]:
                old.prepare(source["name"], max_records=6, reader=reader)
            old.close()
            export_raw(root / "old", root / "snapshot")
            old_hash = digest(root / "snapshot/raw.sqlite")
            manifest = root / "snapshot/manifest.json"
            self.assertEqual(initialize_fresh(root / "fresh", manifest), recipe)
            # The new phase changes the mix; the parent recipe keeps the cursors valid.
            weights = {"fineweb_edu": 70, "finemath_4plus": 30}
            fresh = FreshRawCorpus(root / "fresh", recipe, weights, reader=reader, batch_size=3)
            self.assertEqual(
                sum(s["train_loss_tokens"] for s in fresh.report(400)["sources"].values()), 0
            )
            source = recipe["sources"][0]
            self.assertEqual(fresh.cursor(source["name"]), (0, 6))
            fresh._write(source, fresh._fetch(source, fresh.cursor(source["name"])))
            fresh.close()
            # Restart does not turn already staged fresh records into exclusions.
            with patch("training_data.fresh_raw_corpus.digest", wraps=digest) as checksums:
                initialize_fresh(root / "fresh", manifest)
                self.assertEqual([call.args[0] for call in checksums.call_args_list], [manifest])
            fresh = FreshRawCorpus(root / "fresh", recipe, weights, reader=reader, batch_size=3)
            for _ in range(30):
                if fresh.prepare_parallel(400, max_seconds=0)["ready"]:
                    break
            self.assertTrue(fresh.report(400)["ready"])
            frozen = fresh.export(root / "new-snapshot", 400)
            self.assertEqual(frozen["sources"]["fineweb_edu"]["weight_percent"], 70)
            with sqlite3.connect(root / "new-snapshot/raw.sqlite") as new:
                prior = {r[0] for r in fresh.db.execute("select hash from excluded_prior")}
                current = {
                    r[0] for r in new.execute("select hash from documents where split='train'")
                }
                self.assertTrue(current)
                self.assertFalse(prior & current)
                for name, data in frozen["sources"].items():
                    count = new.execute(
                        "select sum(loss_tokens) from documents where split='train' and source=?",
                        (name,),
                    ).fetchone()[0]
                    self.assertEqual(count, data["splits"]["train"]["loss_tokens"])
                    self.assertGreaterEqual(count * 2, 400 * weights[name] / 100)
            loader = RawTokenMixture(
                root / "new-snapshot/manifest.json", context=8, allow_unreviewed=True
            )
            loader.next_batch(2)
            state = loader.state_dict()
            expected = loader.next_batch(2)
            restored = RawTokenMixture(
                root / "new-snapshot/manifest.json", context=8, allow_unreviewed=True
            )
            restored.load_state_dict(state)
            actual = restored.next_batch(2)
            self.assertEqual([a.tolist() for a in actual], [a.tolist() for a in expected])
            loader.close()
            restored.close()
            self.assertEqual(digest(root / "snapshot/raw.sqlite"), old_hash)
            fresh.close()

    def test_additional_prior_hashes_and_conflicts_never_count_as_fresh(self):
        recipe, reader = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = RawCorpus(root / "old", recipe)
            old.close()
            export_raw(root / "old", root / "snapshot")
            extra = RawCorpus(root / "extra", recipe)
            source = recipe["sources"][0]
            extra.prepare(source["name"], max_records=8, reader=reader)
            hashes = {r[0] for r in extra.db.execute("select hash from documents")}
            extra.close()
            initialize_fresh(
                root / "fresh", root / "snapshot/manifest.json", [root / "extra/raw.sqlite"]
            )
            weights = {s["name"]: 50 for s in recipe["sources"]}
            fresh = FreshRawCorpus(root / "fresh", recipe, weights, reader=reader, batch_size=15)
            fresh._write(source, fresh._fetch(source, fresh.cursor(source["name"])))
            actual = fresh.db.execute(
                "select coalesce(sum(loss_tokens),0) from documents d where split='train' and conflict=0 and not exists(select 1 from excluded_prior e where e.hash=d.hash)"
            ).fetchone()[0]
            self.assertEqual(
                fresh.report(400)["sources"][source["name"]]["train_loss_tokens"], actual
            )
            self.assertTrue(hashes)
            row = fresh.db.execute(
                "select hash,loss_tokens from documents d where split='train' and not exists(select 1 from excluded_prior e where e.hash=d.hash) limit 1"
            ).fetchone()
            self.assertIsNotNone(row)
            with fresh.db:
                fresh.db.execute("update documents set conflict=1 where hash=?", (row[0],))
            self.assertEqual(
                fresh.report(400)["sources"][source["name"]]["train_loss_tokens"], actual - row[1]
            )
            fresh.close()

    def test_weights_must_cover_the_recipe(self):
        recipe, reader = fixture()
        with self.assertRaisesRegex(ValueError, "cover every recipe source"):
            FreshRawCorpus("unused", recipe, {"fineweb_edu": 100}, reader=reader)


if __name__ == "__main__":
    unittest.main()
