import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from training_data.export_raw import export_raw
from training_data.fresh_book_pool import BOOK_SOURCES, FreshBookPoolCorpus, allocate_books
from training_data.fresh_raw_corpus import initialize_fresh
from training_data.raw_corpus import RawCorpus
from training_data.raw_token_stream import RawTokenMixture
from training_data.scholarly import adapt


class BookAllocationTests(unittest.TestCase):
    def test_shortfall_keeps_total_coding_and_all_nonbook_budgets(self):
        original = {
            "code_python": 3000,
            "fineweb_edu": 286,
            "libretexts": 86,
            "pressbooks": 86,
            "doab": 114,
            "gutenberg": 143,
            "pre_1929_books": 200,
            "library_of_congress": 85,
        }
        result = allocate_books(original, {"libretexts": 10, "pressbooks": 20})
        self.assertEqual(sum(result.values()), sum(original.values()))
        self.assertEqual(result["code_python"], 3000)
        self.assertEqual(result["fineweb_edu"], 286)
        self.assertEqual(result["libretexts"], 10)
        self.assertEqual(result["pressbooks"], 20)
        self.assertEqual(
            sum(result[n] for n in BOOK_SOURCES), sum(original[n] for n in BOOK_SOURCES)
        )
        self.assertEqual(allocate_books(original, {}), original)
        self.assertEqual(allocate_books(original, {"libretexts": 0})["libretexts"], 0)

    def test_iterative_caps_and_insufficient_total_capacity(self):
        base = {"libretexts": 20, "doab": 30, "gutenberg": 50}
        self.assertEqual(
            allocate_books(base, {"libretexts": 0, "doab": 35}),
            {"libretexts": 0, "doab": 35, "gutenberg": 65},
        )
        with self.assertRaisesRegex(ValueError, "exhausted"):
            allocate_books(base, {"libretexts": 1, "doab": 2, "gutenberg": 3})
        with self.assertRaises(ValueError):
            allocate_books(base, {"code_python": 2})

    def test_exhausted_empty_source_exports_usable_manifest_and_resumes(self):
        recipe = {
            "sources": [
                {
                    "name": n,
                    "repo": "fixture",
                    "revision": "0" * 40,
                    "files": ["a"],
                    "weight_percent": 50,
                }
                for n in ["libretexts", "doab"]
            ]
        }
        rows = []
        for i in range(200):
            row = {
                "id": str(i),
                "text": f"Fresh book {i}. " + "Example educational text. " * 30,
                "metadata": {"book_id": str(i)},
            }
            rows.append(row)

        def reader(source, shard, start):
            if source["name"] == "doab":
                yield from enumerate(rows[start:], start)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = RawCorpus(root / "prior", recipe)
            # Keep a validation record for the emptied source, as real old staging does.
            for i in range(100):
                row = {
                    "id": str(i),
                    "text": "Heldout textbook. " * 30,
                    "metadata": {"book_url": str(i)},
                }
                document, _ = adapt(recipe["sources"][0], row)
                if document["split"] == "validation":

                    def old_reader(source, shard, start, heldout=row):
                        yield from enumerate([heldout][start:], start)

                    prior.prepare("libretexts", reader=old_reader)
                    break
            prior.close()
            export_raw(root / "prior", root / "old")
            initialize_fresh(root / "fresh", root / "old/manifest.json")

            def open_corpus():
                return FreshBookPoolCorpus(
                    root / "fresh",
                    recipe,
                    {"libretexts": Fraction(50), "doab": Fraction(50)},
                    {"libretexts": 100, "doab": 100},
                    reader=reader,
                    batch_size=3,
                )

            corpus = open_corpus()
            for _ in range(200):
                if corpus.prepare_parallel(200, max_seconds=0)["ready"]:
                    break
            status = corpus.report(200)
            self.assertTrue(status["ready"])
            self.assertEqual(
                status["book_allocation"]["allocated_targets"], {"libretexts": 0, "doab": 200}
            )
            corpus.close()
            corpus = open_corpus()
            self.assertEqual(corpus.report(200)["book_allocation"], status["book_allocation"])
            manifest = corpus.export(root / "snapshot")
            self.assertEqual(set(manifest["sources"]), {"doab"})
            self.assertEqual(manifest["sources"]["doab"]["weight_percent"], 100)
            self.assertEqual(manifest["freshness"]["prior_hash_overlap"], 0)
            loader = RawTokenMixture(
                root / "snapshot/manifest.json", context=8, allow_unreviewed=True
            )
            x, _y = loader.next_batch(2)
            self.assertEqual(tuple(x.shape), (2, 8))
            loader.close()
            corpus.close()


if __name__ == "__main__":
    unittest.main()
