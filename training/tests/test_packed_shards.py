"""Hybrid data preparation, contamination screen and the prepared-shard loader."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from training_data import packed_shards as data

EOS = 99


class CharTokenizer:
    """One id per character, enough to exercise preparation without GLM files."""

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode_batch(self, texts, add_special_tokens=False):
        return [self.Encoding([ord(c) % 90 for c in text]) for text in texts]


def recipe(shards=3):
    return {
        "name": "test-recipe",
        "context": 8,
        "seed": 7,
        "sources": {
            "web": {
                "weight": 0.75,
                "repo": "w",
                "revision": "r",
                "pattern": "web-{index}.parquet",
                "shards": shards,
                "column": "text",
                "max_passes": 1,
            },
            "code": {
                "weight": 0.25,
                "repo": "c",
                "revision": "r",
                "pattern": "code-{index}.parquet",
                "shards": shards,
                "column": "code",
                "permissive_licenses_only": True,
                "max_passes": 1,
            },
        },
        "contamination": {"ngram": 13, "benchmarks": []},
    }


class HybridDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.recipe = recipe()
        for i in range(3):
            web = [f"web document {i}-{j} " * (j % 4 + 1) for j in range(30)] + ["   "]
            pq.write_table(pa.table({"text": web}), self.root / f"web-{i}.parquet")
            code = [f"def f{i}_{j}(): return {j}" for j in range(20)]
            licenses = ["mit" if j % 2 else "gpl-3.0" for j in range(20)]
            pq.write_table(
                pa.table({"code": code, "license": licenses}), self.root / f"code-{i}.parquet"
            )
        self.download = lambda repo, revision, file: self.root / file
        self.tokenizer = CharTokenizer()

    def tearDown(self):
        self.folder.cleanup()

    def prepare(self, screen=None):
        screen = np.empty(0, np.uint64) if screen is None else screen
        return [
            data.prepare_shard(
                self.recipe, name, i, self.root / "out", self.tokenizer, screen, EOS, self.download
            )
            for name in ("web", "code")
            for i in range(3)
        ]

    def test_language_path_and_size_filters(self):
        rows = {
            "code": ["x = 1", "<p>page</p>", "y = 2", "z = " + "3" * 40, "w = 4"],
            "license": ["mit"] * 5,
            "language": ["Python", "HTML", "Python", "Python", "Java"],
            "path": ["a.py", "index.html", "vendor/lib/b.py", "c.py", "D.java"],
        }
        pq.write_table(pa.table(rows), self.root / "filtered-0.parquet")
        base = {
            "weight": 1.0,
            "repo": "c",
            "revision": "r",
            "pattern": "filtered-{index}.parquet",
            "shards": 2,
            "column": "code",
            "permissive_licenses_only": True,
            "exclude_path": "(^|/)vendor/",
            "max_bytes": 20,
            "max_passes": 1,
        }
        expected = {
            "python": ({"languages": ["Python"]}, 1, {"language": 2, "path": 1, "size": 1}),
            "other": ({"exclude_languages": ["HTML", "Python"]}, 1, {"language": 4}),
        }
        for name, (selection, documents, dropped) in expected.items():
            self.recipe["sources"] = {name: {**base, **selection}}
            meta = data.prepare_shard(
                self.recipe,
                name,
                0,
                self.root / "out",
                self.tokenizer,
                np.empty(0, np.uint64),
                EOS,
                self.download,
            )
            self.assertEqual(meta["documents"], documents)
            self.assertEqual({k: meta["dropped"].get(k, 0) for k in dropped}, dropped)

    def test_starred_sources_and_json_lines(self):
        import gzip
        import json

        rows = [
            {"text": f"value_{i} = {i}", "repository": repo, "path": f"f{i}.py"}
            for i, repo in enumerate(["Org/Good", "org/other", "org/good", "x/y"])
        ]
        with gzip.open(self.root / "projects-0.jsonl.gz", "wt") as out:
            out.writelines(json.dumps(row) + "\n" for row in rows)
        pq.write_table(
            pa.table({"repo_name": ["org/good", "a/b", "c/d"], "stars": [12.0, 3.0, None]}),
            self.root / "stars.parquet",
        )
        self.recipe["stars"] = {
            "repo": "s",
            "revision": "r",
            "files": ["stars.parquet"],
            "name_column": "repo_name",
            "count_column": "stars",
            "minimum": 10,
        }
        hashes, report = data.build_stars(self.recipe, self.download)
        self.assertEqual(report, {"minimum": 10, "rows": 3, "repositories": 1})
        base = {
            "weight": 1.0,
            "repo": "c",
            "revision": "r",
            "files": ["projects-0.jsonl.gz", "projects-0.jsonl.gz"],
            "shards": 2,
            "column": "text",
            "split_column": "repository",
            "max_passes": 1,
        }
        self.recipe["sources"] = {"projects": {**base, "exclude_repositories": ["ORG/good"]}}
        meta = data.prepare_shard(
            self.recipe,
            "projects",
            0,
            self.root / "out-excluded",
            self.tokenizer,
            np.empty(0, np.uint64),
            EOS,
            self.download,
        )
        self.assertEqual((meta["documents"], meta["dropped"]["repository"]), (2, 2))
        for wanted, documents in ((True, 2), (False, 2)):
            self.recipe["sources"] = {"projects": {**base, "starred": wanted}}
            meta = data.prepare_shard(
                self.recipe,
                "projects",
                0,
                self.root / f"out-{wanted}",
                self.tokenizer,
                np.empty(0, np.uint64),
                EOS,
                self.download,
                starred=frozenset(hashes.tolist()),
            )
            self.assertEqual((meta["documents"], meta["dropped"]["stars"]), (documents, 2))

    def test_sources_repeat_up_to_max_passes(self):
        self.prepare()
        self.recipe["sources"]["code"]["max_passes"] = 2
        mixture = data.PreparedMixture(self.recipe, self.root / "out")
        tokens = sum(
            len(np.fromfile(data.shard_paths(self.root / "out", "code", i)[0], dtype=np.uint32))
            for i in data.training_order(self.recipe, "code")
        )
        first = mixture._take("code", tokens)
        mixture.state["code"]["shard"] = len(mixture.orders["code"])
        np.testing.assert_array_equal(mixture._take("code", tokens), first)
        mixture.state["code"]["shard"] = 2 * len(mixture.orders["code"])
        with self.assertRaises(StopIteration):
            mixture._take("code", 1)

    def test_repository_documents(self):
        rows = {
            "code": ["b = 2", "a = 1", "z = 26", "solo"],
            "license": ["mit", "mit", "mit", "mit"],
            "repo_name": ["r1", "r1", "r2", "r3"],
            "path": ["b.py", "a.py", "z.py", "s.py"],
        }
        pq.write_table(pa.table(rows), self.root / "repo-12.parquet")
        source = {
            "weight": 1.0,
            "repo": "c",
            "revision": "r",
            "pattern": "repo-{index}.parquet",
            "file_offset": 10,
            "shards": 3,
            "column": "code",
            "permissive_licenses_only": True,
            "group_by": "repo_name",
            "order_by": "path",
            "min_tokens": 20,
            "max_passes": 1,
        }
        self.recipe["sources"] = {"repos": source}
        meta = data.prepare_shard(
            self.recipe,
            "repos",
            2,
            self.root / "out",
            self.tokenizer,
            np.empty(0, np.uint64),
            EOS,
            self.download,
        )
        # r1 joins its files in path order; r2 and r3 are shorter than 20 tokens.
        self.assertEqual((meta["documents"], meta["dropped"]["short"]), (1, 2))
        self.assertEqual(meta["upstream"]["file"], "repo-12.parquet")
        expected = "### a.py\na = 1\n\n### b.py\nb = 2\n\n"
        self.assertEqual(meta["tokens"], len(expected) + 1)
        self.assertEqual(meta["bytes"], len(expected.encode()))

    def test_content_and_repository_splits_across_files(self):
        texts = [f"unique text {i}" for i in range(200)]
        repos = [f"repo-{i}" for i in range(200)]
        table = pa.table({"text": texts + texts, "repo": repos + repos})
        pq.write_table(table, self.root / "split.parquet")
        source = {"column": "text", "hash_split": True, "split_column": "repo"}
        partitions = []
        for heldout in (False, True):
            dropped = {"empty": 0}
            found = [
                text
                for batch in data._text_batches(
                    source, pq.ParquetFile(self.root / "split.parquet"), dropped, heldout=heldout
                )
                for text in batch
            ]
            self.assertEqual(len(found), len(set(found)))
            self.assertGreater(len(found), 0)
            self.assertGreater(dropped["duplicate"], 0)
            for text in found:
                i = texts.index(text)
                self.assertEqual(data.partition(data.document_key(text)), heldout)
                self.assertEqual(data.partition(data.document_key(repos[i])), heldout)
            partitions.append(set(found))
        self.assertFalse(partitions[0] & partitions[1])

    def test_capacity_rejects_incomplete_and_insufficient_shards(self):
        from workflows.checks import check_hybrid_capacity

        self.prepare()
        root = self.root / "out"
        check_hybrid_capacity(self.recipe, root, 16, 8)
        with self.assertRaisesRegex(ValueError, "needs"):
            check_hybrid_capacity(self.recipe, root, 10**9, 8)
        binary = data.shard_paths(root, "code", 0)[0]
        binary.write_bytes(binary.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check_hybrid_capacity(self.recipe, root, 16, 8)

    def test_row_group_parts_cover_upstream_once(self):
        pq.write_table(
            pa.table({"text": [f"row-{i}" for i in range(24)]}),
            self.root / "parts-0.parquet",
            row_group_size=4,
        )
        self.recipe["sources"] = {
            "web": {
                **self.recipe["sources"]["web"],
                "pattern": "parts-{index}.parquet",
                "shards": 3,
                "parts_per_file": 3,
            }
        }
        metas = [
            data.prepare_shard(
                self.recipe,
                "web",
                i,
                self.root / "out",
                self.tokenizer,
                np.empty(0, np.uint64),
                EOS,
                self.download,
            )
            for i in range(3)
        ]
        self.assertEqual(sum(m["documents"] for m in metas), 24)
        self.assertEqual([m["upstream"]["row_groups"] for m in metas], [[0, 1], [2, 3], [4, 5]])

    def test_sources_prepared_under_another_recipe(self):
        self.prepare()
        other = dict(self.recipe, name="other-recipe")
        other["sources"] = {
            "web": dict(self.recipe["sources"]["web"], prepared_in="out", weight=1.0)
        }
        mixture = data.PreparedMixture(other, self.root / "other-recipe")
        name, window = mixture.next_window()
        self.assertEqual(name, "web")
        first = data.training_order(other, "web")[0]
        tokens = np.fromfile(data.shard_paths(self.root / "out", "web", first)[0], np.uint32)
        np.testing.assert_array_equal(window, tokens[: len(window)])

    def test_rendered_message_documents(self):
        messages = [
            [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]
            for i in range(3)
        ]
        pq.write_table(pa.table({"messages": messages}), self.root / "chat-1.parquet")
        source = {
            "weight": 1.0,
            "repo": "s",
            "revision": "r",
            "files": ["x", "chat-1.parquet"],
            "shards": 2,
            "column": "messages",
            "render": "test",
            "max_passes": 1,
        }
        self.recipe["sources"] = {"chat": source}
        args = (self.recipe, "chat", 1, self.root / "out", self.tokenizer, np.empty(0, np.uint64))
        with self.assertRaisesRegex(ValueError, "renderer"):
            data.prepare_shard(*args, EOS, self.download)
        render = lambda m: "".join(f"<{t['role']}>{t['content']}" for t in m)
        meta = data.prepare_shard(*args, EOS, self.download, render=render)
        self.assertEqual(meta["documents"], 3)
        self.assertEqual(meta["bytes"], 3 * len("<user>q0<assistant>a"))

    def test_ngram_hashes_and_screen(self):
        ids = list(range(20))
        hashes = data.ngram_hashes(ids, 13)
        self.assertEqual(len(hashes), 8)
        self.assertEqual(len(data.ngram_hashes(ids[:12], 13)), 0)
        screen = np.unique(data.ngram_hashes(list(range(5, 18)), 13))
        self.assertTrue(data.contaminated(ids, screen, 13))
        self.assertFalse(data.contaminated(list(range(100, 130)), screen, 13))

    def test_prepared_shards(self):
        metas = self.prepare()
        web = metas[0]
        self.assertEqual(web["documents"], 30)
        self.assertEqual(web["dropped"]["empty"], 1)
        self.assertEqual(metas[3]["dropped"]["license"], 10)
        bin_path, docs_path, meta_path = data.shard_paths(self.root / "out", "web", 0)
        tokens = np.fromfile(bin_path, dtype=np.uint32)
        lengths = np.load(docs_path)
        self.assertEqual(len(tokens), lengths[:, 0].sum())
        ends = np.cumsum(lengths[:, 0]) - 1
        self.assertTrue((tokens[ends] == EOS).all())
        self.assertEqual((tokens == EOS).sum(), 30)
        self.assertEqual(json.loads(meta_path.read_text())["tokens"], len(tokens))
        self.assertTrue(metas[2]["heldout"])

    def test_contaminated_documents_are_dropped(self):
        # Only document 0-17 contains "ment 0-17 web" (13 characters).
        ids = self.tokenizer.encode_batch(["ment 0-17 web"])[0].ids
        metas = self.prepare(np.unique(data.ngram_hashes(ids, 13)))
        self.assertEqual(metas[0]["dropped"]["contaminated"], 1)
        self.assertEqual(metas[0]["documents"], 29)

    def test_mixture_weights_order_and_resume(self):
        self.prepare()
        mixture = data.PreparedMixture(self.recipe, self.root / "out")
        self.assertEqual(sorted(mixture.orders["web"]), [0, 1])
        inputs, labels, names = mixture.next_batch(8)
        self.assertEqual(inputs.shape, (8, 8))
        np.testing.assert_array_equal(inputs[:, 1:], labels[:, :-1])
        self.assertEqual(names.count("web"), 6)
        state = json.loads(json.dumps(mixture.state_dict()))
        expected = mixture.next_batch(4)
        resumed = data.PreparedMixture(self.recipe, self.root / "out")
        resumed.load_state_dict(state)
        actual = resumed.next_batch(4)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[2], expected[2])

    def test_stream_is_contiguous_across_shards(self):
        self.prepare()
        mixture = data.PreparedMixture(self.recipe, self.root / "out")
        stream = np.concatenate(
            [
                np.fromfile(data.shard_paths(self.root / "out", "code", i)[0], dtype=np.uint32)
                for i in mixture.orders["code"]
            ]
        )
        windows = []
        while True:
            try:
                name, window = mixture.next_window()
            except StopIteration:
                break
            if name == "code":
                windows.append(window)
        rebuilt = np.concatenate([windows[0]] + [w[1:] for w in windows[1:]])
        np.testing.assert_array_equal(rebuilt, stream[: len(rebuilt)])

    def test_missing_shard_waits_then_fails(self):
        mixture = data.PreparedMixture(self.recipe, self.root / "out", wait_seconds=0)
        with self.assertRaises(TimeoutError):
            mixture.next_window()

    def test_heldout_windows(self):
        self.prepare()
        _, labels, total_bytes = data.heldout_windows(self.recipe, self.root / "out", "web", 8, 5)
        lengths = np.load(data.shard_paths(self.root / "out", "web", 2)[1])[:5]
        self.assertEqual(total_bytes, lengths[:, 1].sum())
        self.assertEqual((labels != -100).sum(), lengths[:, 0].sum() - 1)


if __name__ == "__main__":
    unittest.main()


PRE, SUF, MID = 95, 96, 97
FIM = {
    "rate": 1.0,
    "spm_rate": 0.0,
    "sources": ["code"],
    "seed": 3,
    "tokens": {"prefix": PRE, "suffix": SUF, "middle": MID},
}


def undo_fim(piece):
    """The original segment of a rearranged one. SPM, and PSM with an empty prefix,
    start `PRE SUF`: everything after MID precedes the suffix."""
    piece = list(piece)
    if piece[:2] == [PRE, SUF]:
        middle_at = piece.index(MID)
        return piece[middle_at + 1 :] + piece[2:middle_at]
    suffix_at, middle_at = piece.index(SUF), piece.index(MID)
    return piece[1:suffix_at] + piece[middle_at + 1 :] + piece[suffix_at + 1 : middle_at]


class FimTests(unittest.TestCase):
    def documents(self):
        lengths = np.array([6, 1, 30, 3], dtype=np.int64)
        tokens = np.concatenate(
            [
                np.append(np.arange(1, n, dtype=np.uint32) + 10 * i, EOS)
                for i, n in enumerate(lengths)
            ]
        )
        return tokens, lengths

    def test_psm_segments_restore_their_documents(self):
        tokens, lengths = self.documents()
        out, new = data.fim_rearrange(tokens, lengths, FIM, np.random.default_rng(0), 10)
        self.assertEqual(int(new.sum()), len(out))
        start = 0
        for original, size in zip(np.split(tokens, np.cumsum(lengths)[:-1]), new):
            document = list(out[start : start + size])
            start += size
            self.assertEqual(document[-1], EOS)
            body = document[:-1]
            pieces, restored = [], []
            for i, token in enumerate(body):
                if token == PRE and i:
                    pieces.append(i)
            for piece in np.split(np.array(body), pieces) if body else []:
                restored += undo_fim(piece)
            self.assertEqual(restored, list(original[:-1]))
            # every segment of at most 10 tokens became one FIM example (3 extra tokens)
            segments = -(-(len(original) - 1) // 10) if len(original) > 2 else 0
            self.assertEqual(size, len(original) + 3 * segments)

    def test_spm_moves_the_suffix_first(self):
        tokens = np.append(np.arange(1, 9, dtype=np.uint32), EOS)
        fim = dict(FIM, spm_rate=1.0)
        out, _ = data.fim_rearrange(tokens, np.array([9]), fim, np.random.default_rng(1), 100)
        body = list(out[:-1])
        self.assertEqual(body[:2], [PRE, SUF])
        middle_at = body.index(MID)
        tail, head_and_middle = body[2:middle_at], body[middle_at + 1 :]
        self.assertEqual(head_and_middle + tail, list(range(1, 9)))

    def test_rate_zero_keeps_the_shard(self):
        tokens, lengths = self.documents()
        out, new = data.fim_rearrange(
            tokens, lengths, dict(FIM, rate=0.0), np.random.default_rng(0), 10
        )
        np.testing.assert_array_equal(out, tokens)
        np.testing.assert_array_equal(new, lengths)


class FimMixtureTests(unittest.TestCase):
    setUp, tearDown, prepare = (
        HybridDataTests.setUp,
        HybridDataTests.tearDown,
        HybridDataTests.prepare,
    )

    def mixture(self, fim=FIM):
        return data.PreparedMixture(self.recipe, self.root / "out", fim=fim)

    def test_fim_stream_is_seeded_and_resumes(self):
        self.prepare()
        first, second = self.mixture(), self.mixture()
        np.testing.assert_array_equal(first.next_batch(6)[0], second.next_batch(6)[0])
        state = json.loads(json.dumps(first.state_dict()))
        expected = first.next_batch(5)
        resumed = self.mixture()
        resumed.load_state_dict(state)
        np.testing.assert_array_equal(resumed.next_batch(5)[0], expected[0])

    def test_other_sources_are_unchanged(self):
        self.prepare()
        plain, fim = self.mixture(None), self.mixture()
        for _ in range(4):
            a, b = plain.next_window(), fim.next_window()
            self.assertEqual(a[0], b[0])
            if a[0] == "web":
                np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(
            any(PRE in w for n, w in (fim.next_window() for _ in range(8)) if n == "code")
        )

    def test_continue_state_starts_the_next_document(self):
        self.prepare()
        plain = self.mixture(None)
        while plain.state["code"]["offset"] in (0,) or plain.state["code"]["shard"] > 0:
            plain.next_window()
            if plain.state["code"]["shard"] > 0:
                self.skipTest("first code shard ended before a mid-document cursor")
        raw = plain.state["code"]["offset"]
        fim = self.mixture()
        state = fim.continue_state(plain.state_dict())
        self.assertEqual(state["sources"]["web"], plain.state_dict()["sources"]["web"])
        index = fim.orders["code"][0]
        lengths = np.load(data.shard_paths(self.root / "out", "code", index)[1])[:, 0]
        starts = np.concatenate(([0], np.cumsum(lengths)))
        document = int(np.searchsorted(starts, raw))
        tokens, new = fim._shard("code", 0)
        self.assertEqual(state["sources"]["code"]["offset"], int(new[:document].sum()))
        self.assertEqual(tokens[state["sources"]["code"]["offset"]], PRE)
        with self.assertRaises(ValueError):
            fim.continue_state(dict(plain.state_dict(), recipe="other"))
