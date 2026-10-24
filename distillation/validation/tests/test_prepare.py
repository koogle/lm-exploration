import tempfile
import unittest
import unittest.mock
from pathlib import Path

import helpers  # noqa: F401  (sets sys.path for the project modules below)
import numpy as np
from helpers import chunk_file, record

# isort: split
import prepare
from common import QWEN_VOCAB, TOPK


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
        self.assertEqual(table[30], 2)  # UNK


class CoveringSpellingTest(unittest.TestCase):
    def test_dropped_tokens_are_spelled_from_kept_pieces(self):
        decoder = prepare.byte_decoder()
        vocab = dict(decoder.items()) | {
            "ab": 256,
            "abc": 257,
            "Ġsolve": 258,
            "xyz": 259,
        }
        tokenizer = {
            "model": {"vocab": vocab},
            "added_tokens": [{"id": 300, "content": "<|end|>"}],
        }
        counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
        counts[[256, 258, 300]] = [9, 5, 7]
        kept, pieces, offsets, lengths = prepare.build_covering_vocab(
            counts, 3 + 256 + 3, tokenizer
        )
        strings = {i: s for s, i in vocab.items()} | {300: "<|end|>"}
        for native in (256, 257, 259, 300):
            spelled = "".join(
                strings[int(kept[s - 3])]
                for s in pieces[offsets[native] : offsets[native] + lengths[native]]
            )
            self.assertEqual(spelled, strings[native])
        self.assertEqual(lengths[257], 2)  # "ab" + "c"
        self.assertEqual(lengths[256], 1)


if __name__ == "__main__":
    unittest.main()
