"""N-gram hints: lookup against a brute-force count, causality and the model path."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from model_hybrid import hints, model, optim

from tests.test_hybrid_backbone import tiny_config

EOS, BOS = 2, 1


def packed(rng, rows, length, vocab=7, document=(8, 30)):
    """[rows, length] packed BOS ... EOS documents over a small vocabulary, and labels."""
    stream = []
    while len(stream) < rows * length + 1:
        size = rng.integers(*document)
        stream += [BOS, *rng.integers(3, 3 + vocab, size).tolist(), EOS]
    tokens = np.array(stream[: rows * length + 1])
    inputs = tokens[:-1].reshape(rows, length)
    labels = tokens[1:].reshape(rows, length).copy()
    labels[labels == BOS] = hints.IGNORE_INDEX
    return inputs, labels


def contexts(inputs, labels, n):
    """{(row, t): context tuple} for valid contexts with targets, by explicit scan."""
    found = {}
    for row in range(inputs.shape[0]):
        for t in range(n - 1, inputs.shape[1]):
            window = inputs[row, t - n + 1 : t + 1]
            if labels[row, t] != hints.IGNORE_INDEX and EOS not in window[:-1]:
                found[row, t] = tuple(window)
    return found


def brute_force(corpus, inputs, labels=None):
    """Expected (cell, {token: weight}) per hinted flat position."""
    following = {}
    for n in hints.LENGTHS:
        counts = following.setdefault(n, {})
        for (row, t), context in contexts(*corpus, n).items():
            counts.setdefault(context, Counter())[int(corpus[1][row, t])] += 1
    expected = {}
    unlabeled = np.zeros_like(labels)  # lookups never read labels
    queries = {n: contexts(inputs, unlabeled, n) for n in hints.LENGTHS}
    for row in range(inputs.shape[0]):
        for t in range(inputs.shape[1]):
            for bucket, n in reversed(list(enumerate(hints.LENGTHS, 1))):
                found = following[n].get(queries[n].get((row, t)))
                if found:
                    ranked = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
                    total, top = sum(found.values()), ranked[0][1]
                    count_bin = sum(total >= e for e in hints.COUNT_EDGES)
                    purity_bin = sum(top / total >= e for e in hints.PURITY_EDGES)
                    cell = (bucket * (len(hints.COUNT_EDGES) + 1) + count_bin) * (
                        len(hints.PURITY_EDGES) + 1
                    ) + purity_bin
                    kept = ranked[: hints.ENTRIES]
                    share = sum(c for _, c in kept)
                    expected[row * inputs.shape[1] + t] = (cell, {k: c / share for k, c in kept})
                    break
    return expected


def as_dict(found):
    result = {}
    for position, token, weight in zip(found.position, found.token, found.weight):
        cell = int(found.cells.flat[position])
        result.setdefault(int(position), (cell, {}))[1][int(token)] = float(weight)
    return result


class LookupTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        # A tiny vocabulary repeats contexts of every length.
        self.corpus = packed(rng, 12, 64, vocab=3)
        self.query = packed(rng, 3, 64, vocab=3)
        # Half a query row repeated in the corpus gives 20-token matches.
        self.corpus[0][5, 10:42] = self.query[0][1, 10:42]
        self.corpus[1][5, 10:41] = self.query[0][1, 11:42]

    def tables(self):
        index = hints.HintIndex(EOS, flush_steps=1)
        for step, (x, y) in enumerate(zip(*self.corpus)):
            index.add(step, x[None], y[None])
        index.flush()
        return index

    def test_lookup_matches_brute_force(self):
        index = self.tables()
        found, through = index.hints(len(self.corpus[0]), self.query[0])
        self.assertEqual(through, len(self.corpus[0]) - 1)
        expected = brute_force(self.corpus, self.query[0], self.query[1])
        self.assertGreater(len(expected), 20)
        actual = as_dict(found)
        self.assertEqual(actual.keys(), expected.keys())
        self.assertTrue(set(np.flatnonzero(found.cells)) == set(expected))
        for position, (cell, weights) in expected.items():
            self.assertEqual(actual[position][0], cell)
            self.assertEqual(actual[position][1].keys(), weights.keys())
            for token, weight in weights.items():
                self.assertAlmostEqual(actual[position][1][token], weight, places=6)
        buckets = {cell // 42 for cell, _ in expected.values()}
        self.assertEqual(buckets, {1, 2, 3}, "every length bucket is exercised")

    def test_filtered_tables_give_the_same_hints(self):
        index = self.tables()
        merged = {n: hints.Table.merge(index.tables[n]) for n in hints.LENGTHS}
        keys = hints.query_keys([self.query[0]], EOS)
        filtered = {n: [merged[n].filtered(keys[n])] for n in hints.LENGTHS}
        self.assertLess(len(filtered[6][0]), len(merged[6]))
        self.assertEqual(
            hints.lookup(filtered, self.query[0], EOS),
            hints.lookup(index.tables, self.query[0], EOS),
        )

    def test_hints_never_read_the_future(self):
        index = self.tables()
        inputs = self.query[0].copy()
        before, _ = index.hints(99, inputs)
        cut = 40
        # New tokens, documents boundaries included, from position cut + 1 on.
        inputs[:, cut + 1 :] = packed(np.random.default_rng(9), 3, 64 - cut - 1, vocab=3)[0]
        after, _ = index.hints(99, inputs)
        self.assertFalse(np.array_equal(before.cells[:, cut + 1 :], after.cells[:, cut + 1 :]))
        np.testing.assert_array_equal(before.cells[:, : cut + 1], after.cells[:, : cut + 1])
        for found_a, found_b in ((before, after), (after, before)):
            keep = found_a.position % inputs.shape[1] <= cut
            other = found_b.position % inputs.shape[1] <= cut
            np.testing.assert_array_equal(found_a.position[keep], found_b.position[other])
            np.testing.assert_array_equal(found_a.token[keep], found_b.token[other])

    def test_contexts_stop_at_document_and_row_starts(self):
        inputs = np.array([[BOS, 3, 4, 5, 6, 7, 8, EOS, BOS, 3, 4, 5, 6, 7, 8, 9]])
        keys = hints.context_keys(inputs, EOS)
        _, valid = keys[6]
        # A 6-token context ending at t needs t >= 5 and no EOS among x_(t-5) .. x_(t-1).
        self.assertEqual(np.flatnonzero(valid[0]).tolist(), [5, 6, 7, 13, 14, 15])
        context, _ = keys[6]
        self.assertEqual(context[0, 6], context[0, 14], "same tokens, same key")

    def test_index_refuses_steps_it_holds(self):
        index = hints.HintIndex(EOS, flush_steps=2)
        x, y = self.corpus
        index.add(0, x[:1], y[:1])
        # Step 0 is pending, not yet indexed: step 1 sees nothing.
        found, through = index.hints(1, x[1:2])
        self.assertEqual(through, -1)
        self.assertFalse(found.cells.any())
        index.add(1, x[1:2], y[1:2])
        index.flush()
        with self.assertRaises(RuntimeError):
            index.hints(1, x[2:3])
        with self.assertRaises(ValueError):
            index.add(1, x[2:3], y[2:3])
        self.assertEqual(index.hints(2, x[2:3])[1], 1)

    def test_saved_tables_round_trip(self):
        index = self.tables()
        merged = {n: hints.Table.merge(index.tables[n]) for n in hints.LENGTHS}
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "tables.npz"
            hints.save_tables(file, merged, tokens=123)
            loaded = hints.load_tables(file)
        self.assertEqual(
            hints.lookup(loaded, self.query[0], EOS),
            hints.lookup(index.tables, self.query[0], EOS),
        )


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.config = tiny_config(n_layer=4, layer_pattern=("full",), hints=True)
        self.lm = model.HybridLM(self.config).double()
        rng = np.random.default_rng(1)
        inputs, labels = packed(rng, 2, 16, vocab=2, document=(10, 16))
        self.inputs, self.labels = torch.tensor(inputs), torch.tensor(labels)
        index = hints.HintIndex(EOS, flush_steps=1)
        corpus = packed(rng, 40, 16, vocab=2, document=(10, 16))
        for step, (x, y) in enumerate(zip(*corpus)):
            index.add(step, x[None], y[None])
        index.flush()
        self.hints, _ = index.hints(40, inputs)
        self.assertTrue(self.hints.cells.any())

    def test_no_match_equals_no_hints(self):
        empty = hints.lookup({}, self.inputs.numpy(), EOS)
        with torch.no_grad():
            plain = self.lm(self.inputs)
            unmatched = self.lm(self.inputs, hints=empty.to("cpu"))
        torch.testing.assert_close(plain, unmatched, rtol=0, atol=0)
        reference = model.HybridLM(tiny_config(n_layer=4, layer_pattern=("full",))).double()
        reference.load_state_dict(
            {k: v for k, v in self.lm.state_dict().items() if not k.startswith("hint.")}
        )
        with torch.no_grad():
            torch.testing.assert_close(reference(self.inputs), plain, rtol=0, atol=0)

    def test_hint_vector_is_the_weighted_candidate_embedding(self):
        found = self.hints.to("cpu")
        vector = self.lm.hint(found, self.lm.output_weight)
        position = int(found["position"][0])
        cell = int(found["cells"].flatten()[position])
        mine = found["position"] == position
        embedding = self.lm.output_weight[found["token"][mine]]
        expected = (embedding * found["weight"][mine, None].double()).sum(0) * 4.0
        torch.testing.assert_close(vector.flatten(0, 1)[position], expected)
        self.assertGreater(cell, 0)
        unmatched = (found["cells"] == 0).flatten()
        self.assertTrue((vector.flatten(0, 1)[unmatched] == 0).all())

    def test_hints_change_logits_and_train_their_parameters(self):
        found = self.hints.to("cpu")
        self.lm.hint.cell_embedding.data.normal_(0, 0.02)
        losses = self.lm(self.inputs, self.labels, hints=found)
        losses["loss"].backward()
        for name, parameter in self.lm.hint.named_parameters():
            self.assertTrue(parameter.grad.abs().sum() > 0, name)
        with torch.no_grad():
            self.assertFalse(torch.equal(self.lm(self.inputs), self.lm(self.inputs, hints=found)))

    def test_hints_need_a_hinted_model(self):
        plain = model.HybridLM(tiny_config()).double()
        with self.assertRaises(ValueError):
            plain(self.inputs, hints=self.hints.to("cpu"))

    def test_fast_group_for_trust_scales(self):
        roles = {
            name: role
            for role, entries in optim.parameter_groups(self.lm).items()
            for name, _ in entries
        }
        self.assertEqual(roles["hint.next_scale"], "adamw_fast")
        self.assertEqual(roles["hint.site_scales"], "adamw_fast")
        self.assertEqual(roles["hint.cell_embedding"], "adamw_no_decay")
        optimizer = optim.HybridOptimizer(self.lm)
        fast = optimizer.adamw.param_groups[2]
        self.assertAlmostEqual(fast["lr"], 3e-3)
        self.assertEqual(fast["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
