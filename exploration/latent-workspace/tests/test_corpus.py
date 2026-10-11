import pytest
import torch

from latent_workspace.corpus import DOMAINS, SPLITS, Record, build, record_windows
from latent_workspace.study import nested_indices


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def test_balanced_splits_quarantine_duplicates():
    records = [
        Record("code", split, "duplicate-" + split, "shared duplicate content")
        for split in ("train", "test")
    ]
    records += [
        Record(
            domain,
            split,
            domain + split,
            split + " " + domain + " unique content",
            "source " + split if domain == "translation" else "",
        )
        for split in SPLITS
        for domain in DOMAINS
    ]
    data, meta = build(records, Tokenizer(), 8, 4, dict.fromkeys(SPLITS, 1))
    assert all(rows.shape == (3, 12) for rows in data.values())
    assert len(meta["quarantined_duplicate_hashes"]) == 1
    assert not set(meta["window_hashes"]["train"]) & set(meta["window_hashes"]["test"])


def test_translation_labels_are_target_only():
    record = Record("translation", "train", "pair", "German answer", "English source")
    windows = list(record_windows(record, Tokenizer(), 128, 4))
    target = list((" " + record.content).encode())
    assert windows[0][0] == -1
    assert windows[0][-4:] == target[:4]
    assert windows[1][-8:] == target[:8]


def test_learning_curve_subsets_are_balanced_and_nested():
    ranges = {domain: (i * 12, (i + 1) * 12) for i, domain in enumerate(DOMAINS)}
    subsets = nested_indices(ranges, (6, 18, 36), 42)
    assert set(subsets[6].tolist()) < set(subsets[18].tolist()) < set(subsets[36].tolist())
    for size, subset in subsets.items():
        for start, end in ranges.values():
            assert int(((subset >= start) & (subset < end)).sum()) == size // 3
    torch.testing.assert_close(subsets[18], nested_indices(ranges, (18,), 42)[18])


@pytest.mark.parametrize("mode", ("diffusion:multi", "diffusion:jspace", "scan", "compact"))
def test_study_runs_all_sizes_without_evaluating_test(tmp_path, monkeypatch, mode):
    from dataclasses import replace
    from types import SimpleNamespace

    from test_backbone import tiny_backbone
    from test_core import tiny_config

    from latent_workspace import study

    cfg = replace(
        tiny_config(),
        corpus="mixed",
        train_examples=12,
        eval_examples=6,
        test_examples=6,
        study_sizes=(6,) if mode == "compact" else (6, 12),
        study_checkpoints=(1, 2),
        variants=(mode if mode not in ("scan", "compact") else "diffusion:multi",),
        study_scan_only=mode == "scan",
        scan_attention=mode == "scan",
        scan_steps=1,
        jspace_prompts=1,
        jspace_top_k=5,
        jspace_dim_batch=4,
        batch_size=2,
        max_seconds=60,
    )
    data = {
        "train": torch.randint(31, (12, 28)),
        "validation": torch.randint(31, (6, 28)),
        "test": torch.arange(6 * 28).reshape(6, 28) + 99,
    }
    ranges = {
        split: {d: [i * len(rows) // 3, (i + 1) * len(rows) // 3] for i, d in enumerate(DOMAINS)}
        for split, rows in data.items()
    }

    def factory(config, device):
        backbone = tiny_backbone()
        backbone.cfg = config
        backbone.revision = "test"
        backbone.tokenizer = SimpleNamespace(eos_token_id=30, decode=lambda ids: str(ids))
        return backbone

    monkeypatch.setattr(study, "FrozenBackbone", factory)
    monkeypatch.setattr(study, "prepare", lambda *a: (data, {"domain_ranges": ranges}))
    result = study.run_study(cfg, tmp_path)
    assert result["completed"]
    assert "no test losses" in result["test_status"]
    if mode == "scan":
        assert set(result["layer_scan"]["layers"]) == {"1", "2", "attention1", "attention2"}
        assert result["curves"] == {}
    if mode == "diffusion:jspace":
        assert len(result["calibration_indices"]) == 1
        assert torch.load(tmp_path / "size-12-best.pt", weights_only=True)["jacobians"] is not None
    for curve in result["curves"].values():
        assert len(curve["checkpoints"]) == 2
        assert curve["checkpoints"][-1]["masked_supervised_tokens_seen"] <= 16
        assert set(curve["validation_samples"]) == set(DOMAINS)


def test_expansion_preserves_frozen_rows_and_subset_order():
    from latent_workspace.corpus import extend_frozen_corpus

    def corpus(per_domain):
        rows = torch.tensor([[d * 100 + i, i] for d in range(3) for i in range(per_domain)])
        ranges = {d: [i * per_domain, (i + 1) * per_domain] for i, d in enumerate(DOMAINS)}
        meta = {
            "sources": {"immutable": "revision"},
            "domain_ranges": {"train": ranges},
            "window_groups": {
                "train": [f"g{x[0]}" for x in rows.tolist()],
                "validation": ["held-v"],
                "test": ["held-t"],
            },
            "window_hashes": {
                "train": [f"h{x[0]}" for x in rows.tolist()],
                "validation": ["hash-v"],
                "test": ["hash-t"],
            },
        }
        return {"train": rows, "validation": torch.tensor([[8]]), "test": torch.tensor([[9]])}, meta

    old, old_meta = corpus(2)
    fresh, fresh_meta = corpus(4)
    data, meta, priority_maps = extend_frozen_corpus(
        {"tensors": old, "metadata": old_meta}, (fresh, fresh_meta), 12
    )
    priority = {d: list(reversed(list(mapping.values()))) for d, mapping in priority_maps.items()}
    subsets = nested_indices(meta["domain_ranges"]["train"], (6, 12), 42, priority)
    expected = torch.cat([old["train"][i * 2 : (i + 1) * 2].flip(0) for i in range(3)])
    torch.testing.assert_close(data["train"][subsets[6]], expected)
    for split in ("validation", "test"):
        assert data[split] is old[split]
    assert old_meta["domain_ranges"]["train"]["code"] == [2, 4]
    fresh_meta["window_hashes"]["train"][3] = "hash-t"
    import pytest

    with pytest.raises(ValueError, match="leaks"):
        extend_frozen_corpus({"tensors": old, "metadata": old_meta}, (fresh, fresh_meta), 12)


def test_different_documents_with_same_windows_do_not_inflate_data_size():
    # Full-document fingerprints differ, but their retained token windows match.
    records = [
        Record(
            domain,
            split,
            domain + split + str(i),
            f"{domain}{split} " * 16 + str(i),
            f"source-{split}-{i}" if domain == "translation" else "",
        )
        for split in SPLITS
        for domain in DOMAINS
        for i in range(3)
    ]
    # Translation sources differ; only prose/code duplicates are relevant here.
    import pytest

    with pytest.raises(ValueError, match="insufficient disjoint"):
        build(records, Tokenizer(), 8, 4, dict.fromkeys(SPLITS, 5))


def test_expansion_cleans_legacy_duplicate_and_held_out_windows():
    from latent_workspace.corpus import extend_frozen_corpus

    def pack(rows, count):
        data = {
            "train": torch.tensor(rows),
            "validation": torch.tensor([[20, 1]]),
            "test": torch.tensor([[99, 1]]),
        }
        meta = {
            "sources": {},
            "domain_ranges": {
                "train": {d: [i * count, (i + 1) * count] for i, d in enumerate(DOMAINS)}
            },
            "window_groups": {
                "train": [f"train-{i}" for i in range(len(rows))],
                "validation": ["valid"],
                "test": ["test"],
            },
            "window_hashes": {
                "train": [f"hash-{i}" for i in range(len(rows))],
                "validation": ["valid-hash"],
                "test": ["test-hash"],
            },
        }
        return data, meta

    old, meta = pack([[10, 1], [10, 1], [20, 1], [21, 1], [30, 1], [31, 1]], 2)
    fresh = pack([[d * 10 + i, 1] for d in (1, 2, 3) for i in range(4)], 4)
    data, audit, _ = extend_frozen_corpus({"tensors": old, "metadata": meta}, fresh, 9)
    assert len(set(map(tuple, data["train"].tolist()))) == 9
    assert not set(map(tuple, data["train"].tolist())) & set(
        map(tuple, data["validation"].tolist())
    )
    assert audit["parent_training_cleanup"]["duplicate_windows_removed"] == 1
    assert audit["parent_training_cleanup"]["held_out_overlapping_windows_removed"] == 1
    assert data["validation"] is old["validation"]


@pytest.mark.parametrize("dropout,null", [(0.0, False), (0.25, True)])
def test_study_continuation_matches_uninterrupted_updates(tmp_path, monkeypatch, dropout, null):
    from dataclasses import replace
    from types import SimpleNamespace

    from test_backbone import tiny_backbone
    from test_core import tiny_config

    from latent_workspace import study

    cfg = replace(
        tiny_config(),
        corpus="mixed",
        layer_dropout=dropout,
        null_attention=null,
        train_examples=12,
        eval_examples=6,
        test_examples=6,
        study_sizes=(12,),
        study_checkpoints=(1, 2),
        variants=("diffusion:multi",),
        batch_size=2,
        max_seconds=60,
    )
    data = {
        "train": torch.randint(31, (12, 28)),
        "validation": torch.randint(31, (6, 28)),
        "test": torch.arange(6 * 28).reshape(6, 28) + 99,
    }
    ranges = {
        s: {d: [i * len(rows) // 3, (i + 1) * len(rows) // 3] for i, d in enumerate(DOMAINS)}
        for s, rows in data.items()
    }

    def factory(config, device):
        backbone = tiny_backbone()
        backbone.cfg, backbone.revision = config, "test"
        backbone.tokenizer = SimpleNamespace(eos_token_id=30, decode=lambda ids: str(ids))
        return backbone

    monkeypatch.setattr(study, "FrozenBackbone", factory)
    monkeypatch.setattr(study, "prepare", lambda *a: (data, {"domain_ranges": ranges}))
    study.run_study(replace(cfg), tmp_path / "direct")
    study.run_study(replace(cfg, study_checkpoints=(1,)), tmp_path / "parent")
    result = study.run_study(
        replace(cfg, study_parent=str(tmp_path / "parent"), study_resume=str(tmp_path / "parent")),
        tmp_path / "resumed",
    )
    direct = torch.load(tmp_path / "direct/size-12.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/size-12.pt", weights_only=True)
    assert result["completed"] and resumed["step"] == 2
    for key in direct["head"]:
        torch.testing.assert_close(direct["head"][key], resumed["head"][key], atol=0, rtol=0)
