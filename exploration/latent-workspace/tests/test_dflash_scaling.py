from dataclasses import replace

import pytest
import torch
from test_dflash import fixture

from latent_workspace.dflash_scaling import (
    buffer_shuffle,
    build_batched_teacher,
    continuation_slice,
    generate_batch,
    nested_subset,
)


def test_seeded_buffer_shuffle_preserves_all_rows_and_changes_order():
    a = list(buffer_shuffle(range(100), 42, 7))
    assert a == list(buffer_shuffle(range(100), 42, 7))
    assert sorted(a) == list(range(100))
    assert a != list(buffer_shuffle(range(100), 43, 7))


def test_continuation_offsets_are_stable_varied_and_leave_source_suffix():
    ids = list(range(300))
    offsets = []
    for group in range(20):
        piece = continuation_slice(ids, 128, str(group), 42)
        assert piece == continuation_slice(ids, 128, str(group), 42)
        tokens, offset = piece
        assert tokens == ids[offset : offset + 128]
        assert offset + 128 + 8 <= len(ids)
        offsets.append(offset)
    assert len(set(offsets)) > 10


def test_subsets_are_balanced_nested_and_independent_of_cache_order():
    rows = [
        {"domain": d, "group": f"{d}-{i}"}
        for d in ("prose", "code", "translation")
        for i in range(10)
    ]
    small = nested_subset(rows, 6, 42)
    large = nested_subset(rows, 18, 42)
    assert small == nested_subset(list(reversed(rows)), 6, 42)
    assert {e["group"] for e in small} <= {e["group"] for e in large}
    assert all(sum(e["domain"] == d for e in small) == 2 for d in ("prose", "code", "translation"))
    with pytest.raises(ValueError):
        nested_subset(rows, 33, 42)


def test_left_padded_batch_greedy_matches_unpadded_cpu_reference():
    target, _ = fixture()
    records = [{"prompt_ids": [3, 7]}, {"prompt_ids": [4, 8, 9, 10]}]
    actual = generate_batch(target, records, 5, 30)
    for record, result in zip(records, actual):
        expected = generate_batch(target, [record], 5, 30)[0]
        assert result == expected
        assert result[: len(record["prompt_ids"])] == record["prompt_ids"]


def test_batched_cache_excludes_test_and_removes_padding_from_features(tmp_path, monkeypatch):
    target, draft = fixture()
    cfg = replace(draft.cfg, teacher_batch_size=5)
    records = []
    for split in ("train", "validation", "test"):
        for domain in ("prose", "code", "translation"):
            for i in range(2):
                token = len(records) + 1
                records.append(
                    {
                        "split": split,
                        "domain": domain,
                        "group": str(token),
                        "prompt_ids": [token] * (i + 2),
                    }
                )

    def generated(target, batch, response_tokens, eos_id):
        assert all(r["split"] != "test" for r in batch)
        return [r["prompt_ids"] + [r["prompt_ids"][0]] * 4 for r in batch]

    monkeypatch.setattr("latent_workspace.dflash_scaling.generate_batch", generated)

    class Tokenizer:
        eos_token_id = 30

    data, audit = build_batched_teacher(
        cfg, target, Tokenizer(), {"records": records}, tmp_path, float("inf")
    )
    assert set(data) == {"train", "validation"}
    assert len(data["train"]) == len(data["validation"]) == 6
    for rows in data.values():
        for e in rows:
            assert len(e["memory"]) == len(e["tokens"])
            from latent_workspace.dflash import features

            with torch.no_grad():
                expected = features(
                    target.model(e["tokens"][None], output_hidden_states=True), cfg.layers
                )[0]
            torch.testing.assert_close(e["memory"], expected, atol=1e-6, rtol=1e-5)
    assert audit["by_split_domain"]["train"]["code"]["eligible_anchors"] == 2


def test_fresh_training_from_teacher_cache_does_not_restore_a_model(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import latent_workspace.dflash_train as trainer
    from latent_workspace.dflash import features
    from latent_workspace.dflash_scaling import teacher_statistics

    target, draft = fixture()
    parent = tmp_path / "pool"
    parent.mkdir()
    cfg = replace(
        draft.cfg,
        teacher_parent=str(parent),
        train_limit=6,
        steps=1,
        eval_every=1,
        batch_size=3,
        anchors_per_sequence=1,
    )
    data = {"train": [], "validation": []}
    for split, rows in data.items():
        for domain in ("prose", "code", "translation"):
            for i in range(2):
                tokens = torch.tensor([3, 5, 7, 9, 1, 2, 4, 6])
                with torch.no_grad():
                    memory = features(
                        target.model(tokens[None], output_hidden_states=True), cfg.layers
                    )[0]
                rows.append(
                    {
                        "tokens": tokens,
                        "memory": memory,
                        "prompt_length": 4,
                        "group": f"{split}-{domain}-{i}",
                        "domain": domain,
                    }
                )
    torch.save(data, parent / "teacher.pt")
    for name, value in (
        ("config", cfg.to_dict()),
        ("report", {"completed": True}),
        ("teacher-audit", teacher_statistics(data, cfg.block_size)),
        ("prompts", {"records": []}),
    ):
        (parent / f"{name}.json").write_text(json.dumps(value))
    monkeypatch.setattr(trainer.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: target)
    monkeypatch.setattr(
        trainer.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(get_vocab=lambda: {"<|fim_pad|>": 30}, eos_token_id=30),
    )
    monkeypatch.setattr(trainer, "benchmark", lambda *a: [{"identical_to_target_greedy": True}])
    report = trainer.run_dflash(cfg, tmp_path / "fit")
    assert report["training_completed"]
    assert report["curve"][0]["step"] == 0
    assert report["curve"][-1]["step"] == 1
    assert report["curve"][-1]["supervised_target_exposures"] == 9
    assert report["frozen_target_has_no_gradients"]
    assert "parent_run" not in report
    resumed = trainer.run_dflash(
        replace(cfg, continue_from=str(tmp_path / "fit")), tmp_path / "resumed"
    )
    assert resumed["curve"][-1]["step"] == 2
    assert resumed["teacher_sequence_sha256"] == report["teacher_sequence_sha256"]
    assert resumed["parent_run"] == str(tmp_path / "fit")
