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

    original_rows = [e for rows in data.values() for e in rows]
    originals = {e["group"]: (e["tokens"].clone(), e["memory"].clone()) for e in original_rows}
    new_record = {"split": "train", "domain": "code", "group": "new", "prompt_ids": [22, 23]}

    def only_new(target, batch, response_tokens, eos_id):
        assert batch == [new_record]  # Neither old labels nor heldouts may be regenerated.
        return [[22, 23, 24, 25, 26, 27]]

    monkeypatch.setattr("latent_workspace.dflash_scaling.generate_batch", only_new)
    expanded, extension_audit = build_batched_teacher(
        cfg,
        target,
        Tokenizer(),
        {"records": records + [new_record]},
        tmp_path,
        float("inf"),
        existing=data,
    )
    assert len(expanded["train"]) == 7 and len(expanded["validation"]) == 6
    assert extension_audit["retained_examples"] == 12
    for e in original_rows:
        expected_tokens, expected_memory = originals[e["group"]]
        torch.testing.assert_close(e["tokens"], expected_tokens, atol=0, rtol=0)
        torch.testing.assert_close(e["memory"], expected_memory, atol=0, rtol=0)


@pytest.mark.parametrize("schedule_steps", [0, 4])
def test_fresh_training_from_teacher_cache_does_not_restore_a_model(
    tmp_path, monkeypatch, schedule_steps
):
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
        schedule_steps=schedule_steps,
        quality_only=bool(schedule_steps),
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
    assert resumed["initial_drafter_sha256"] == report["initial_drafter_sha256"]

    warmed = trainer.run_dflash(
        replace(cfg, warm_start_from=str(tmp_path / "fit")), tmp_path / "warmed"
    )
    assert warmed["curve"][0]["step"] == 0
    expected_validation = {
        k: v for k, v in report["selected_validation"].items() if k != "examples"
    }
    assert warmed["curve"][0]["validation"] == expected_validation
    assert warmed["warm_start"]["step"] == report["best_step"]
    assert warmed["curve"][-1]["supervised_target_exposures"] == 9
    warmed_state = torch.load(tmp_path / "warmed" / "latest.pt", weights_only=True)
    assert all(int(s["step"]) == 1 for s in warmed_state["optimizer"]["state"].values())
    with pytest.raises(ValueError, match="qk_norm"):
        trainer.run_dflash(
            replace(cfg, warm_start_from=str(tmp_path / "fit"), qk_norm=True),
            tmp_path / "incompatible-warm-start",
        )

    if schedule_steps:
        with pytest.raises(ValueError, match="global schedule"):
            trainer.run_dflash(
                replace(cfg, continue_from=str(tmp_path / "fit"), steps=schedule_steps),
                tmp_path / "beyond-schedule",
            )
        uninterrupted = trainer.run_dflash(replace(cfg, steps=2), tmp_path / "uninterrupted")
        assert uninterrupted["verification_passed"] is None
        a = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=True)
        b = torch.load(tmp_path / "uninterrupted" / "latest.pt", weights_only=True)
        for key in a["drafter"]:
            torch.testing.assert_close(a["drafter"][key], b["drafter"][key], atol=0, rtol=0)
        assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
        assert torch.equal(a["sampler_rng"], b["sampler_rng"])

        from latent_workspace.dflash_diagnostics import replay_exposures

        exposures = replay_exposures(data, cfg, 2, a["sampler_rng"])
        assert exposures["sampled_blocks"] == 6
        assert exposures["supervised_new_token_exposures"] == 18
        assert exposures["conditioning_position_exposures"] == 24


def test_expanded_subset_retains_required_training_groups_and_rejects_incompatibility():
    rows = [
        {"domain": d, "group": f"{d}-{i}"}
        for d in ("prose", "code", "translation")
        for i in range(20)
    ]
    required = {f"{d}-19" for d in ("prose", "code", "translation")}
    selected = nested_subset(rows, 12, 42, required)
    assert required <= {r["group"] for r in selected}
    assert all(
        sum(r["domain"] == d for r in selected) == 4 for d in ("prose", "code", "translation")
    )
    assert selected == nested_subset(list(reversed(rows)), 12, 42, required)
    with pytest.raises(ValueError, match="missing"):
        nested_subset(rows, 12, 42, {"not-present"})
    with pytest.raises(ValueError, match="exceed"):
        nested_subset(rows, 3, 42, {"code-18", "code-19"})
