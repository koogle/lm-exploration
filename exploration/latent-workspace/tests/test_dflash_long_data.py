import json
from dataclasses import replace

import pytest
import torch
from test_dflash import fixture

from latent_workspace.dflash import features
from latent_workspace.dflash_large_data import extend_response, prepare_chunk
from latent_workspace.dflash_train import collate, frozen_memory


def test_online_frozen_features_match_individual_unpadded_prefixes():
    target, draft = fixture()
    tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    memory = frozen_memory(target, tokens, torch.tensor([5, 3]), draft.cfg.layers)
    assert not memory.requires_grad
    for i, length in enumerate((5, 3)):
        with torch.no_grad():
            expected = features(
                target.model(tokens[i : i + 1, :length], output_hidden_states=True),
                draft.cfg.layers,
            )
        torch.testing.assert_close(memory[i : i + 1, :length], expected)
    changed = tokens.clone()
    changed[0, 3:] = 9
    perturbed = frozen_memory(target, changed, torch.tensor([5, 3]), draft.cfg.layers)
    torch.testing.assert_close(memory[0, :3], perturbed[0, :3], rtol=0, atol=0)
    rows = [
        {"tokens": tokens[0], "prompt_length": 1},
        {"tokens": tokens[1, :3], "prompt_length": 1},
    ]
    _ids, actual, anchors = collate(
        rows, 2, 2, torch.Generator().manual_seed(42), "cpu", target=target, layers=draft.cfg.layers
    )
    torch.testing.assert_close(actual, memory)
    assert anchors.shape == (2, 2)
    assert all(p.grad is None for p in target.parameters())


def test_extending_teacher_response_preserves_old_prefix_and_stops_at_eos(monkeypatch):
    from types import SimpleNamespace

    seen = []

    def generate(target, records, count, eos):
        seen.append((records, count))
        return [r["prompt_ids"] + [10] * count for r in records]

    monkeypatch.setattr("latent_workspace.dflash_large_data.generate_batch", generate)
    tokenizer = SimpleNamespace(eos_token_id=30)
    old = {"tokens": torch.tensor([1, 2, 3, 4]), "prompt_length": 2}
    assert extend_response(None, tokenizer, {"prompt_ids": [1, 2]}, old, 6) == [
        1,
        2,
        3,
        4,
        10,
        10,
        10,
        10,
    ]
    assert seen[0][1] == 4
    old["tokens"] = torch.tensor([1, 2, 30])
    assert extend_response(None, tokenizer, {}, old, 6) == [1, 2, 30]
    assert len(seen) == 1


def test_bounded_teacher_chunks_resume_without_regenerating_rows_or_touching_test(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from latent_workspace.dflash_data import load_teacher_cache, token_hash

    _, draft = fixture()
    cfg = replace(
        draft.cfg,
        online_teacher_features=True,
        response_tokens=12,
        train_prompts=6,
        teacher_batch_size=2,
        max_seconds=210,
    )
    seed, manifest, first, second = (tmp_path / s for s in ("seed", "manifest", "first", "second"))
    seed.mkdir()
    manifest.mkdir()
    records, validation = [], []
    for d in ("prose", "code", "translation"):
        for i in range(2):
            number = len(records) + 1
            records.append(
                {
                    "domain": d,
                    "split": "train",
                    "group": f"g{number}",
                    "prompt_ids": [number, number + 10],
                }
            )
            tokens = torch.tensor([number, number + 10] + [number + 200] * 6)
            validation.append(
                {
                    "domain": d,
                    "split": "validation",
                    "group": f"v{number}",
                    "tokens": tokens,
                    "prompt_length": 2,
                    "response_sha256": token_hash(tokens[2:].tolist()),
                    "memory": torch.randn(8, 32),
                }
            )
    old = {**records[0], "tokens": torch.tensor([1, 11, 20, 21, 22, 23]), "prompt_length": 2}
    old["response_sha256"] = token_hash(old["tokens"][2:].tolist())
    torch.save({"train": [old], "validation": validation}, seed / "teacher.pt")
    for folder in (seed, manifest):
        (folder / "config.json").write_text(json.dumps(cfg.to_dict()))
    (manifest / "prompts.json").write_text(
        json.dumps({"records": records + [{"split": "test", "group": "sealed"}]})
    )

    class FakeTarget:
        def cuda(self):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **kw: FakeTarget()
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: SimpleNamespace(eos_token_id=999),
    )
    visited = []

    def generate(target, batch, count, eos):
        visited.extend(r["prompt_ids"][0] for r in batch)
        return [r["prompt_ids"] + [r["prompt_ids"][0] + 100] * count for r in batch]

    monkeypatch.setattr("latent_workspace.dflash_large_data.generate_batch", generate)
    tick = iter(range(0, 1000, 10))
    monkeypatch.setattr("latent_workspace.dflash_large_data.time.monotonic", lambda: next(tick))
    a = prepare_chunk(cfg, first, manifest, seed)
    assert not a["completed"] and 0 < a["cursor"] < 6
    tick = iter(range(0, 1000, 10))
    b = prepare_chunk(cfg, second, manifest, seed, first)
    assert b["completed"] and b["cursor"] == 6
    assert sorted(visited) == list(range(1, 7))  # No retry regenerates committed rows.
    data = load_teacher_cache(second / "teacher.pt")
    assert len(data["train"]) == len(data["validation"]) == 6
    assert all("memory" not in r for r in data["train"])
    torch.testing.assert_close(
        data["train"][0]["tokens"][: len(old["tokens"])], old["tokens"], rtol=0, atol=0
    )
    for actual, original in zip(data["validation"], validation):
        torch.testing.assert_close(actual["tokens"], original["tokens"], rtol=0, atol=0)
        torch.testing.assert_close(actual["memory"], original["memory"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="changes manifest or configuration"):
        prepare_chunk(replace(cfg, response_tokens=16), tmp_path / "bad", manifest, seed, second)
