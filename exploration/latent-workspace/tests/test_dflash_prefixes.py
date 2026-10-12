import pytest
import torch

from latent_workspace.corpus import digest, repository_split
from latent_workspace.dflash_data import PromptRegistry, load_teacher_cache
from latent_workspace.dflash_prefixes import training_prose_prefixes


def source_for(split):
    return next(
        digest(str(i)) for i in range(1000) if repository_split(f"prose:{digest(str(i))}") == split
    )


def canonical(source_hash, split="train", ids=None):
    return {
        "domain": "prose",
        "group": f"prose:{source_hash}",
        "source_hash": source_hash,
        "split": split,
        "prompt_ids": list(range(64)) if ids is None else ids,
    }


def extras(source_hash=None, **kwargs):
    return list(
        training_prose_prefixes(
            list(range(2048)),
            source_hash or source_for("train"),
            context=64,
            seed=42,
            existing_offsets=[0],
            **kwargs,
        )
    )


def test_prefixes_are_deterministic_nonoverlapping_and_keep_article_identity():
    rows = extras()
    assert rows == extras()
    assert len(rows) == 3  # Four total including the retained prompt.
    offsets = [0] + [r["source_token_offset"] for r in rows]
    assert all(abs(a - b) >= 64 for i, a in enumerate(offsets) for b in offsets[i + 1 :])
    for row in rows:
        offset = row["source_token_offset"]
        assert row["source_group"] == f"prose:{row['source_hash']}"
        assert row["group"] == f"{row['source_group']}:prefix:{offset}"
        assert row["split"] == "train"
        assert row["prompt_ids"] == list(range(offset, offset + 64))
        assert offset + 64 + 8 <= 2048


@pytest.mark.parametrize("split", ["validation", "test"])
def test_heldout_article_never_becomes_training_via_new_example_id(split):
    assert extras(source_for(split)) == []


@pytest.mark.parametrize("split", ["validation", "test"])
def test_registry_rejects_unregistered_hash_holdout_mislabeled_as_training(split):
    source_hash = source_for(split)
    record = {
        **canonical(source_hash),
        "group": f"prose:{source_hash}:prefix:128",
        "source_group": f"prose:{source_hash}",
        "source_token_offset": 128,
    }
    assert not PromptRegistry(allow_training_prefixes=True).add(record)


def test_historical_holdout_is_quarantined_even_when_hash_assignment_is_train():
    source_hash = source_for("train")
    assert extras(source_hash, excluded_groups={f"prose:{source_hash}"}) == []


def test_short_articles_and_exhausted_prefix_limit_are_skipped():
    assert extras(max_prefixes=1) == []
    assert list(training_prose_prefixes(range(70), source_for("train"), context=64, seed=42)) == []
    with pytest.raises(ValueError):
        list(training_prose_prefixes(range(70), source_for("train"), context=0, seed=42))
    with pytest.raises(ValueError, match="nonnegative integers"):
        list(
            training_prose_prefixes(
                range(300), source_for("train"), context=64, seed=42, existing_offsets=[0, "bad"]
            )
        )


def test_registry_requires_opt_in_and_keeps_exact_prompt_deduplication():
    source_hash = source_for("train")
    extra = extras(source_hash)[0]
    default = PromptRegistry()
    assert default.add(canonical(source_hash))
    assert not default.add(dict(extra))
    registry = PromptRegistry(allow_training_prefixes=True)
    assert registry.add(canonical(source_hash))
    assert registry.add(dict(extra))
    assert len(registry.sources) == 1
    assert len(registry.groups) == 2
    duplicate = {**extras(source_hash)[1], "prompt_ids": list(extra["prompt_ids"])}
    assert not registry.add(duplicate)
    # Opt-in does not relax duplicate-source rejection for canonical records.
    assert not registry.add({**canonical(source_hash), "group": "different-id"})


@pytest.mark.parametrize("prefix_first", [False, True])
def test_registry_preserves_article_holdout_in_either_insertion_order(prefix_first):
    source_hash = source_for("train")
    prefix = extras(source_hash)[0]
    heldout = canonical(source_hash, "validation", list(range(5000, 5064)))
    registry = PromptRegistry(allow_training_prefixes=True)
    first, second = (prefix, heldout) if prefix_first else (heldout, prefix)
    assert registry.add(first)
    assert not registry.add(second)
    assert registry.rejected["cross_split_source_group"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"split": "test"},
        {"domain": "code"},
        {"source_group": "unrelated-article"},
        {"source_token_offset": -1},
        {"source_token_offset": True},
        {"group": "unrelated-example"},
    ],
)
def test_registry_rejects_invalid_prefix_provenance(change):
    registry = PromptRegistry(allow_training_prefixes=True)
    assert not registry.add({**extras()[0], **change})
    assert not registry.groups


def test_prefix_opt_in_retains_cross_source_near_duplicate_quarantine():
    registry = PromptRegistry(allow_training_prefixes=True)
    row = extras()[0]
    assert registry.add(row)
    offset = row["source_token_offset"]
    other = canonical(source_for("validation"), "validation", list(range(offset + 1, offset + 65)))
    assert not registry.add(other)
    assert registry.rejected["cross_split_near_duplicate_prompt"] == 1


def test_mapped_cache_keeps_frozen_labels_private_even_under_shared_global_default(tmp_path):
    import mmap

    path = tmp_path / "teacher.pt"
    tokens = torch.arange(80)
    memory = torch.randn(80, 12, dtype=torch.bfloat16)
    torch.save({"train": [{"tokens": tokens, "memory": memory, "group": "article"}]}, path)
    original = path.read_bytes()
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        cache = load_teacher_cache(path)
        assert torch.serialization.get_default_mmap_options() == mmap.MAP_SHARED
    row = cache["train"][0]
    assert torch.equal(row["tokens"], tokens)
    assert torch.equal(row["memory"], memory)
    # Accidental in-memory edits cannot corrupt the source labels or feature file.
    row["tokens"][0] = -1
    row["memory"].zero_()
    assert path.read_bytes() == original
    restored = load_teacher_cache(path)["train"][0]
    assert torch.equal(restored["tokens"], tokens)
    assert torch.equal(restored["memory"], memory)


@pytest.mark.parametrize("mode", ["enabled", "default", "historical_holdout", "unknown_offset"])
def test_manifest_prefix_extension_preserves_article_split_and_existing_records(
    tmp_path, monkeypatch, mode
):
    import copy
    from types import SimpleNamespace

    from latent_workspace.corpus import DOMAINS, SOURCES
    from latent_workspace.dflash import DraftConfig
    from latent_workspace.dflash_scaling import diverse_manifest

    article = next(
        f"article-{i}"
        for i in range(1000)
        if repository_split(f"prose:{digest(f'article-{i}')}") == "train"
    )
    source_hash = digest(article)
    source_group = f"prose:{source_hash}"
    repository = next(str(i) for i in range(1000) if repository_split(str(i)) == "train")
    english = next(
        f"english-{i}"
        for i in range(1000)
        if repository_split(f"translation:{digest(f'english-{i}')}") == "train"
    )
    records = []
    for split in ("train", "validation", "test"):
        for domain in DOMAINS:
            i = len(records) + 1
            records.append(
                {
                    "domain": domain,
                    "split": split,
                    "group": f"old-{i}",
                    "source_hash": f"old-source-{i}",
                    "source_token_offset": 0,
                    "prompt_ids": list(range(10000 * i, 10000 * i + 64)),
                }
            )
    first = next(r for r in records if r["domain"] == "prose" and r["split"] == "train")
    first.update(group=source_group, source_hash=source_hash, prompt_ids=list(range(64)))
    if mode == "unknown_offset":
        first.pop("source_token_offset")
    historical = {
        "window_groups": {
            "train": [],
            "validation": [source_group] if mode == "historical_holdout" else [],
            "test": [],
        },
        "sources": {d: {"revision": "pinned"} for d in DOMAINS},
    }
    rows = {
        SOURCES["prose"][0]: [{"text": article}],
        SOURCES["code"][0]: [{"content": "new-code", "repo_name": repository}],
        SOURCES["translation"][0]: [{"translation": {"en": english}}],
    }
    monkeypatch.setattr("datasets.load_dataset", lambda repo, *a, **kw: iter(rows[repo]))
    monkeypatch.setattr(
        "latent_workspace.dflash_scaling.documents", lambda rows: (r["text"] for r in rows)
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: (
            list(range(2048))
            if text == article
            else list(range(2000, 2600))
            if text == "new-code"
            else list(range(3000, 3032))
        )
    )
    cfg = DraftConfig(
        train_prompts=6,
        validation_prompts=3,
        test_prompts=3,
        prompt_tokens=64,
        prose_prefixes_per_article=1 if mode == "default" else 4,
    )
    original = copy.deepcopy(records)
    if mode != "enabled":
        with pytest.raises(ValueError, match="insufficient diverse prompts"):
            diverse_manifest(
                cfg, tokenizer, tmp_path, float("inf"), historical, {"records": records}, records
            )
        assert records == original
        return
    manifest = diverse_manifest(
        cfg, tokenizer, tmp_path, float("inf"), historical, {"records": records}, records
    )
    assert manifest["records"][: len(records)] == original
    assert records == original
    extra = [r for r in manifest["records"][len(records) :] if r["domain"] == "prose"]
    assert len(extra) == 1
    assert extra[0]["source_group"] == source_group
    assert extra[0]["source_token_offset"] >= 64
    assert manifest["source_group_counts"]["train"]["prose"] == 1
    assert all(r["split"] == "train" for r in manifest["records"][len(records) :])


@pytest.mark.parametrize("cap", [0, 9, True, 1.5])
def test_config_rejects_unbounded_or_noninteger_prefix_cap(cap):
    from latent_workspace.dflash import DraftConfig

    with pytest.raises(ValueError, match="prose prefix cap"):
        DraftConfig(prose_prefixes_per_article=cap).validate()
