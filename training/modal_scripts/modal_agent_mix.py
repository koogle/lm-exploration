"""Agent trajectory mixes: public and our own trajectories, prepared for chat SFT.

Run from training/:
    modal run --detach -m modal_scripts.modal_agent_mix::prepare --mix <name> --chat qwen|glm
    modal run -m modal_scripts.modal_agent_mix::target --mix <name> --chat qwen|glm --config <run config>
    modal run --detach -m modal_scripts.modal_agent_mix::publish_mix --mix <name> --repo <owner/name>

`MIXES` names each mix: sources, counts and the seed. Sources (`SOURCES`), all pinned:
- `open-swe`: NVIDIA Open-SWE-Traces, mini-swe-agent v2 subset (real SWE-rebench V2 and
  Scale-SWE tasks; Qwen3.6-27B, Qwen3.8-27B and Qwen3.8-Flash-Next teachers), resolved
  trajectories only, at most two per task, in its own bash-tool format
  (`training_data.open_swe_format`).
- `mini-coder`: `ricdomolm/mini-coder-trajs-400k` (MIT), the mini-coder models' training
  data: Qwen3-Coder-30B mini-swe-agent v1 rollouts on SWE-smith tasks, verified ones
  preferred, converted to the bash-tool format (`open_swe_format.from_mini_swe`).
- `terminal-lego`: Terminal-Lego DeepSeek-V3.2 trajectories on real Stack Overflow
  tasks, reward 1 only (the dataset carries no licence on HF), Terminus 2 format.
- `nemotron-terminal`: NVIDIA Nemotron-Terminal-Corpus skill-based synthetic tasks
  (DeepSeek-V3.2), Terminus 2 format.
- `pr-glm`: our GLM-5.3 trajectories on the PR tasks (resolved, training repositories),
  converted to the Open-SWE bash-only layout; mixes can repeat it.

Every source drops trajectories from the SWE-bench Verified repositories and from the
repositories held out from our PR tasks (`pr_task_agent.heldout`), and drops reasoning.

`prepare` (CPU) chooses rows from metadata, uniformly over each source's files, streams
them through conversion and one chat format (`--chat qwen`: Qwen3 ChatML for the Qwen
reference bases; `--chat glm`: GLM's for the hybrid) and writes one chat SFT subset per
source (`training_data.chat_sft.SubsetWriter`) into the recipe's folder
`agent-<mix>-<chat>` on the hybrid data volume. The recipe
(`data/recipes/agent-<mix>-<chat>.json`) lists the subsets with the mix's repeats; both
chat formats see the same candidates in the same order. `target` prints a run
config's `target_tokens` for one pass of the packed recipe. `publish_mix` uploads the
Qwen-format conversations of a mix to a private HF dataset. Every call has
`retries=0`, a timeout under 24 hours and a deadline supervisor
(`modal_agentic._spawn_all`).
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks

from modal_scripts.modal_agentic import (
    REFERENCE_MODELS,
    _spawn_all,
    hf_cache,
    reference_image,
)
from modal_scripts.modal_agentic import (
    volume as data_volume,
)

app = modal.App("lm-exploration-agent-mix")
app.include(modal_benchmarks.app)  # the deadline supervisor
CONTEXT = 40960  # Open-SWE trajectories are long; the reference runs train at this length
PREPARE_SECONDS = 10 * 60 * 60
TOKENIZER = "Qwen/Qwen3-1.7B-Base"  # the Qwen3 0.6B-4B models share it
SWE_BENCH_VERIFIED_REPOS = {
    "django/django",
    "sympy/sympy",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
    "pydata/xarray",
    "pytest-dev/pytest",
    "pylint-dev/pylint",
    "psf/requests",
    "mwaskom/seaborn",
    "pallets/flask",
}
SOURCES = {
    "open-swe": {
        "repo": "nvidia/Open-SWE-Traces",
        "revision": "4fa4c2ae9afc6b397bc50391d43450c79993c138",
        "prefixes": [
            f"data/minisweagent/{teacher}/{tasks}/"
            for teacher in ("qwen38_27b", "qwen36_27b", "qwen38_flash_next")
            for tasks in ("swe-rebench-v2", "scale-swe")
        ],
    },
    "terminal-lego": {
        "repo": "Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k",
        "revision": "922c2fcb019854911fc05eb1a208d2685ac6e006",
        "files": ["terminal-lego-deepseek-v3-2-15k.json"],
    },
    # Our GLM-5.3 trajectories on the PR tasks (resolved, training repositories),
    # converted to the bash-only Open-SWE layout (`open_swe_format.from_harness`).
    "pr-glm": {"path": "pr-tasks-pilot/agentic/glm-5.3-1793298716/trajectories.jsonl"},
    # mini-coder's own training data (MIT): Qwen3-Coder-30B mini-swe-agent v1 rollouts on
    # SWE-smith tasks, 16% verified, converted to the bash-tool format.
    "mini-coder": {
        "repo": "ricdomolm/mini-coder-trajs-400k",
        "revision": "90b240e09807bbfdead1a18c0a861e2c2d20fa14",
        "prefixes": ["data/"],
    },
    "nemotron-terminal": {
        "repo": "nvidia/Nemotron-Terminal-Corpus",
        "revision": "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9",
        "prefixes": ["synthetic_tasks/skill_based/"],
    },
}
MIXES = {
    "smoke-200": {
        "seed": 0,
        "parts": {
            "open-swe": 100,
            "mini-coder": 50,
            "terminal-lego": 50,
            "nemotron-terminal": 50,
        },
    },
    "openswe-20k": {"seed": 0, "parts": {"open-swe": 20000}},
    "openswe-20k-terminal-30k": {
        "seed": 0,
        "parts": {"open-swe": 20000, "terminal-lego": 15000, "nemotron-terminal": 15000},
    },
    # Our own trajectories, all of them, each read 5 times.
    "openswe-20k-pr5": {
        "seed": 0,
        "parts": {"open-swe": 20000, "pr-glm": {"count": None, "repeat": 5}},
    },
    "openswe-20k-terminal-30k-pr5": {
        "seed": 0,
        "parts": {
            "open-swe": 20000,
            "terminal-lego": 15000,
            "nemotron-terminal": 15000,
            "pr-glm": {"count": None, "repeat": 5},
        },
    },
    # About 500k trajectories, mini-coder's scale: every trajectory of each source that
    # passes its filters (about 29k Open-SWE, 314k mini-coder, 140k Nemotron, 12k
    # Terminal-Lego fit the context in `smoke-200`'s candidate counts), our own five times.
    "scale-500k": {
        "seed": 0,
        "parts": {
            "open-swe": None,
            "mini-coder": None,
            "nemotron-terminal": None,
            "terminal-lego": None,
            "pr-glm": {"count": None, "repeat": 5},
        },
    },
}


def excluded(repo):
    from training_data import pr_task_agent

    return repo in SWE_BENCH_VERIFIED_REPOS or pr_task_agent.heldout(repo)


def _files(source):
    from huggingface_hub import HfApi

    spec = SOURCES[source]
    if "files" in spec:
        return spec["files"]
    names = HfApi().list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"])
    return sorted(
        n for n in names if n.endswith(".parquet") and n.startswith(tuple(spec["prefixes"]))
    )


def _download(source, name):
    from huggingface_hub import hf_hub_download

    spec = SOURCES[source]
    return hf_hub_download(spec["repo"], name, repo_type="dataset", revision=spec["revision"])


# Columns read to choose rows before any trajectory is loaded.
LIGHT_COLUMNS = {
    "open-swe": ["instance_id", "repo", "resolved"],
    "mini-coder": ["instance_id", "verified"],
}
TRAJECTORY_COLUMNS = {
    "open-swe": "messages",
    "mini-coder": "messages",
    "nemotron-terminal": "conversations",
}
CHUNK = 50000  # candidates encoded per pass over a source's files


def mini_coder_repo(instance_id):
    """`owner/repo` from a SWE-smith instance id (`owner__repo.commit.mutation`)."""
    return instance_id.split(".")[0].replace("__", "/", 1)


def _candidates(source, rng):
    """(file, row) keys for every row a source may contribute, in random order with
    preferred rows first: verified mini-coder trajectories, at most two Open-SWE
    trajectories per task."""
    from pyarrow import parquet

    keys = []  # (file, row, priority, task)
    if source == "pr-glm":
        from training_data import pr_task_agent

        lines = Path("/data", SOURCES[source]["path"]).read_text().splitlines()
        for index, line in enumerate(lines):
            row = json.loads(line)
            if row["resolved"] and not pr_task_agent.heldout(row["repository"]):
                keys.append((None, index, 0, None))
    elif source == "terminal-lego":
        name = SOURCES[source]["files"][0]
        rows = json.loads(Path(_download(source, name)).read_text())
        keys = [
            (name, index, 0, None)
            for index, row in enumerate(rows)
            if (row.get("metadata") or {}).get("reward") == 1
        ]
    else:
        for name in _files(source):
            path = _download(source, name)
            if source == "nemotron-terminal":
                count = parquet.ParquetFile(path).metadata.num_rows
                keys.extend((name, index, 0, None) for index in range(count))
                continue
            rows = parquet.read_table(path, columns=LIGHT_COLUMNS[source]).to_pylist()
            for index, row in enumerate(rows):
                if source == "open-swe" and row["resolved"] and not excluded(row["repo"]):
                    keys.append((name, index, 0, row["instance_id"]))
                elif source == "mini-coder" and not excluded(mini_coder_repo(row["instance_id"])):
                    keys.append((name, index, 0 if row["verified"] else 1, None))
    rng.shuffle(keys)
    keys.sort(key=lambda key: key[2])
    if source == "open-swe":
        per_task, capped = {}, []
        for key in keys:
            per_task[key[3]] = per_task.get(key[3], 0) + 1
            if per_task[key[3]] <= 2:
                capped.append(key)
        keys = capped
    return [(name, index) for name, index, _, _ in keys]


def _rows(source, name, indices):
    """The rows at `indices` of one source file."""
    if source == "pr-glm":
        lines = Path("/data", SOURCES[source]["path"]).read_text().splitlines()
        return [json.loads(lines[index]) for index in indices]
    if source == "terminal-lego":
        rows = json.loads(Path(_download(source, name)).read_text())
        return [rows[index] for index in indices]
    from pyarrow import parquet

    column = TRAJECTORY_COLUMNS[source]
    table = parquet.read_table(_download(source, name), columns=[column])
    return table.take(indices).to_pylist()


def _convert(source, row):
    """(chat format, training messages or None) for one raw row."""
    from training_data import open_swe_format, terminus_format

    if source == "pr-glm":
        return "open-swe", open_swe_format.from_harness(row["messages"])
    if source == "open-swe":
        return "open-swe", open_swe_format.training_messages(row)
    if source == "mini-coder":
        return "open-swe", open_swe_format.from_mini_swe(row["messages"])
    return "terminus", terminus_format.training_messages(row["conversations"])


CHAT_FORMATS = ("qwen", "glm")
QWEN_EOS = "<|endoftext|>"
_tokenizers = {}


def _tokenizer(chat_format):
    if chat_format not in _tokenizers:
        if chat_format == "glm":
            from tokenizer import glm

            _tokenizers[chat_format] = glm.load()
        else:
            from transformers import AutoTokenizer

            _tokenizers[chat_format] = AutoTokenizer.from_pretrained(
                TOKENIZER, revision=REFERENCE_MODELS[TOKENIZER]
            )
    return _tokenizers[chat_format]


def _encode(job):
    """(rank, (token ids, loss mask, held out) or None when the row does not convert or
    fit a window). Every conversation ends with its tokenizer's EOS, which
    `ChatMixture` packing and the document masks read as the boundary: GLM's chat format
    (`training_data.chat`) adds it; for Qwen3 ChatML it is appended, a target when the
    conversation ends with an assistant turn, as in GLM's."""
    import numpy as np
    from training_data import chat, chat_sft, mini_swe_format, open_swe_format, qwen_agent_chat

    source, rank, row, chat_format, heldout_fraction = job
    kind, messages = _convert(source, row)
    if not messages:
        return rank, None
    tokenizer = _tokenizer(chat_format)
    tools = open_swe_format.TOOLS if kind == "open-swe" else None
    if chat_format == "glm":
        try:
            ids, mask = chat.encode(tokenizer, messages, tools)
        except (ValueError, StopIteration, TypeError, KeyError):
            return rank, None
    else:
        tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False)
        if kind == "open-swe":
            ids, mask = qwen_agent_chat.encode(tokenize, messages, tools)
        else:
            ids, mask = mini_swe_format.encode(tokenize, messages)
        ids = [*ids, tokenizer.convert_tokens_to_ids(QWEN_EOS)]
        mask = [*mask, int(messages[-1]["role"] == "assistant")]
    if len(ids) > CONTEXT + 1 or not sum(mask):  # a window holds context + 1 tokens
        return rank, None
    heldout = bool(heldout_fraction) and chat_sft.is_heldout(messages, heldout_fraction)
    return rank, (np.asarray(ids, dtype=np.uint32), np.asarray(mask, dtype=np.uint8), heldout)


def _stream_part(source, count, rng, pool, chat_format, heldout_fraction, write):
    """Encode a source's candidates in rank order, a chunk at a time, until `count`
    training examples fit (all of them when `count` is None); `write(rank, ids, mask,
    heldout)` stores each example that fits. Returns (candidates, encoded)."""
    keys, encoded, trained, begin = _candidates(source, rng), 0, 0, 0
    while begin < len(keys) and (count is None or trained < count):
        # A small margin over what the part still needs covers rows that do not fit.
        size = CHUNK if count is None else min(CHUNK, (count - trained) * 5 // 4 + 64)
        by_file = {}
        for rank, (name, index) in enumerate(keys[begin : begin + size], begin):
            by_file.setdefault(name, []).append((index, rank))
        for name, rows in by_file.items():
            raw = _rows(source, name, [index for index, _ in rows])
            jobs = [
                (source, rank, row, chat_format, heldout_fraction)
                for (_, rank), row in zip(rows, raw, strict=True)
            ]
            encoded += len(jobs)
            for rank, example in pool.imap_unordered(_encode, jobs, chunksize=8):
                if example is not None:
                    write(rank, *example)
                    trained += not example[2]
        begin += size
    return len(keys), encoded


def _part(part):
    """(count or None for all, repeat) of a mix part."""
    return (part["count"], part["repeat"]) if isinstance(part, dict) else (part, 1)


# Sources whose trajectories are trained on but not republished (no licence on HF).
NOT_REPUBLISHED = {"terminal-lego"}
EXPORT_SHARD_ROWS = 20000


def _export_row(job):
    """(rank, row for the published mix, or None when the row is not in the mix):
    the training messages as JSON, with the Qwen3 token count the mix's fit test used."""
    source, rank, row, name, index = job
    _, example = _encode((source, rank, row, "qwen", 0))
    if example is None:
        return rank, None
    kind, messages = _convert(source, row)
    return rank, {
        "source": source,
        "source_file": name,
        "source_row": index,
        "format": kind,
        "messages": json.dumps(messages, ensure_ascii=False),
        "qwen3_tokens": len(example[0]),
    }


@app.function(
    image=reference_image,
    cpu=32,
    memory=131072,
    timeout=PREPARE_SECONDS,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/hf": hf_cache, "/data": data_volume},
)
def export_mix(mix, repo, deadline):
    """A mix's trajectories as conversations, to the private HF dataset `repo`: the same
    candidates in the same order as `prepare_mix` (every part's random draws, also for
    sources not republished) and its fit test, as parquet shards
    `data/<source>/<shard>.parquet` of `EXPORT_SHARD_ROWS` rows, uploaded as written.
    Parts with a count, and repeats, are recorded in `mix.json`, not applied."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import random
    import tempfile
    from multiprocessing import Pool

    import pyarrow as pa
    from huggingface_hub import HfApi
    from pyarrow import parquet

    spec = MIXES[mix]
    if any(_part(part)[0] is not None for part in spec["parts"].values()):
        raise ValueError("Only mixes that keep every fitting trajectory export exactly")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    rng, counts, scratch = random.Random(spec["seed"]), {}, Path(tempfile.mkdtemp())

    def upload(source, shard, rows):
        path = scratch / f"{shard:05d}.parquet"
        parquet.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        api.upload_file(
            path_or_fileobj=path, path_in_repo=f"data/{source}/{shard:05d}.parquet",
            repo_id=repo, repo_type="dataset", commit_message=f"{source} shard {shard}",
        )  # fmt: skip
        path.unlink()

    with Pool(30) as pool:
        for source, part in spec["parts"].items():
            keys = _candidates(source, rng)
            if source in NOT_REPUBLISHED:
                counts[source] = {"candidates": len(keys), "published": 0}
                continue
            kept, rows, shard = [], [], 0
            by_file = {}
            for rank, (name, index) in enumerate(keys):
                by_file.setdefault(name, []).append((index, rank))
            for name, entries in by_file.items():
                raw = _rows(source, name, [index for index, _ in entries])
                jobs = [(source, rank, row, name, index)
                        for (index, rank), row in zip(entries, raw, strict=True)]  # fmt: skip
                for _, row in pool.imap(_export_row, jobs, chunksize=8):
                    if row is None:
                        continue
                    rows.append(row)
                    if len(rows) == EXPORT_SHARD_ROWS:
                        upload(source, shard, rows)
                        kept.append(len(rows))
                        rows, shard = [], shard + 1
            if rows:
                upload(source, shard, rows)
                kept.append(len(rows))
            counts[source] = {"candidates": len(keys), "published": sum(kept),
                              "repeat": _part(part)[1]}  # fmt: skip
            print(json.dumps({source: counts[source]}), flush=True)
    manifest = {
        "mix": mix,
        "spec": spec,
        "sources": {s: SOURCES[s] for s in spec["parts"]},
        "not_republished": sorted(NOT_REPUBLISHED & set(spec["parts"])),
        "tokenizer": [TOKENIZER, REFERENCE_MODELS[TOKENIZER]],
        "context": CONTEXT,
        "counts": counts,
    }
    api.upload_file(
        path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode(), path_in_repo="mix.json",
        repo_id=repo, repo_type="dataset", commit_message="mix.json",
    )  # fmt: skip
    return manifest


def mix_recipe(mix, chat_format):
    """The chat SFT recipe (`data/recipes/agent-<mix>-<chat>.json`) of a mix."""
    import training_data

    path = Path(training_data.__file__).parent / "recipes" / f"agent-{mix}-{chat_format}.json"
    recipe = json.loads(path.read_text())
    parts = MIXES[recipe["agent_mix"]]["parts"]
    listed = {s["name"]: s.get("repeat", 1) for s in recipe["subsets"]}
    if listed != {source: _part(part)[1] for source, part in parts.items()}:
        raise ValueError(f"{path.name} must list the mix's sources with its repeats")
    if recipe["context"] != CONTEXT:
        raise ValueError(f"{path.name} must use the mix context {CONTEXT}")
    return recipe


@app.function(
    image=reference_image,
    cpu=32,
    memory=131072,
    timeout=PREPARE_SECONDS,
    retries=0,
    volumes={"/hf": hf_cache, "/data": data_volume},
)
def prepare_mix(mix, chat_format, deadline):
    """A mix's candidates, in rank order a chunk at a time, encoded in `chat_format` and
    written as one chat SFT subset per source into `/data/agent-<mix>-<chat>/` (the
    first `count` that fit per part, the recipe's hash-selected held-out fraction aside),
    with a manifest of the counts."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import random
    from multiprocessing import Pool

    from training_data import chat, chat_sft

    from tokenizer import glm

    recipe, started = mix_recipe(mix, chat_format), time.time()
    # Downloaded and verified once, before the workers load them.
    _tokenizer(chat_format)
    if chat_format == "glm":
        glm.tokenizer_file()
        chat.template()
    spec = MIXES[mix]
    rng = random.Random(spec["seed"])
    out = Path("/data", recipe["name"])
    counts = {}
    with Pool(30) as pool:
        for source, part in spec["parts"].items():
            count, _ = _part(part)
            writer = chat_sft.SubsetWriter(out, source)
            candidates, encoded = _stream_part(
                source, count, rng, pool, chat_format, recipe["heldout_fraction"],
                lambda rank, ids, mask, heldout, writer=writer: writer.write(ids, mask, heldout, rank),
            )  # fmt: skip
            counts[source] = {"candidates": candidates, "encoded": encoded, **writer.close(count)}
            print(json.dumps({source: counts[source]}), flush=True)
            data_volume.commit()
    manifest = {
        "recipe": recipe["name"],
        "mix": mix,
        "chat": chat_format,
        "spec": spec,
        "sources": {s: SOURCES[s] for s in spec["parts"]},
        "context": CONTEXT,
        "counts": counts,
        "seconds": round(time.time() - started),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    data_volume.commit()
    return manifest


@app.function(
    image=reference_image, cpu=4, memory=32768, timeout=1800, retries=0,
    volumes={"/data": data_volume},
)  # fmt: skip
def mix_pass(mix, chat_format, seed, rows, deadline):
    """One pass of a mix's recipe as `ChatMixture` reads it with `seed`: the windows it
    packs, and the budget tokens of its whole batches of `rows`."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from training_data import chat_sft

    recipe = mix_recipe(mix, chat_format)
    mixture = chat_sft.ChatMixture(recipe, Path("/data", recipe["name"]), seed)
    windows = chat_sft.pass_windows(mixture)
    batches = windows // rows
    return {
        "recipe": recipe["name"],
        "conversations": len(mixture.order),
        "windows": windows,
        "batches": batches,
        "target_tokens": batches * rows * recipe["context"],
    }


@app.local_entrypoint()
def prepare(mix: str, chat: str):
    """Prepare a mix in one chat format (`qwen` or `glm`)."""
    if chat not in CHAT_FORMATS:
        raise ValueError(f"--chat is one of {CHAT_FORMATS}")
    print(json.dumps({"results": f"lm-exploration-hybrid-data/{mix_recipe(mix, chat)['name']}/"}))
    _spawn_all([(prepare_mix, (mix, chat))])


@app.local_entrypoint()
def target(mix: str, chat: str, config: str):
    """The `target_tokens` of one pass of a mix's recipe for a run config (its `seed`
    and `global_rows`)."""
    settings = json.loads(Path(config).read_text())
    print(json.dumps(mix_pass.remote(mix, chat, settings["seed"], settings["global_rows"],
                                     time.time() + 3600)))  # fmt: skip


@app.local_entrypoint()
def publish_mix(mix: str, repo: str):
    """Publish a mix's trajectories as conversations to a private HF dataset."""
    _spawn_all([(export_mix, (mix, repo))])
