"""Agent SFT on mixes of public trajectories, on eight H100s.

Run from training/:
    modal run --detach -m modal_scripts.modal_agent_sft::prepare --mix <name> [--chat glm]
    modal run --detach -m modal_scripts.modal_agent_sft::train --mix <name> --base B [--tag 01] [--hours 20]
    modal run --detach -m modal_scripts.modal_agent_sft::serve --run <run> [--hours 3]
    modal run --detach -m modal_scripts.modal_agent_sft::bench --run <run> [--attempts 1]

`MIXES` names each mix: sources, counts and the seed. Sources (`SOURCES`), all pinned:
- `open-swe`: NVIDIA Open-SWE-Traces, mini-swe-agent v2 subset (real SWE-rebench V2 and
  Scale-SWE tasks; Qwen3.6-27B, Qwen3.8-27B and Qwen3.8-Flash-Next teachers), resolved
  trajectories only, at most two per task, in its own bash-tool format
  (`training_data.open_swe_format`).
- `terminal-lego`: Terminal-Lego DeepSeek-V3.2 trajectories on real Stack Overflow
  tasks, reward 1 only (the dataset carries no licence on HF; see
  `plans/AGENTIC_TRAJECTORIES.md`), Terminus 2 format.
- `pr-glm`: our GLM-5.3 trajectories on the PR tasks (resolved, training repositories),
  converted to the Open-SWE bash-only layout; mixes can repeat it.
- `mini-coder`: `ricdomolm/mini-coder-trajs-400k` (MIT), the mini-coder models' training
  data: Qwen3-Coder-30B mini-swe-agent v1 rollouts on SWE-smith tasks, verified ones
  preferred, converted to the bash-tool format (`open_swe_format.from_mini_swe`).
- `nemotron-terminal`: NVIDIA Nemotron-Terminal-Corpus skill-based synthetic tasks
  (DeepSeek-V3.2), Terminus 2 format.

Every source drops trajectories from the SWE-bench Verified repositories and from the
repositories held out from our PR tasks (`pr_task_agent.heldout`), and drops reasoning.

`prepare` (CPU) chooses rows from metadata, uniformly over each source's files, and
streams them through the Qwen3 tokenizer into `lm-exploration-hybrid-runs/agent-mix/<mix>/`
(tokens, loss mask, offsets, the shuffled training order with repeats, manifest).
`train` fully fine-tunes a pinned Qwen3 base on it with `torchrun` over eight GPUs
(`modal_scripts.agent_sft_worker`) and writes the model, metrics and summary to
`agentic-reference/<run>/`, where `modal_agentic.py::reference_evaluate` finds it. It
checkpoints every `checkpoint_steps` and when its time budget ends; launching `train`
again for the same run continues from the checkpoint, so a mix longer than one launch
trains over consecutive launches.
`serve` runs a trained model behind vLLM's OpenAI API through a Modal tunnel; `bench`
runs Terminal-Bench 2.1 (Harbor, Terminus 2, Modal sandboxes) against a trained model
served inside the same function. All have `retries=0` and a timeout under 24 hours;
training stops and saves 30 minutes before its timeout; one supervisor per launch
cancels the call at 24 hours.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks

from modal_scripts.modal_agentic import (
    LAUNCH_LIMIT_SECONDS,
    REFERENCE_FOLDER,
    REFERENCE_MODELS,
    hf_cache,
    reference_base,
    reference_image,
    runs_volume,
)
from modal_scripts.modal_agentic import (
    volume as data_volume,
)
from workflows.images import add_project

app = modal.App("lm-exploration-agent-sft")
app.include(modal_benchmarks.app)  # the deadline supervisor
MIX_FOLDER = "agent-mix"
CONTEXT = 40960  # Qwen3's position limit; Open-SWE trajectories are long
TRAIN_SECONDS = 20 * 60 * 60  # one launch; longer mixes resume from the checkpoint
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
    # Checkpoints every five steps, to exercise resuming.
    "smoke-200": {
        "seed": 0,
        "settings": {"checkpoint_steps": 5},
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
SETTINGS = {
    "context": CONTEXT,
    "gpus": 8,
    "accumulation": 2,  # trajectories per GPU per step: 16 per step in all
    "learning_rate": 2e-5,
    "warmup_steps": 30,
    "min_learning_rate_fraction": 0.1,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "loss_chunk": 4096,
    "epochs": 1,
    "checkpoint_steps": 2000,  # about two hours of 16 trajectories per step
}
cpu_image = reference_image


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
    fit). Qwen3 ChatML for the Qwen bases; GLM's chat format (`training_data.chat`) with
    a conversation's final EOS, as the hybrid's chat SFT reads it, for the hybrid."""
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
        limit = CONTEXT + 1  # a window holds context + 1 tokens
    else:
        tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False)
        if kind == "open-swe":
            ids, mask = qwen_agent_chat.encode(tokenize, messages, tools)
        else:
            ids, mask = mini_swe_format.encode(tokenize, messages)
        limit = CONTEXT
    if len(ids) > limit or not sum(mask):
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


@app.function(
    image=cpu_image,
    cpu=32,
    memory=131072,
    timeout=PREPARE_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache, "/data": data_volume},
)
def prepare_mix(mix, deadline):
    """Tokenize a mix for the Qwen bases, streaming: each source's candidates are
    encoded in rank order, a chunk at a time, and appended to `tokens.u32`/`mask.u8`
    until the part's count fits; `offsets.npy` bounds the stored examples and
    `order.npy` is the training order (the kept examples of every part, repeated,
    shuffled)."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import random
    from multiprocessing import Pool

    import numpy as np

    spec, started = MIXES[mix], time.time()
    rng = random.Random(spec["seed"])
    out = Path("/runs", MIX_FOLDER, mix)
    out.mkdir(parents=True, exist_ok=True)
    lengths, targets, counts, order = [], [], {}, []
    with (
        open(out / "tokens.u32", "wb") as tokens_file,
        open(out / "mask.u8", "wb") as mask_file,
        Pool(30) as pool,
    ):
        for source, part in spec["parts"].items():
            count, repeat = _part(part)
            first, fitted = len(lengths), []

            def write(rank, ids, mask, heldout, fitted=fitted):
                ids.tofile(tokens_file)
                mask.tofile(mask_file)
                fitted.append((rank, len(lengths)))
                lengths.append(len(ids))
                targets.append(int(mask.sum()))

            candidates, encoded = _stream_part(source, count, rng, pool, "qwen", 0, write)
            fitted.sort()
            kept = [example for _, example in (fitted if count is None else fitted[:count])]
            order.extend(kept * repeat)
            counts[source] = {
                "candidates": candidates,
                "encoded": encoded,
                "fit": len(fitted),
                "kept": len(kept),
                "repeat": repeat,
                "stored": [first, len(lengths)],
            }
            print(json.dumps({source: counts[source]}), flush=True)
    order = np.random.default_rng(spec["seed"]).permutation(np.array(order, dtype=np.int64))
    np.save(out / "order.npy", order)
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]))
    lengths, targets = np.array(lengths), np.array(targets)
    manifest = {
        "mix": mix,
        "spec": spec,
        "sources": {s: SOURCES[s] for s in spec["parts"]},
        "tokenizer": [TOKENIZER, REFERENCE_MODELS[TOKENIZER]],
        "context": CONTEXT,
        "counts": counts,
        "examples": len(order),
        "stored_examples": len(lengths),
        "tokens": int(lengths[order].sum()),
        "target_tokens": int(targets[order].sum()),
        "seconds": round(time.time() - started),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runs_volume.commit()
    return manifest


class _ChatSubset:
    """One chat SFT subset written while streaming (`training_data.chat_sft` layout):
    tokens and masks go to raw files, and `close` writes the `.npy` arrays with the
    training conversations (the first `count` by rank) and the held-out ones."""

    def __init__(self, folder, name):
        self.folder, self.name = Path(folder), name
        self.files = {
            # Closed in `close`, after the whole source has streamed through.
            kind: open(self.folder / f"{name}.{kind}.raw", "wb")  # noqa: SIM115
            for kind in ("tokens", "mask")
        }
        self.position, self.rows = 0, []  # (rank, start, length, targets, held out)

    def write(self, rank, ids, mask, heldout):
        ids.tofile(self.files["tokens"])
        mask.tofile(self.files["mask"])
        self.rows.append((rank, self.position, len(ids), int(mask.sum()), heldout))
        self.position += len(ids)

    def close(self, count):
        import numpy as np

        for handle in self.files.values():
            handle.close()
        self.rows.sort()
        trained = [r for r in self.rows if not r[4]]
        trained = trained if count is None else trained[:count]
        heldout = [r for r in self.rows if r[4]]
        arrays = {
            kind: np.memmap(self.folder / f"{self.name}.{kind}.raw", dtype=dtype, mode="r")
            for kind, dtype in (("tokens", np.uint32), ("mask", np.uint8))
        }
        # Training conversations point into the full arrays; held-out ones get their own.
        for kind, array in arrays.items():
            np.save(self.folder / f"{self.name}.{kind}.npy", array)
        docs = np.array([(start, length) for _, start, length, _, _ in trained], np.int64)
        np.save(self.folder / f"{self.name}.docs.npy", docs.reshape(-1, 2))
        held_docs, position = [], 0
        for kind, array in arrays.items():
            pieces = [np.asarray(array[s : s + n]) for _, s, n, _, _ in heldout]
            empty = np.zeros(0, array.dtype)
            np.save(
                self.folder / f"{self.name}.heldout.{kind}.npy", np.concatenate(pieces or [empty])
            )
        for _, _, length, _, _ in heldout:
            held_docs.append((position, length))
            position += length
        np.save(
            self.folder / f"{self.name}.heldout.docs.npy",
            np.array(held_docs, np.int64).reshape(-1, 2),
        )
        for kind in arrays:
            (self.folder / f"{self.name}.{kind}.raw").unlink()
        meta = {
            "subset": self.name,
            "conversations": len(trained),
            "tokens": int(sum(r[2] for r in trained)),
            "target_tokens": int(sum(r[3] for r in trained)),
            "heldout_conversations": len(heldout),
            "heldout_tokens": int(sum(r[2] for r in heldout)),
            "heldout_target_tokens": int(sum(r[3] for r in heldout)),
            "stored_conversations": len(self.rows),
        }
        (self.folder / f"{self.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta


def glm_recipe(mix):
    """The hybrid chat SFT recipe (`data/recipes/agent-<mix>-glm.json`) for a mix."""
    import training_data

    path = Path(training_data.__file__).parent / "recipes" / f"agent-{mix}-glm.json"
    recipe = json.loads(path.read_text())
    parts = MIXES[recipe["agent_mix"]]["parts"]
    listed = {s["name"]: s.get("repeat", 1) for s in recipe["subsets"]}
    if listed != {source: _part(part)[1] for source, part in parts.items()}:
        raise ValueError(f"{path.name} must list the mix's sources with its repeats")
    if recipe["context"] != CONTEXT:
        raise ValueError(f"{path.name} must use the mix context {CONTEXT}")
    return recipe


@app.function(
    image=cpu_image,
    cpu=32,
    memory=131072,
    timeout=PREPARE_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache, "/data": data_volume},
)
def prepare_glm_mix(mix, deadline):
    """The same mix for the hybrid: the same candidates in the same rank order, encoded
    in GLM's chat format with the GLM tokenizer and written as one chat SFT subset per
    source into the recipe's folder on the hybrid data volume (`agent-<mix>-glm`), with
    the recipe's hash-selected held-out fraction. The recipe carries the repeats."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import random
    from multiprocessing import Pool

    from training_data import chat

    from tokenizer import glm

    # Downloaded and verified once, before the workers load them.
    glm.tokenizer_file()
    chat.template()
    recipe, started = glm_recipe(mix), time.time()
    spec = MIXES[mix]
    rng = random.Random(spec["seed"])
    out = Path("/data", recipe["name"])
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with Pool(30) as pool:
        for source, part in spec["parts"].items():
            count, _ = _part(part)
            subset = _ChatSubset(out, source)
            candidates, encoded = _stream_part(
                source, count, rng, pool, "glm", recipe["heldout_fraction"], subset.write
            )
            counts[source] = {
                "candidates": candidates,
                "encoded": encoded,
                **subset.close(count),
            }
            print(json.dumps({source: counts[source]}), flush=True)
            data_volume.commit()
    manifest = {
        "recipe": recipe["name"],
        "mix": mix,
        "spec": spec,
        "sources": {s: SOURCES[s] for s in spec["parts"]},
        "tokenizer": [glm.SOURCE["repo"], glm.SOURCE["revision"]],
        "context": CONTEXT,
        "counts": counts,
        "seconds": round(time.time() - started),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    data_volume.commit()
    return manifest


@app.function(
    image=reference_image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=TRAIN_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def train_mix(base, mix, run, seconds, deadline):
    """Train (or continue) `run` for at most `seconds` of this launch."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    runs_volume.reload()
    config = {
        "base": base,
        "revision": REFERENCE_MODELS[base],
        "mix": str(Path("/runs", MIX_FOLDER, mix)),
        "out": str(Path("/runs", REFERENCE_FOLDER, run)),
        "settings": {**SETTINGS, **MIXES[mix].get("settings", {})},
        "stop_unix": min(deadline - 1800, time.time() + min(seconds, TRAIN_SECONDS) - 1800),
    }
    Path(config["out"]).mkdir(parents=True, exist_ok=True)
    config_path = Path(config["out"], "train-config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={SETTINGS['gpus']}",
            "-m",
            "modal_scripts.agent_sft_worker",
            str(config_path),
        ],
        check=True,
        env=env,
    )
    runs_volume.commit()
    return json.loads(Path(config["out"], "summary.json").read_text())


SERVE_SECONDS = 6 * 60 * 60
IM_END_ID = 151645  # <|im_end|> in the Qwen3 tokenizer
endpoints = modal.Dict.from_name("lm-exploration-agent-endpoints", create_if_missing=True)


def _start_server(run):
    """vLLM's OpenAI server for `agentic-reference/<run>/model` on localhost:8000."""
    runs_volume.reload()
    model = Path("/runs", REFERENCE_FOLDER, run, "model")
    # The models end every turn with <|im_end|>; the Qwen3 base generation config only
    # stops at <|endoftext|>, so vLLM's server would generate past each turn.
    generation = json.loads((model / "generation_config.json").read_text())
    stops = generation.get("eos_token_id")
    stops = stops if isinstance(stops, list) else [stops]
    if IM_END_ID not in stops:
        generation["eos_token_id"] = [IM_END_ID, *[s for s in stops if s is not None]]
        (model / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
        runs_volume.commit()
    # Trained at CONTEXT, beyond the Qwen3 base configs' 32,768 positions (RoPE).
    env = {**os.environ, "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    return subprocess.Popen(
        ["vllm", "serve", str(model), "--served-model-name", run, "--port", "8000",
         "--max-model-len", str(CONTEXT), "--enable-prefix-caching", "--seed", "0",
         # vLLM sizes the RoPE cache from max_position_embeddings; positions past the
         # base config's 32,768 would index outside it.
         "--hf-overrides", json.dumps({"max_position_embeddings": CONTEXT})],
        env=env,
    )  # fmt: skip


def _wait_healthy(server, end):
    import urllib.request

    while time.time() < end:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            return
        except OSError:
            time.sleep(10)
    raise TimeoutError("vLLM did not become healthy")


@app.function(
    image=reference_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=SERVE_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def serve_model(run, seconds, deadline):
    """Serve `agentic-reference/<run>/model` with vLLM's OpenAI API through a Modal tunnel
    for `seconds` (bounded by the timeout and the launch deadline). The tunnel URL is
    stored in the `lm-exploration-agent-endpoints` Dict under the run name."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    server = _start_server(run)
    end = min(time.time() + seconds, deadline - 600, time.time() + SERVE_SECONDS - 300)
    try:
        with modal.forward(8000) as tunnel:
            _wait_healthy(server, end)
            endpoints[run] = {"url": tunnel.url, "until_unix": end}
            print(json.dumps({"run": run, "url": tunnel.url, "until_unix": end}), flush=True)
            while time.time() < end and server.poll() is None:
                time.sleep(30)
    finally:
        endpoints.pop(run, None)
        server.terminate()
    return {"run": run, "served_until": time.time()}


TERMINAL_BENCH = "terminal-bench/terminal-bench-2-1"
TERMINAL_BENCH_SECONDS = 9 * 60 * 60
# Harbor (the Terminal-Bench runner) in its own tool environment, next to vLLM.
terminal_bench_image = add_project(
    reference_base.run_commands(
        "pip install uv==0.9.5",
        "uv tool install 'harbor[modal]==0.25.0' --with 'modal[api-proxy-support]'",
    ).env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
)


@app.function(
    image=terminal_bench_image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=TERMINAL_BENCH_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def terminal_bench(run, attempts, concurrency, deadline):
    """Terminal-Bench 2.1 for a trained run: vLLM on localhost and Harbor's Terminus 2 in
    the same container, task sandboxes on Modal (each capped below this call's end),
    temperature 0.7. Harbor's job folder goes to `terminal-bench/<run>/` on the runs
    volume; returns the reward summary."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    started = time.time()
    end = min(deadline - 900, started + TERMINAL_BENCH_SECONDS - 900)
    server = _start_server(run)
    out = Path("/runs", "terminal-bench", run)
    out.mkdir(parents=True, exist_ok=True)
    try:
        _wait_healthy(server, end)
        model_info = {"max_input_tokens": CONTEXT, "max_output_tokens": 4096,
                      "input_cost_per_token": 0, "output_cost_per_token": 0}  # fmt: skip
        command = [
            "harbor", "run", "-d", TERMINAL_BENCH, "-a", "terminus-2",
            "-m", f"hosted_vllm/{run}", "--ak", "api_base=http://localhost:8000/v1",
            "--ak", "temperature=0.7", "--ak", f"model_info={json.dumps(model_info)}",
            "-e", "modal", "--ek", f"sandbox_timeout_secs={int(end - time.time())}",
            "-k", str(attempts), "-n", str(concurrency), "-o", str(out),
        ]  # fmt: skip
        # Harbor's preflight looks for a Modal token file or variables; inside a Modal
        # container the SDK authenticates with the container's own credentials.
        Path.home().joinpath(".modal.toml").touch()
        env = {**os.environ, "HOSTED_VLLM_API_KEY": "none"}
        subprocess.run(command, env=env, timeout=max(60, end - time.time()), check=False)
    finally:
        server.terminate()
        runs_volume.commit()
    rewards, exceptions = [], {}
    for result in out.glob("*/*/result.json"):
        data = json.loads(result.read_text())
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rewards.append(reward == 1)
        kind = (data.get("exception_info") or {}).get("exception_type") or "none"
        exceptions[kind] = exceptions.get(kind, 0) + 1
    summary = {"run": run, "dataset": TERMINAL_BENCH, "trials": len(rewards),
               "solved": sum(rewards), "exceptions": exceptions,
               "seconds": round(time.time() - started)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs_volume.commit()
    return summary


def _launch(function, *arguments):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = function.spawn(*arguments, deadline)
    supervisor = modal_benchmarks.enforce_deadline.spawn([call.object_id], deadline)
    print(
        json.dumps(
            {"call": call.object_id, "supervisor": supervisor.object_id, "deadline_unix": deadline}
        )
    )


@app.local_entrypoint()
def bench(run: str, attempts: int = 1, concurrency: int = 24):
    """Terminal-Bench 2.1 for a trained run, entirely on Modal."""
    _launch(terminal_bench, run, attempts, concurrency)


@app.local_entrypoint()
def serve(run: str, hours: float = 3.0):
    """Serve a trained run for Harbor; read the URL from the endpoints Dict."""
    _launch(serve_model, run, int(hours * 3600))


@app.local_entrypoint()
def prepare(mix: str, chat: str = "qwen"):
    """Prepare a mix for the Qwen bases (`--chat qwen`) or for the hybrid (`--chat glm`,
    the chat SFT recipe `agent-<mix>-glm` on the hybrid data volume)."""
    if chat not in CHAT_FORMATS:
        raise ValueError(f"--chat is one of {CHAT_FORMATS}")
    if chat == "glm":
        name = glm_recipe(mix)["name"]
        print(json.dumps({"results": f"lm-exploration-hybrid-data/{name}/"}))
        _launch(prepare_glm_mix, mix)
    else:
        print(json.dumps({"results": f"lm-exploration-hybrid-runs/{MIX_FOLDER}/{mix}/"}))
        _launch(prepare_mix, mix)


@app.local_entrypoint()
def train(mix: str, base: str = "Qwen/Qwen3-1.7B-Base", tag: str = "01", hours: float = 20.0):
    """Train a run, or continue it from its checkpoint; `hours` bounds this launch."""
    run = f"{base.split('/')[1].lower()}-{mix}-{tag}"
    print(json.dumps({"results": f"lm-exploration-hybrid-runs/{REFERENCE_FOLDER}/{run}/"}))
    _launch(train_mix, base, mix, run, int(hours * 3600))
