"""Prepare a chat SFT recipe (data/recipes/chat-sft-*.json) on Modal CPUs.

Run from training/:
    modal run -m modal_scripts.modal_chat_sft_data [--recipe data/recipes/chat-sft-v1.json]

One call per parquet or JSON-lines file, all in parallel: download it at the recipe's
pinned revision, convert rows to OpenAI messages (smoltalk2's `chat_sft.convert`, or
the recipe's `converter` from `contest_sft.CONVERTERS`, run on every CPU since contest
converters execute each candidate program against its tests; with `questions`, the
converter looks each problem up in a table built once by `questions`; files may be
parquet, JSON lines or Arrow streams; and
`per_question` keeps the first conversations per problem in each file, or with
`per_question_order: "shortest"` those with the shortest reasoning and answer;
`starter_copies` repeats the conversations whose prompt gives starter code), render
them in GLM's chat format with
loss masks, drop conversations longer than the context or sharing a 13-gram with
the pretraining recipe's benchmark screen, hold out a hash-selected fraction, and
write the subset to the `lm-exploration-hybrid-data` volume under `<recipe name>/`
(see chat_sft.py). Each call has `retries=0`, a 1-hour timeout and refuses to start
24 hours after launch; finished subsets are skipped. `summary.json` collects the
per-subset counts. A recipe without `files`, made only of subsets prepared in other
recipes' folders, is composed instead: each subset is checked and the recipe's folder
gets a `summary.json` of counts per pass.
"""

import io
import json
import time
from pathlib import Path

import modal

from workflows.images import add_project

app = modal.App("lm-exploration-hybrid-sft-data")
volume = modal.Volume.from_name("lm-exploration-hybrid-data", create_if_missing=True)
secret = modal.Secret.from_name("simple-llm-hf")
image = add_project(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "pyarrow==26.0.0", "tokenizers==0.22.2", "huggingface-hub==1.33.0", "jinja2==3.1.6"
    )
    .env({"PYTHONPATH": "/project", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
ROOT = Path("/data")
DEFAULT_RECIPE = "data/recipes/chat-sft-v1.json"
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=3600,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def prepare(recipe, file, deadline):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from training_data import chat_sft

    from tokenizer import glm as tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    folder = ROOT / recipe["name"]
    name = Path(file).stem
    if (folder / f"{name}.json").exists():
        return json.loads((folder / f"{name}.json").read_text())
    path = hf_hub_download(
        recipe["repo"], file, revision=recipe["revision"], repo_type="dataset", local_dir="/tmp/sft"
    )
    if "converter" in recipe:
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        from training_data import contest_sft

        if file.endswith(".jsonl"):
            # JSON lines end at "\n" only; statements can hold other line separators.
            rows = [json.loads(line) for line in Path(path).read_text().split("\n") if line]
        elif file.endswith(".arrow"):
            import pyarrow as pa

            rows = pa.ipc.open_stream(pa.memory_map(path)).read_all().to_pylist()
        else:
            rows = pq.read_table(path).to_pylist()
        convert = contest_sft.CONVERTERS[recipe["converter"]]
        if recipe.get("questions"):
            questions = json.loads((folder / "questions.json").read_text())
            convert = partial(convert, questions=questions)
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(convert, rows))
        if "per_question" in recipe:
            shortest = recipe.get("per_question_order", "first") == "shortest"
            rows = per_question(rows, recipe["per_question"], shortest)
        if recipe.get("starter_copies", 1) > 1:
            rows = repeat_starter(rows, recipe["starter_copies"])
        converter = lambda conversation: conversation
    else:
        rows = pq.read_table(path, columns=["messages", "chat_template_kwargs"]).to_pylist()
        converter = chat_sft.convert
    screen = np.load(ROOT / recipe["screen"])
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    meta = chat_sft.prepare_subset(recipe, name, rows, folder, glm, screen, converter)
    volume.commit()
    return meta


def per_question(rows, limit, shortest=False):
    """Converted rows with at most `limit` kept per problem (the conversation's prompt):
    the first ones, or with `shortest` those with the shortest assistant turn (reasoning
    and answer). Dropped ones become None, so they count as unparsed."""
    order = list(range(len(rows)))
    if shortest:
        order.sort(key=lambda i: _assistant_length(rows[i]) if rows[i] else 0)
    seen, kept = {}, set()
    for i in order:
        if rows[i]:
            key = rows[i]["messages"][0]["content"]
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= limit:
                kept.add(i)
    return [row if i in kept else None for i, row in enumerate(rows)]


def _assistant_length(row):
    reply = row["messages"][-1]
    return len(reply.get("reasoning_content") or "") + len(reply["content"])


def repeat_starter(rows, copies):
    """Rows whose LiveCodeBench prompt gives starter code (a function or class to
    complete) appear `copies` times; the others once."""
    from evaluation.benchmarks import livecodebench

    starter = [
        row
        for row in rows
        if row and livecodebench.STDIN_FORMAT not in row["messages"][0]["content"]
    ]
    return rows + starter * (copies - 1)


@app.function(
    image=image,
    cpu=2,
    memory=16384,
    timeout=3600,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def questions(recipe, deadline):
    """Write `<recipe>/questions.json`: "dataset/split/index" -> {"question",
    "starter_code"} from `contest_sft.QUESTION_SOURCES`, for converters that look
    problems up by index; with `questions` "leetcode", LeetCodeDataset train problems
    by statement (`contest_sft.leetcode_questions`)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from training_data import contest_sft

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    path = ROOT / recipe["name"] / "questions.json"
    if path.exists():
        return len(json.loads(path.read_text()))
    table = {}
    if recipe["questions"] == "leetcode":
        repo, revision = contest_sft.LEETCODE
        local = hf_hub_download(
            repo, "LeetCodeDataset-train.jsonl", revision=revision, repo_type="dataset"
        )
        rows = [json.loads(line) for line in Path(local).read_text().split("\n") if line]
        table = contest_sft.leetcode_questions(rows)
    sources = {} if table else contest_sft.QUESTION_SOURCES
    for dataset, (repo, revision, splits) in sources.items():
        for split, files in splits.items():
            rows = []
            for file in files:
                local = hf_hub_download(
                    repo, file, revision=revision, repo_type="dataset", local_dir="/tmp/questions"
                )
                if file.endswith(".jsonl"):
                    text = Path(local).read_text().split("\n")
                    rows += [json.loads(line) for line in text if line]
                else:
                    columns = (
                        ["description"]
                        if dataset == "code_contests"
                        else ["question", "starter_code"]
                    )
                    rows += pq.read_table(local, columns=columns).to_pylist()
                Path(local).unlink()
            for index, row in enumerate(contest_sft.question_rows(dataset, rows)):
                table[f"{dataset}/{split}/{index}"] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table))
    volume.commit()
    return len(table)


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/data": volume})
def compose(recipe):
    """A recipe made only of subsets prepared in other recipes' folders: check that each
    is there and write the recipe's folder with `summary.json` (counts per pass)."""
    from training_data import chat_sft

    rows = []
    for name, prepared_in, repeat in chat_sft.subsets(recipe):
        path = ROOT / (prepared_in or recipe["name"]) / f"{name}.json"
        if not path.is_file():
            raise ValueError(f"Subset {name} is not prepared in {path.parent}")
        rows.append({**json.loads(path.read_text()), "prepared_in": prepared_in, "repeat": repeat})
    keys = ("conversations", "tokens", "target_tokens")
    summary = {
        "recipe": recipe["name"],
        "subsets": rows,
        "per_pass": {key: round(sum(r[key] * r["repeat"] for r in rows)) for key in keys},
    }
    folder = ROOT / recipe["name"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary["per_pass"]


@app.local_entrypoint()
def main(recipe: str = DEFAULT_RECIPE):
    settings = json.loads(Path(recipe).read_text())
    if not settings.get("files"):
        print(json.dumps(compose.remote(settings), indent=2))
        return
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    if settings.get("questions"):
        print(f"questions: {questions.remote(settings, deadline)}")
    metas = list(prepare.starmap([(settings, file, deadline) for file in settings["files"]]))
    keys = ("conversations", "tokens", "target_tokens", "heldout_conversations")
    summary = {
        "recipe": settings["name"],
        "subsets": metas,
        "totals": {key: sum(m[key] for m in metas) for key in keys},
        "dropped": {key: sum(m["dropped"][key] for m in metas) for key in metas[0]["dropped"]},
    }
    with volume.batch_upload(force=True) as upload:
        upload.put_file(
            io.BytesIO(json.dumps(summary, indent=2).encode()),
            f"{settings['name']}/summary.json",
        )
    print(json.dumps(summary["totals"] | {"dropped": summary["dropped"]}, indent=2))
