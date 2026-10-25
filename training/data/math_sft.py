"""Strict JSONL math word problems for answer-only SFT, and the GSM8K + MetaMathQA builder.

Each row is `{"id", "group", "split", "bucket", "question", "answer", "source", "license"}`.
The prompt is `Question: {question}\\nAnswer:` (the evaluation format in
`analysis/evaluation`); the answer, after one space, and EOS are supervised. Answers end
with `#### N`.
"""

import hashlib
import json
import re
from pathlib import Path

from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, LOSS_IGNORE_INDEX

FORMAT = "math-sft-v1"
FIELDS = ("id", "group", "split", "bucket", "question", "answer", "source", "license")
# Word n-gram length for test-set decontamination.
NGRAM = 13
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
METAMATH = ("meta-math/MetaMathQA", "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18")
SVAMP = ("ChilleD/SVAMP", "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef")


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def prompt(row):
    return f"Question: {row['question'].strip()}\nAnswer:"


def encode(row, tokenizer, context):
    prefix = [BOS_TOKEN_ID] + tokenizer.encode(prompt(row), add_special_tokens=False).ids
    answer = tokenizer.encode(" " + row["answer"].strip(), add_special_tokens=False).ids
    answer.append(EOS_TOKEN_ID)
    if len(prefix) + len(answer) - 1 > context:
        raise ValueError(f"Example {row['id']} exceeds context; refusing to truncate the answer")
    tokens = prefix + answer
    labels = [LOSS_IGNORE_INDEX] * len(prefix) + answer
    return {"id": row["id"], "bucket": row["bucket"], "x": tokens[:-1], "y": labels[1:]}


def prepare(path, tokenizer, context):
    """Tokenized train/validation splits and the file's SHA256; rejects split leaks."""
    raw = Path(path).read_bytes()
    splits = {"train": [], "validation": []}
    ids, groups, questions = set(), {}, {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in FIELDS:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty string: {key}")
        split = row["split"]
        if split not in splits or row["id"] in ids:
            raise ValueError("Invalid split or duplicate ID")
        ids.add(row["id"])
        # A group is one original problem and all of its rewrites.
        if groups.setdefault(row["group"], split) != split:
            raise ValueError("Group crosses train/validation")
        if questions.setdefault(normalize(row["question"]), split) != split:
            raise ValueError("Question duplicate crosses train/validation")
        if not re.search(r"####\s*\S", row["answer"]):
            raise ValueError(f"Answer of {row['id']} lacks a final #### line")
        splits[split].append(encode(row, tokenizer, context))
    if not all(splits.values()):
        raise ValueError("Both train and validation examples are required")
    return splits, hashlib.sha256(raw).hexdigest()


def ngrams(text):
    words = normalize(text).split()
    return {" ".join(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def gsm8k_solution(answer):
    """A GSM8K solution without its `<<a*b=c>>` calculator annotations."""
    return re.sub(r"<<[^>]*>>", "", answer).strip()


def metamath_solution(response):
    """A MetaMathQA GSM response ending in one `#### N` line.

    Responses close with `The answer is: N`, most after a `#### N` line already.
    """
    body, found, final = response.strip().rpartition("The answer is:")
    if not found or not final.strip():
        return None
    body = body.strip()
    if re.search(r"(^|\n)####\s*\S[^\n]*$", body):
        return body
    return f"{body}\n#### {final.strip()}"


def build(gsm8k_train, metamath, test_questions, output, validation_count=500):
    """Write `examples.jsonl` and `report.json` to `output`.

    The first `validation_count` GSM8K training problems (in file order) are validation;
    MetaMathQA rewrites of them are dropped from training. Only MetaMathQA's GSM-derived
    rows are used. Any row sharing a 13-word n-gram with a test question is dropped.
    """
    output = Path(output)
    blocked = set().union(*(ngrams(q) for q in test_questions))
    counts = {
        "gsm8k": 0,
        "metamath": 0,
        "contaminated": 0,
        "validation_rewrites": 0,
        "duplicates": 0,
    }
    skipped = {"metamath_not_gsm": 0, "metamath_no_final_answer": 0}
    rows, validation = [], set()
    for index, row in enumerate(gsm8k_train):
        split = "validation" if index < validation_count else "train"
        key = normalize(row["question"])
        if split == "validation":
            validation.add(key)
        if ngrams(row["question"]) & blocked:
            counts["contaminated"] += 1
            continue
        rows.append(
            {
                "id": f"gsm8k-train-{index}",
                "group": hashlib.sha256(key.encode()).hexdigest()[:16],
                "split": split,
                "bucket": "gsm8k",
                "question": row["question"],
                "answer": gsm8k_solution(row["answer"]),
                "source": f"{GSM8K[0]}@{GSM8K[1]} main/train row {index}",
                "license": "MIT",
            }
        )
        counts["gsm8k"] += 1
    # Answer augmentation gives one question several solutions; drop only exact repeats.
    seen = {(normalize(r["question"]), r["answer"]) for r in rows}
    for index, row in enumerate(metamath):
        if not row["type"].startswith("GSM_"):
            skipped["metamath_not_gsm"] += 1
            continue
        original = normalize(row["original_question"])
        if original in validation:
            counts["validation_rewrites"] += 1
            continue
        answer = metamath_solution(row["response"])
        if answer is None:
            skipped["metamath_no_final_answer"] += 1
            continue
        key = normalize(row["query"])
        if key in validation or (key, answer) in seen:
            counts["duplicates"] += 1
            continue
        if ngrams(row["query"]) & blocked or ngrams(row["original_question"]) & blocked:
            counts["contaminated"] += 1
            continue
        seen.add((key, answer))
        rows.append(
            {
                "id": f"metamath-{index}",
                "group": hashlib.sha256(original.encode()).hexdigest()[:16],
                "split": "train",
                "bucket": "metamath",
                "question": row["query"],
                "answer": answer,
                "source": f"{METAMATH[0]}@{METAMATH[1]} row {index} ({row['type']})",
                "license": "MIT",
            }
        )
        counts["metamath"] += 1
    output.mkdir(parents=True, exist_ok=False)
    path = output / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "format": FORMAT,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {"gsm8k": GSM8K, "metamath": METAMATH, "svamp": SVAMP},
        "decontamination": f"{NGRAM}-word n-grams of GSM8K and SVAMP test questions",
        "splits": {
            split: sum(r["split"] == split for r in rows) for split in ("train", "validation")
        },
        "kept": counts,
        "skipped": skipped,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
