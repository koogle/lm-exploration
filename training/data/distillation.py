"""Student answers scored by a teacher, for on-policy distillation (`distill-answers-v1`).

A set is a folder `/runs/<data-id>/` on the training volume:

    tokens.npy            uint32, every answer's prompt, then its answer tokens (with the
                          stop token that ended a finished answer)
    answers.npy           int64 [answers, 4]: offset into tokens.npy, prompt length,
                          answer length and whether the answer finished
    teacher_ids.npy       int32 [answer tokens, k]: the teacher's k likeliest next tokens
                          at each answer token, in answer order
    teacher_logprobs.npy  float16 [answer tokens, k]: their log-probabilities
    report.json           counts, the student checkpoint and sampling settings, the
                          teacher and the SHA256 of every array

The student and the teacher share a tokenizer and a chat format, so the student's token
ids are the teacher's input and the teacher's distribution is over the student's
vocabulary. `chat_prompts` takes the prompts from a prepared chat SFT recipe: each
distinct conversation prefix up to its first assistant turn, ending in the opened
reasoning span (`chat.THINKING_PROMPT`). The sampling and scoring jobs
(`modal_scripts/modal_distill.py`) write shards; `merge_answers` and `merge_scores`
combine them. `Answers` reads a finished set through memory maps for the RL loop's
`scored` trajectory source.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from tokenizer import glm

FORMAT = "distill-answers-v1"
ARRAYS = ("tokens", "answers", "teacher_ids", "teacher_logprobs")
ASSISTANT = glm.SPECIAL_TOKENS["<|assistant|>"]
THINK = glm.SPECIAL_TOKENS["<think>"]


def thinking_prompt(ids, mask):
    """A conversation's tokens up to its first assistant turn, ending in the opened
    reasoning span; None when the first turn the model writes is not the assistant's."""
    targets = np.flatnonzero(mask)
    if not len(targets):
        return None
    prefix = [int(t) for t in ids[: targets[0]]]
    if ASSISTANT not in prefix[-2:]:
        return None
    return prefix[: len(prefix) - prefix[::-1].index(ASSISTANT)] + [THINK]


def chat_prompts(recipe, root, longest):
    """Distinct thinking prompts of at most `longest` tokens from the prepared training
    conversations of chat SFT `recipe`, prepared in the folder `root`, in subset order."""
    from training_data import chat_sft

    seen, prompts = set(), []
    for name, _, _ in chat_sft.subsets(recipe):
        tokens, mask, docs = chat_sft._load(chat_sft.subset_folder(recipe, root, name), name)
        for start, length in docs:
            prompt = thinking_prompt(tokens[start : start + length], mask[start : start + length])
            if prompt is None or len(prompt) > longest:
                continue
            key = hashlib.sha256(np.asarray(prompt, dtype=np.uint32).tobytes()).digest()
            if key not in seen:
                seen.add(key)
                prompts.append(prompt)
    return prompts


def pack(prompts, completions, stops):
    """(tokens, answers) arrays of sampled `completions` (without their stop tokens) to
    `prompts`; `stops[i]` is the token that ended answer i, None if it never ended."""
    tokens, answers, offset = [], [], 0
    for prompt, completion, stop in zip(prompts, completions, stops, strict=True):
        answer = list(completion) + ([stop] if stop is not None else [])
        tokens += [*prompt, *answer]
        answers.append((offset, len(prompt), len(answer), int(stop is not None)))
        offset += len(prompt) + len(answer)
    return np.asarray(tokens, dtype=np.uint32), np.asarray(answers, dtype=np.int64).reshape(-1, 4)


def merge_answers(shards):
    """One (tokens, answers) pair from per-shard pairs, dropping empty answers."""
    tokens, answers, offset = [], [], 0
    for shard_tokens, shard_answers in shards:
        for start, prompt, answer, finished in shard_answers:
            if answer == 0:
                continue
            tokens.append(shard_tokens[start : start + prompt + answer])
            answers.append((offset, prompt, answer, finished))
            offset += prompt + answer
    return np.concatenate(tokens), np.asarray(answers, dtype=np.int64)


def score_offsets(answers):
    """Each answer's first row in the teacher arrays, and the total row count."""
    ends = np.cumsum(answers[:, 2])
    return ends - answers[:, 2], int(ends[-1]) if len(ends) else 0


def merge_scores(parts):
    """The teacher arrays of consecutive answer ranges, concatenated in order."""
    ids = np.concatenate([part[0] for part in parts])
    logprobs = np.concatenate([part[1] for part in parts])
    return ids.astype(np.int32), logprobs.astype(np.float16)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 24):
            hasher.update(block)
    return hasher.hexdigest()


def write_report(folder, details):
    """`report.json` of a finished set: `details`, counts and every array's SHA256."""
    folder = Path(folder)
    answers = np.load(folder / "answers.npy")
    ids = np.load(folder / "teacher_ids.npy", mmap_mode="r")
    _, rows = score_offsets(answers)
    if ids.shape[0] != rows:
        raise ValueError("The teacher arrays do not cover every answer token")
    report = {
        "format": FORMAT,
        **details,
        "answers": len(answers),
        "answer_tokens": rows,
        "finished": int(answers[:, 3].sum()),
        "top_k": int(ids.shape[1]),
        "sha256": {name: digest(folder / f"{name}.npy") for name in ARRAYS},
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    return report


class Answers:
    """A finished set, read through memory maps. `verify` checks the arrays against the
    report (once per run; the files are large)."""

    def __init__(self, folder, verify=False):
        folder = Path(folder)
        self.report = json.loads((folder / "report.json").read_text())
        if self.report["format"] != FORMAT:
            raise ValueError(f"{folder} is not a {FORMAT} set")
        if verify:
            for name, expected in self.report["sha256"].items():
                if digest(folder / f"{name}.npy") != expected:
                    raise ValueError(f"{name}.npy differs from its report")
        self.tokens = np.load(folder / "tokens.npy", mmap_mode="r")
        self.answers = np.load(folder / "answers.npy")
        self.ids = np.load(folder / "teacher_ids.npy", mmap_mode="r")
        self.logprobs = np.load(folder / "teacher_logprobs.npy", mmap_mode="r")
        self.offsets, _ = score_offsets(self.answers)
        self.sha256 = hashlib.sha256(
            json.dumps(self.report["sha256"], sort_keys=True).encode()
        ).hexdigest()

    def __len__(self):
        return len(self.answers)

    def lengths(self, index):
        """(prompt tokens, answer tokens) of answer `index`."""
        return int(self.answers[index, 1]), int(self.answers[index, 2])

    def answer(self, index):
        """Answer `index`: its prompt ids, its tokens without the stop token, the stop
        token (None if unfinished) and the teacher's (ids, log-probabilities) at every
        answer token, the stop token included."""
        start, prompt, length, finished = (int(v) for v in self.answers[index])
        tokens = self.tokens[start : start + prompt + length].tolist()
        first = int(self.offsets[index])
        teacher = (
            np.asarray(self.ids[first : first + length]),
            np.asarray(self.logprobs[first : first + length], dtype=np.float32),
        )
        stop = tokens[-1] if finished else None
        completion = tokens[prompt : prompt + length - bool(finished)]
        return tokens[:prompt], completion, stop, teacher
