"""Benchmark tasks scored by log-likelihood or greedy generation, independent of Modal.

A model is any object with `forward(tokens) -> logits` over a `[batch, time]` LongTensor,
`encode(text) -> ids` (no special tokens), `decode(ids) -> text`, `bos_token_id`,
`eos_token_id`, `pad_token_id`, `block_size` and `device`.

Multiple-choice tasks follow the EleutherAI evaluation harness formats: each choice is a
continuation of a shared context, scored by its summed log-probability (`acc`) and by
that sum divided by the continuation's UTF-8 byte length (`acc_norm`). Generation tasks
use fixed few-shot exemplars from the task's training split and greedy decoding.
"""

import math
import re

# Requests per forward pass are bounded by total padded tokens, not by count.
TOKENS_PER_BATCH = 16384


def loglikelihoods(model, requests):
    """(summed log-probability, greedy match) of each (context ids, continuation ids)."""
    import torch

    results = [None] * len(requests)
    sequences = []
    for index, (context, continuation) in enumerate(requests):
        if not continuation:
            raise ValueError("Empty continuation")
        tokens = [model.bos_token_id, *context, *continuation]
        # Keep the continuation and as much context as fits, plus one predicting token.
        tokens = tokens[-(model.block_size + 1) :]
        sequences.append((index, tokens, len(continuation)))
    sequences.sort(key=lambda item: -len(item[1]))
    start = 0
    while start < len(sequences):
        width = len(sequences[start][1]) - 1
        count = max(1, TOKENS_PER_BATCH // width)
        batch = sequences[start : start + count]
        start += count
        inputs = torch.full((len(batch), width), model.pad_token_id, dtype=torch.long)
        for row, (_, tokens, _) in enumerate(batch):
            inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
        with torch.no_grad():
            logits = model.forward(inputs.to(model.device)).float()
        for row, (index, tokens, length) in enumerate(batch):
            end = len(tokens) - 1
            scores = torch.log_softmax(logits[row, end - length : end], dim=-1)
            targets = torch.tensor(tokens[-length:], device=scores.device)
            total = scores.gather(1, targets[:, None]).sum().item()
            greedy = bool((scores.argmax(-1) == targets).all())
            results[index] = (total, greedy)
    return results


def generate(model, prompt, max_new_tokens, stop):
    """Greedy continuation of `prompt` until EOS, `max_new_tokens` or any `stop` string."""
    import torch

    tokens = [model.bos_token_id, *model.encode(prompt)]
    generated = []
    for _ in range(max_new_tokens):
        window = torch.tensor([tokens[-model.block_size :]], device=model.device)
        with torch.no_grad():
            token = int(model.forward(window)[0, -1].argmax())
        if token == model.eos_token_id:
            break
        tokens.append(token)
        generated.append(token)
        text = model.decode(generated)
        if any(s in text for s in stop):
            break
    text = model.decode(generated)
    cut = min((text.index(s) for s in stop if s in text), default=len(text))
    return text[:cut]


def hellaswag_text(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def multiple_choice(task, row):
    """(context, [continuations], gold index) for one row of a multiple-choice task."""
    if task == "hellaswag":
        context = hellaswag_text(
            row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
        )
        return context, [" " + hellaswag_text(e) for e in row["endings"]], int(row["label"])
    if task in ("arc_easy", "arc_challenge"):
        choices = row["choices"]
        context = f"Question: {row['question']}\nAnswer:"
        gold = list(choices["label"]).index(row["answerKey"])
        return context, [" " + t for t in choices["text"]], gold
    if task == "openbookqa":
        choices = row["choices"]
        gold = list(choices["label"]).index(row["answerKey"])
        return row["question_stem"], [" " + t for t in choices["text"]], gold
    raise ValueError(f"Unknown multiple-choice task {task}")


def score_multiple_choice(model, task, rows):
    requests, spans = [], []
    for row in rows:
        context, choices, gold = multiple_choice(task, row)
        ids = model.encode(context)
        spans.append((len(requests), choices, gold))
        requests += [(ids, model.encode(choice)) for choice in choices]
    scores = loglikelihoods(model, requests)
    correct = normalized = 0
    for start, choices, gold in spans:
        totals = [scores[start + i][0] for i in range(len(choices))]
        lengths = [len(c.encode()) for c in choices]
        correct += max(range(len(totals)), key=totals.__getitem__) == gold
        normalized += max(range(len(totals)), key=lambda i: totals[i] / lengths[i]) == gold
    return {"examples": len(rows), "acc": correct / len(rows), "acc_norm": normalized / len(rows)}


def score_winogrande(model, rows):
    """Partial scoring: the shared continuation after the blank, under each filled context."""
    requests = []
    for row in rows:
        sentence = row["sentence"]
        blank = sentence.index("_")
        continuation = model.encode(" " + sentence[blank + 1 :].strip())
        for option in (row["option1"], row["option2"]):
            requests.append((model.encode(sentence[:blank] + option), continuation))
    scores = loglikelihoods(model, requests)
    correct = sum(
        (scores[2 * i][0] < scores[2 * i + 1][0]) == (row["answer"] == "2")
        for i, row in enumerate(rows)
    )
    return {"examples": len(rows), "acc": correct / len(rows)}


def score_lambada(model, rows):
    """Last-word accuracy (greedy match of every target token) and target perplexity."""
    requests = []
    for row in rows:
        context, word = row["text"].rsplit(" ", 1)
        requests.append((model.encode(context), model.encode(" " + word)))
    scores = loglikelihoods(model, requests)
    tokens = sum(len(continuation) for _, continuation in requests)
    return {
        "examples": len(rows),
        "acc": sum(greedy for _, greedy in scores) / len(rows),
        "perplexity": math.exp(-sum(total for total, _ in scores) / tokens),
    }


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def number(text):
    if text is None:
        return None
    value = float(text.replace(",", ""))
    return int(value) if value.is_integer() else value


def gsm8k_answer(text):
    """The number after `####`, else the last number in the text."""
    marked = re.search(r"####\s*(-?[\d,\.]+)", text)
    found = marked.group(1).rstrip(".") if marked else (NUMBER.findall(text) or [None])[-1]
    try:
        return number(found)
    except ValueError:
        return None


def math_prompt(task, row):
    """(question text, gold answer, worked exemplar answer) for a math row."""
    if task == "gsm8k":
        return row["question"], gsm8k_answer(row["answer"]), row["answer"]
    if task == "svamp":
        question = f"{row['Body'].strip()} {row['Question'].strip()}"
        gold = number(str(row["Answer"]))
        return question, gold, str(gold)
    raise ValueError(f"Unknown math task {task}")


def score_math(model, task, rows, exemplars, max_new_tokens):
    """Greedy few-shot answers; exact match on the extracted number."""
    shots = "".join(
        f"Question: {q}\nAnswer: {worked}\n\n"
        for q, _, worked in (math_prompt(task, row) for row in exemplars)
    )
    outcomes = []
    for row in rows:
        question, gold, _ = math_prompt(task, row)
        completion = generate(
            model, f"{shots}Question: {question}\nAnswer:", max_new_tokens, ["\n\n", "Question:"]
        )
        found = gsm8k_answer(completion) if task == "gsm8k" else first_number(completion)
        outcomes.append(
            {"question": question, "gold": gold, "answer": found, "completion": completion}
        )
    correct = sum(o["answer"] is not None and o["answer"] == o["gold"] for o in outcomes)
    return {"examples": len(rows), "shots": len(exemplars), "acc": correct / len(rows)}, outcomes


def first_number(text):
    found = NUMBER.search(text)
    try:
        return number(found.group()) if found else None
    except ValueError:
        return None


def packed_windows(documents, bos, eos, context, count):
    """The first `count` training-style windows of tokenized `documents` (lists of ids).

    Documents are concatenated as `BOS ids EOS`; each window holds `context + 1` tokens
    and the next starts at its last target, as in the raw loader.
    """
    buffer, windows = [], []
    for ids in documents:
        buffer += [bos, *ids, eos]
        while len(buffer) > context and len(windows) < count:
            windows.append(buffer[: context + 1])
            del buffer[:context]
        if len(windows) == count:
            break
    return windows


def bits_per_byte(model, windows):
    """Summed target loss in bits over the UTF-8 bytes the targets decode to.

    BOS is never a target, as in training; EOS targets count toward the loss but decode
    to no bytes.
    """
    import torch

    bits = 0.0
    tokens = text_bytes = 0
    for start in range(0, len(windows), max(1, TOKENS_PER_BATCH // (model.block_size + 1))):
        batch = windows[start : start + max(1, TOKENS_PER_BATCH // (model.block_size + 1))]
        width = max(len(w) for w in batch) - 1
        inputs = torch.full((len(batch), width), model.pad_token_id, dtype=torch.long)
        targets = torch.full((len(batch), width), -100, dtype=torch.long)
        for row, window in enumerate(batch):
            inputs[row, : len(window) - 1] = torch.tensor(window[:-1])
            labels = [t if t != model.bos_token_id else -100 for t in window[1:]]
            targets[row, : len(window) - 1] = torch.tensor(labels)
        with torch.no_grad():
            logits = model.forward(inputs.to(model.device)).float()
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.to(logits.device).flatten(), reduction="sum"
        )
        bits += loss.item() / math.log(2)
        for window in batch:
            kept = [t for t in window[1:] if t != model.bos_token_id]
            tokens += len(kept)
            text_bytes += len(model.decode(kept).encode())
    return {
        "windows": len(windows),
        "target_tokens": tokens,
        "bytes": text_bytes,
        "bits_per_byte": bits / text_bytes,
        "nats_per_token": bits * math.log(2) / tokens,
    }
