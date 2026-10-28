"""Model-agnostic scoring: continuation log-likelihoods, batched greedy generation,
few-shot answer checking, code pass rates and bits per byte. Dataset formats live in
`benchmarks/`.

A model is any object with `forward(tokens) -> logits` over a `[batch, time]` LongTensor,
`encode(text) -> ids` (no special tokens), `decode(ids) -> text`, `bos_token_id`,
`eos_token_id`, `pad_token_id`, `block_size` and `device`; generation also uses
`network`, the decoder itself, which the shared sampler (`trainer/rollout.py`) runs.
"""

import math

# Requests per forward pass are bounded by total padded tokens, not by count.
TOKENS_PER_BATCH = 16384
# Prompts decoded together by the shared sampler.
GENERATION_BATCH = 64


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


def greedy_completions(model, prompts, max_new_tokens, stops):
    """Greedy continuations of text `prompts`, each ending at EOS or `max_new_tokens` and
    cut before the first of the `stops` strings. Prompts keep their last tokens if they
    would not fit the context with the new tokens."""
    from trainer.rollout import sample

    def stopped(ids):
        text = model.decode(ids)
        return any(stop in text for stop in stops)

    limit = model.block_size - max_new_tokens
    texts = []
    for start in range(0, len(prompts), GENERATION_BATCH):
        chunk = [
            [model.bos_token_id, *model.encode(prompt)][-limit:]
            for prompt in prompts[start : start + GENERATION_BATCH]
        ]
        samples = sample(model.network, chunk, max_new_tokens, model.eos_token_id, 0, stop=stopped)
        for ids in samples.completions:
            text = model.decode(ids)
            cut = min((text.index(stop) for stop in stops if stop in text), default=len(text))
            texts.append(text[:cut])
    return texts


def choice_accuracy(model, items):
    """Accuracy over (context, [continuations], gold index) items, EleutherAI harness style.

    `acc` picks the continuation with the highest summed log-probability; `acc_norm`
    divides that sum by the continuation's UTF-8 byte length first.
    """
    requests, spans = [], []
    for context, choices, gold in items:
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
    count = len(items)
    return {"examples": count, "acc": correct / count, "acc_norm": normalized / count}


def few_shot_answers(model, problems, exemplars, max_new_tokens):
    """Greedy answers to (question, gold answer text) problems after worked exemplars.

    `exemplars` are (question, worked answer) pairs, shown as `Question: ...\nAnswer: ...`
    blocks before the problem's own `Question: ...\nAnswer:`. Generation stops at a blank
    line, `Question:` or EOS. `math_verify` extracts the final answer from the completion
    and the gold text and checks that they are equal.
    """
    from training_data.math_sft import answer_matches, question_prompt

    shots = "".join(f"{question_prompt(q)} {worked}\n\n" for q, worked in exemplars)
    outcomes = []
    completions = greedy_completions(
        model,
        [shots + question_prompt(question) for question, _ in problems],
        max_new_tokens,
        ["\n\n", "Question:"],
    )
    for (question, gold), completion in zip(problems, completions, strict=True):
        answer, correct = answer_matches(completion, gold)
        outcomes.append(
            {
                "question": question,
                "gold": gold,
                "answer": answer,
                "correct": correct,
                "completion": completion,
            }
        )
    accuracy = sum(o["correct"] for o in outcomes) / len(outcomes)
    return {"examples": len(problems), "shots": len(exemplars), "acc": accuracy}, outcomes


def code_answers(model, problems, sandbox, max_new_tokens):
    """Greedy pass@1 of (question, declaration, tests, preamble) function problems.

    The prompt is the code SFT format (`training_data.code_sft.code_prompt`) and the
    completion, up to EOS, is the program; `preamble` (the benchmark's imports and
    helpers) goes before it. A program passes if every test passes in `sandbox`
    (`training_data.sandbox`); a completion without EOS fails.
    """
    from training_data.code_sft import code_prompt

    from trainer.rollout import sample

    completions, finished = [], []
    for start in range(0, len(problems), GENERATION_BATCH):
        chunk = problems[start : start + GENERATION_BATCH]
        prompts = [
            [model.bos_token_id, *model.encode(code_prompt(p["question"], p["declaration"]))]
            for p in chunk
        ]
        samples = sample(model.network, prompts, max_new_tokens, model.eos_token_id, 0)
        completions += [model.decode(i) for i in samples.completions]
        finished += samples.finished
    programs = [p["preamble"] + c for p, c in zip(problems, completions, strict=True)]
    results = sandbox.run([(c, p["tests"]) for c, p in zip(programs, problems, strict=True)], 20)
    outcomes = [
        {
            "question": p["question"],
            "completion": c,
            "finished": f,
            "passed": f and r["passed"],
            "status": r["status"],
        }
        for p, c, f, r in zip(problems, completions, finished, results, strict=True)
    ]
    accuracy = sum(o["passed"] for o in outcomes) / len(outcomes)
    return {"examples": len(problems), "pass@1": accuracy}, outcomes


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
