"""Matched correctness and timing for device-side token feedback/logit capture."""

import time

import torch


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    return {
        "nrmse": (
            (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8)
        ).item(),
        "max_abs": (a - b).abs().max().item(),
        "top1_fraction": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
        "kl_reference_to_actual": torch.nn.functional.kl_div(
            a.log_softmax(-1), b.log_softmax(-1), log_target=True, reduction="batchmean"
        ).item(),
        "finite": bool(a.isfinite().all()),
    }


@torch.inference_mode()
def sequential_capture(decoder, ids, count):
    decoder.reset()
    for token in ids:
        logits = decoder.step(token)
    tokens, rows = [], []
    for i in range(count):
        token = int(logits.argmax())
        tokens.append(token)
        rows.append(logits.cpu())
        if i + 1 < count:
            logits = decoder.step(token)
    return tokens, torch.stack(rows)


@torch.inference_mode()
def qualify(decoder, tokenizer, out):
    prompts = [
        "Write a Python factorial function. Explain why factorial(0) is 1.",
        "Write Python that removes duplicates preserving the first occurrence. Include empty input.",
        "Return JSON with keys count and valid, containing integer 3 and boolean true.",
    ]
    report = {
        "rows": [],
        "thresholds": {
            "capture_max_abs": 0.0,
            "hidden_projection_nrmse": 0.005,
            "reference_nrmse": 0.04,
            "reference_min_top1": 0.95,
            "reference_max_kl": 0.02,
        },
        "sampling": "greedy, fixed length for matched timing; EOS tested separately",
    }
    for j, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        for count in (32, 128):
            tokens, baseline = sequential_capture(decoder, ids, count)
            # True reference on precisely the same prefix; not free-running divergence.
            cache = None
            ref_rows = []
            for position, token in enumerate(ids + tokens[:-1]):
                result = decoder.model(
                    input_ids=torch.tensor([[token]], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = result.past_key_values
                if position >= len(ids) - 1:
                    ref_rows.append(
                        torch.nn.functional.linear(
                            result.last_hidden_state[0, -1], decoder.head
                        )
                        .float()
                        .cpu()
                    )
            reference = torch.stack(ref_rows)
            reference_error = metrics(baseline, reference)
            del cache, result, reference, ref_rows
            for capture in ("logits", "hidden"):
                got, values = decoder.generate_buffered(ids, count, capture=capture)
                error = metrics(values, baseline)
                times = {"sequential_capture": [], "buffered": []}
                for repeat in range(3):
                    modes = ["sequential_capture", "buffered"]
                    if repeat % 2:
                        modes.reverse()
                    for mode in modes:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        if mode == "sequential_capture":
                            sequential_capture(decoder, ids, count)
                        else:
                            decoder.generate_buffered(ids, count, capture=capture)
                        torch.cuda.synchronize()
                        times[mode].append(time.perf_counter() - started)
                row = {
                    "prompt": prompt,
                    "prompt_tokens": len(ids),
                    "output_tokens": count,
                    "capture": capture,
                    "tokens_equal": tokens == got,
                    "capture_error": error,
                    "reference_error": reference_error,
                    "seconds": times,
                    "output": tokenizer.decode(got),
                    "token_ids": got,
                }
                report["rows"].append(row)
                print(
                    "BUFFER_CHECK",
                    j,
                    count,
                    capture,
                    error,
                    reference_error,
                    flush=True,
                )
                if j == 0 and count == 32 and capture == "logits":
                    torch.save(
                        {"prompt_ids": ids, "output_ids": got, "logits": values},
                        out / "buffered-logits-sample.pt",
                    )
            # EOS at first generated position must return no token or target row.
            eos_ids, eos_values = decoder.generate_buffered(ids, 8, eos=tokens[0])
            if eos_ids or eos_values.shape[0] != 0:
                raise AssertionError("EOS trimming failed")
    report["correctness_passed"] = all(
        r["tokens_equal"]
        and r["capture_error"]["finite"]
        and (
            r["capture_error"]["max_abs"] == 0
            if r["capture"] == "logits"
            else r["capture_error"]["nrmse"] <= 0.005
        )
        and r["reference_error"]["finite"]
        and r["reference_error"]["nrmse"] <= 0.04
        and r["reference_error"]["top1_fraction"] >= 0.95
        and r["reference_error"]["kl_reference_to_actual"] <= 0.02
        for r in report["rows"]
    )
    from megakernel.targets import trajectory

    ids = tokenizer.encode(
        "Explain factorial zero in Python.", add_special_tokens=False
    )
    sampled = trajectory(decoder, ids, max_tokens=32, block_tokens=8, seed=471)
    repeated = trajectory(decoder, ids, max_tokens=32, block_tokens=8, seed=471)
    decoder.reset()
    for token in ids:
        logits = decoder.step(token)
    sparse_ok = sampled == repeated
    max_sparse_error = 0.0
    for i, token in enumerate(sampled["token_ids"]):
        logp = logits.log_softmax(-1).cpu()
        target = sampled["targets"]
        indices = torch.tensor(target["topk_ids"][i])
        delta = (
            (logp[indices] - torch.tensor(target["topk_logprobs"][i]))
            .abs()
            .max()
            .item()
        )
        selected_delta = abs(logp[token].item() - target["selected_logprobs"][i])
        max_sparse_error = max(max_sparse_error, delta, selected_delta)
        sparse_ok &= (
            abs(logp[indices].exp().sum().item() + target["tail_mass"][i] - 1) < 2e-5
        )
        if i + 1 < len(sampled["token_ids"]):
            logits = decoder.step(token)
    eos = trajectory(
        decoder,
        ids,
        max_tokens=32,
        block_tokens=8,
        seed=471,
        eos_ids=[sampled["token_ids"][0]],
    )
    sparse_ok &= len(eos["token_ids"]) == 1 and eos["finish_reason"] == "stop"
    report["sparse_targets"] = {
        "passed": bool(sparse_ok and max_sparse_error < 2e-5),
        "max_logprob_error": max_sparse_error,
        "sample": sampled,
    }
    report["correctness_passed"] &= report["sparse_targets"]["passed"]
    return report
