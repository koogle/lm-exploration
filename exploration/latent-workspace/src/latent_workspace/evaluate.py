import time

import torch
from torch.nn import functional as F

from .models import loss_and_logits, sample_block


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(head, backbone, data, cfg, device, *, anchor_diagnostics=False):
    head.eval()
    backbone.conditioning = head.condition
    nll = torch.zeros(cfg.block_size, device=device)
    accuracy = torch.zeros_like(nll)
    exact, count, generated_exact = 0, 0, 0
    generated_accuracy = torch.zeros_like(nll)
    per_example = []
    prefix_agreement = []
    anchored_agreement = []
    ordered_agreement = []
    chain_nll = torch.zeros(cfg.block_size, device=device)
    for rows in data.split(cfg.batch_size):
        rows = rows.to(device)
        prefix, target = rows[:, : cfg.context_length], rows[:, cfg.context_length :]
        memory, _ = backbone.encode(prefix)
        # Identical all-masked context-only metric for EVERY architecture.
        _, logits, _ = loss_and_logits(head, backbone, memory, target, all_masked=True)
        ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
        hits = logits.argmax(-1) == target
        proposed = sample_block(head, backbone, memory)
        generated_hits = proposed == target
        teacher_choices = backbone.teacher_logits(prefix, proposed).argmax(-1)
        prefix_agreement.extend(
            (proposed == teacher_choices).long().cumprod(-1).sum(-1).cpu().tolist()
        )
        if anchor_diagnostics and head.kind == "diffusion":
            anchored = sample_block(head, backbone, memory, anchor_first=True)
            expected = backbone.teacher_logits(prefix, anchored).argmax(-1)
            anchored_agreement.extend(
                (anchored == expected).long().cumprod(-1).sum(-1).cpu().tolist()
            )
            ordered = sample_block(head, backbone, memory, anchor_first=True, left_to_right=True)
            expected = backbone.teacher_logits(prefix, ordered).argmax(-1)
            ordered_agreement.extend(
                (ordered == expected).long().cumprod(-1).sum(-1).cpu().tolist()
            )
            for position in range(cfg.block_size):
                masked = (
                    torch.arange(cfg.block_size, device=device)[None].expand(len(rows), -1)
                    >= position
                )
                features = head(
                    memory,
                    backbone.embed(target.masked_fill(masked, 0)),
                    masked,
                    masked.float().mean(-1),
                )
                conditional_logits = backbone.project(features)[:, position].float()
                chain_nll[position] += F.cross_entropy(
                    conditional_logits, target[:, position], reduction="sum"
                )
        generated_accuracy += generated_hits.sum(0)
        generated_exact += generated_hits.all(-1).sum().item()
        nll += ce.sum(0)
        accuracy += hits.sum(0)
        exact += hits.all(-1).sum().item()
        count += len(rows)
        per_example.extend(ce.mean(-1).cpu().tolist())
    values = torch.tensor(per_example)
    result = {
        "nll": float(nll.sum() / (count * cfg.block_size)),
        "nll_by_position": (nll / count).tolist(),
        "accuracy_by_position": (accuracy / count).tolist(),
        "exact_block_accuracy": exact / count,
        "generated_accuracy_by_position": (generated_accuracy / count).tolist(),
        "generated_exact_block_accuracy": generated_exact / count,
        "qwen_greedy_prefix_agreement": sum(prefix_agreement) / count,
        "qwen_greedy_full_block_agreement": sum(n == cfg.block_size for n in prefix_agreement)
        / count,
        "qwen_greedy_prefix_lengths": prefix_agreement,
        "agreement_note": "Read-only comparison along proposed blocks; no verifier in generation and no speed claim.",
        "examples": count,
        "example_mean_nll": per_example,
        "nll_standard_error": float(values.std(unbiased=True) / count**0.5) if count > 1 else None,
    }
    if anchored_agreement:
        result["anchored_qwen_prefix_agreement"] = sum(anchored_agreement) / count
        result["anchored_new_draft_agreement"] = (
            sum(max(0, n - 1) for n in anchored_agreement) / count
        )
        result["anchored_full_block_agreement"] = (
            sum(n == cfg.block_size for n in anchored_agreement) / count
        )
        result["anchored_prefix_lengths"] = anchored_agreement
        result["anchored_left_to_right_prefix_agreement"] = sum(ordered_agreement) / count
        result["anchored_left_to_right_new_draft_agreement"] = (
            sum(max(0, n - 1) for n in ordered_agreement) / count
        )
        result["teacher_forced_drafter_nll_by_position"] = (chain_nll / count).tolist()
        result["teacher_forced_drafter_note"] = (
            "Earlier reference tokens visible, later tokens masked; comparable information to teacher-forced Qwen, not all-masked prediction or actual generation quality."
        )
        result["anchor_note"] = (
            "One token supplied by Qwen, then three drafted tokens; not four newly drafted tokens. Same trained checkpoint, changed inference policy only."
        )
    return result


@torch.no_grad()
def teacher_baseline(backbone, data, cfg, device):
    nll = torch.zeros(cfg.block_size, device=device)
    hits = torch.zeros_like(nll)
    for rows in data.split(cfg.batch_size):
        rows = rows.to(device)
        prefix, target = rows[:, : cfg.context_length], rows[:, cfg.context_length :]
        logits = backbone.teacher_logits(prefix, target).float()
        nll += F.cross_entropy(logits.transpose(1, 2), target, reduction="none").sum(0)
        hits += (logits.argmax(-1) == target).sum(0)
    return {
        "nll_by_position": (nll / len(data)).tolist(),
        "accuracy_by_position": (hits / len(data)).tolist(),
        "note": "Teacher-forced AR reference sees true earlier target tokens; not a context-only comparator.",
    }


@torch.no_grad()
def generate(head, backbone, prefix, max_new_tokens=32, temperature=0.0, anchor_first=False):
    head.eval()
    backbone.conditioning = head.condition
    memory, cache = backbone.encode(prefix)
    generated = []
    eos = backbone.tokenizer.eos_token_id
    while len(generated) < max_new_tokens:
        block = sample_block(head, backbone, memory, temperature, anchor_first=anchor_first)[0]
        tokens = block.tolist()[: max_new_tokens - len(generated)]
        if eos in tokens:
            generated.extend(tokens[: tokens.index(eos) + 1])
            break
        generated.extend(tokens)
        if len(generated) < max_new_tokens:
            fresh, cache = backbone.encode(block[None], past_key_values=cache)
            memory = torch.cat([memory, fresh], dim=2)[:, :, -head.cfg.memory_tokens :]
    return generated


@torch.no_grad()
def benchmark(head, backbone, prefix, tokens=32, repeats=3):
    """Warm, end-to-end latency includes Qwen prefix and committed-block updates."""
    device = prefix.device

    def ar():
        return backbone.model.generate(
            prefix.clamp(min=0),
            attention_mask=prefix.ge(0).long(),
            max_new_tokens=tokens,
            min_new_tokens=tokens,
            do_sample=False,
            pad_token_id=backbone.tokenizer.eos_token_id,
        )

    def block():
        return generate(head, backbone, prefix, tokens)

    report = {}
    for name, fn in (("autoregressive", ar), ("block", block)):
        fn()
        times, lengths = [], []
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            output = fn()
            sync(device)
            times.append(time.perf_counter() - start)
            lengths.append(
                output.shape[-1] - prefix.shape[-1] if name == "autoregressive" else len(output)
            )
        report[name] = {
            "seconds": times,
            "output_tokens": lengths,
            "seconds_per_token": sum(times) / sum(lengths),
        }
    report["note"] = (
        "Unverified block quality; timing alone does not establish efficiency at matched quality."
    )
    return report


@torch.no_grad()
def repeated_next_token_baseline(backbone, data, cfg, device):
    """Zero-training, prefix-only baseline: repeat Qwen's next-token distribution."""
    nll = torch.zeros(cfg.block_size, device=device)
    hits = torch.zeros_like(nll)
    for rows in data.split(cfg.batch_size):
        rows = rows.to(device)
        memory, _ = backbone.encode(rows[:, : cfg.context_length])
        final = memory[:, len(cfg.layers) - 1, -1]
        logits = backbone.project(final).float()[:, None].expand(-1, cfg.block_size, -1)
        target = rows[:, cfg.context_length :]
        nll += F.cross_entropy(logits.transpose(1, 2), target, reduction="none").sum(0)
        hits += (logits.argmax(-1) == target).sum(0)
    return {
        "nll": float(nll.mean() / len(data)),
        "nll_by_position": (nll / len(data)).tolist(),
        "accuracy_by_position": (hits / len(data)).tolist(),
        "note": "Prefix-only; repeat the same Qwen next-token distribution at every block position. No training or target access.",
    }
