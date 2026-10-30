"""One-H100 gate for document-masked attention in the shared 10M/100M decoder.

Prints one JSON report and raises on any failed check.
"""

import json
import time

import torch
from model_100m import model as modern
from torch.nn import functional as F

from trainer.gpu_benchmark import explicit_document_attention

EOS = 2


def packed(rows, length, every):
    tokens = torch.randint(4, 32768, (rows, length), device="cuda")
    tokens[:, every - 1 :: every] = EOS
    return tokens


def attention_against_explicit():
    """bf16 varlen kernel vs fp32 per-document softmax attention, outputs and gradients."""
    torch.manual_seed(0)
    tokens = torch.full((3, 1000), 5, device="cuda")
    tokens[0, [99, 450, 451, 998]] = EOS  # includes a one-token document
    tokens[2, 999] = EOS  # a final EOS starts no document inside the row
    tensors = [
        torch.randn(3, 1000, 12, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    documents = modern.find_documents(tokens, EOS)
    actual = modern.document_attention(*tensors, documents)
    expected = explicit_document_attention(*tensors, tokens, EOS)
    upstream = torch.randn_like(actual)
    errors = {"out": (actual.float() - expected.float()).abs().max().item()}
    for name, a, b in zip(
        ("dq", "dk", "dv"),
        torch.autograd.grad(actual, tensors, upstream),
        torch.autograd.grad(expected, tensors, upstream),
    ):
        errors[name] = (a.float() - b.float()).abs().max().item()
    if max(errors.values()) > 0.06:
        raise AssertionError(f"varlen attention differs from the explicit reference: {errors}")
    return errors


def compiled_model():
    """The trainer's compile(dynamic=True) path vs eager, and document isolation, in bf16."""
    torch.manual_seed(1)
    config = modern.ModelConfig(32768, 1024, 2, 4, 256, eos_id=EOS)
    eager = modern.GPT(config).cuda()
    compiled = modern.GPT(config).cuda()
    compiled.load_state_dict(eager.state_dict())
    compiled.compile(dynamic=True)
    b = torch.randint(4, 32768, (1, 300), device="cuda")
    first = torch.cat((packed(1, 200, 200), b), 1)
    second = torch.cat((packed(1, 500, 500), b), 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        isolation = (compiled(first)[0, 200:] - compiled(second)[0, 500:]).abs().max().item()
        tokens = packed(4, 1024, 300)
        labels = torch.randint(4, 32768, tokens.shape, device="cuda")
        losses = [model(tokens, labels) / labels.numel() for model in (eager, compiled)]
    for loss in losses:
        loss.backward()
    grads = [
        torch.cat([p.grad.flatten() for p in model.parameters()]) for model in (eager, compiled)
    ]
    cosine = F.cosine_similarity(grads[0], grads[1], dim=0).item()
    report = {
        "isolation_max_logit_difference": isolation,
        "eager_loss": losses[0].item(),
        "compiled_loss": losses[1].item(),
        "gradient_cosine": cosine,
    }
    if isolation > 0.05 or abs(losses[0].item() - losses[1].item()) > 1e-2 or cosine < 0.999:
        raise AssertionError(f"compiled document masking differs: {report}")
    return report


def timing(rows=16, length=2048, heads=12, width=64, steps=20):
    """Forward+backward of attention alone: varlen with 1 or ~2 documents/row vs causal SDPA."""
    tensors = [
        torch.randn(rows, length, heads, width, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    tensors = [t.requires_grad_() for t in tensors]

    def measure(function):
        for _ in range(3):
            function().sum().backward()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            function().sum().backward()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - start) / steps

    def causal():
        q, k, v = (t.transpose(1, 2) for t in tensors)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    results = {"shape": [rows, length, heads, width], "causal_sdpa_ms": measure(causal)}
    for label, every in (("varlen_1_doc_ms", length + 1), ("varlen_1000_token_docs_ms", 1000)):
        tokens = torch.full((rows, length), 5, device="cuda")
        tokens[:, every - 1 :: every] = EOS
        documents = modern.find_documents(tokens, EOS)
        results[label] = measure(lambda d=documents: modern.document_attention(*tensors, d))
    return results


report = {
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "attention_vs_explicit_bf16": attention_against_explicit(),
    "compiled_model": compiled_model(),
    "timing": timing(),
}
print(json.dumps(report, indent=2))
