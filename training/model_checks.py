"""Bounded validation harness; no test or run infrastructure lives in model.py."""

import json
import math
import tempfile
from pathlib import Path

import torch
from model_100m.model import GPT, ModelConfig, causal_attention
from torch.nn import functional as F
from training_data.loading import CharacterTokenizer

from run_utils import load_checkpoint
from train import TrainConfig, build_parser, sample, train


def self_test():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    options = build_parser().parse_args([])
    assert options.config is None and options.resume is None
    assert set(vars(options)) == {"config", "resume"}
    print("PASS no-argument training CLI; architecture is not a command-line setting")
    with torch.device("meta"):
        configured = GPT(ModelConfig(256))
    count = sum(parameter.numel() for parameter in configured.parameters())
    assert 95_000_000 <= count <= 105_000_000, count
    assert all(parameter.is_meta for parameter in configured.parameters())
    print(
        f"PASS 100M default architecture: {count:,} parameters (meta device, no weight allocation)"
    )

    tensors = [torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3)]
    reference = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    actual = causal_attention(*tensors)
    expected = F.scaled_dot_product_attention(*reference, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for tensor, ref in zip(tensors, reference):
        torch.testing.assert_close(tensor.grad, ref.grad, rtol=1e-5, atol=1e-6)
    print("PASS attention outputs and Q/K/V gradients vs SDPA")

    model = GPT(ModelConfig(4, 8, 1, 2, 16))
    x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 4
    torch.testing.assert_close(model(x)[0][:, :4], model(changed)[0][:, :4], rtol=0, atol=0)
    y = (x + 1) % 4
    initial = model(x, y)[1].item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    final = model(x, y)[1].item()
    assert final < initial * 0.4, (initial, final)
    assert torch.isfinite(model(x)[0]).all()
    tokenizer = CharacterTokenizer(list("abcd"))
    before_sampling = torch.get_rng_state().clone()
    assert len(sample(model, tokenizer, "ab", 12, 123)) == 14
    assert len(sample(model, tokenizer, "", 12, 123)) == 12
    assert torch.equal(before_sampling, torch.get_rng_state())
    print(
        f"PASS causality, fixed-batch fit ({initial:.4f} -> {final:.4f}), generation and isolated RNG"
    )

    def assert_same(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_same(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_same(a, b)
        else:
            assert left == right

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "text.txt"
        data.write_text("abcd abc dab cab\n" * 40, encoding="utf-8")

        def run(directory, steps, resume=None):
            config = TrainConfig(
                data=str(data),
                out_dir=str(root / directory),
                steps=steps,
                batch_size=2,
                lr=0.003,
                seed=42,
                eval_interval=2,
                log_interval=2,
                eval_batches=2,
                checkpoint_interval=2,
                resume=str(resume) if resume else None,
            )
            tiny = ModelConfig(len(set(data.read_text())), 8, 1, 2, 16)
            train(config, model_config=tiny)
            saved = load_checkpoint(root / directory / "checkpoint.pt")
            count = sum(tensor.numel() for tensor in saved["model"].values())
            assert saved["config"]["parameter_count"] == count
            assert (
                json.loads((root / directory / "config.json").read_text())["parameter_count"]
                == count
            )
            return saved

        full = run("full", 8)
        split = run("split", 3)
        assert split["step"] == 3
        resumed = run("split", 8, root / "split" / "checkpoint.pt")
        assert resumed["step"] == full["step"] == 8
        for key in ("model", "optimizer", "rng", "dataset_fingerprint", "train_loss"):
            assert_same(full[key], resumed[key])
        reloaded = GPT(ModelConfig(**full["config"]["model"]))
        reloaded.load_state_dict(full["model"])
        assert torch.isfinite(reloaded(x)[0]).all()
        for directory in ("full", "split"):
            records = [
                json.loads(line)
                for line in (root / directory / "metrics.jsonl").read_text().splitlines()
            ]
            assert all(math.isfinite(value) for record in records for value in record.values())
            assert records[-1]["tokens_seen"] == 8 * 2 * 8
            assert len((root / directory / "sample.txt").read_text()) == 80
        try:
            run("full", 9)
        except ValueError as error:
            assert "already has a run" in str(error)
        else:
            raise AssertionError("existing run was overwritten without --resume")
        data.write_text(data.read_text() + "a", encoding="utf-8")
        try:
            run("rejected", 9, root / "split" / "checkpoint.pt")
        except ValueError as error:
            assert "identical data" in str(error)
        else:
            raise AssertionError("changed corpus was accepted")
    print(
        "PASS checkpoint roundtrip, exact CPU resume, metrics, data identity and overwrite protection"
    )
