"""Bounded validation harness; no test or run infrastructure lives in model.py."""

import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from model_100m.model import GPT, ModelConfig, causal_attention
from torch.nn import functional as F
from training_data.loading import CharacterTokenizer

from cli import build_parser
from trainer.checkpoints import load_checkpoint, validate_resume
from trainer.config import TrainConfig
from trainer.loop import train
from trainer.sampling import sample


def self_test():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    options = build_parser().parse_args([])
    assert options.config is None and options.resume is None
    assert set(vars(options)) == {"config", "resume"}
    assert (
        Path(TrainConfig().data) == Path(__file__).resolve().parent / "data/raw/tinyshakespeare.txt"
    )
    subprocess.run(
        [sys.executable, "-c", "import cli, sys; assert 'trainer.loop' not in sys.modules"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    help_result = subprocess.run(
        [sys.executable, "-m", "cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "--config" in help_result.stdout and "--resume" in help_result.stdout
    print("PASS thin training CLI, no training on import, and default dataset path")
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
    logits = model(x)
    assert isinstance(logits, torch.Tensor) and logits.shape == (1, 8, 4)
    torch.testing.assert_close(logits[:, :4], model(changed)[:, :4], rtol=0, atol=0)
    y = (x + 1) % 4
    initial = F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        optimizer.step()
    final = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    assert final < initial * 0.4, (initial, final)
    assert torch.isfinite(model(x)).all()
    tokenizer = CharacterTokenizer(list("abcd"))
    before_sampling = torch.get_rng_state().clone()
    assert len(sample(model, tokenizer, "ab", 12, 123)) == 14
    assert len(sample(model, tokenizer, "", 12, 123)) == 12
    assert torch.equal(before_sampling, torch.get_rng_state())
    print(
        f"PASS causality, fixed-batch fit ({initial:.4f} -> {final:.4f}), generation and isolated RNG"
    )

    def model_from_checkpoint(checkpoint):
        model = GPT(ModelConfig(**checkpoint["config"]["model"]))
        model.load_state_dict(checkpoint["model"])
        return model

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
            saved = load_checkpoint(root / directory / "checkpoints" / f"step-{steps:06d}.pt")
            count = sum(tensor.numel() for tensor in saved["model"].values())
            assert saved["config"]["parameter_count"] == count
            assert (
                json.loads((root / directory / "config.json").read_text())["parameter_count"]
                == count
            )
            return saved

        full = run("full", 8)
        assert full["format_version"] == 1
        hashes = full["config"]["runtime"]["source_hashes"]
        for name, filename in (
            ("training", "trainer/loop.py"),
            ("training_config", "trainer/config.py"),
            ("checkpoints", "trainer/checkpoints.py"),
            ("sampling", "trainer/sampling.py"),
            ("cli", "cli.py"),
        ):
            source = Path(__file__).resolve().parent / filename
            assert hashes[name] == hashlib.sha256(source.read_bytes()).hexdigest()
        invalid_path = root / "unsupported.pt"
        torch.save({"format_version": 2}, invalid_path)
        try:
            load_checkpoint(invalid_path)
        except ValueError as error:
            assert "unsupported checkpoint format" in str(error)
        else:
            raise AssertionError("unsupported checkpoint format was accepted")
        changed = copy.deepcopy(full["config"])
        changed["training"]["lr"] *= 2
        for config, steps, message in (
            (changed, 9, "identical data"),
            (full["config"], 7, "smaller than"),
        ):
            try:
                validate_resume(full, config, full["dataset_fingerprint"], steps)
            except ValueError as error:
                assert message in str(error)
            else:
                raise AssertionError("incompatible resume settings were accepted")
        settings_path = root / "training.json"
        settings_path.write_text(json.dumps({"data": str(data), "steps": 12}))
        with (
            patch(
                "sys.argv", ["lm-train", "--config", str(settings_path), "--resume", "trusted.pt"]
            ),
            patch("trainer.loop.train") as launch,
        ):
            from cli import main

            main()
            launched = launch.call_args.args[0]
            assert launched.data == str(data) and launched.steps == 12
            assert launched.resume == "trusted.pt"
        generated = subprocess.run(
            [
                sys.executable,
                "-c",
                "from cli import generate; generate()",
                "--checkpoint",
                str(root / "full" / "checkpoints" / "step-000008.pt"),
                "--prompt",
                "ab",
                "--max-new-tokens",
                "12",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (
            generated.stdout
            == sample(
                model_from_checkpoint(full),
                CharacterTokenizer(full["tokenizer"]["characters"]),
                "ab",
                12,
                full["config"]["training"]["seed"] + 2,
            )
            + "\n"
        )
        print(
            "PASS checkpoint format, resume guards, source hashes, CLI dispatch and generation CLI"
        )
        # Check UI dispatch without allocating the full-sized architecture or starting a server.
        from ui import APIError, RunManager

        ui_root = (root / "ui").resolve()
        (ui_root / "data").mkdir(parents=True)
        (ui_root / "trainer").mkdir()
        (ui_root / "cli.py").touch()
        (ui_root / "trainer" / "loop.py").touch()
        (ui_root / "data" / "fixture.txt").write_text(data.read_text())
        (ui_root / "data" / "datasets.json").write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "id": "fixture",
                            "name": "Test fixture",
                            "path": "fixture.txt",
                            "description": "Temporary test data",
                            "source": "Original test text",
                            "license_notes": "Test-only fixture",
                            "languages": ["en"],
                            "training_eligible": True,
                        }
                    ]
                }
            )
        )
        manager = RunManager(ui_root)
        try:
            with patch("ui.subprocess.Popen", side_effect=OSError("test launch")) as launch:
                try:
                    manager.start({"dataset_id": "fixture", "steps": 2})
                except APIError as error:
                    assert error.status == 503 and "test launch" in error.message
                else:
                    raise AssertionError("expected test launch failure")
                argv = launch.call_args.args[0]
                assert argv[:4] == [sys.executable, "-u", str(ui_root / "cli.py"), "--config"]
                assert launch.call_args.kwargs["cwd"] == ui_root
                ui_settings = json.loads(Path(argv[4]).read_text())
                assert ui_settings["steps"] == 2
                assert ui_settings["data"] == str(ui_root / "data" / "fixture.txt")
        finally:
            manager.close()
        print("PASS UI launches the thin CLI with dataset and run settings")
        split = run("split", 3)
        assert split["step"] == 3
        resumed = run("split", 8, root / "split" / "checkpoints" / "step-000003.pt")
        assert resumed["step"] == full["step"] == 8
        for key in ("model", "optimizer", "rng", "dataset_fingerprint", "train_loss"):
            assert_same(full[key], resumed[key])
        reloaded = GPT(ModelConfig(**full["config"]["model"]))
        reloaded.load_state_dict(full["model"])
        assert torch.isfinite(reloaded(x)).all()
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
            run("rejected", 9, root / "split" / "checkpoints" / "step-000008.pt")
        except ValueError as error:
            assert "identical data" in str(error)
        else:
            raise AssertionError("changed corpus was accepted")
    print(
        "PASS checkpoint roundtrip, exact CPU resume, metrics, data identity and overwrite protection"
    )
