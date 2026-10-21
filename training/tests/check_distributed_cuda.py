"""Bounded integration checks executed on a three-GPU Modal worker, not CPU training."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch
from training_data.subword import TOKENIZER_SHA256
from training_data.token_shards import FORMAT, file_entry, write_shard

from distributed.config import DistributedConfig


def fixture(root, documents=60):
    root.mkdir()
    audit = root / "audit.json"
    audit.write_text("{}")
    sources = {}
    for name, weight in (("short", 40), ("long", 60)):
        splits = {}
        for split in ("train", "validation"):
            path = root / f"{name}-{split}.tokens"
            count = 0

            def rows(split=split, name=name):
                nonlocal count
                for i in range(documents if split == "train" else 2):
                    ids = list(range(4, 4 + (7 + i % 13 if name == "short" else 45 + i % 19)))
                    count += len(ids) + 1
                    yield {"ids": ids, "hash": hashlib.sha256(f"{name}-{i}".encode()).hexdigest()}

            shard = write_shard(path, rows())
            splits[split] = {
                "loss_tokens": count,
                "shards": [shard],
            }
        sources[name] = {"weight_percent": weight, "splits": splits}
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "token_format": FORMAT,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "training_ready": False,
                "max_passes": 2,
                "sources": sources,
                "audit": file_entry(audit),
            }
        )
    )
    return manifest


def check_equal(a, b, floating_tolerance=False):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(
            a, b, rtol=1e-5 if floating_tolerance else 0, atol=1e-7 if floating_tolerance else 0
        )
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            check_equal(a[key], b[key], floating_tolerance)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            check_equal(x, y, floating_tolerance)
    else:
        assert a == b


def checks(output):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    from trainer.loop import source_hashes

    summary = {
        "source_sha256": source_hashes(),
        "gradient_reference": {},
        "resume": {},
        "token_limit": {},
        "exhaustion": {},
        "failures": {},
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = fixture(root / "data")
        small = fixture(root / "exhaustion", documents=2)

        def run(label, config, *, steps=None, resume=None, source=manifest, fail=False):
            path = root / label
            settings = root / f"{label}.json"
            settings.write_text(json.dumps(config.as_dict()))
            command = [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={config.world_size}",
                "-m",
                "trainer.loop",
                "--config",
                str(settings),
                "--manifest",
                str(source),
                "--output",
                str(path),
                "--max-seconds",
                "120",
            ]
            if steps:
                command.extend(["--max-steps", str(steps)])
            if resume:
                command.extend(["--resume", str(resume)])
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=180
            )
            (output / f"{label}.log").write_text(result.stdout + result.stderr)
            if fail:
                assert result.returncode != 0, "Invalid training request unexpectedly succeeded"
                return result.stdout + result.stderr
            if result.returncode:
                raise RuntimeError(result.stderr[-5000:])
            report = json.loads((path / "report.json").read_text())
            checkpoint = torch.load(
                report["final_checkpoint"], map_location="cpu", weights_only=False
            )
            return report, checkpoint

        for world in (1, 2, 3):
            gradient = subprocess.run(
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={world}",
                    "/project/tests/check_distributed_gradients.py",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            (output / f"w{world}-gradient.log").write_text(gradient.stdout + gradient.stderr)
            if gradient.returncode:
                raise RuntimeError(gradient.stderr[-5000:])
            summary["gradient_reference"][world] = "passed"
            config = DistributedConfig(
                world_size=world,
                global_windows=12,
                microbatch=2,
                context=32,
                target_tokens=5000,
                warmup_tokens=100,
                pilot=True,
                allow_unreviewed_data=True,
                n_layer=1,
                n_head=2,
                n_embd=32,
                checkpoint_every=2,
                evaluate_every=2,
            )
            whole, full = run(f"w{world}-whole", config, steps=4)
            part, _ = run(f"w{world}-part", config, steps=2)
            resumed, restored = run(
                f"w{world}-resumed", config, steps=2, resume=part["final_checkpoint"]
            )
            for key in ("model", "optimizer", "data", "step", "tokens"):
                check_equal(
                    full[key], restored[key], floating_tolerance=key in ("model", "optimizer")
                )
            for a, b in zip(full["rng"], restored["rng"]):
                for key in ("python", "torch", "cuda"):
                    check_equal(a[key], b[key])
                assert a["numpy"][0] == b["numpy"][0]
                assert (a["numpy"][1] == b["numpy"][1]).all()
                assert a["numpy"][2:] == b["numpy"][2:]
            assert resumed["session_steps"] == 2 and resumed["steps"] == 4
            summary["resume"][world] = {
                "status": "numerically_equal",
                "max_parameter_difference": max(
                    float((full["model"][k] - restored["model"][k]).abs().max())
                    for k in full["model"]
                ),
                "tokens": full["tokens"],
                "validation": whole["final_validation"]["weighted_loss"],
            }
            print(json.dumps({"world": world, "resume": "numerically_equal"}), flush=True)
            # Stop immediately before an over-budget batch; restore its cursor.
            limit_config = DistributedConfig(
                **{**config.as_dict(), "target_tokens": full["tokens"] + 1}
            )
            limited, state = run(f"w{world}-limit", limit_config)
            assert limited["stop_reason"] == "token_limit" and limited["steps"] == 4
            assert state["data"] == full["data"] and state["tokens"] == full["tokens"]
            summary["token_limit"][world] = "no_overshoot_cursor_rolled_back"
            exhausted, state = run(f"w{world}-exhaustion", config, source=small)
            assert exhausted["stop_reason"] == "source_two_pass_limit"
            assert sum(state["data"]["loss_token_exposures"].values()) == state["tokens"]
            summary["exhaustion"][world] = "clean_checkpoint_no_deadlock"
            if world == 2:
                bad = DistributedConfig(**{**config.as_dict(), "peak_lr": 0.001})
                error = run("reject-resume", bad, resume=part["final_checkpoint"], fail=True)
                assert "identical training settings" in error
                summary["failures"]["incompatible_resume"] = "rejected_on_all_ranks"
                substantial = DistributedConfig(world_size=2, microbatch=24)
                error = run("reject-audits", substantial, fail=True)
                assert "pending audits" in error
                summary["failures"]["unreviewed_manifest"] = "rejected_on_all_ranks"
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
