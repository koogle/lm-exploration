"""One end-to-end export/inference check and fail-closed input checks."""

import importlib
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers

from publication.export import digest, export_checkpoint


class PublicationTests(unittest.TestCase):
    def fixture(self, folder, name="model_10m.model", width=8, heads=2):
        torch.set_num_threads(1)
        module = importlib.import_module(name)
        cfg = module.ModelConfig(
            vocab_size=8, block_size=4, n_layer=1, n_head=heads, n_embd=width, eos_id=2
        )
        model = module.GPT(cfg).eval()
        tokenizer = Tokenizer(
            models.WordLevel(
                {
                    s: i
                    for i, s in enumerate(
                        ("<pad>", "<unk>", "hello", "world", "<bos>", "<eos>", "other", ".")
                    )
                },
                unk_token="<unk>",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
        path = folder / "tokenizer.json"
        tokenizer.save(str(path))
        saved = {
            "architecture": {"module": name, "config": asdict(cfg)},
            "special_tokens": {"bos": 4, "eos": 5},
            "model": model.state_dict(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "source_sha256": {
                k: digest(v)
                for k, v in {"model": module.__file__, **module.MODEL_DEPENDENCIES}.items()
            },
            "data_identity": {"tokenizer_sha256": digest(path)},
            "optimizer": {"private": "not-for-inference"},
            "rng": "not-for-inference",
            "data": "not-for-inference",
        }
        checkpoint = folder / "checkpoint.pt"
        torch.save(saved, checkpoint)
        return checkpoint, path, saved, model, tokenizer

    def test_export_and_standalone_generation_use_checkpoint_metadata(self):
        for name, width, heads in (("model_10m.model", 8, 2), ("model_100m.model", 12, 3)):
            with self.subTest(module=name), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                checkpoint, tokenizer_path, saved, model, tokenizer = self.fixture(
                    folder, name, width, heads
                )
                output = folder / "bundle"
                provenance = export_checkpoint(checkpoint, output, tokenizer_path)
                self.assertEqual(provenance["parameters"], saved["parameters"])
                self.assertNotIn("not-for-inference", (output / "provenance.json").read_text())
                for key, value in load_file(str(output / "model.safetensors")).items():
                    torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
                # HF snapshot files may be symlinks to a separate blobs directory.
                blobs = folder / "blobs"
                blobs.mkdir()
                for name in ("generate.py", "config.json", "model.safetensors"):
                    (output / name).rename(blobs / name)
                    (output / name).symlink_to(blobs / name)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(output / "generate.py"),
                        "--prompt",
                        "hello",
                        "--tokens",
                        "3",
                        "--temperature",
                        "0",
                    ],
                    cwd=folder,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                ids = [4, 2]
                with torch.inference_mode():
                    for _ in range(3):
                        ids.append(int(model(torch.tensor([ids[-4:]]))[0, -1].argmax()))
                        if ids[-1] == 5:
                            break
                self.assertEqual(
                    result.stdout.strip(), tokenizer.decode(ids, skip_special_tokens=True)
                )

    def test_invalid_inputs_do_not_leave_partial_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint, tokenizer, saved, _, _ = self.fixture(folder)
            for broken in (
                {**saved, "parameters": 0},
                {**saved, "source_sha256": {"model": "0" * 64}},
                {**saved, "data_identity": {"tokenizer_sha256": "0" * 64}},
            ):
                torch.save(broken, checkpoint)
                with self.assertRaises(ValueError):
                    export_checkpoint(checkpoint, folder / "output", tokenizer)
                self.assertFalse((folder / "output").exists())
