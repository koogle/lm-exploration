"""Run the shared generator as a standalone exported bundle, without project imports."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from model_100m.model import GPT, ModelConfig
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]


class PublicationGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.bundle = self.folder / "bundle"
        self.bundle.mkdir()
        for source, package, names in (
            ("100m", "model_100m", ("__init__.py", "model.py")),
            ("10m", "model_10m", ("__init__.py", "model.py", "legacy.py")),
        ):
            (self.bundle / package).mkdir()
            for name in names:
                shutil.copyfile(ROOT / source / name, self.bundle / package / name)
        shutil.copyfile(ROOT / "publication/generate.py", self.bundle / "generate.py")
        self.config = ModelConfig(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8)
        (self.bundle / "config.json").write_text(json.dumps(asdict(self.config)))
        torch.manual_seed(7)
        self.model = GPT(self.config).eval()
        self.weights = {k: v.contiguous().clone() for k, v in self.model.state_dict().items()}
        save_file(self.weights, str(self.bundle / "model.safetensors"))
        vocabulary = {
            s: i
            for i, s in enumerate(
                ("<pad>", "<bos>", "<eos>", "<unk>", "hello", "world", "other", ".")
            )
        }
        self.tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.add_special_tokens(["<pad>", "<bos>", "<eos>", "<unk>"])
        (self.bundle / "tokenizer").mkdir()
        self.tokenizer.save(str(self.bundle / "tokenizer/tokenizer.json"))
        self.prompt = "hello world hello world hello"

    def run_cli(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-I",
                str(self.bundle / "generate.py"),
                "--prompt",
                self.prompt,
                *args,
            ],
            cwd=self.folder,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def reference(self, temperature, top_k, count=6):
        # Reproduce constructor RNG consumption before loading the stored weights.
        torch.manual_seed(42)
        model = GPT(self.config).eval()
        model.load_state_dict(self.weights, strict=True)
        ids = [1] + self.tokenizer.encode(self.prompt, add_special_tokens=False).ids
        with torch.inference_mode():
            for _ in range(count):
                logits = model(torch.tensor([ids[-self.config.block_size :]]))[:, -1]
                if temperature == 0:
                    token = int(logits.argmax())
                elif top_k:
                    values, indices = torch.topk(logits[0] / temperature, min(top_k, 8))
                    token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                else:
                    # Original smoke sampler used a batch-shaped full distribution.
                    token = int(torch.multinomial((logits / temperature).softmax(-1), 1).item())
                ids.append(token)
                if token == 2:
                    break
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def test_greedy_top_k_and_original_smoke_sampling(self):
        for temperature, top_k in ((0, 50), (0.8, 3), (1, 0), (0.8, 1000)):
            with self.subTest(temperature=temperature, top_k=top_k):
                result = self.run_cli(
                    "--tokens", "6", "--temperature", str(temperature), "--top-k", str(top_k)
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), self.reference(temperature, top_k))

    def test_zero_tokens_returns_prompt(self):
        result = self.run_cli("--tokens", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.prompt)

    def test_symlinked_hf_snapshot(self):
        blobs = self.folder / "blobs"
        blobs.mkdir()
        for name in ("generate.py", "config.json", "model.safetensors", "tokenizer/tokenizer.json"):
            path = self.bundle / name
            blob = blobs / name.replace("/", "-")
            path.rename(blob)
            path.symlink_to(blob)
        result = self.run_cli("--tokens", "6", "--temperature", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.reference(0, 50))

    def test_invalid_settings_and_strict_weight_loading(self):
        for args in (
            ("--tokens", "-1"),
            ("--top-k", "-1"),
            ("--temperature", "-1"),
            ("--temperature", "nan"),
            ("--temperature", "inf"),
        ):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("finite and nonnegative", result.stderr)
        save_file({}, str(self.bundle / "model.safetensors"))
        result = self.run_cli("--tokens", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing key", result.stderr)
