import json
import os
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


class ArchiveGenerationTests(unittest.TestCase):
    def test_cli_loads_a_symlinked_hub_style_snapshot_without_project_imports(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = base / "snapshot"
            bundle.mkdir()
            blobs = base / "blobs"
            blobs.mkdir()
            shutil.copyfile(
                root / "publication/generate_checkpoint_archive.py", blobs / "generator"
            )
            (bundle / "generate.py").symlink_to(blobs / "generator")
            for package, folder in [("model_100m", "100m"), ("model_10m", "10m")]:
                (bundle / package).mkdir()
                for name in ["__init__.py", "model.py"]:
                    shutil.copyfile(root / folder / name, bundle / package / name)
            config = ModelConfig(vocab_size=32768, block_size=16, n_layer=1, n_head=2, n_embd=8)
            torch.manual_seed(1)
            model = GPT(config)
            save_file(
                {k: v.clone() for k, v in model.state_dict().items()},
                str(bundle / "model.safetensors"),
            )
            (bundle / "config.json").write_text(json.dumps(asdict(config)))
            (bundle / "tokenizer").mkdir()
            shutil.copyfile(
                root / "tokenizer/structured-code-v1/tokenizer.json",
                bundle / "tokenizer/tokenizer.json",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(bundle / "generate.py"),
                    "--prompt",
                    "hello",
                    "--tokens",
                    "2",
                    "--temperature",
                    "0",
                ],
                cwd=base,
                env={**os.environ, "PYTHONPATH": ""},
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("hello"))
