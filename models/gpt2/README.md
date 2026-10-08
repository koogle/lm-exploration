# GPT-2 small checkpoint

[Source](https://huggingface.co/stratalab-org/gpt2-GGUF): a community GGUF conversion
of `openai-community/gpt2`, licensed under MIT. See [LICENSE](LICENSE) and
[manifest.json](manifest.json) for attribution, pinned download URL, and checksum.

The checkpoint is **106,554,880 bytes (106.6 MB / 101.6 MiB)** and is stored in Git LFS.
After cloning, install Git LFS and run:

```sh
git lfs install
git lfs pull
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('models/gpt2')
m = json.loads((root / 'manifest.json').read_text())
p = root / m['file']
assert p.stat().st_size == m['size_bytes'], 'Size mismatch; run git lfs pull'
with p.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == m['sha256'], 'Hash mismatch'
print('Checkpoint verified')
PY
```

The verification command requires Python 3.11 or newer. Run it from the repository root.

GGUF is a tensor container, not a GPU executable. This file contains 12 GPT-2
blocks, 12 attention heads, width 768, a 1,024-token context, and a 50,257-entry
vocabulary with BPE merges. Its 149 tensors use F32 (99), Q4_0 (49), and Q6_K (1).
Despite the filename, the output projection uses Q6_K; the loader must support
all three types. Quantization approximates the original weights.

The small download does not imply a 107 MB runtime memory requirement. Expanding
all stored tensors to F32 takes roughly 652 MB, before activations, attention
state, and temporary copies. A later packed-weight implementation can reduce
this. Check device memory and buffer limits and split large matrices as needed.

The source identifies the base model but does not pin its revision or converter
version. We verified the downloaded bytes and tensor layout; generation quality
and on-device inference have not yet been validated.
