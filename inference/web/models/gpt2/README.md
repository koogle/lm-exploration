# GPT-2 small checkpoint

[Source](https://huggingface.co/stratalab-org/gpt2-GGUF): a community GGUF conversion
of `openai-community/gpt2`, licensed under MIT. See [LICENSE](LICENSE) and
[manifest.json](manifest.json) for attribution, pinned download URL, and checksum.

The checkpoint is **106,554,880 bytes (106.6 MB / 101.6 MiB)**. Download it only
when needed; weights are ignored in this consolidated checkout. From `inference/web/`:

```sh
curl -fL 'https://huggingface.co/stratalab-org/gpt2-GGUF/resolve/98b6b378440596a4c1f7eae0c7f74168ba1c5823/gpt2.Q4_0.gguf' -o models/gpt2/gpt2.Q4_0.gguf
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('models/gpt2')
m = json.loads((root / 'manifest.json').read_text())
p = root / m['file']
assert p.stat().st_size == m['size_bytes'], 'Size mismatch; download is incomplete'
with p.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == m['sha256'], 'Hash mismatch'
print('Checkpoint verified')
PY
```

The verification command requires Python 3.11 or newer. Run it from `inference/web/`.

The original Git LFS object remains in `koogle/web-inference`. This environment
could read it but could not complete its upload to the destination because the
LFS upload-verification endpoint is blocked. Original Git commits in the
[history bundle](../../../../docs/history/README.md) retain its LFS pointer;
the imported code history omits that pointer. The current tree keeps the manifest and download instructions so a
normal clone does not depend on an unavailable destination LFS object. The model
bytes were verified against this manifest during consolidation.

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
