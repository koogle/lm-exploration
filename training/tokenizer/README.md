# Frozen scholarly/code tokenizer

Version `scholarly-code-v1` is a 16,384-token byte-level BPE trained from scratch.
Read the [current training plan](../plans/SCHOLARLY_CODE_PRETRAINING.md) for the
mixture, measured comparison and remaining work. Files in this version are immutable.

From `training/`, verify all artifact hashes and load the tokenizer:

```sh
uv run --with tokenizers==0.22.2 python - <<'PY'
from pathlib import Path
import hashlib
import json
from tokenizers import Tokenizer

folder = Path('tokenizer/scholarly-code-v1')
manifest = json.loads((folder / 'freeze-manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, name
assert manifest['files']['tokenizer.json'] == '2a6bd2343ab29e3955dd7a5e17330e706afc8c650618b1339ceedab5e77fde04'
tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
assert tokenizer.get_vocab_size() == 16384
text = 'def square(x):\r\n\treturn x ** 2\r\n'
ids = tokenizer.encode(text, add_special_tokens=False).ids
assert tokenizer.decode(ids, skip_special_tokens=False) == text
assert 3 not in ids
print('PASS: hashes, vocabulary and exact code round trip')
PY
```

PAD/BOS/EOS/UNK IDs are 0/1/2/3. The tokenizer adds no boundaries automatically.
Future training code must verify this hash when loading shards and checkpoints.
No model weights, raw corpus, credentials or budget ledger are stored here.
