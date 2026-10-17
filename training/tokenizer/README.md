# Default tokenizer

The repository default is `structured-code-v1`: the code-focused 32,768-entry BPE
trained on approximately 64 MiB on Modal. It was selected for code and tool calling.

```python
from training_data.subword import load_frozen, VOCAB_SIZE, TOKENIZER_SHA256

tokenizer = load_frozen()  # verifies the frozen artifact hashes
assert VOCAB_SIZE == tokenizer.get_vocab_size() == 32768
```

The [training plan](../plans/SCHOLARLY_CODE_PRETRAINING.md) records the selection,
trade-offs, special IDs and retokenization requirements. The frozen version's
`publication.json` pins its Hub revision. Original scholarly-code-v1 assets remain
available as an immutable historical version; existing tokenized snapshots cannot
be reused with the new tokenizer.

Published resource: [jfrick/scholarly-code-tokenizer](https://huggingface.co/jfrick/scholarly-code-tokenizer).
