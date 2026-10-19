# Tokenizer tools and frozen assets

Run from `training/`. Python tools live alongside the assets in this folder and
are imported as `tokenizer.<module>`; `tokenizers` (plural) remains the external
BPE library. Modal entrypoints live separately in
[`../modal_scripts/`](../modal_scripts/README.md).

| Module | Responsibility |
| --- | --- |
| `tokenizer_research.py` | Corpus collection, candidate fitting and evaluation |
| `tokenizer_evaluation.py` | Broader held-out evaluation without refitting |
| `tokenizer_report.py` | Learning-curve plots and reports |
| `structured_tokenizer.py` | Code/JSON/diff/tool-call workloads and focused fitting |
| `structured_tokenizer_report.py` | Structured-output comparison report |
| `package_tokenizer_research.py` | Hash-verified publication packaging |
| `verify_tokenizer_publication.py` | Standalone local/Hub publication verification |

For example:

```sh
uv run --extra corpus --extra research python -m tokenizer.tokenizer_report \
  results/tokenizer-learning-curve/<run-id>
```

See the [research guide](research/README.md) for fitting, reporting and publication
commands. Frozen version directories and historical results are unchanged by
this source reorganization. Data-token staging and preparation remain in
[`../data/`](../data/README.md), not in tokenizer tooling.

## Default tokenizer

The repository default is `structured-code-v1`: the code-focused 32,768-entry BPE
trained on approximately 64 MiB on Modal. It was selected for code and tool calling.

```python
from training_data.subword import load_frozen, VOCAB_SIZE, TOKENIZER_SHA256

tokenizer = load_frozen()  # verifies the frozen artifact hashes
assert VOCAB_SIZE == tokenizer.get_vocab_size() == 32768
```

The [training plan](../plans/SIMPLE_LLM_PRETRAINING.md) records the selection,
trade-offs, special IDs and retokenization requirements. The frozen version's
`publication.json` pins its Hub revision. Original assets now live in
[`simple-llm-v1/`](simple-llm-v1/) with their bytes and hashes unchanged;
existing tokenized snapshots cannot
be reused with the new tokenizer.

Published resource: [Simple LLM tokenizer](https://huggingface.co/jfrick/scholarly-code-tokenizer).
