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
| `mixture_tokenizer.py` | Sample a corpus recipe's training split, fit and compare |
| `structured_tokenizer.py` | Code/JSON/diff/tool-call workloads and focused fitting |
| `structured_tokenizer_report.py` | Structured-output comparison report |
| `package_tokenizer_research.py` | Hash-verified publication packaging |
| `verify_tokenizer_publication.py` | Standalone local/Hub publication verification |

For example:

```sh
uv run --extra corpus --extra research python -m tokenizer.tokenizer_report \
  artifacts/tokenizer-learning-curve/<run-id>
```

See the [research guide](research/README.md) for fitting, reporting and publication
commands. Frozen version directories and historical results are unchanged by
this source reorganization. Data-token staging and preparation remain in
[`../data/`](../data/README.md), not in tokenizer tooling.

## Default tokenizer

The repository default is `climbmix-code-v1`, a 32,768-entry byte-level BPE fitted
on 256 MiB sampled from the training split of the
[`climbmix-code-v1` recipe](../data/recipes/climbmix-code-v1.json): 85% ClimbMix and
15% code. It keeps the GPT-2-style pre-tokenizer, except that every digit is its
own token so arithmetic sees one token per digit. Special IDs are unchanged.
Fit it with `python -m tokenizer.mixture_tokenizer` (see the module docstring).

On 20 MiB of held-out validation text from the same sources (`evaluation.json`):

| Source | `climbmix-code-v1` bytes/token | `structured-code-v1` bytes/token |
| --- | ---: | ---: |
| ClimbMix | 4.583 | 3.666 |
| Curated open-source code | 3.598 | 3.519 |
| MIT code | 3.444 | 3.370 |

Both tokenizers round-trip every evaluated document exactly. The earlier
code-focused [`structured-code-v1`](structured-code-v1/) stays frozen for its
published models; data and checkpoints made with it must be retokenized.

```python
from training_data.subword import load_frozen, VOCAB_SIZE, TOKENIZER_SHA256

tokenizer = load_frozen()  # verifies the frozen artifact hashes
assert VOCAB_SIZE == tokenizer.get_vocab_size() == 32768
```

The [training plan](../plans/SIMPLE_LLM_PRETRAINING.md) records the selection.
`structured-code-v1` is published at
[`jfrick/scholarly-code-tokenizer`](https://huggingface.co/jfrick/scholarly-code-tokenizer);
`climbmix-code-v1` is not yet published. The original assets live in
[`simple-llm-v1/`](simple-llm-v1/) with their bytes and hashes unchanged.
