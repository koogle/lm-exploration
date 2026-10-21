# Publication

Git holds export code and an immutable artifact index, not generated results.
Keep new model cards, reports, audits, curves and samples on Hugging Face datasets.

## Export an inference bundle

One function, `publication.export.export_checkpoint`, handles any native GPT
architecture described by the checkpoint. From `training/`:

```sh
uv run --extra corpus --extra research python -m publication.export \
  --checkpoint /path/to/checkpoint.pt --output artifacts/model \
  --tokenizer /path/to/tokenizer.json
```

Use only trusted checkpoints: loading training state uses Python pickle. New DDP
checkpoints record the architecture module, complete model configuration and
BOS/EOS IDs. The module exposes `GPT`, `ModelConfig` and declared local
`MODEL_DEPENDENCIES`. Export checks current model-source and tokenizer hashes,
loads tensors strictly, and verifies exact safetensors/logit round-trips. It does
not reconstruct old code or infer missing metadata from experiment names.

The standalone bundle includes selected model code, safetensors, configuration,
tokenizer, generation script and provenance, but no optimizer/RNG/cursor state.
Optional `--data-manifest` includes a checksum-matching provenance manifest.
Historical cards and measured reports are not prerequisites for exporting weights.

```sh
python artifacts/model/generate.py --prompt Hello --tokens 32 \
  --seed 42 --temperature 1 --top-k 0
```

Generation defaults to 128 tokens, seed 42, temperature 0.8, top-k 50 and CPU.
`--temperature 0` selects greedy decoding; `--top-k 0` samples the full vocabulary.

## Historical evidence

Completed experiments' results, reports and original model cards live on HF, not in Git:

- Result JSONs: private dataset `jfrick/lm-exploration-results`, under `pr-13/` and `pr-16/`.
- PR #16 repetition report and plot: model repository
  `jfrick/scholarly-code-modern-gpt-124m-step2`, under `analysis/repetition/`.
- Published model cards: each model repository's `README.md` and commit history.

Download what you need into the ignored `artifacts/` directory and pass explicit paths.
New inference or repetition runs write to `artifacts/pr-16/results/` by default.
The former experiment-specific exporters have been replaced by the generic export
function above. Card rendering is independent of weight export.

## Model-card source

[`cards/completed.md`](cards/completed.md) is the editable source document, not a
generated card. Its TOML front matter (between `+++` lines) selects the pinned
audit, validation and training-segment artifacts, baseline report key/label and
number of displayed samples. The Markdown body owns the publication prose and
`${field}` placeholders. `--document` selects another source document; the
completed document above is the default.

The renderer derives dimensions, parameter/update/token counts, batch accumulation,
learning-rate settings, runtime versions, exposure shares, evaluation counts,
losses, diagnostic loss, checkpoint path and sample text from verified reports.
It handles the listed segment count rather than assuming three segments. Edit
Markdown to change the wording, sections or examples shown; no Python prose edits
are needed. Unknown placeholders, failed checksums, failed verification flags and
inconsistent run identities/counts fail before writing output.

Historical reports omitted settings that were fixed in their trainer. Those
values are documented explicitly in the Markdown's `historical_training` table;
recorded training config values take precedence. Architecture/tokenizer details
and evaluation limitations absent from structured reports remain authored prose
in this experiment's Markdown. Do not infer historical settings from current
recipe defaults or reuse those descriptions for a different architecture without
review. The original archived cards and reports remain unchanged.

Rebuild the Markdown card from downloaded `pr-13/training/` files and copy its evidence into an ignored output directory:

```sh
uv run --extra corpus python -m publication.build_completed_card \
  --artifacts artifacts/pr-13 --output artifacts/pr-13/rebuilt-card \
  --document publication/cards/completed.md
```

The checked-in Markdown is authored source. Generated `README.md` files and copied
result JSONs belong in the HF artifact archive, not Git. Rendering does not publish
or upload anything and does not modify the original evidence.
