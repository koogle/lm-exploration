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

[`pr13-artifacts.json`](pr13-artifacts.json) indexes the original PR #13 results
and model cards in the private **dataset** `jfrick/lm-exploration-results` at
immutable revision `292e06c89b8965f565d33b9c7ed52531030c6086`. It records original
paths, byte sizes, SHA256 checksums and source Git commit. Every remote file was
re-downloaded and verified before its Git copy was removed. Original identifiers
and bytes are retained; no historical measurements are claimed as new results.

Download verified copies explicitly, using an authenticated account with access:

```sh
uv run --extra corpus python -m publication.artifacts --output artifacts/pr-13
```

`artifact_path()` verifies local bytes before use. It never silently fetches a
mutable revision. Existing model repositories and Modal volumes are unchanged.
The experiment-specific card builder and exporters are no longer maintained.
