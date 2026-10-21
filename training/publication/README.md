# Publication logic and dataset artifacts

Scripts live here; generated cards, metrics, audits, reports and samples belong on
Hugging Face **datasets**, not in the code repository. Run commands from `training/`.

## PR #13

The completed modern 9.85M experiment processed 4,999,934,057 targets over 72,805
updates. Broader exploratory validation loss was 2.6926 nats/token; generations
remained repetitive and factually weak. Pending corpus audits prevent a sealed
benchmark claim. The CPU preparation benchmarks do not establish end-to-end speedup.

All 19 generated PR #13 artifacts (170,805 bytes), including poor generations and
historical publication cards, are archived byte-for-byte in the private dataset
[`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/292e06c89b8965f565d33b9c7ed52531030c6086/pr-13)
at immutable revision `292e06c89b8965f565d33b9c7ed52531030c6086`.
The checked-in [artifact index](pr13-artifacts.json) records the source Git commit,
original paths, dataset paths, byte sizes and SHA256 checksums. Existing model
repositories and training volumes were not changed by this migration.

## Download and use

Authenticate locally with dataset read access; never place tokens in Git or CLI
arguments. Download and verify the pinned artifacts into the ignored local cache:

```sh
uv run --extra corpus python -m publication.artifacts --output artifacts/pr-13

# Rebuild a card and its evidence bundle without writing results into Git.
uv run --extra corpus python -m publication.build_completed_card \
  --artifacts artifacts/pr-13 --output artifacts/pr-13/rebuilt-card
```

Consumers verify checksums and require an explicit artifact directory. They do not
silently fetch a mutable latest revision. The completed-checkpoint exporter also
requires `--artifacts`; the smoke exporter takes `--artifacts` and an explicit
`--training-report` path. Invoke these as `python -m publication.export_completed`
and `python -m publication.export_ddp_smoke` from `training/` with their other
required checkpoint, data-manifest and output arguments. These export tools require
`safetensors` in addition to the corpus environment and only accept trusted
checkpoints. They do not launch training or maintain an archived-model evaluator.

## New outputs

The preparation benchmark defaults to `artifacts/unpublished/`, which is ignored.
Upload new outputs to a new dataset path/commit, retain provenance and failures,
and record their immutable revision and checksums before deleting local copies or
changing references. Do not replace the PR #13 index with an unpinned `main` link.
Do not upload credentials, private corpus rows or unsafe checkpoint state.
