# Modal launchers

This folder owns the training project's general Modal entrypoints, including
bounded one-off research and benchmark jobs. Distributed subword training and
its launchers live separately in
[`../distributed_training/`](../distributed_training/README.md), including one
GPU. Reusable implementations belong in `trainer/`, `data/` or `tokenizer/`,
not here. The separate EU corpus project's
launchers remain scoped to `data/eu/` and its own environment.

**Run every command from `training/`**, not from this folder. Image source paths,
recipes, local results and budget ledgers are relative to the training root.

```sh
uv sync --locked --extra corpus --extra research
uv run modal run modal_scripts/modal_train.py \
  --recipe recipes/shakespeare-10m.json --run-id <fresh-run-id> --smoke
```

| Script | Purpose / owning guide |
| --- | --- |
| `modal_train.py` | Character-level recipe training; [recipes](../recipes/README.md) |
| `modal_prepare_simple_llm.py` | CPU corpus staging/export; [preparation](../data/SIMPLE_LLM_STREAMING.md) |
| `modal_gpu_benchmark.py` | Single-GPU hardware comparison |
| `modal_multi_gpu_benchmark.py` | Two-/three-GPU DDP comparison |
| `modal_gpu_confirmation.py` | Longer H100 throughput confirmation with explicit benchmark replay |
| `modal_tokenizer_research.py` | Tokenizer fitting grid; [research](../tokenizer/research/README.md) |
| `modal_tokenizer_evaluation.py` | Held-out parent coverage and rescoring; same research guide |
| `modal_structured_tokenizer.py` | Structured-code/tokenizer comparison; same research guide |

The benchmark launchers retain their historical defaults, resource bounds and
measured outputs in `../results/`; they do not start full pretraining. Inspect
those defaults and choose fresh run IDs before any deliberate rerun. Moving a
script does not authorize a job, change budgets or establish new measurements.
Some launchers resolve Modal Volume handles on import; imports are not a cloud
execution or deployment check. Use local correctness tests without importing
Modal entrypoints when no cloud access is intended.
