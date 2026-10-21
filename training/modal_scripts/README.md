# Modal launchers

This folder owns the training project's general Modal entrypoints, including
bounded one-off research and benchmark jobs. Training launch support lives in
[`../distributed/`](../distributed/README.md), including one GPU, and executes
the loop in [`../trainer/`](../trainer/README.md). Reusable implementations belong in `trainer/`, `data/` or `tokenizer/`,
not here. The separate EU corpus project's
launchers remain scoped to `data/eu/` and its own environment.

**Run every command from `training/`**, not from this folder. Image source paths,
recipes, local results and budget ledgers are relative to the training root.

```sh
uv sync --locked --extra corpus --extra research
uv run modal run -m workflows.modal_train \
  --workflow workflows/h100x1-pilot.json --run-id <fresh-run-id>
```

| Script | Purpose / owning guide |
| --- | --- |
| `modal_prepare_simple_llm.py` | CPU corpus staging/export; [preparation](../data/SIMPLE_LLM_STREAMING.md) |
| `modal_gpu_benchmark.py` | Single-GPU hardware comparison |
| `modal_multi_gpu_benchmark.py` | Two-/three-GPU DDP comparison |
| `modal_gpu_confirmation.py` | Longer H100 throughput confirmation with explicit benchmark replay |
| `modal_tokenizer_research.py` | Tokenizer fitting grid; [research](../tokenizer/research/README.md) |
| `modal_tokenizer_evaluation.py` | Held-out parent coverage and rescoring; same research guide |
| `modal_structured_tokenizer.py` | Structured-code/tokenizer comparison; same research guide |
| `modal_simple_llm_8k.py` | Eight-H100 124M / 8K / 5B controller; [workflow](../distributed/SIMPLE_LLM_8K.md) |
| `modal_infer_simple_llm.py` | Fixed-prompt samples from the final 124M / 8K checkpoint |
| `modal_validate_repetition.py` | Four-checkpoint repetition comparison; [report](https://huggingface.co/jfrick/scholarly-code-modern-gpt-124m-step2/blob/a5111f064480c8b39b7d6e791a6f7746290619ab/analysis/repetition/simple-llm-repetition-validation-20261018.md) |
| `modal_training_optimizations.py` | Opt-in padding crop/chunked loss/compile/RoPE benchmark suite; [guide](../distributed/OPTIMIZATIONS.md) |
| `modal_single_gpu_optimizations.py` | Single-H100 optimization suite; same guide |
| `modal_eight_gpu_finalists.py` | Eight-H100 finalist comparison; same guide |
| `modal_muon_comparison.py` | Matched 100M-token Muon vs AdamW comparison; [guide](../distributed/MUON_COMPARISON.md); outputs on HF under `pr-24/training/results/` |
| `modal_overlap_checks.py` | Bounded three-GPU equivalence/resume checks for the overlapped [`../modal_streaming_train.py`](../modal_streaming_train.py) launcher |

The benchmark launchers retain their historical defaults, resource bounds and
measured outputs in the HF dataset `jfrick/lm-exploration-results` (`training/results/`); new
reports go to the ignored `artifacts/results/`. They do not start full pretraining. Inspect
those defaults and choose fresh run IDs before any deliberate rerun. Moving a
script does not authorize a job, change budgets or establish new measurements.
Some launchers resolve Modal Volume handles on import; imports are not a cloud
execution or deployment check. Use local correctness tests without importing
Modal entrypoints when no cloud access is intended.

The legacy character-level `modal_train.py` and its loop have been removed;
historical recipes and published checkpoint loading remain available.
