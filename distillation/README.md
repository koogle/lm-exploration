# Qwen-teacher distillation

Distill a Hugging Face teacher (Qwen3 so far) into the repository's small GPT students
(`training/100m`). The package holds the objectives, data readers, curriculum generation
and verification, trainers, evaluation, Llama/vLLM export and a sizing calculator.
Recorded run recipes, run-specific orchestration and findings belong in
`distillation/EXPERIMENTS.md` and `distillation/recipes/` (stacked experiments PR).

| Module | Role |
| --- | --- |
| `alignment.py` | Byte-level alignment of two BPE vocabularies; 257-bucket first-byte marginals and their KL |
| `data.py` | One-pass code windows from an immutable snapshot database, preserving splits and exclusions |
| `train.py` | `byte`: CE + first-byte-marginal KL against an online teacher with a different tokenizer |
| `shared.py`, `shared_train.py` | `shared`: switch the student to the teacher's tokenizer (embedding transfer), then CE + full-vocabulary KL |
| `curriculum.py` | Deterministic Python/math task specs, hidden reference tests and a restricted subprocess verifier |
| `curriculum_train.py` | `students`: fresh 26.8M / 48.3M students, one pass over a verified curriculum corpus |
| `evaluate.py` | `evaluate`: token CE of several checkpoints on one fixed document-capped holdout |
| `export.py` | Native checkpoint to `LlamaForCausalLM` (RoPE permutation, split QKV) with logit parity |
| `scaling.py` | Parameter counts and token budgets: `python -m distillation.scaling --accepted-tokens N` |
| `recipe.py` | Recipe validation for both launchers |
| `modal_app.py` | Modal launcher for `byte`, `shared`, `students`, `evaluate` and `generations` recipes |
| `modal_vllm.py` | Modal launcher for `curriculum` recipes and the vLLM export-parity check |
| `scale_data.py`, `dataset_pipeline.py`, `dataset_archive.py` | Streaming synthetic-corpus shards, append-only publication and HF dataset archive (from the dataset PR) |

## Objectives

- **byte** (different tokenizers): reconstruct raw bytes from the student's ByteLevel BPE
  pieces, tokenize with the teacher and keep positions with identical text prefixes.
  The teacher's full next-token distribution is summed into 256 first-byte buckets plus
  one special bucket; loss is `0.5 * CE + 0.5 * KL(teacher_byte || student_byte)`. This is
  a lossy marginal, not token KL. Windows that cut UTF-8 characters or change under the
  teacher's normalizer are skipped and counted.
- **shared** (teacher tokenizer): the student backbone keeps its weights; byte-identical
  vocabulary rows are copied and other rows initialized from their old-token
  decomposition. Loss is equal-weight CE and full-vocabulary KL at temperature 1,
  normalized by the unmasked targets of the whole optimizer batch.
- **students** (sequence distillation): all-token CE on verified
  `Problem -> Python -> Explanation` records. Explanations are unverified teacher prose.

Teacher probabilities are computed online on the training GPU; nothing is cached.
`byte` and `shared` checkpoints include weights, optimizer, RNG, exact data cursor,
settings and source hashes. Resume writes to a new run directory and rejects changed
settings or source. `students` runs are bounded one-pass qualifications without resume.

## Recipes

One JSON file per job. `kind` selects the job; every input is explicit:

```json
{
  "name": "shared-example",
  "kind": "shared",
  "run_id": "UNIQUE_RUN_ID",
  "teacher": {"repo": "Qwen/Qwen3-4B-Base", "revision": "<full commit>"},
  "parent": "PARENT_RUN/step-000153342.pt",
  "data": {"manifest": "SNAPSHOT/manifest.json", "database": "SNAPSHOT/raw.sqlite"},
  "target_tokens": 1000000,
  "max_seconds": 3600
}
```

| Kind | Required fields | Launcher, GPU, timeout |
| --- | --- | --- |
| `byte` | `teacher`, `parent`, `data`, `target_tokens`, `max_seconds` (at most 1,200) | `modal_app`, L40S, 30 min |
| `shared` | same; `max_seconds` at most 82,000 | `modal_app`, H100, 24 h |
| `students` | `corpus` (completed curriculum run) | `modal_app`, H100, 2 h |
| `evaluate` | `checkpoints` (`training:` or `distillation:` prefix), `data`; optional `context`, `max_documents`, `windows_per_document` | `modal_app`, L4, 15 min |
| `generations` | `checkpoints`, `precision`, `max_new_tokens`, `prompts`, `modes` | `modal_app`, L4, 15 min |
| `curriculum` | `teacher`, `count` (32 to 4,096); optional `sampling` | `modal_vllm`, H100, 2 h |

`byte` and `shared` also take `resume`. Optional `name`, `description` and `provenance`
fields are kept with the result. Paths are relative to Modal volumes: `parent` to
`lm-exploration-training`, `data` to `lm-exploration-mixture-data`, and run IDs, `resume`,
`corpus` and generation checkpoints to `lm-exploration-distillation`.

`generations` recipes use the [sampling toolkit](../training/analysis/sampling/README.md)
prompt/mode format and decoding loop. Their result JSON follows the toolkit's result
schema, so `python -m sampling.report RESULT.json --output-dir DIR` (from `training/`)
renders tables and completions. Shared-tokenizer checkpoints load the tokenizer archived
at their run root; frozen-tokenizer checkpoints use the 32K tokenizer.

## Run

From the repository root, with `modal[api-proxy-support]==1.6.2` installed (in a managed
environment, launch through `truststore.inject_into_ssl()` as in the training docs):

```sh
modal run --detach -m distillation.modal_app --recipe RECIPE.json \
  [--run-id NEW_ID] [--parent RUN/step-N.pt] [--resume RUN/step-N.pt] [--output OUT.json]
modal run --detach -m distillation.modal_vllm --recipe CURRICULUM.json [--run-id NEW_ID]
modal run -m distillation.modal_vllm::parity --checkpoint RUN/step-000001000.pt
```

Each launch makes one function call with `retries=0` and records a deadline 24 hours
after launch. Calls with timeouts of two hours or less refuse to start unless the full
timeout still fits before the deadline. The 24-hour `shared` trainer caps its
cooperative training time to stop 30 minutes before the deadline, then evaluates and
checkpoints. The returned report is written to the ignored `distillation/artifacts/`;
run folders on the volume keep settings, source snapshots, metrics and checkpoints.

The parity check exports a checkpoint next to itself (`step-N-hf/`) and compares greedy
tokens and log probabilities from Transformers and vLLM 0.11.0 on five code prompts. Use
raw completion prompts with exactly one BOS and identical stop IDs; do not apply a chat
template. Serving parity alone does not establish evaluation-harness parity.

## Tests

```sh
python -m unittest discover -s distillation/tests -v
ruff check distillation && ruff format --check distillation
```

Tests need torch, transformers 4.57.1, tokenizers 0.22.2 and numpy (CPU is enough). The
curriculum verifier relies on Linux resource limits; its execution and timeout tests are
skipped elsewhere, so a macOS pass does not qualify the Linux sandbox.
