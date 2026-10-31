# Checkpoint evaluation

Benchmarks for our own training checkpoints on the `lm-exploration-training` volume. One
Modal call scores a list of checkpoints and writes structured JSON; findings are written
by hand in the owning study's docs. uv installs this folder as `evaluation`.

| File | Role |
| --- | --- |
| `configs/benchmarks.json` | Benchmark datasets (HF repository, pinned commit, files) and settings |
| `benchmarks/` | One module per benchmark: its prompt format and `score` |
| `scoring.py` | The scored model interface of a checkpoint (loaded by [`trainer/checkpoints.py`](../../trainer/checkpoints.py) with its run's verified model code), log-likelihood batching, greedy generation, answer checking, bits per byte |
| `heldout.py` | Validation windows of a raw snapshot |
| `modal_evaluate.py` | Modal launcher: one H100, 3-hour timeout, no retries; a code sandbox for code benchmarks |
| `benchmarks/livecodebench.py`, `modal_livecodebench.py` | LiveCodeBench v6 for checkpoints of any family and reference models ([below](#livecodebench)) |

## Run

From `training/`:

```sh
uv run --all-extras modal run -m evaluation.modal_evaluate --name NAME \
  --snapshot climbmix-code/VERSION/snapshots/RUN \
  --checkpoints RUN-s01/step-000050000.pt,RUN-s02/step-000100000.pt \
  [--config analysis/evaluation/configs/benchmarks.json]
```

`--config` names the datasets and settings (default `configs/benchmarks.json`); the result
records it and the SHA256 of every evaluation source file.

Each checkpoint loads the model code its run copied, after checking the source and
tokenizer hashes the checkpoint recorded, so any checkout can evaluate any checkpoint:
pretraining checkpoints from `RUN-sNN/source`, SFT and RL checkpoints
([`sft/`](../../sft/README.md), [`rl/`](../../rl/README.md)) from the `model.py` and `shared_model.py` beside them.
`--shots` sets the math exemplar count (default 5; use 0 for SFT models, whose training
prompt is the zero-shot format). `--tasks` limits a run to a comma-separated subset,
such as `gsm8k,svamp`. The call refuses new work 24 hours after launch. Results go to
`/runs/evaluation/NAME.json` after each checkpoint and to `--output` (default
`artifacts/evaluation/NAME.json`, ignored by Git); the HF results archive keeps them.

## What is measured

- **Held-out bits per byte.** The first 64 packed validation windows (2,049 tokens) of
  each snapshot source, in the raw loader's hash order. Bits are summed over every target
  except BOS (EOS included) and divided by the UTF-8 bytes the targets decode to, so the
  number compares across tokenizers. Also reported as nats per token.
- **Zero-shot multiple choice**, in the EleutherAI harness formats, scored by summed
  continuation log-probability (`acc`) and by that sum per continuation byte
  (`acc_norm`): HellaSwag validation (10,042), ARC-Easy test (2,376), ARC-Challenge test
  (1,172) and OpenBookQA test (500). Winogrande XL validation (1,267) uses partial
  scoring: the text after the blank, under each filled-in option.
- **LAMBADA** (OpenAI test, 5,153): last-word accuracy (every target token greedy) and
  last-word perplexity.
- **Few-shot math, greedy:** GSM8K test (first 500) and SVAMP test (300), each with the
  first `--shots` training examples as `Question: ...\nAnswer: ...` exemplars. Generation stops
  at a blank line, `Question:` or EOS (256 new tokens).
  [Math-Verify](https://github.com/huggingface/Math-Verify) extracts the final answer
  from the completion and the gold answer and checks they are equal. Answers are kept in the
  result.

- **Code, greedy pass@1:** MBPP+ (378) and HumanEval+ (164) from EvalPlus, in the code
  SFT prompt format (`training_data.code_sft.code_prompt`: the problem, or HumanEval's
  docstring, and the function's signature). The completion up to EOS is the program
  (after HumanEval's imports and helpers); it passes if every EvalPlus test passes, run
  only in a network-blocked Modal Sandbox (`training_data.sandbox`, 20 s per
  program). A completion without EOS within 512 tokens fails. SFT and RL models only.

Datasets are pinned to HF commits in `configs/benchmarks.json`. Inference uses BF16 autocast with FP32
weights. Chance levels: about 25% for HellaSwag, ARC and OpenBookQA, 50% for Winogrande.

## LiveCodeBench

[`modal_livecodebench.py`](modal_livecodebench.py) scores LiveCodeBench v6
(`test6.jsonl`, 175 problems, pinned revision) for checkpoints of any model family and
for pinned reference base models (the hybrid benchmark references and the dense Qwen3
1.7–14B and Qwen3.5 2–9B base models). A checkpoint loads through its family's adapter
([`trainer/families.py`](../../trainer/families.py)): a checkpoint trained to answer
instructions (any post-training stage, or a chat-format midtraining recipe such as hybrid
chat SFT) gets LiveCodeBench's system text and question as the family's user turn
(`instruction`), any other its plain prompt after one worked example of the same test
mode (`base_prompt`). Reference models get the plain prompt through transformers. One
greedy completion per problem, up to 2,048 tokens (half the context for shorter-context
models); the first fenced Python block is
graded inside the call's container by
[`benchmarks/livecodebench.py`](benchmarks/livecodebench.py) on every public and private
test (stdin programs compare output tokens, LeetCode-style problems call the starter
method; each test in its own `python -I` subprocess with time and memory limits). One
H100 call per model, `retries=0`, 6-hour timeout, refused 24 hours after launch with an
independent deadline supervisor. Results go to
`lm-exploration-hybrid-runs/benchmarks/lcb-<launch>/`. Checkpoint paths start with their
volume, `runs/` (`lm-exploration-hybrid-runs`) or `training/` (`lm-exploration-training`):

```sh
uv run --all-extras modal run --detach -m evaluation.modal_livecodebench \
  --checkpoints runs/RUN/step-N.pt,training/RUN/step-M.pt [--references M,...|all] [--limit N]
```

Findings: [hybrid 720M plan](../../plans/HYBRID_720M.md#livecodebench-v6).
