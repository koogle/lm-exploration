# Checkpoint evaluation

Benchmarks for our own training checkpoints on the `lm-exploration-training` volume. One
Modal call scores a list of checkpoints and writes structured JSON; findings are written
by hand in the owning study's docs. uv installs this folder as `evaluation`.

| File | Role |
| --- | --- |
| `tasks.py` | Task formats, log-likelihood batching, greedy generation, bits per byte |
| `modal_evaluate.py` | Modal launcher: one H100, 3-hour timeout, no retries |

## Run

From `training/`:

```sh
uv run --all-extras modal run -m evaluation.modal_evaluate --name NAME \
  --snapshot climbmix-code/VERSION/snapshots/RUN \
  --checkpoints RUN-s01/step-000050000.pt,RUN-s02/step-000100000.pt
```

Each checkpoint loads the model code its run copied to `RUN-sNN/source`, after checking
the source and tokenizer hashes the checkpoint recorded, so any checkout can evaluate any
checkpoint. The call refuses new work 24 hours after launch. Results go to
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
  first five training examples as `Question: ...\nAnswer: ...` exemplars. Generation stops
  at a blank line, `Question:` or EOS (256 and 32 new tokens). GSM8K takes the number
  after `####`, else the last number; SVAMP takes the first number. Answers are kept in the
  result.

Datasets are pinned to HF commits in `DATASETS`. Inference uses BF16 autocast with FP32
weights. Chance levels: about 25% for HellaSwag, ARC and OpenBookQA, 50% for Winogrande.
