# Distillation validation ladder

A bounded experiment testing whether the Qwen3.5-9B Python/math trajectories
(private HF dataset `jfrick/qwen35-python-math-trajectories-20261022`) teach a small
model anything measurable. It is added in stacked parts; these parts prepare the data, the evaluation and the
students. The Modal campaign and its findings follow in the last pull request.

## Data preparation

`prepare.py` converts accepted teacher records (stop-terminated, JSON/syntax-checked)
into flat NumPy shards: output token IDs (native Qwen3.5 IDs; the generation prompt is
dropped), document offsets, the teacher's top-16 IDs and log probabilities per
position (99.7% of the mass in a sample), and native ID counts. Held-out shards also
keep the problem/code/explanation text.

- **Split.** `common.is_heldout`: chunks whose salted path hash falls in the lowest
  2% are held out; `common.training_order` fixes one salted chunk order, so every
  token budget is a prefix of the same stream.
- **Trimmed vocabulary.** `build_vocab`: PAD, BOS and UNK plus the most frequent
  native IDs; unseen IDs map to UNK.
- **Covering vocabulary.** `build_covering_vocab`: the same size, but it keeps all 256
  byte tokens and spells every other native token as kept pieces (greedy longest
  match), so any text encodes without UNK.
- **Byte lengths.** `byte_lengths` gives each native token's UTF-8 length, for loss per
  byte across vocabularies.

`shards-card.md` is the dataset card for the archived shards
(`jfrick/qwen35-distillation-shards-20261023`).

## Evaluation sets and grading

`evalsets.py` builds two execution-graded sets in the training data's JSON format:

- `mbpp`: all 427 sanitized MBPP problems (human-written tests); the tested function is
  renamed to `solve` and the problem shows its signature and first test.
- `curriculum`: exercises from held-out chunks with teacher-written asserts, kept only
  when the teacher's original solution passes all of them (`curriculum_items` counts
  the rest, an estimate of how self-consistent the data is).

It also holds the student prompt format, the parser for the generated code string and
the teacher prompts. `grader.py` runs each case in its own isolated interpreter with
CPU, memory, file-size and process limits and a random pass marker. Model output is
only ever graded inside a network-blocked sandbox; the tests run fixed snippets.

## Students and inference

- `student.py`: the repository's modern decoder (`training/10m/model.py`) trained on a
  packed, fixed-order stream (a BOS per document, no repeats) with cross-entropy or
  `0.5 · CE + 0.5 · soft CE` against the teacher's {top-16, residual} partition. The soft
  loss uses only the gathered teacher logits and one log-sum-exp per position.
  Vocabularies: trimmed, covering (pieces expanded in training and prompts) or the full
  248K native IDs.
- `vllm_model.py`: the same decoder written for vLLM (paged attention, interleaved RoPE,
  biased projections, tied embeddings, identical parameter names). vLLM loads the
  training checkpoint `model.pt` directly with `load_format="pt"`.
- `vllm_eval.py`: greedy generation on vLLM for the teacher (test writing and the
  ceiling) and for students (from token IDs, decoded through the student vocabulary),
  with a vLLM-vs-PyTorch prompt log-probability parity check.
- `jobs.py`: the container job body that trains (or loads) a student, validates it and
  evaluates it through vLLM; pass@1 scoring with Wilson intervals.

## Tests

```sh
python -m unittest discover -s distillation/validation/tests -v
ruff check distillation && ruff format --check distillation
```
