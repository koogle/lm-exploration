# Distillation validation ladder

A bounded experiment testing whether the Qwen3.5-9B Python/math trajectories
(private HF dataset `jfrick/qwen35-python-math-trajectories-20261022`) teach a small
model anything measurable. It is added in stacked parts; this part prepares the data.
Evaluation, students and the Modal campaign follow in later pull requests.

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

## Tests

```sh
python -m unittest discover -s distillation/validation/tests -v
ruff check distillation && ruff format --check distillation
```
