# Fresh Qwen3-8B curriculum generation

Two new generation runs completed on one Modal H100 each. Teacher revision
`b968826d9c46dd6066d109eabc6255188de91218`; vLLM 0.11.0, BF16, thinking disabled,
temperature 0.7, top-p 0.95. Raw rollouts, hidden tests, accepted records and
source snapshots are on the `lm-exploration-distillation` volume under the run IDs.

| Run | Attempts | Accepted | Train tokens | Validation tokens | Generation + verification time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen8b-python-curriculum-20261021-01` | 512 | 456 (89.1%) | 66,732 | 21,754 | 89.96s |
| `qwen8b-python-curriculum-20261021-02` | 512 | 500 (97.7%) | 74,672 | 25,516 | 91.80s |

These are original-32K-tokenizer counts including prompts, code, explanation
and EOS. Raw teacher output counts were 83,246 and 84,320 respectively. End-to-end
times including initialization were 183.18s and 122.42s. The initial deployment
attempt failed image validation before a GPU worker started; it generated no data.

The two batches share problem families and constants. They are **not disjoint
datasets** and must not be concatenated as independent new examples. Only the
second batch feeds the fresh-student qualification. Each has 384 attempted train
specifications and 128 validation specifications, selected by transformation
variant before generation and verification.

## A failure caught before training

Run 01 accepted only 20/64 factorial solutions. A representative rejected answer
computed `n!` when asked for the factorial of `n + 1`. The second run changed
the specification to explicitly define `t = n + offset`, then request `t!`.
All 64 factorial solutions in run 02 passed. The seed and sampling recipe stayed
fixed, but the prompt changed; this is a generation-quality intervention, not
evidence of a change in teacher weights or general reasoning ability.

Fresh accepted example from run 02, for `(n + 3)! mod 488`:

```python
def solve(n):
    t = n + 3
    result = 1
    for i in range(1, t + 1):
        result = (result * i) % 488
    return result
```

This passed the 20 independently specified inputs n=0..19. It is a teacher
generation, not a student generation. Its explanation gives the correct n=0 and
n=1 examples, but uses imprecise “prevent overflow” language: Python integers
have arbitrary precision; intermediate reduction bounds their size. All prose
retains the `unverified_teacher_prose` label, including this answer. Code tests
alone do not qualify the explanation corpus for large-scale training.

The remaining 12 rejected answers in run 02 were 11 GCD responses and one prime
response. Rejection covers syntax/API-contract failures as well as wrong outputs;
it is not uniformly a mathematical error rate. The rejected prime answer tested
divisors up to sqrt(n) while classifying n+1, despite prose claiming sqrt(n+1).
The independent oracle caught the boundary error.

## Completed fresh-student qualification

Both students completed exactly one pass over the 372 accepted training records:
74,672 loss-bearing tokens and 24 updates each, with identical tokenizer, corpus
order and optimizer recipe. They start from random weights. This is a pipeline
smoke test, many orders of magnitude short of the proposed scaling budgets.

| Student | Initial validation CE | At 38,567 tokens | At 74,672 tokens | Final diagnostics |
| --- | ---: | ---: | ---: | ---: |
| 26.78M | 10.4658 | 8.0962 | 7.3972 | 0/8 |
| 48.31M | 10.5231 | 7.3101 | 6.3994 | 0/8 |

All validation losses use the same 25,516 tokens. Both models pass 0/8 at all
three checkpoints. The 26.78M final factorial continuation consists of repeated
spaces. Lower CE after a few updates is not evidence of Python competence,
capacity saturation, or which model is preferable at a full training budget.
The larger model also uses more compute; this is an equal-data qualification,
not an equal-compute scaling comparison.

Each model saved initial, middle and final checkpoints with optimizer state.
Elapsed times (including evaluation/generation/checkpoint work) were 58.24s and
61.37s; peak allocated GPU memory was 1.79GB and 2.76GB. Do not extrapolate these
overhead-dominated timings as sustained training throughput.

- [Student training report](python-curriculum-students-20261021-01.json).
- [All 48 checkpoint continuations](python-curriculum-students-20261021-01-generations.json).
- Checkpoints on the distillation volume:
  `python-curriculum-students-20261021-01/student-{26776960,48308408}/step-{000000000,000000012,000000024}.pt`.

## Scale implications

Run 02 produced about 1,091 accepted tokens/sec including validation, or 813
**training** tokens/sec. A naive linear projection to 500M accepted training
tokens is roughly 171 H100 hours for generation/verification at this recipe,
excluding startup, failures, student training and wider curriculum verification.
This is a small-batch extrapolation, not a quote or optimized-throughput promise.
These short records average about 200 tokens: a million accepted answers would
be about 200M tokens at this length, not automatically 500M.

The eight template families qualify the pipeline. Scaling only their constants
to millions of answers would not supply the diversity needed for the proposed
capacity experiment. A 500M-token launch remains pending corpus expansion and
explanation verification, rather than repeating this small corpus to reach the
number. The 27M/48M one-pass qualification is a separate actual GPU run.

Machine-readable reports and samples:

- [Run 01 report](qwen8b-python-curriculum-20261021-01.json),
  [accepted/rejected samples](qwen8b-python-curriculum-20261021-01-samples.json).
- [Run 02 report](qwen8b-python-curriculum-20261021-02.json),
  [accepted/rejected samples](qwen8b-python-curriculum-20261021-02-samples.json).
- Run 02 Modal: https://modal.com/apps/radiantai/main/ap-0NMPOayGGF1sQ6tN21B7GW
- Student qualification: https://modal.com/apps/radiantai/main/ap-NXrAJZ7xBGBAjmt7wPwF2k
