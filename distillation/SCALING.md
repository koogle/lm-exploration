# Choose the budget from learning curves

The objective is a small model that generalizes on Python tasks at low serving
cost. Chinchilla provides a starting hypothesis for allocating training compute;
it does not measure how many facts a parameter stores, establish a maximum useful
token count, or prescribe an additional fine-tuning budget.

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) study compute-optimal
training from scratch. The often-used approximation is roughly 20 training
tokens per parameter, with model size and data growing together as compute
increases. The ratio depends on the data, architecture, optimization and fitted
regime. Narrow synthetic Python data needs its own measurements. For a fixed
deployment size, training beyond the compute-optimal point can still improve
quality and be worthwhile over many inference requests.

## Our actual sizes

Run `python -m distillation.scaling --accepted-tokens 500000000` to reproduce the
counts and rough budgets. The modern GPT uses tied embeddings, so count the
embedding/output matrix once. These illustrative 20N figures use **total**
parameters; they are not calibrated predictions for vocabulary-heavy models.

| Architecture | Vocabulary | Total parameters | Embeddings | Illustrative 20N |
| --- | ---: | ---: | ---: | ---: |
| 8 layers, width 384 | 32,768 | 26.78M | 12.58M | 536M tokens |
| 10 layers, width 512 | 32,768 | 48.31M | 16.78M | 966M tokens |
| 14 layers, width 768 | 32,768 | 124.39M | 25.17M | 2.49B tokens |
| 8 layers, width 384 | 151,936 | 72.54M | 58.34M | 1.45B tokens |
| 10 layers, width 512 | 151,936 | 109.32M | 77.79M | 2.19B tokens |
| 14 layers, width 768 | 151,936 | 215.91M | 116.69M | 4.32B tokens |

Thus 500M tokens is a reasonable scale to investigate for a fresh approximately
27M model; it is not evidence that a 216M model has exhausted useful learning.
Sharing the large Qwen vocabulary is particularly expensive for small students.
Generated prompt/code/explanation text can be retokenized with our original 32K
tokenizer. Exact token-probability KL needs aligned token IDs, but sequence
distillation does not require a shared tokenizer. Tokenizer choices also change
sequence lengths: compare task accuracy, bytes and measured compute alongside
token counts, never raw token CE across tokenizers.

The existing backbone already received about 20.879B tokens before distillation.
Its 50M-token shared-vocabulary run is continued training after a vocabulary
change. Neither the new 50M exposures nor the lifetime count can be inserted
into the from-scratch heuristic to predict its remaining learning. Its factorial
failures could involve data coverage, optimization or capacity; the generations
do not isolate those causes.

## Proposed experiment, not launched by the calculator

1. Qualify a bounded synthetic corpus pilot using
   `Qwen/Qwen3-8B`, revision
   `b968826d9c46dd6066d109eabc6255188de91218` (Apache 2.0).
   This is the post-trained generation teacher, distinct from the 4B base model
   used for online KL. Its 8.19B BF16 weights occupy roughly 16.4GB before KV
   cache and runtime memory; one Modal H100 is a reasonable starting host.
   Measure accepted student tokens/sec and verification yield before estimating
   the price of a million answers or 500M accepted tokens.
2. Build prompt + solution code + concise explanation + worked-example records.
   Cover arithmetic, loop invariants, recursion, indexing, accumulation and data
   structures. Pair cases such as factorial versus square, inclusive versus
   exclusive bounds, and order-preserving deduplication versus sorting.
   Independently executable tests verify code; passing tests do not verify prose.
   Generated tests alone are insufficient. Track explanation review separately.
3. Use the same frozen 32K tokenizer and data order for fresh 26.78M and 48.31M
   students. Save checkpoints at approximately 5, 10, 20 and 40 tokens per total
   parameter. These are sampling points around a hypothesis, not established
   optima. Independently train at matched compute budgets with budget-appropriate
   LR decay before claiming a compute optimum: intermediate checkpoints of one
   long schedule are not equivalent to optimally scheduled short runs.
4. Compare the current pretrained student on the same curriculum as a separate
   adaptation track. Include a matched code-only versus code-plus-explanation
   comparison at equal loss-bearing token exposure. An independently verified
   boundary-case mixture can then be compared at equal measured training compute.
   Do not confound the tokenizer change with an explanation benefit.
5. Hold out entire problem templates and combinations of conditions. Freeze
   tests before generation; deduplicate and keep related variants in the same
   split. Report executable pass@1, boundary-case accuracy, repetition and
   held-out loss at every checkpoint. Compare generations on fixed prompts.
   Use a separate final test set after choosing recipes on validation results.

Record unique accepted records, unique student tokens, repeated exposures,
prompt versus answer loss tokens, teacher-generated tokens including rejects,
and the total teacher + verification + training GPU/CPU cost. Repeating 50M
tokens ten times is not equivalent to 500M tokens of diverse verified data.
For initial all-token language-model training count prompts as targets; if
switching to assistant-only loss, report both context exposures and loss-bearing
targets so the change is visible.

Select the smallest model meeting the held-out task target, then extend its
training while the marginal quality gain justifies the cost. Fit a scaling curve
only after observing several sizes and budgets with stable optimization and
repeat seeds near the apparent optimum. This estimates useful generalization
per parameter; it does not establish a literal information-storage capacity.

Status: parameter calculator and [fresh curriculum qualification](CURRICULUM.md)
implemented. Fresh generation has run; the full token-budget scaling sweep above
remains a separate experiment, not something the calculator launches. The
completed 50M distillation run and its negative coding examples remain the
pretrained baseline.
