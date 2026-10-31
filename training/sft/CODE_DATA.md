# Code SFT data, teacher solutions and the RL prompt pool

How the function-writing data for SFT and the problems for RL are built. All of it is
generated on Modal into the private run volume (`/runs/<data-id>`); nothing is trained here.

## Problems

[`../data/code_sft.py`](../data/code_sft.py), `modal_sft_data::code`, format `code-sft-v1`:

- **Prompt:** ``Question: {problem}\nWrite a Python function `def f(args):`.\nAnswer:\n``
  (or ``class `class C:` ``); **answer:** the complete solution, then EOS.
- **Sources:** NVIDIA OpenCodeInstruct (CC-BY-4.0) problems from shards 0–5 of 50, each
  with ten unit tests, kept only if its own Qwen2.5-Coder-32B solution passed all ten; and
  MBPP train (CC-BY-4.0). Solutions keep their imports, functions and classes but not
  their example-usage code. Two in 1,000 OpenCodeInstruct problems, by hash of their ID,
  are validation.
- **Decontamination:** rows sharing a 13-word n-gram with an MBPP test, validation or
  prompt problem, an MBPP+ or a HumanEval+ problem are dropped.
- **Tests:** every row keeps its tests as one module that imports the solution as
  `solution`; SFT ignores them, RL and the teacher run them.

`code-oci-mbpp-01`: 227,084 OpenCodeInstruct and 306 MBPP rows (453 validation); 80 were
contaminated. On a 500-row sample, 98% of the kept reference solutions passed their
tests locally; the rest read files or globals their example code created.

## Running model-written code

Code runs only in the sandbox described in [`../data/sandbox/`](../data/sandbox/README.md).

## Teacher solutions

[`../data/teacher.py`](../data/teacher.py),
[`../modal_scripts/modal_teacher.py`](../modal_scripts/modal_teacher.py): Qwen3.5-35B-A3B
(Apache-2.0, FP8, thinking off, vLLM 0.31 on H100s) answers every problem `--samples`
times. Each answer's code runs against the problem's tests in the sandbox, and the
shortest passing answer replaces the row's solution; problems without one are dropped.
On a 64-problem trial (`code-teacher-smoke-04`, 2 samples) 59 problems kept a solution,
with a median of 311 characters against 593 for the OpenCodeInstruct solutions. The full
run (`code-teacher-qwen35-02`, 8 H100 shards, 2 samples) kept 204,554 of the 227,390
problems (204,149 train, 405 validation); 22,758 had no passing answer and 78 no
extractable code.

Each shard's answers and kept rows are saved under `/runs/teacher-work/<data-id>/` as
soon as they exist, so rerunning the same command after a failure resumes. The
orchestrator spawns one H100 call per shard only while the shard's timeout fits in
the 24-hour launch budget, cancels unfinished shard calls when it stops, and its sandbox
times out at the deadline.

## Mixtures

[`../data/mixture_sft.py`](../data/mixture_sft.py), `modal_sft_data::mixture`, format
`mixture-sft-v1`: the rows of a math and a code dataset in one file, each tagged with its
task, which decides its prompt and answer. The trainer samples rows uniformly, so each
task's share is its share of rows; validation buckets are `task/bucket`.
`mixture-math-code-02` holds the 224,533 math rows of `math-gsm8k-metamath-02` and the
204,554 code rows of `code-teacher-qwen35-02`.

## RL prompt pool

[`../data/rl_tasks.py`](../data/rl_tasks.py), `modal_sft_data::rl_pool`, format
`rl-pool-v1`: problems with checkable answers and their rewards (1 if the answer ends in
EOS and is correct, else 0). Math answers are checked with Math-Verify against the GSM8K
final answer; code answers by running the tests in the sandbox. Each problem holds its
prompt text and answer format, here the `sft` style (the SFT prompts, plain answers).
`rl-pool-math-code-04` (the problems of `rl-pool-math-code-03`, which predates the prompt
fields):

| Split | Math | Code |
| --- | ---: | ---: |
| Train | 6,971 GSM8K training problems after the first 500 (SFT's validation) | 10,000 OpenCodeInstruct problems from shard 49 |
| Validation | 199 of the first 200 GSM8K training problems | 77 MBPP validation problems |

## Run

From `training/`:

```sh
uv run --all-extras modal run -m modal_scripts.modal_sft_data::code --data-id code-oci-mbpp-01
uv run --all-extras modal run -m modal_scripts.modal_teacher --source code-oci-mbpp-01 \
  --data-id code-teacher-qwen35-02 --shards 8 --samples 2
uv run --all-extras modal run -m modal_scripts.modal_sft_data::mixture --data-id mixture-math-code-02 \
  --math math-gsm8k-metamath-02 --code code-teacher-qwen35-02
uv run --all-extras modal run -m modal_scripts.modal_sft_data::rl_pool --data-id rl-pool-math-code-04
```

Preparation reports are in `jfrick/lm-exploration-results` under `pr-87/data/`.
