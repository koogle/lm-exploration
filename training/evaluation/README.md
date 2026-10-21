# Fixed Python evaluation

`python-tasks.json` contains 32 reproducibly selected function tasks from the
sanitized **test** split of Google Research MBPP, pinned at Hugging Face revision
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`.

Source and attribution: [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
[dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp).
The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Prompts are adapted from task descriptions and reference function signatures;
reference solutions and tests retain the dataset content. Task selection is
SHA-256 ordered, fixed before training, and not based on model performance.

Tests and reference implementations are hidden from model prompts. Exact reference
hashes for all 257 test records are excluded from training. This is a bounded
exclusion, not an exhaustive contamination audit or benchmark-clean claim.
Execute generated code only in credential-free, network-blocked sandboxes with CPU,
memory and wall-clock limits: `trainer.python_evaluation.execution_script` builds the
program to run there, and `reference_cases` adds the reference solutions and an
explicit network/environment isolation probe that must pass the same harness first.
`evaluate_checkpoint` produces the fixed generations without executing them.
