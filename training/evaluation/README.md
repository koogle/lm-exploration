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
Generated code executes only in credential-free, network-blocked Modal sandboxes
with CPU, memory and wall-clock limits. Reference solutions and an explicit
network/environment isolation probe must pass the same harness first.
