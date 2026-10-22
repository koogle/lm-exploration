# Fixed Python evaluation

`python-tasks.json` holds 32 function tasks from the sanitized **test** split of Google
Research MBPP, pinned at Hugging Face revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
Task selection is SHA-256 ordered, fixed before training and not based on model
performance. `reference_hashes` lists the exact reference hashes of all 257 test records,
for excluding them from training corpora. That is a bounded exclusion, not an exhaustive
contamination audit.

Source and attribution: [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
[dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp), licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Prompts are adapted from
the task descriptions and reference function signatures. Reference solutions and tests
keep the dataset content and are never shown in prompts.

## Run

1. **Generate with the [sampling toolkit](../analysis/sampling/README.md).** Use a recipe
   with `"prompt_file": "<path to this file, relative to the recipe>"` (optionally with
   `"prompt_group"`). `load_recipe` appends the 32 task prompts. Greedy decoding with
   `max_new_tokens` 256 matches the original setup.
2. **Grade.** From `training/`, with Modal credentials:

   ```sh
   uv run python -m trainer.python_evaluation --samples artifacts/sampling/NAME.json \
     --output artifacts/python-evaluation/NAME.json
   ```

   - **Isolation:** grading runs in one Modal sandbox with `block_network=True`, no secrets
     or volumes, and a one-hour hard timeout. Each case runs in its own process with a
     10-second limit, CPU, memory, file and descriptor rlimits, an empty environment and
     discarded output.
   - **Completions:** each completion is cut at the first new top-level statement.
   - **Harness check:** the reference solutions and a network/environment isolation probe
     must pass the harness before any sample is graded.
   - **Pass rule:** a case passes only if every hidden test passes and the process prints
     its random marker.
   - **Output:** per-case results and pass/syntax fractions per model and decoding mode,
     written to the ignored `artifacts/` folder. Results belong on Hugging Face.

`subprocess_executor` runs cases locally. Use it only for the reference solutions and
tests, never for model output.
