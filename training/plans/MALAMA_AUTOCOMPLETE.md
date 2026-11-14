# Malama autocomplete: training plan

Updated November 14, 2026. This is the active path for the Malama models. The second
path, a sub-1B [coding reasoner](CODING_REASONER_1B.md), shares the same foundation
model and is deferred.

## Goal

A small model that is competitive for its size and training compute at code
completion: filling in code between a prefix and a suffix, and continuing code from a
short prompt (a signature, a docstring, a comment). The claim to test is a score
against training FLOPs, with Malama above the line the comparison models form.

## Foundation

[Malama Base](https://huggingface.co/jfrick/malama-base): the 960M KDA hybrid trained
on 100B tokens (50% ClimbMix, 40% code, 10% math), about 5.8e20 training FLOPs
(6 × parameters × tokens). Details and results are in
[foundation constraints](KDA_960M_PRETRAINING.md) and the model card. The base model has
not seen fill-in-the-middle (FIM) data: the GLM tokenizer's `<|code_prefix|>`,
`<|code_suffix|>` and `<|code_middle|>` tokens are untrained.

## Stages

1. **FIM midtraining branch (running).** `workflows/kda-960m-fim-midtrain.json`
   continues the foundation run's last constant-learning-rate checkpoint (84.8B tokens)
   with weights, optimizer, RNG and data cursor. Half of all code documents are
   rearranged for FIM, half of those suffix-first (`fim` in
   `hybrid/configs/kda-960m-fim-midtrain.json`). The learning rate decays at the same
   token positions as the foundation run's own decay to 100B, and the data is otherwise
   identical, so Malama Base is the matched control.
2. **Final midtraining mix.** Chosen from the branch's results:
   - the FIM rate, and character-level instead of token-level split points if
     token-boundary artifacts show up in the scores;
   - fresh high-quality code rather than further passes over the curated projects,
     which are already read five times;
   - short convolutions with and without, added at identity
     ([128M study](KDA_960M_PRETRAINING.md#architecture-options-from-the-marin-notes-128m-study));
   - longer context for repository-level completion, if the repository-level scores
     justify it.
3. **Light instruction tuning.** Short completion prompts: comment or docstring to
   code, and infilling requests phrased as a user would. The model should stay a
   completion model, not a chat model.
4. **Reinforcement learning on executable completions.** Rewards from tests passing on
   infilled or completed code, on tasks held out from the evaluation sets.

Each stage starts only if the one before it improves the scores below without an
unacceptable left-to-right cost.

## Evaluation

| Benchmark | Measures | Role |
| --- | --- | --- |
| SAFIM (2024) | Infilling of algorithm blocks, control flow and API calls in Python, Java, C++ and C#; mostly execution-graded | Primary |
| HumanEval-Infilling (single-line, multi-line) | Python infilling, test-graded | Secondary; reported for most small FIM models |
| HumanEval, MBPP | Left-to-right completion | The cost of FIM against Malama Base |
| Held-out bits per byte | Left-to-right prediction per source | The cost of FIM, measured during training |
| CrossCodeEval or RepoBench | Repository-level completion with retrieved context | Later, with stage 2's context decision |

- Score the comparison models with the same harness and prompt format rather than
  copying published numbers: SantaCoder 1.1B, StarCoderBase-1B, DeepSeek-Coder-1.3B,
  Qwen2.5-Coder-0.5B and 1.5B. Their training FLOPs come from their papers and are
  stated with sources.
- Extend the training data's contamination screen to SAFIM and HumanEval-Infilling
  before any score counts. HumanEval is already screened; SAFIM is not.
- Keep the scoring code in Git and the scores, samples and reports on Hugging Face,
  with findings written up in this folder.

## Budget

The FIM branch is about 15B tokens, roughly 11 hours on 8 × H100 (about 92 H100-hours).
Evaluation of each model fits on one H100 per benchmark. Reconcile spending before
stage 2.
