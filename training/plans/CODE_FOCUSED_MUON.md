# Code-focused 124M experiment with Muon

Status: implementation and CPU corpus preparation in progress. The tested Muon implementation from PR #24 is integrated, stacked on #27. Modal and Hugging Face access have been verified. No GPU optimizer step or model publication is claimed for this experiment. See [Measured status](#measured-status-2026-10-21-0314-utc) below.

## Measured status (2026-10-21 03:14 UTC)

Corpus frozen; eight-H100 qualification queued (call `fc-01M5E6S46XR9X2H8CH7JWH1820`,
controller `fc-01M5E6S52WN7SZ9C9B0CQSWX85`). No eight-GPU optimizer update, checkpoint
or model publication yet.

| Category | Target | Unique train targets | Required passes |
|---|---:|---:|---:|
| python_implementation | 3,250,000,000 | 3,344,840,434 | 0.972 |
| python_tests_examples | 750,000,000 | 1,455,959,517 | 0.515 |
| python_documentation | 500,000,000 | 638,756,689 | 0.783 |
| cs_math | 250,000,000 | 255,001,393 | 0.980 |
| general_education | 150,000,000 | 153,000,047 | 0.980 |
| books | 100,000,000 | 102,061,503 | 0.980 |

- Final corpus: 5,949,619,583 unique usable targets after filtering, global dedup and
  validation exclusions; all categories `max_passes=1`. Collected Git snapshots
  contribute 108,063,930 targets (31,016 files), consumed first. 192 fixed validation
  windows from distinct held-out parents.
- Acquisition: 53 code shards (5,742,078,521 candidate targets) and 95 documentation
  shards (617,694,542) before global merge.
- Negative results kept: the repository-only census gave only 109,455,272 targets;
  a partial supplemental census left documentation at 29,876,984 vs the 250M two-pass
  minimum; `codeparrot/github-jupyter-parsed` has at most 186,211,988 targets (upper
  bound, not added); The Stack / StarCoderData probes returned `GatedRepoError`. The
  first full-code preparation failed on a parser `MemoryError` (fixed, shards resumed).
- Single-H100 124M diagnostic: tiny-set loss 10.464919 → 0.001435 in 21 updates,
  greedy `return a + b`; logit agreement, causality, cached RoPE and Muon parameter
  assignment passed. All 32 MBPP reference tasks and the isolation probe passed the
  network-blocked sandbox harness. Partial Modal billing snapshot: USD 6.18 (not total).

The dated execution record, capacity tables/JSON, frozen manifest, qualification,
harness and billing evidence are archived in
[`jfrick/lm-exploration-results` / `pr-30/training/results/python-124m-5b-20261020/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-30/training/results/python-124m-5b-20261020)
(HF commit `b63cb65d6d3cc24f683355bf8771a70e586af93d`). The PR #24 Muon vs AdamW results
this branch carried are archived once under
[`pr-24/training/results/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-24/training/results).

## Objective and evidence

Target useful short Python functions and completion. All code allocation is Python. Do not promise a general coding assistant at 124M parameters. The previous 20.879B lifetime checkpoint still produced invalid simple code and severe repetition; more aggregate training tokens are not a sufficient success criterion.

Patch records represented only about 94.7M of 9.73B prepared code tokens (about 1% of code capacity), not the majority. Consumed proportions need measurement. Removing patches is an experiment in task alignment, not an established explanation of the previous failure.

## Proposed token mix

Weights are fractions of nonpadding training targets, not record counts. Categories are exclusive; count every record once.

| Category | Weight | Selection |
| --- | ---: | --- |
| Repository implementation code | 65% | Complete source files from pinned repository snapshots, coherent function/class fragments for oversized files |
| Tests and executable examples | 15% | Repository tests and examples, associated implementation context when it fits |
| Programming documentation and tutorials | 10% | API docs, READMEs, explanations with code; rendered duplicates excluded |
| CS/math educational text | 5% | Algorithms, data structures, discrete math, practical computing |
| General educational prose | 3% | Selected FineWeb-Edu or equivalent |
| Books | 2% | Clean public-domain/licensed prose; avoid OCR noise and boilerplate |

This proposes raising code from 60% to 80% for the new specialization; it is not a claim that 80% is optimal. General old books are optional support for prose, not a known cure for coding errors. Programming textbooks belong to the technical categories.

All implementation, test, and executable-example code is Python. Train from fresh initialization for **5,000,000,000 nonpadding target tokens**, preserving the six category weights above. Prefer one pass and never exceed two passes per category. Insufficient capacity blocks launch; it does not authorize changing the mixture or repeating a small corpus.

## Source shortlist

Prioritize complete repositories updated recently, ordered by snapshot commit date, with complete pinned file snapshots instead of patch strings. Verify whether the private datasets contain full files or only patches; when necessary fetch the corresponding repository revision. Do not infer a complete source file from a diff.

The Pile's GitHub component is a candidate supplement, not a quality label or guaranteed repository checkout. Inspect available parent/revision metadata and overlap with current collections before assigning it volume. Prefer sources with usable repository identity for validation exclusion.

Other candidates: The Stack v2 / StarCoderData for broader repository coverage; OpenCoder's RefineCode for a curation recipe and candidate corpus; official language/library documentation; repositories with meaningful unit tests and examples. Availability, record quality and eligibility for these candidates have not been verified in this session. Preserve existing license policy and provenance. Candidate data links:

- https://huggingface.co/datasets/bigcode/the-stack-v2
- https://huggingface.co/datasets/bigcode/starcoderdata
- https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus
- https://github.com/OpenCoder-llm/OpenCoder-llm
- https://pile.eleuther.ai/

Do not take all of The Pile for this specialized recipe. Retain original comments, indentation, tests and docstrings. Remove vendored/minified/generated files, lockfile dumps and obvious repeated boilerplate. Deduplicate file hashes across corpus sources and cap any single repository's token contribution. Record total and unique tokens separately. Related but nonidentical versions are permitted under the user's existing policy; prefer one canonical snapshot per repository for this experiment.

Freeze held-out repositories before window creation, exclude their files/hashes across all sources, and sample representative validation by language and task. Keep this bounded: the user has explicitly declined exhaustive benchmark-contamination and tokenizer-quality audits. Do not describe results as benchmark-clean. Use the existing tokenizer initially; add no tokenizer-fragmentation gate.

## Establish pipeline correctness first

Check label shift, ignored padding, BOS/EOS boundaries, causal masking including cropped batches, and position handling. Compare teacher-forced training logits to generation logits for identical prefixes, with relevant numerical tolerances. Demonstrate tiny-set overfit and known-function completion. Check loss aggregation weights against the actual target counts.

Keep unrelated packing changes separate from the initial data experiment. Current cropping plus cached RoPE and torch.compile, microbatch 2 on eight H100s, is the measured starting configuration. Preserve 8K support; log actual cropped lengths, useful targets/sec and GPU time. No throughput claim transfers automatically to Muon.

## Selected optimizer: Muon

PR #24 (`1b9a7379e361d92f7e2ace0b15758a7e6eeed2f5`) contains the tested implementation and results. Its Muon run completed 99,988,281 targets / 1,296 updates at 248,500 steady nonpadding targets/sec and 13.993 GiB maximum allocated GPU memory. Diagnostic validation loss was 2.7204; this is a tiny validation set, not broad capability evidence.

Reuse `trainer/optimizers.py`: hidden block matrices use five-step BF16 Newton–Schulz Muon with original aspect-ratio scaling, momentum 0.95, peak LR 0.02. Tied embedding/output weights, biases and norms use fused AdamW, peak LR 0.0003 and betas 0.9/0.95. Both use decay 0.1, clipping 1.0, 20M-target warmup and cosine decay to 10%. Global batch 32, microbatch 2, eight H100s, context 8192, cropping, cached RoPE and dynamic torch.compile are already measured together. Do not repeat an optimizer comparison.

Start from fresh weights and a fresh cursor. Preserve both optimizer states, per-rank RNG and consumed data cursors across worker segments. Reuse the prior passing eight-rank Muon checks; verify new integration and restore before the substantial run.

## Evaluation and promotion

Evaluate at initialization, around 50M targets, and regularly thereafter: held-out loss by language/task, syntax validity, executable tests for unseen functions, completion length/EOS rate, exact n-gram repetition and long loops. Freeze development prompts and separate final test tasks; use matched decoding settings and seeds, both greedy and sampled. Run generated code in an isolated, time/resource-limited environment without network access or credentials. Include the existing 50-generation suite for continuity, but expand coding evaluation beyond three Python prompts.

Report uncertainty and raw examples, not only aggregate scores. Compare to the completed model under the same prompts. Continue the authorized 5B run unless there is a concrete correctness failure or unusable data. Do not require an early fresh checkpoint to outperform the prior 20B model.

A later separate fill-in-the-middle ablation (e.g. 25% of code examples) may improve editing/completion; it is not a repetition cure. It requires explicit sentinel/token handling and matching inference/evaluation. Keep the initial run causal-only. Instruction tuning on tested problem/solution pairs is a separate stage if the target becomes following coding requests.

Publish loss, task scores, repetition curves and sample outputs alongside checkpoints, with corpus manifest, code revision, initialization and optimizer provenance. A falling mixed loss alone is not grounds to extend training.
