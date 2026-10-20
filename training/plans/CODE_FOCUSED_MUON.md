# Code-focused 124M experiment with Muon

Status: proposed experiment, documentation only. No optimizer implementation, new corpus, or run is claimed by this PR. Execution environment was unavailable when this plan was written. Stacked on #27 to retain its training and publication infrastructure.

## Objective and evidence

Target useful short Python functions and completion first, with secondary JS/TS and systems-language coverage. Do not promise a general coding assistant at 124M parameters. The previous 20.879B lifetime checkpoint still produced invalid simple code and severe repetition; more aggregate training tokens are not a sufficient success criterion.

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

Within the 80% code allocation start with Python 50%, JS/TS 25%, C/C++/Rust/Go 20%, and SQL/shell/other 5%. These are hypotheses to test against usable capacity and intended tasks. Preserve language diversity without trying to weight dozens of languages equally in a small model. Avoid repeating a tiny test/example pool just to hit its quota: report shortfall and revise the explicit recipe before launching.

## Source shortlist

Prioritize the collected repositories, with complete pinned file snapshots instead of patch strings. Verify whether the private datasets contain full files or only patches; when necessary fetch the corresponding repository revision. Do not infer a complete source file from a diff.

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

The user confirms Muon has already been tested and selects it for this run. Reuse the tested implementation, parameter groups, learning rates, scaling and decay settings; locate and pin that existing experiment before execution. Do not repeat an AdamW-versus-Muon comparison or invent replacement hyperparameters. The prior test artifacts have not yet been inspected in this session.

Preserve the tested treatment of hidden matrices and auxiliary parameters, including tied embedding/head identity. Check only integration changes needed for this corpus/launcher, including distributed checkpoint restore. Adam moments cannot be relabeled as Muon momentum.

Proposed first stage: one 250M-target Muon run from fresh initialization, evaluated every 50M targets, focused on corpus and pipeline quality. This is a proposed bounded first stage, not a launched job or the final token budget. Keep the completed 20.879B model as a reference. Fresh initialization is a recommendation for the new experiment; record initialization explicitly. If using an existing checkpoint instead, record optimizer resets and lineage. No new optimizer comparison is planned.

## Evaluation and promotion

Evaluate at initialization and every 50M pilot targets: held-out loss by language/task, syntax validity, executable tests for unseen functions, completion length/EOS rate, exact n-gram repetition and long loops. Freeze development prompts and separate final test tasks; use matched decoding settings and seeds, both greedy and sampled. Run generated code in an isolated, time/resource-limited environment without network access or credentials. Include the existing 50-generation suite for continuity, but expand coding evaluation beyond three Python prompts.

Report uncertainty and raw examples, not only aggregate scores. Compare to the completed model under the same prompts. Select a longer run only after pipeline correctness and useful coding results support it. A 250M first stage cannot establish the final potential of a fresh model or support a matched-budget comparison with the 20B checkpoint.

A later separate fill-in-the-middle ablation (e.g. 25% of code examples) may improve editing/completion; it is not a repetition cure. It requires explicit sentinel/token handling and matching inference/evaluation. Keep the initial run causal-only. Instruction tuning on tested problem/solution pairs is a separate stage if the target becomes following coding requests.

Publish loss, task scores, repetition curves and sample outputs alongside checkpoints, with corpus manifest, code revision, initialization and optimizer provenance. A falling mixed loss alone is not grounds to extend training.
