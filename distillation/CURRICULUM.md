# Fresh teacher generation and student qualification

`modal_curriculum.py` actually samples Qwen3-8B with vLLM on one H100. It does
not reuse the previous code corpus or earlier student generations. Teacher pin:
`b968826d9c46dd6066d109eabc6255188de91218`. Thinking is disabled, temperature
0.7, top-p 0.95, output capped at 1,024 teacher tokens. Responses contain Python
code and a concise mathematical explanation with a worked example and edge case.

```sh
modal run --detach -m distillation.modal_curriculum \
  --run-id UNIQUE_CORPUS_ID --count 512
modal run --detach -m distillation.modal_curriculum_train \
  --corpus-id COMPLETED_CORPUS_ID --run-id UNIQUE_STUDENT_COMPARISON_ID
```

Run from the repository root, with the usual managed-environment truststore
wrapper where needed. Both workers persist to `lm-exploration-distillation`.
The corpus worker saves settings, source, specifications, hidden test cases,
every raw rollout, rejections, accepted records and incremental counters. The
first invocation failed during Modal image validation before starting a worker;
the corrected invocation uses the same untouched corpus ID.

## What is verified

Eight deterministic task families cover factorials, sums of powers, counting
residue classes, GCD, primality, deduplication, prefix sums and comparisons.
Each task has independent Python-reference tests including boundaries. The
teacher does not receive these tests. A held-out transformation variant in every
family supplies validation (25% of attempted records); this is a compositional
diagnostic, not a broad Python benchmark or an untouched final test set.

Verification accepts a single `solve` function within a restricted AST, then
runs it in a separate isolated Python process with restricted builtins, no
inherited environment, a five-second wall limit, two-second CPU limit, and
256MiB address-space limit. Supported operations are intentionally limited to
pure Python functions. Importing, arbitrary attribute access and I/O calls are
rejected. The verifier checks result type, equality and preservation of inputs.
This is a constrained execution harness, not a general-purpose code sandbox.

Passing finite tests does not prove correctness on every input. In particular,
explanations and their worked examples are **unverified teacher prose** and are
labeled that way. They need a separate review/verification stage before a large
production corpus. Format failures, truncated outputs, prohibited operations and
incorrect solutions remain in the raw rollout archive; no silent retry changes
the attempted denominator. Full-record hashes and specification hashes track
exact duplicates. Different constants can still yield near-duplicate lessons.

## What the first student comparison means

Fresh 26,776,960- and 48,308,408-parameter students use the original frozen 32K
tokenizer, exactly the same shuffled training records, and one exposure per
target token. Each record is `Problem -> Python -> Explanation`, prefixed by BOS
and terminated by EOS. All tokens, including prompts, receive CE loss. This is
sequence distillation; it does not claim online probability KL.

Document-local windows have context 512; every target is counted once, including
the final short window. Microbatch eight, accumulation two, AdamW peak LR 3e-4,
10% token warmup followed by cosine decay to 10% of peak, gradient norm cap one.
The initial, approximately halfway and final checkpoints retain optimizer, RNG,
configuration, corpus identity and tokenizer identity. This bounded qualification
does not implement resume. Final reports retain actual token counts; the small
corpus is never repeated to manufacture a large exposure count.

At each checkpoint, report loss on all accepted validation records and greedy
160-token continuations on the same eight held-out prompts (one per family).
These diagnostic prompts come from the original specifications even if the
teacher failed them; filtering the training answers must not hide hard tasks.
These continuations go through the same verifier; malformed code fails. Retain
raw IDs, text and test result. This is enough to qualify data, training and
evaluation plumbing, not to compare asymptotic capacity or fit a scaling law.

Before the [scaling sweep](SCALING.md), expand semantic/template diversity and
verify explanations, measure corpus yield and generation cost, and freeze a
broader evaluation suite. A million constant substitutions are not a million
independent programming lessons. The launcher bounds generation at 4,096
attempts per invocation and does not automatically start a 500M-token job.
