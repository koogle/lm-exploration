# Qwen3.5-2B drafting quality experiment

User direction: improve drafting quality, not speed; test the 2B Qwen.
Use [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), pinned at
`15852e8c16360a2fea060d615a32b45270f8a8fc`. This is a model-family comparison,
not a parameter-count-only ablation against Qwen2.5-0.5B. Tokenizer, vocabulary,
backbone architecture, learned weights and draft width differ.

Retain a two-layer bidirectional Qwen2-style drafter, frozen target embeddings
and output weights, fixed post-block features 2/22, seven masked NEW tokens after
a known anchor, and the existing decaying cross-entropy objective. The 2B target
has 2048-dimensional states versus 896 before, so draft capacity also increases.
The drafter retains Qwen2-style biased attention projections and RMSNorm, not the
target's recurrent blocks or gated attention. Its rotary embedding follows the
target's partial rotary width; unrotated channels pass through unchanged.

Measure training and validation loss, per-position accuracy and consecutive NEW
matches by domain. Test remains sealed. Do not compare raw cross-tokenizer NLL
as if it were the same task, or call target imitation human answer quality.

For matched source prompts, decode the archived 0.5B prompt IDs and retokenize
without truncation. Retain original groups/splits and re-audit
cross-split prompt duplication. Generate fresh 2B continuations/features. This
initially used raw continuation prompts. The smoke sample audit found repetitive
prose and appended thinking text after translations, so the raw full preparation
was stopped before training. The revised protocol uses the official chat template
with `enable_thinking=False`, explicit continuation-only prose/code requests, and
translation-only instructions. Source text/groups are preserved, but prompt format
is an additional cross-model difference. Samples remain archived.
Record actual usable counts; short and duplicate responses can differ by target.

Quality-only runs explicitly omit rollout verification: Qwen3.5's recurrent
state cannot be rolled back using the existing KV-only crop verifier. They
report verification as null, not successful. No latency qualification is intended.

Status: bounded 40-update Modal smoke completed in 116.2 seconds, with finite
training and no target gradients. It has 104,876,032 trainable parameters.
Chat-formatted teacher smoke passed: 23 usable train and nine validation responses
in 13.6 seconds. Sampled prose/translation improve in coherence, but a code sample
still repeats the partial function; this is not a semantic-quality benchmark.
Full preparation completed in 623.1 seconds: 6390 usable training responses
(2173 prose, 2054 code, 2163 translation), and 184 validation responses
(63/57/64). Dropped 211 short and seven duplicate responses. The balanced 6144
training subset is now fitting; the 192 test prompts remain unlabelled. Raw
preparation was deliberately stopped.
The first chat preparation failed before generation because the tokenizer returned
a BatchEncoding; explicit `return_dict=False` fixes serialization. No trained
2B generalization conclusion yet. The smoke tests loading, feature extraction, gradients
and held-out scoring; it cannot establish generalization or adequate training.

The planned fit uses a fixed 20000-update global cosine schedule. Bounded-worker
continuations restore weights, optimizer and sampler/RNG state while retaining
this global schedule (`schedule_steps=20000`); the default historical continuation
behavior remains a new cycle. Exact split-versus-uninterrupted training is tested.

Prompt audit: all 6984 ordered source groups/splits/domains match the 0.5B
manifest, with zero cross-split group overlap. The text audit finds 6440 exact source-text
round trips and 544 with trailing whitespace removed by the official chat template;
there are zero other text mismatches. This whitespace normalization, which can
affect code continuation, is another declared prompt-format difference. See
[identity audit](../results/dflash-qwen35-2b-prompt-audit.json).

Training is running as `dflash-qwen35-2b-fit-01`, with 104,876,032 trainable
parameters and 295691 distinct response-token positions in the selected 6144
responses. At early checkpoints the 2B setup is not ahead; do not treat this
as a final result or stop the fixed-budget fit based on that observation.
`scripts/plot_dflash_targets.py` compares shared measured update counts and,
when final reports are available, validation-selected heads on shared source
prompts. It marks incomplete curves and refuses to overwrite input reports.

A bounded real-model teacher audit sampled one train and validation response per
domain (six total). Full-forward argmax agrees with all sampled teacher tokens;
cached feature cosine is 0.999896–0.999930, and perturbing future tokens changes
prior features by exactly zero. This reduces concern about a gross cache/label
mismatch; it is not an exhaustive corpus audit. See
[teacher consistency](../results/dflash-qwen35-2b-teacher-audit-01.json).
