# Drafter quality for frozen Qwen3.5-2B

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

Implementation checks: bounded 40-update Modal smoke completed in 116.2 seconds, with finite
training and no target gradients. It has 104,876,032 trainable parameters.
Chat-formatted teacher smoke passed: 23 usable train and nine validation responses
in 13.6 seconds. The inspected prose/translation samples are coherent, but a code sample
still repeats the partial function. Raw and chat smoke samples use different
prompt subsets: this is neither a paired formatting ablation nor a semantic-quality
benchmark. The official template is the intended instruction-model interface.
Full preparation completed in 623.1 seconds: 6390 usable training responses
(2173 prose, 2054 code, 2163 translation), and 184 validation responses
(63/57/64). Dropped 211 short and seven duplicate responses. The balanced 6144
training subset completed 20000 updates; the 192 test prompts remain unlabelled. Raw
preparation was deliberately stopped.
The first chat preparation failed before generation because the tokenizer returned
a BatchEncoding; explicit `return_dict=False` fixes serialization. These smoke
checks validate loading, feature extraction, gradients and scoring. The completed
trained-model results below provide the quality evidence.

The completed fit used a fixed 20000-update global cosine schedule. Bounded-worker
continuations restore weights, optimizer and sampler/RNG state while retaining
this global schedule (`schedule_steps=20000`); the default historical continuation
behavior remains a new cycle. Exact split-versus-uninterrupted training is tested;
no continuation was needed for this completed fit.

Prompt audit: all 6984 ordered source groups/splits/domains match the 0.5B
manifest, with zero cross-split group overlap. The text audit finds 6440 exact source-text
round trips and 544 with trailing whitespace removed by the official chat template;
there are zero other text mismatches. This whitespace normalization, which can
affect code continuation, is another declared prompt-format difference. See
[identity audit](../results/dflash-qwen35-2b-prompt-audit.json).

Training completed as `dflash-qwen35-2b-fit-01`, with 104,876,032 trainable
parameters and 295691 distinct response-token positions in the selected 6144
responses. The 2B setup was behind at early checkpoints; the full fixed budget
was completed before drawing the final conclusion.
`scripts/plot_dflash_targets.py` compares shared measured update counts and
validation-selected heads on shared source
prompts. It marks incomplete curves and refuses to overwrite input reports.

A bounded real-model teacher audit sampled one train and validation response per
domain (six total). Full-forward argmax agrees with all sampled teacher tokens;
cached feature cosine is 0.999896–0.999930, and perturbing future tokens changes
prior features by exactly zero. This reduces concern about a gross cache/label
mismatch; it is not an exhaustive corpus audit. See
[teacher consistency](../results/dflash-qwen35-2b-teacher-audit-01.json).

The final fitting audit also shows unverified seven-token proposals from the
selected checkpoint: first validation response per domain, at its first anchor
and up to 16 response tokens later. This fixed selection is illustrative, not a
new benchmark or verified generation. Teacher tokens, draft tokens, supplied
anchor and context tail are retained together.

[Halfway curve](../results/dflash-qwen35-2b-halfway-comparison.png) and
[raw snapshot](../results/dflash-qwen35-2b-halfway.json) are explicitly incomplete.
At 10000 updates, validation NEW matches are 0.136 for 2B versus 0.313 for the
0.5B baseline. Both preparation and training package hashes match committed
source `4d3fa54`; see [launch audit](../results/dflash-qwen35-2b-launch-audit.json).

Training completed in 2505.4 seconds, with 3.36M supervised NEW-token exposures,
no target gradients, and verification explicitly unrun. Selected step: 16750.
On the 184 shared validation source prompts, selected NEW matches are 0.17017
for 2B versus 0.38658 for 0.5B. All three domains are worse for this 2B setup.
The full-training-set fitting audit completed: selected train/validation NEW
matches are 0.17250/0.17017, and final matches are 0.17537/0.16789.
Both validation audits match the original curves exactly; saved sampler state
replay and the downloaded checkpoint checksum also match. [Completed curves](../results/dflash-qwen35-2b-comparison.png).

Completed interpretation and artifacts: [findings](../results/dflash-qwen35-2b-findings.md).
All jobs ended; conservative reservations total $258.25/$500, not billed usage.
