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
without truncation or chat wrapping. Retain original groups/splits and re-audit
cross-split prompt duplication. Generate fresh 2B continuations/features. This
uses raw continuation prompts, not instruction-template evaluation of this model.
Record actual usable counts; short and duplicate responses can differ by target.

Quality-only runs explicitly omit rollout verification: Qwen3.5's recurrent
state cannot be rolled back using the existing KV-only crop verifier. They
report verification as null, not successful. No latency qualification is intended.

Status: implementation and bounded 40-update Modal smoke underway. No trained
2B quality conclusion yet. The smoke tests loading, feature extraction, gradients
and held-out scoring; it cannot establish generalization or adequate training.
