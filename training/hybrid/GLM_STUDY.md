# KDA comparison on the large hybrid branch's prepared corpus

This study compares original KDA, KDA without short convolutions, and fused RoPE
without short convolutions, with seeds 17 and 29. Each run targets 999,997,440
input positions (122,070 updates), using batches of eight 1,024-token windows.

The data loader and recipe come from `claude/hybrid-720m-40b-01-run`. They read
existing GLM-tokenized shards on `lm-exploration-hybrid-data`: 70% ClimbMix,
20% licensed code, and 10% math. Training follows the branch's deterministic
weighted loader and shard order. Validation uses 256 fixed windows from its
separate held-out shards, with the same weighting. Preparation checks the existence
and sizes of all shards needed for the full budget before any GPU worker starts.

The GLM vocabulary differs from the earlier 32K-tokenizer pilot, so these are new
runs from matched initialization. Their losses are not directly comparable to that
pilot. To retain the 100M parameter class with GLM's large embedding table, the
model uses 12 layers, width 448, seven 64-wide query/KDA heads, one KV head, and
MLP width 1152. The KDA/KDA/KDA/full pattern is unchanged. All variants retain the
same optimizer, data order, loss, and initialization of shared parameters.

Training uses AdamW with peak LR 3e-4, 400-update warmup and cosine decay to 3e-5,
betas (0.9, 0.95), weight decay 0.1, and gradient clipping at 1. Evaluation is
every 512 updates, checkpoints every 2,048 and at completion. State and metrics
are preserved on `lm-exploration-training`; the independent remote coordinator
records terminal failures and completion. The independent deadline supervisor
cancels all tracked calls ten hours from launch, including queued time. GPU
workers have a nine-hour execution limit and save partial checkpoints before
shutdown. Original checkpoints and the failed continuation's evidence remain.

Run `python launch_kda_branch.py` from `training/`. The current run is
`kda-glm-1b-1793107241`; its artifact folder is `training/results/` under the
private HF dataset `jfrick/lm-exploration-results`. Detailed runtime provenance
and source hashes live in that folder's launch manifest. Use
`python collect_kda_continuation.py <run>` to collect and archive durable metrics.
