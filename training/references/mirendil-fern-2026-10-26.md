# Mirendil Fern 0.1 and 10M-token context

Reviewed 2026-10-27. [Introducing Fern](https://mirendil.com/news/introducing-fern/)
(Mirendil, 2026-10-26). Figures below are read from the data the post's charts are
drawn from. All results are Mirendil's; we have not reproduced them. For the
pretraining post on Mirendil-Base, see [the 2026-10-21 note](lab-training-2026-10-21.md).

## What the post establishes

Fern 0.1 is Mirendil-Base, pretrained from scratch, after "a short round of
post-training". Its architecture, training and serving support up to 10M tokens of
context (about 23 MB of code). The post gives sizes only as ranges:

| Model | Parameters | Status |
| --- | --- | --- |
| Fern 0.1 (Mirendil-Base) | 0.1–0.5T | Released; access "coming soon", early access by request |
| Mirendil-Base-Large | 0.5–2T | In pretraining, 80%+ |
| Mirendil-Base-XL | 2–10T | In pretraining, 20%+ |

The post does not disclose the architecture, how the long context is achieved,
the token count or the post-training recipe. Weights and license are not mentioned.

### Bits per byte against context length

Mirendil concatenates its own repositories into one 42.9 MB stream and scores it
with overlapping sliding windows of W bytes. Each window advances W/8 and scores
only its newest W/8 bytes. Context is measured in bytes so tokenizers compare
fairly. Selected points (BPB, lower is better; "context" is the mean context behind
each scored byte):

| Window | Mean context | Fern (base) | Kimi-K3 | Beam | GLM-5.3 | DeepSeek-V4-Pro-Base | Nemotron 3 Ultra Base | Inkling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 KiB | 30.7 KB | 0.2456 | 0.2183 | 0.2398 | 0.2644 | 0.2693 | 0.2657 | 0.3176 |
| 1 MiB | 0.97 MB | 0.1647 | 0.1400 | 0.1595 | 0.1703 | 0.1697 | 0.1760 | 0.2292 |
| 3.5 MiB | 3.30 MB | 0.1589 | 0.1395 | 0.1565 | 0.1643 | 0.1627 | 0.1719 | 0.2386 |
| 23 MiB | 16.7 MB | 0.1552 | — | — | — | — | — | — |

Kimi-K3, GLM-5.3, Beam and Inkling are post-trained models, which the post says
usually score slightly worse than their bases. Other models were run up to about 1M
tokens; Fern continues to about 10.2M tokens.

- Fern's BPB falls 37% from a 32 KiB to a 23 MiB window, but most of the gain is
  before 1 MiB: 4 MiB to 23 MiB improves it only from 0.1584 to 0.1552.
- At every matched window, Kimi-K3 and Beam score lower than Fern. Kimi-K3 at
  3.5 MiB (0.1395) beats Fern at 23 MiB.
- Inkling gets worse beyond a 512 KiB window (0.2267 to 0.2386): an advertised
  window is not evidence that a model uses it.
- The text is Mirendil's own code. The post does not say whether it was excluded
  from Mirendil-Base's training data.

### RULER on Mirendil-Base

Standard RULER setup: answer prefix, greedy decoding, thinking off, exact match.
100 examples per cell, except 20 at 10M.

| Task | 32K | 128K | 512K | 1M | 2M | 4M | 10M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single needle | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Multi-key retrieval | 100 | 95 | 98 | 98 | 100 | 98 | 90 |
| Multi-key, all needles | 100 | 100 | 100 | 100 | 95 | 100 | 100 |
| Variable tracking (recall) | 100 | 100 | 99 | 95 | 83 | 75 | 100 |
| Multi-hop QA (HotpotQA) | 65 | 62 | 65 | 65 | 62 | 60 | 45 |

Retrieval holds to 10M. Variable tracking falls to 75 at 4M, and multi-hop QA drops
to 45 at 10M. With 20 examples, each 10M cell has a 95% interval of roughly
±20 points, so the rise to 100 on variable tracking at 10M is not evidence of
recovery.

### Agentic benchmarks

Pass rate (%). Fern was run by Mirendil with OpenCode, one attempt. Other scores
come from model cards and leaderboards with different harnesses (mini-swe-agent,
Codex, Claude Code), effort levels and attempt counts, and some were run by Mirendil.
The rows are not a controlled comparison.

| Benchmark | Fern 0.1 | Beam | Inkling | Nemotron 3 Ultra | GLM-5.3 | Kimi-K3 | GPT-6 Astra | Claude Opus 5.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSWE v1.1 | 44.2 | 43.3 | 5.0 | 3.5 | 66.9 | 67.3 | 74.1 | 74.2 |
| Terminal Bench 4.0 | 6.1 | 3.0 | 1.0 | 0.0 | 41.9 | 12.6 | 57.9 | 66.4 |
| SWE Atlas Codebase QnA | 53.2 | 31.8 | 21.2 | 16.1 | 59.4 | 66.1 | 59.1 | 66.4 |
| SWE-Bench Pro v2-Hard | 76.5 | 76.5 | 56.9 | 31.4 | 84.3 | 88.2 | 90.2 | 100.0 |
| SWE-Bench Verified | 73.4 | 81.6 | 77.6 | 70.7 | 95.4 | 93.4 | 86.0 | 98.2 |

Fern scores at or above Beam, Inkling and Nemotron 3 Ultra on four of five rows
(SWE-Bench Verified is the exception) and below GLM-5.3 and Kimi-K3 on every row. SWE-Bench Pro v2-Hard values are all multiples of
1/51, consistent with a 51-task subset, so one task moves a score by about two points.
Codebase QnA, the task closest to "read a whole repository", is where Fern is
furthest ahead of Beam, Inkling and Nemotron.

## What this means for our models

Our models are the 124M Simple LLM (trained at 8K tokens) and the planned
[~720M KDA/full-attention hybrid](../plans/HYBRID_720M.md); Fern is 100B–500B
parameters. Its results say nothing directly about models at our scale, and the post
discloses no technique we could adopt. The useful part is the evaluation design.
The following are our proposals, not lab claims or changes to the plans:

| Priority | Proposal | What would count as useful evidence? |
| --- | --- | --- |
| First | Add a context-length BPB curve to [`analysis/evaluation`](../analysis/evaluation/README.md), using Mirendil's protocol: byte-sized windows, stride W/8, scoring only the newest W/8 bytes, x-axis as mean context in bytes. It fits the [long-context plan](../plans/SIMPLE_LLM_LONG_CONTEXT.md)'s "held-out loss at matched lengths". | A curve per checkpoint from about 1 KB to the training context and beyond it, on held-out long code files and repositories excluded from training. A curve that flattens or rises (as Inkling's does) marks the context the model actually uses. |
| First | Use bytes, not tokens, for context in every comparison between the 124M model (32K tokenizer) and the hybrid (GLM tokenizer). | Same byte windows and held-out text for both tokenizers; report bytes per token alongside. |
| Next | Gate the 8K→16K extension and the hybrid's KDA layout on synthetic retrieval, not just loss: RULER-style single needle, multi-key and variable tracking at 2K–32K. Skip multi-hop QA, which needs capability our models lack. | At least 100 examples per cell, with confidence intervals; compare the 124M baseline against the hybrid at equal training tokens, including lengths beyond those trained. |
| Later | When we compare against other models, run them in our own harness. | Same prompts, harness, attempts and scorer for every model; record anything taken from external tables as such. |

The post supports what our plans already assume: accepting a long input is not
using it, and BPB gains from context flatten. It does not justify changing our
context targets, adopting a specific long-context architecture or launching
training.
