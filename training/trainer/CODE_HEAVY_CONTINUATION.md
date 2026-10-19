# Additional 5B tokens with coding doubled

The requested phase targets another 5,000,000,000 loss-bearing tokens on the
original 12 sources. Coding rises from 30% to 60%; all other shares are multiplied
by exactly 4/7, preserving their relative proportions. Integer token allocations
use largest-remainder rounding and sum exactly to 5B. See the machine-readable
[plan](../plans/code-heavy-continuation-5b-20261019.json).

| Category | New token share | Phase target |
|---|---:|---:|
| Python-heavy code | 60% | 3B |
| Educational web | 5.714286% | ~285.7M |
| Mathematics | 11.428571% | ~571.4M |
| Scholarly papers | 8.571429% | ~428.6M |
| Books/textbooks | 14.285714% | ~714.3M |

The proposed parent is the latest EU-trained checkpoint:
`simple-llm-124m-eu-onepass-20261018-s02/step-000003838.pt`, whose lifetime count is
5,878,591,538. Completing the new phase would reach approximately 10.879B lifetime
tokens. Starting instead from the original 5B checkpoint is a separate option.
No model weights or data have been modified, and this phase is not launched.

## Data decision

The original cached corpus has 787,568,652 unique coding tokens. It already
contributed 1,499,987,929 coding exposures to the original run. The new 3B coding
target would require 3.809 additional effective coding passes, or 5.714 cumulative
passes if the same cache is reused.

The recorded pretraining plan says: "Initial policy: at most two passes per source;
expand eligible sources or explicitly revise the plan if capacity cannot support
a target share." [Source](../plans/SCHOLARLY_CODE_PRETRAINING.md).

Two execution choices are awaiting the user's preference:

- Expand with fresh data from the same source mix and maintain the two-pass limit.
  A conservative fresh-only phase needs at least 1.5B new eligible coding tokens
  plus 1B new tokens across the other categories. Prior training hashes must be
  excluded and existing evaluation splits retained. Upstream capacity must be
  measured; it is not established by this plan.
- Reuse the immutable cached corpus with an explicit phase repetition override.
  Retain source-specific pass caps and exposure accounting; record the old/new
  limits in the derived manifest. Existing default two-pass behavior must remain.
  This avoids redoing preparation but changes the earlier repetition policy.

## Checkpoint and evaluation

The optimizer-preserving migration helper verifies architecture/GPU count,
tokenizer, framework, parameter ordering, finite model and finite Adam moments.
It resets the phase cursor/counters and retains nested checkpoint lineage so
lifetime counters include both the original and EU phases. Strict ordinary resume
checks remain unchanged. Parent files are immutable.

Proposed execution retains eight H100s, 8K context, global batch 32, compiled
microbatch two, padding cropping and cached RoPE. A constant 3e-5 learning rate
continues the last phase's rate; this is a conservative proposal, not a tuned
schedule. Fresh-process two-step restore qualification is required before the
full run. Actual trained counts, source shares, throughput and runtime must be
measured, not inferred from the token allocation.

Reuse the already prepared 7,620,187-target / 2,782-family validation sample from
`simple-llm-eu-expanded-validation-20261019`. Compare the new checkpoint against
both the original 5B model and the EU checkpoint using the same FP32 scoring
protocol, including ECB/EMA retention. Private checkpoints and samples stay on
Modal; public HF publication is a separate action.
